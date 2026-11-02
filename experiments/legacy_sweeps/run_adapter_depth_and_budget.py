#!/usr/bin/env python3
"""
experiments/run_adapter_depth_and_budget.py

Addresses Claim C2 & Gap A12:
- A12. Adapter Capacity at Other Depths and Extended Training Budget:
  1. Trains r=2048 high-capacity bottleneck MLP adapters at deeper layers:
     * L=24 (67% depth)
     * L=30 (83% depth)
  2. Runs extended training (10,000+ steps) at L=18 (50% depth).
  3. Evaluates validation PPL, KL divergence, and Top-1 agreement against Full-Depth baseline.

Proves:
Whether the 18% top-1 disagreement at L=18 is an asymptotic structural/geometric manifold ceiling
or merely an artifact of insufficient training steps / adapter underparameterization.

Usage:
  python experiments/run_adapter_depth_and_budget.py --model-id Qwen/Qwen2.5-3B --depths 18,24,30 --steps 10000 --device cuda
  python experiments/run_adapter_depth_and_budget.py --quick --device cpu
"""

import os
import sys
import math
import argparse
import csv
import json
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.qwen_dynamic import QwenDynamicEngine
from src.exit_adapter import ExitCalibrationModule
from src.eval_dataset import get_tokenized_wikitext2


def parse_args():
    parser = argparse.ArgumentParser(description="Train and Evaluate Adapters at Deeper Layers and Longer Budgets (A12)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--depths", type=str, default="18,24,30", help="Comma-separated depths")
    parser.add_argument("--rank", type=int, default=2048, help="Adapter bottleneck rank")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps (default: 10,000)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/adapter_depth_budget")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test with 10 steps")
    return parser.parse_args()


def train_and_eval_adapter_depth(
    engine: QwenDynamicEngine,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    depth: int,
    rank: int,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Trains a high-capacity adapter at depth `depth` for `steps` steps,
    then evaluates on held-out validation tokens.
    """
    d_model = engine.config.hidden_size
    adapter = ExitCalibrationModule(
        d_model=d_model,
        candidate_depths=[depth],
        adapter_type="mlp",
        rank=rank,
    ).to(device)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=0.01)
    criterion_kd = nn.KLDivLoss(reduction="batchmean")

    print(f"\n[*] Training r={rank} adapter at depth L={depth} for {steps} steps...")
    adapter.train()

    t_start = time.perf_counter()
    n_train = len(train_tokens) - seq_len - 1

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        # Random chunk
        st = np.random.randint(0, n_train)
        input_ids = train_tokens[st : st + seq_len].unsqueeze(0).to(device)

        # Forward through layers up to exit depth
        with torch.no_grad():
            h = engine.model.model.embed_tokens(input_ids)
            for l in range(depth):
                layer = engine.model.model.layers[l]
                residual = h
                norm_h = layer.input_layernorm(h)
                attn_out = layer.self_attn(norm_h, is_causal=True)[0]
                h = residual + attn_out
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            h_exit = h

            # Target logits from full model
            h_target = h
            for l in range(depth, engine.num_layers):
                layer = engine.model.model.layers[l]
                residual = h_target
                norm_h = layer.input_layernorm(h_target)
                attn_out = layer.self_attn(norm_h, is_causal=True)[0]
                h_target = residual + attn_out
                h_target = h_target + layer.mlp(layer.post_attention_layernorm(h_target))

            target_logits = engine.model.lm_head(engine.model.model.norm(h_target))

        # Pass h_exit through adapter
        h_adapted = adapter.forward_layer(h_exit, depth=depth)
        adapted_logits = engine.model.lm_head(engine.model.model.norm(h_adapted))

        # Loss: 0.5 CE + 0.5 KD
        target_toks = input_ids[:, 1:].contiguous()
        loss_ce = F.cross_entropy(adapted_logits[:, :-1, :].reshape(-1, engine.config.vocab_size), target_toks.reshape(-1))

        p_target = F.softmax(target_logits[:, :-1, :] / 2.0, dim=-1)
        logp_adapted = F.log_softmax(adapted_logits[:, :-1, :] / 2.0, dim=-1)
        loss_kd = criterion_kd(logp_adapted, p_target) * 4.0

        loss = 0.5 * loss_ce + 0.5 * loss_kd
        loss.backward()
        optimizer.step()

        if step % 500 == 0 or step == steps:
            print(f"  Step {step}/{steps} | Loss: {loss.item():.4f} (CE: {loss_ce.item():.4f}, KD: {loss_kd.item():.4f})")

    elapsed_train = time.perf_counter() - t_start

    # Validation
    adapter.eval()
    print(f"[*] Evaluating adapted readout on validation tokens...")

    n_val_chunks = 20
    val_max = len(val_tokens) - seq_len - 1
    val_step = max(1, val_max // n_val_chunks)

    total_kl = 0.0
    total_agreed = 0
    total_tokens = 0
    total_nll = 0.0

    with torch.no_grad():
        for i in range(n_val_chunks):
            st = i * val_step
            input_ids = val_tokens[st : st + seq_len].unsqueeze(0).to(device)

            # Target logits
            outputs_target = engine.model(input_ids)
            target_logits = outputs_target.logits[:, :-1, :]

            # Intermediate h_exit
            h = engine.model.model.embed_tokens(input_ids)
            for l in range(depth):
                layer = engine.model.model.layers[l]
                residual = h
                norm_h = layer.input_layernorm(h)
                attn_out = layer.self_attn(norm_h, is_causal=True)[0]
                h = residual + attn_out
                h = h + layer.mlp(layer.post_attention_layernorm(h))

            h_adapted = adapter.forward_layer(h, depth=depth)
            adapted_logits = engine.model.lm_head(engine.model.model.norm(h_adapted))[:, :-1, :]

            p_target = F.softmax(target_logits.float(), dim=-1)
            logp_target = F.log_softmax(target_logits.float(), dim=-1)
            logp_adapted = F.log_softmax(adapted_logits.float(), dim=-1)
            kl = (p_target * (logp_target - logp_adapted)).sum(dim=-1).mean().item()
            total_kl += kl

            pred_target = torch.argmax(target_logits, dim=-1)
            pred_adapted = torch.argmax(adapted_logits, dim=-1)
            agree = (pred_target == pred_adapted).float().mean().item()
            total_agreed += agree

            target_toks = input_ids[:, 1:].contiguous()
            nll = F.cross_entropy(adapted_logits.reshape(-1, engine.config.vocab_size).float(), target_toks.reshape(-1)).item()
            total_nll += nll
            total_tokens += 1

    val_kl = total_kl / total_tokens
    val_top1 = (total_agreed / total_tokens) * 100.0
    val_ppl = math.exp(min(total_nll / total_tokens, 15.0))

    return {
        "depth": depth,
        "rank": rank,
        "steps": steps,
        "val_kl": val_kl,
        "val_top1": val_top1,
        "val_ppl": val_ppl,
        "train_time_sec": elapsed_train,
        "adapter": adapter,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"=== Adapter Capacity, Depths & Budget Scaling (A12) on {args.model_id} ===")
    engine = QwenDynamicEngine(model_id=args.model_id, device=args.device)
    device = torch.device(args.device)

    depths = [int(d) for d in args.depths.split(",")]
    steps = 10 if args.quick else args.steps

    # Load tokens
    try:
        val_tokens = get_tokenized_wikitext2(engine.tokenizer, split="validation")
        train_tokens = get_tokenized_wikitext2(engine.tokenizer, split="train")
    except Exception as e:
        print(f"[!] Fallback to synthetic corpus: {e}")
        text = "Bottleneck adapters calibrate intermediate transformer representations into terminal vocabulary space. " * 60
        val_tokens = engine.tokenizer(text, return_tensors="pt")["input_ids"][0]
        train_tokens = val_tokens

    out_csv = os.path.join(args.output_dir, f"adapter_depth_budget_{args.model_id.replace('/', '_')}.csv")
    rows = []

    print(f"\n{'Depth':<6} | {'Rank (r)':<10} | {'Steps':<8} | {'Val PPL':<9} | {'Val KL (nats)':<14} | {'Top-1 (%)':<9}")
    print("-" * 72)

    for d in depths:
        res = train_and_eval_adapter_depth(
            engine=engine,
            train_tokens=train_tokens,
            val_tokens=val_tokens,
            depth=d,
            rank=args.rank,
            steps=steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            device=device,
        )

        print(f"L={d:<4} | {res['rank']:<10} | {res['steps']:<8} | {res['val_ppl']:9.2f} | {res['val_kl']:14.4f} | {res['val_top1']:8.2f}%")

        # Save checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"adapter_L{d}_r{args.rank}_{steps}steps.pt")
        torch.save(res["adapter"].state_dict(), ckpt_path)

        rows.append({
            "model_id": args.model_id,
            "depth": d,
            "rank": res["rank"],
            "steps": res["steps"],
            "val_ppl": round(res["val_ppl"], 2),
            "val_kl": round(res["val_kl"], 4),
            "val_top1": round(res["val_top1"], 2),
            "train_time_sec": round(res["train_time_sec"], 1),
            "checkpoint": ckpt_path,
        })

    fieldnames = ["model_id", "depth", "rank", "steps", "val_ppl", "val_kl", "val_top1", "train_time_sec", "checkpoint"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Adapter depth & budget scaling results saved to: {out_csv}")


if __name__ == "__main__":
    main()
