#!/usr/bin/env python3
"""
experiments/ablate_adapter_capacity.py

Milestone 2.5: Adapter Capacity Ceiling Ablation across Log-Spaced Ranks.
Evaluates MLP Bottleneck Adapters across ranks:
    r in {64, 128, 256, 512, 1024, 2048}
at intermediate exit depth L = 18 on Qwen/Qwen2.5-3B.

Core Scientific Goal:
Prove that the failure of intermediate early exit is a structural, informational
manifold limitation rather than underparameterization of the adapter.
"""

import os
import sys
import time
import math
import argparse
import csv
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.exit_adapter import ExitCalibrationModule
from src.qwen_dynamic import QwenDynamicEngine
from experiments.isolate_kv_ablation import (
    evaluate_qwen_sequence,
    prepare_eval_sequences,
    CANONICAL_TEST_TEXTS
)


def parse_args():
    parser = argparse.ArgumentParser(description="Milestone 2.5: Adapter Capacity Scaling Ablation")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Model ID")
    parser.add_argument("--exit-depth", type=int, default=18, help="Target exit depth to calibrate (default: 18)")
    parser.add_argument(
        "--ranks",
        type=str,
        default="64,128,256,512,1024,2048",
        help="Comma-separated ranks for MLP hidden bottleneck"
    )
    parser.add_argument("--steps", type=int, default=300, help="Training steps per rank")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--seq-len", type=int, default=64, help="Tokens evaluated per sequence")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/canonical_sweeps/qwen2.5_3b",
        help="Destination directory for CSV results"
    )
    return parser.parse_args()


def create_training_batches(tokenizer, batch_size: int, seq_len: int, device: torch.device):
    """Generates training batches cycling through canonical texts."""
    all_toks = []
    for text in CANONICAL_TEST_TEXTS:
        toks = tokenizer.encode(text)
        if isinstance(toks, torch.Tensor):
            toks = toks.tolist()
        all_toks.extend(toks)

    tok_tensor = torch.tensor(all_toks, dtype=torch.long, device=device)
    total_tokens = len(tok_tensor)

    offset = 0
    while True:
        batch_inputs = []
        batch_targets = []
        for _ in range(batch_size):
            if offset + seq_len + 1 > total_tokens:
                offset = 0
            chunk = tok_tensor[offset : offset + seq_len + 1]
            if len(chunk) < seq_len + 1:
                chunk = F.pad(chunk, (0, seq_len + 1 - len(chunk)), value=0)
            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])
            offset = (offset + seq_len) % (total_tokens - seq_len - 1)

        x_batch = torch.stack(batch_inputs, dim=0)
        y_batch = torch.stack(batch_targets, dim=0)
        yield x_batch, y_batch


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    ranks = [int(r.strip()) for r in args.ranks.split(",") if r.strip()]
    d_target = args.exit_depth

    print("=" * 80)
    print("MILESTONE 2.5: ADAPTER CAPACITY CEILING ABLATION")
    print(f"Model: {args.model_id} | Exit Depth: L = {d_target} | Device: {device}")
    print(f"Testing Capacity Ranks: r in {ranks}")
    print(f"Training Budget: {args.steps} steps per rank (batch_size={args.batch_size}, seq_len={args.seq_len})")
    print("=" * 80)

    # 1. Initialize engine
    print(f"[*] Initializing QwenDynamicEngine '{args.model_id}'...")
    engine = QwenDynamicEngine(
        model_id=args.model_id,
        device=args.device,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        candidate_layers=[d_target]
    )
    tokenizer = engine.tokenizer
    d_model = engine.hidden_size
    model_dtype = engine.dtype

    # Prepare eval sequences
    eval_seqs = prepare_eval_sequences("canonical", tokenizer, args.seq_len, num_samples=6, device=device)

    # Prepare training stream
    train_stream = create_training_batches(tokenizer, args.batch_size, args.seq_len, device)

    results = []

    for rank in ranks:
        print("\n" + "-" * 80)
        print(f"TRAINING & EVALUATING MLP BOTTLENECK ADAPTER AT RANK r = {rank}")
        print("-" * 80)

        adapter = ExitCalibrationModule(
            d_model=d_model,
            candidate_depths=[d_target],
            adapter_type="mlp",
            rank=rank
        ).to(device=device, dtype=model_dtype)

        trainable_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
        print(f"  -> Rank: {rank:<5} | Trainable Params: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)

        # Quick calibration loop
        adapter.train()
        t0 = time.perf_counter()
        final_loss = 0.0

        for step in range(1, args.steps + 1):
            x_b, y_b = next(train_stream)
            optimizer.zero_grad()

            with torch.no_grad():
                out = engine.model(input_ids=x_b, output_hidden_states=True, return_dict=True)
                h_exit = out.hidden_states[d_target]
                teacher_logits = out.logits

            calibrated_h = adapter.forward_layer(h_exit, depth=d_target)
            norm_cal = engine.model.model.norm(calibrated_h)
            logits = engine.model.lm_head(norm_cal)

            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y_b.reshape(-1))
            p_teacher = F.softmax((teacher_logits / 2.0).float(), dim=-1)
            log_p_student = F.log_softmax((logits / 2.0).float(), dim=-1)
            loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * 4.0

            loss = 0.5 * loss_ce + 0.5 * loss_kd
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()

            final_loss = loss.item()
            if step % 100 == 0 or step == args.steps:
                print(f"  [Step {step:3d}/{args.steps}] Loss: {final_loss:.4f} (CE: {loss_ce.item():.4f}, KD: {loss_kd.item():.4f})")

        train_time = time.perf_counter() - t0
        adapter.eval()

        # Teacher-Forced Validation Evaluation
        print(f"  [*] Evaluating teacher-forced validation perplexity at L = {d_target}...")
        ppl_samples = []
        kl_samples = []
        top1_samples = []

        for seq in eval_seqs:
            cond_res, _ = evaluate_qwen_sequence(engine, seq, d_target, device, adapter=adapter)
            # Evaluate readout-only with adapter
            ppl_samples.append(cond_res["readout_only"]["ppl"])
            kl_samples.append(cond_res["readout_only"]["mean_kl"])
            top1_samples.append(cond_res["readout_only"]["top1_acc"])

        mean_ppl = float(np.mean(ppl_samples))
        mean_kl = float(np.mean(kl_samples))
        mean_top1 = float(np.mean(top1_samples))

        print(f"  => RESULT (r={rank}): Val PPL = {mean_ppl:.2f} | KL = {mean_kl:.4f} nats | Top-1 = {mean_top1:.1f}%")

        results.append({
            "exit_depth": d_target,
            "rank": rank,
            "trainable_params": trainable_params,
            "train_loss": final_loss,
            "val_ppl": mean_ppl,
            "val_kl": mean_kl,
            "top1_acc": mean_top1,
            "train_time_sec": train_time
        })

    # Save to CSV
    csv_path = os.path.join(args.output_dir, "adapter_capacity_ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print("ACADEMIC PAPER TABLE: ADAPTER CAPACITY SATURATION ABLATION")
    print("=" * 80)

    md_table = f"""
### Table 4: Exit Adapter Capacity Saturation Ablation at $L = {d_target}$ (50% Relative Depth)

$$\\begin{{array}}{{cccccc}}
\\hline
\\textbf{{Capacity Rank }} (r) & \\textbf{{Parameters}} & \\textbf{{Train Loss}} & \\textbf{{Validation PPL}} & \\bar{{D}}_{{\\text{{KL}}}}\\text{{ (nats)}} & \\textbf{{Top-1 Match}} \\\\
\\hline
"""
    for r in results:
        md_table += f"{r['rank']} & {r['trainable_params']:,} & {r['train_loss']:.4f} & {r['val_ppl']:.2f} & {r['val_kl']:.4f} & {r['top1_acc']:.1f}\\% \\\\\n"

    md_table += """\\hline
\\end{array}$$
"""
    print(md_table)

    table_path = os.path.join(args.output_dir, "paper_table_adapter_capacity.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(md_table)

    print(f"[+] Saved master capacity CSV to: {csv_path}")
    print(f"[+] Saved markdown table to: {table_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
