#!/usr/bin/env python3
"""
experiments/sol2_residual_unembed.py

Solution 2: Residual Unembedding Adapter (Attacks E_readout).

Problem:
  Standard MLP adapters (Tuned Lens) try to learn an unconstrained mapping
  from intermediate representation space to a 100k+ token vocabulary, causing
  slow convergence and plateaus around 41-55% Top-1.

Solution:
  Force the adapter to learn a low-rank structural *correction* to the final
  unembedding manifold:
    P_t = softmax( W_unembed * LN(h_L) + (alpha/r) * B * GELU(A * LN(h_L)) )
  with B initialized to exact zero. The model starts at the baseline zero-shot
  point and only learns the residual delta, drastically reducing the hypothesis
  space and speeding up convergence.

Usage:
  # Quick CPU self-test:
  python experiments/sol2_residual_unembed.py --selftest

  # GPU training & evaluation on Qwen2.5-3B:
  python experiments/sol2_residual_unembed.py --model Qwen/Qwen2.5-3B --depth 18 --train_steps 1000
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local imports work whether run from repo root or experiments/
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)
sys.path.insert(0, os.path.dirname(curr_dir))

from vkr2 import Ctx, Data, boot_ci, kl_tokens, make_scaler, n_params, set_seed


# -----------------------------------------------------------------------------
# Modules: Residual Unembedding Adapters
# -----------------------------------------------------------------------------
class ResidualLogitAdapter(nn.Module):
    """
    Low-rank residual delta directly on vocabulary logits:
      logits = W_unembed * LN_final(h_L) + (alpha/r) * B * Activation(A * LN(h_L))
    Initialized with B = 0 so training starts exactly at zero-shot readout.
    """
    def __init__(self, d_model, vocab_size, rank=64, alpha=16.0):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.rank = rank
        self.scale = alpha / rank

        self.ln = nn.LayerNorm(d_model)
        self.A = nn.Linear(d_model, rank, bias=False)
        self.B = nn.Linear(rank, vocab_size, bias=False)

        # Kaiming init for down-projection, exact zero init for up-projection
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, ctx, h_L):
        # 1. Base terminal readout from intermediate state
        base_logits = ctx.head(h_L)

        # 2. Low-rank residual correction
        delta = self.B(F.gelu(self.A(self.ln(h_L.float())))) * self.scale
        return base_logits + delta.to(base_logits.dtype)


class ResidualManifoldAdapter(nn.Module):
    """
    Low-rank manifold alignment in representation space prior to frozen unembedding:
      h_adj = h_L + W_up * SwiGLU(W_down * LN(h_L))
      logits = ctx.head(h_adj)
    Initialized with W_up = 0.
    """
    def __init__(self, d_model, rank=128):
        super().__init__()
        self.d_model = d_model
        self.rank = rank

        self.ln = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, rank, bias=False)
        self.up_proj = nn.Linear(d_model, rank, bias=False)
        self.down_proj = nn.Linear(rank, d_model, bias=False)

        nn.init.kaiming_uniform_(self.gate_proj.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.up_proj.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, ctx, h_L):
        x = self.ln(h_L.float())
        h_adj = h_L + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)).to(h_L.dtype)
        return ctx.head(h_adj)


# -----------------------------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------------------------
def train_adapter(ctx, adapter, L, data, args, device, name="Adapter"):
    adapter.train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = make_scaler(ctx.dtype == torch.float16 and device.type == "cuda", device)
    warmup_steps = min(50, args.train_steps // 10)

    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        if step <= warmup_steps:
            lr_scale = float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, args.train_steps - warmup_steps))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_scale

        ids = data.sample(args.batch).to(device)
        orc = ctx.oracle(ids)
        h_L = orc["hs"][L]

        logits_pred = adapter(ctx, h_L)[:, :-1]
        logits_orc = orc["logits"][:, :-1]

        loss = kl_tokens(logits_orc, logits_pred).mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % max(1, args.train_steps // 10) == 0 or step == args.train_steps:
            print(f"  [{name} L={L}] Step {step:4d}/{args.train_steps} | KL: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")


@torch.no_grad()
def evaluate_adapters(ctx, adapters_dict, L, chunks, bs, device):
    """
    Evaluates adapters against full oracle logits:
      - Forward KL (nats)
      - Top-1 Agreement (%)
    """
    results = {name: [] for name in adapters_dict}

    for s in range(0, len(chunks), bs):
        ids = chunks[s : s + bs].to(device)
        orc = ctx.oracle(ids)
        h_L = orc["hs"][L]
        logits_orc = orc["logits"][:, :-1]
        orc_preds = logits_orc.argmax(-1)

        # Direct Zero-shot (baseline)
        if "zero_shot" in adapters_dict:
            lg_zs = ctx.head(h_L)[:, :-1]
            kl_zs = kl_tokens(logits_orc, lg_zs).mean(1).cpu().tolist()
            t1_zs = (lg_zs.argmax(-1) == orc_preds).float().mean(1).cpu().tolist()
            results["zero_shot"].append((kl_zs, t1_zs))

        # Trained adapters
        for name, adapter in adapters_dict.items():
            if name == "zero_shot" or adapter is None:
                continue
            adapter.eval()
            lg_ad = adapter(ctx, h_L)[:, :-1]
            kl_ad = kl_tokens(logits_orc, lg_ad).mean(1).cpu().tolist()
            t1_ad = (lg_ad.argmax(-1) == orc_preds).float().mean(1).cpu().tolist()
            results[name].append((kl_ad, t1_ad))

    out = {}
    for name, batch_list in results.items():
        if not batch_list:
            continue
        all_kl = [x for b in batch_list for x in b[0]]
        all_t1 = [x for b in batch_list for x in b[1]]
        out[name] = {
            "kl": float(np.mean(all_kl)),
            "top1": float(np.mean(all_t1)) * 100.0,
            "kl_ci": boot_ci(all_kl, np.random.RandomState(0)),
            "top1_ci": boot_ci(all_t1, np.random.RandomState(0)),
        }
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Solution 2: Residual Unembedding Adapter")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depth", type=str, default="18", help="Exit layer depth (e.g. 18 or 0.5)")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--train_steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=64, help="Rank for residual adapter")
    p.add_argument("--eval_n", type=int, default=60)
    p.add_argument("--eval_bs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="vkr_sol2_out")
    p.add_argument("--no_c4", action="store_true")
    p.add_argument("--c4_docs", type=int, default=400)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.selftest else "cpu")
    os.makedirs(args.out, exist_ok=True)

    if args.selftest:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(
            vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False
        )
        model = Qwen2ForCausalLM(cfg).float().eval()
        tok = None
        args.depth = "4"
        args.train_steps = 30
        args.rank = 16
        args.eval_n, args.eval_bs, args.seq_len = 12, 4, 32
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        dt = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(device).eval()

    for param in model.parameters():
        param.requires_grad_(False)

    ctx = Ctx(model)
    L = int(round(float(args.depth) * ctx.N)) if "." in args.depth else int(args.depth)
    print(f"\n=======================================================")
    print(f" Solution 2: Residual Unembedding Adapter at Layer L={L}/{ctx.N}")
    print(f" Model: {args.model if not args.selftest else 'tiny-random'} | Device: {device}")
    print(f"=======================================================")

    data = Data(tok, args, model.config.vocab_size, need_train=True)

    # 1. Residual Logit Bypass Adapter
    adapter_logit = ResidualLogitAdapter(
        d_model=ctx.d,
        vocab_size=model.config.vocab_size,
        rank=args.rank
    ).to(device)
    print(f"1. Residual Logit Adapter (rank {args.rank}):    {n_params(adapter_logit):,} parameters")

    # 2. Residual Manifold Alignment Adapter
    adapter_manifold = ResidualManifoldAdapter(
        d_model=ctx.d,
        rank=args.rank * 2
    ).to(device)
    print(f"2. Residual Manifold Adapter (rank {args.rank*2}): {n_params(adapter_manifold):,} parameters")

    # Train both adapters
    print(f"\n--- Training Residual Logit Adapter ---")
    train_adapter(ctx, adapter_logit, L, data, args, device, name="Residual-Logit")

    print(f"\n--- Training Residual Manifold Adapter ---")
    train_adapter(ctx, adapter_manifold, L, data, args, device, name="Residual-Manifold")

    # Evaluate across domains
    adapters_dict = {
        "zero_shot": None,
        "residual_logit": adapter_logit,
        "residual_manifold": adapter_manifold,
    }

    print(f"\n--- Evaluating Readout Quality across Domains ---")
    for dom, chunks in data.eval.items():
        res = evaluate_adapters(ctx, adapters_dict, L, chunks, args.eval_bs, device)
        print(f"\n[Domain: {dom.upper()}] (N={len(chunks)} sequences x {args.seq_len} tokens)")
        print(f"  {'Method':<28s} | {'Forward KL (nats)':<18s} | {'Top-1 Agreement':<18s}")
        print(f"  {'-'*28}-|-{'-'*18}-|-{'-'*18}")
        for method, m_data in res.items():
            kl_str = f"{m_data['kl']:.4f} [{m_data['kl_ci'][1]:.4f}, {m_data['kl_ci'][2]:.4f}]"
            t1_str = f"{m_data['top1']:.2f}%"
            print(f"  {method:<28s} | {kl_str:<18s} | {t1_str:<18s}")

    print("\n[Done] Solution 2 Residual Unembedding experiment complete.")


if __name__ == "__main__":
    main()
