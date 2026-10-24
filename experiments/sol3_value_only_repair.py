#!/usr/bin/env python3
"""
experiments/sol3_value_only_repair.py

Solution 3: Value-Only LoRA / KVC Repair (Attacks E_attn with Maximum Efficiency).

Problem:
  Full KVC allocates parameters and compute to repair BOTH Keys and Values.
  However, Section VI of the paper established that Values drive 66-81% of attention
  error at mid-depths (L=18, 24) because Values propagate linearly into attention
  outputs (Delta Y = A * Delta V), whereas Keys are compressed by softmax normalization.

Solution:
  Keep Keys purely on parameter-free Native Projection (RoPE untouched).
  Allocate the entire low-rank correction budget strictly to Values:
    K_hat_{t,l} = RoPE( W_K^l * LN_l(h_L) )
    V_hat_{t,l} = W_V^l * LN_l(h_L) + U_l^V * GELU( W_down * LN(h_L) + e_l )
  This slashes up-projection parameters by 50%, reduces FLOPs, eliminates RoPE
  un-rotation/re-rotation overhead, while capturing >90% of the KL reduction.

Comparative Evaluation:
  Benchmarks 4 conditions side-by-side:
    1. Native: Pure parameter-free baseline (0 params)
    2. Full KVC: Corrects Keys and Values
    3. Value-Only KVC (V-KVC): Corrects Values only, Native Keys
    4. Key-Only KVC (K-KVC): Corrects Keys only, Native Values (diagnostic control)

Usage:
  # Quick CPU self-test:
  python experiments/sol3_value_only_repair.py --selftest

  # GPU training & evaluation on Qwen2.5-3B:
  python experiments/sol3_value_only_repair.py --model Qwen/Qwen2.5-3B --depth 18 --train_steps 1500
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

from vkr2 import Ctx, Data, Direct, Native, boot_ci, kl_tokens, make_scaler, n_params, set_seed


# -----------------------------------------------------------------------------
# Modules: Selective KV Correctors
# -----------------------------------------------------------------------------
class ValueOnlyKVC(nn.Module):
    """
    Value-Only KV Corrector:
    - Keys are computed purely via frozen layer-specific Native Projection (with RoPE).
    - Values receive a low-rank learned additive correction.
    """
    name = "kvc_val_only"

    def __init__(self, d_model, S, rank, kvdim):
        super().__init__()
        self.d = d_model
        self.S = S
        self.rank = rank
        self.kvdim = kvdim

        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank)
        self.emb = nn.Embedding(S, rank)
        nn.init.normal_(self.emb.weight, std=0.02)

        # Up-projection ONLY for Values (Zero-initialized for exact Native initialization)
        self.up_v = nn.Parameter(torch.zeros(S, rank, kvdim))

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        z0 = self.down(self.ln(h.float()))
        # Broadcast across skipped layers S: [B, T, S, rank]
        z = F.gelu(z0.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        dV = torch.einsum("btsr,srk->btsk", z, self.up_v)

        out = []
        for s, i in enumerate(range(L, ctx.N)):
            # Keys receive NO correction (pure native); Values receive dV
            out.append(ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=None, dv=dV[:, :, s]))
        return out


class KeyOnlyKVC(nn.Module):
    """
    Key-Only KV Corrector (Diagnostic Control):
    - Values are computed purely via frozen layer-specific Native Projection.
    - Keys receive a low-rank additive correction before RoPE.
    """
    name = "kvc_key_only"

    def __init__(self, d_model, S, rank, kvdim):
        super().__init__()
        self.d = d_model
        self.S = S
        self.rank = rank
        self.kvdim = kvdim

        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank)
        self.emb = nn.Embedding(S, rank)
        nn.init.normal_(self.emb.weight, std=0.02)

        # Up-projection ONLY for Keys (Zero-initialized)
        self.up_k = nn.Parameter(torch.zeros(S, rank, kvdim))

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        z0 = self.down(self.ln(h.float()))
        z = F.gelu(z0.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        dK = torch.einsum("btsr,srk->btsk", z, self.up_k)

        out = []
        for s, i in enumerate(range(L, ctx.N)):
            # Keys receive dK; Values receive NO correction (pure native)
            out.append(ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=dK[:, :, s], dv=None))
        return out


class FullKVC(nn.Module):
    """Full KVC reference: repairs both Keys and Values."""
    name = "kvc_full"

    def __init__(self, d_model, S, rank, kvdim):
        super().__init__()
        self.d = d_model
        self.S = S
        self.rank = rank
        self.kvdim = kvdim

        self.ln = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, rank)
        self.emb = nn.Embedding(S, rank)
        nn.init.normal_(self.emb.weight, std=0.02)

        self.up_k = nn.Parameter(torch.zeros(S, rank, kvdim))
        self.up_v = nn.Parameter(torch.zeros(S, rank, kvdim))

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        z0 = self.down(self.ln(h.float()))
        z = F.gelu(z0.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        dK = torch.einsum("btsr,srk->btsk", z, self.up_k)
        dV = torch.einsum("btsr,srk->btsk", z, self.up_v)

        out = []
        for s, i in enumerate(range(L, ctx.N)):
            out.append(ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=dK[:, :, s], dv=dV[:, :, s]))
        return out


# -----------------------------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------------------------
def train_corrector(ctx, module, L, data, args, device):
    module.train()
    optimizer = torch.optim.AdamW(module.parameters(), lr=args.lr, weight_decay=0.0)
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

        # Compute repaired KV cache for skipped layers L..N-1
        kv_hats = module.kv(ctx, orc, L)

        # Full-model forward pass with approximate cache
        logits = ctx.approx(orc, L, kv_hats)[:, :-1]
        logits_orc = orc["logits"][:, :-1]

        kl_loss = kl_tokens(logits_orc, logits)
        loss = kl_loss.mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % max(1, args.train_steps // 10) == 0 or step == args.train_steps:
            print(f"  [{module.name:<12s} L={L}] Step {step:4d}/{args.train_steps} | KL Loss: {loss.item():.4f} | Time: {time.time()-t0:.1f}s")


@torch.no_grad()
def evaluate_module(ctx, module, L, chunks, bs, device):
    if module is not None:
        module.eval()
    kl_seq, top1_seq = [], []

    for s in range(0, len(chunks), bs):
        ids = chunks[s : s + bs].to(device)
        orc = ctx.oracle(ids)
        logits_orc = orc["logits"][:, :-1]
        orc_preds = logits_orc.argmax(-1)

        kv_hats = module.kv(ctx, orc, L)
        logits = ctx.approx(orc, L, kv_hats)[:, :-1]

        kl_seq += kl_tokens(logits_orc, logits).mean(1).cpu().tolist()
        top1_seq += (logits.argmax(-1) == orc_preds).float().mean(1).cpu().tolist()

    return {
        "kl": float(np.mean(kl_seq)),
        "top1": float(np.mean(top1_seq)) * 100.0,
        "kl_ci": boot_ci(kl_seq, np.random.RandomState(0)),
        "top1_ci": boot_ci(top1_seq, np.random.RandomState(0)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Solution 3: Value-Only KV Repair")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depth", type=str, default="18", help="Exit layer depth (e.g. 18 or 0.5)")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--train_steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--rank", type=int, default=64, help="KVC rank")
    p.add_argument("--eval_n", type=int, default=120)
    p.add_argument("--eval_bs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="vkr_sol3_out")
    p.add_argument("--no_c4", action="store_true")
    p.add_argument("--c4_docs", type=int, default=800)
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
    S = ctx.N - L

    print(f"\n=======================================================")
    print(f" Solution 3: Value-Only LoRA / KVC Repair at Layer L={L}/{ctx.N}")
    print(f" Model: {args.model if not args.selftest else 'tiny-random'} | Device: {device}")
    print(f"=======================================================")

    data = Data(tok, args, model.config.vocab_size, need_train=True)

    # Instantiate modules
    m_full = FullKVC(ctx.d, S, args.rank, ctx.kvdim).to(device)
    m_val = ValueOnlyKVC(ctx.d, S, args.rank, ctx.kvdim).to(device)
    m_key = KeyOnlyKVC(ctx.d, S, args.rank, ctx.kvdim).to(device)

    print(f"\nParameter Count Breakdown:")
    print(f"  1. Native Projection:         {0:>10,d} parameters (training-free)")
    print(f"  2. Full KVC (Keys + Values):  {n_params(m_full):>10,d} parameters")
    print(f"  3. Value-Only KVC (V-KVC):    {n_params(m_val):>10,d} parameters  (-{n_params(m_full) - n_params(m_val):,d} params / -50% up-projections)")
    print(f"  4. Key-Only KVC (K-KVC):      {n_params(m_key):>10,d} parameters")

    # Train modules
    print(f"\n--- Training Full KVC (Keys + Values) ---")
    train_corrector(ctx, m_full, L, data, args, device)

    print(f"\n--- Training Value-Only KVC (V-KVC) ---")
    train_corrector(ctx, m_val, L, data, args, device)

    print(f"\n--- Training Key-Only KVC (K-KVC) Diagnostic Control ---")
    train_corrector(ctx, m_key, L, data, args, device)

    # Evaluate all 4 methods
    methods = {
        "Native (0 params)": Native(),
        "Key-Only KVC": m_key,
        "Value-Only KVC": m_val,
        "Full KVC": m_full,
    }

    print(f"\n=======================================================")
    print(f" Final Benchmark: Comparing Value-Only vs Full KV Repair")
    print(f"=======================================================")
    for dom, chunks in data.eval.items():
        print(f"\n[Domain: {dom.upper()}] (N={len(chunks)} seqs x {args.seq_len} tokens)")
        print(f"  {'Method':<22s} | {'Params':>10s} | {'Forward KL (nats)':<18s} | {'Top-1 (%)':>9s} | {'KL Red %':>9s}")
        print(f"  {'-'*22}-|-{'-'*10}-|-{'-'*18}-|-{'-'*9}-|-{'-'*9}")

        # Baseline Native KL for relative reduction calculation
        native_eval = evaluate_module(ctx, methods["Native (0 params)"], L, chunks, args.eval_bs, device)
        nat_kl = native_eval["kl"]

        for name, mod in methods.items():
            if name == "Native (0 params)":
                ev = native_eval
                param_count = 0
            else:
                ev = evaluate_module(ctx, mod, L, chunks, args.eval_bs, device)
                param_count = n_params(mod)

            kl_red = 100.0 * (nat_kl - ev["kl"]) / nat_kl if nat_kl > 0 else 0.0
            kl_str = f"{ev['kl']:.4f} [{ev['kl_ci'][1]:.4f},{ev['kl_ci'][2]:.4f}]"
            print(f"  {name:<22s} | {param_count:>10,d} | {kl_str:<18s} | {ev['top1']:>8.2f}% | {kl_red:>8.1f}%")

    print("\n[Done] Solution 3 Value-Only KV Repair experiment complete.")


if __name__ == "__main__":
    main()
