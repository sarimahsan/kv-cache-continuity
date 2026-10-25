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
        dt = self.down.weight.dtype
        z0 = self.down(self.ln(h.to(dt)))
        # Broadcast across skipped layers S: [B, T, S, rank]
        z = F.gelu(z0.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        dV = torch.einsum("btsr,srk->btsk", z, self.up_v)

        out = []
        for s, i in enumerate(range(L, ctx.N)):
            # Keys receive NO correction (pure native); Values receive dV
            out.append(ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=None, dv=dV[:, :, s].to(h.dtype)))
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
        dt = self.down.weight.dtype
        z0 = self.down(self.ln(h.to(dt)))
        z = F.gelu(z0.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        dK = torch.einsum("btsr,srk->btsk", z, self.up_k)

        out = []
        for s, i in enumerate(range(L, ctx.N)):
            # Keys receive dK; Values receive NO correction (pure native)
            out.append(ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=dK[:, :, s].to(h.dtype), dv=None))
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
        dt = self.down.weight.dtype
        z0 = self.down(self.ln(h.to(dt)))
        z = F.gelu(z0.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        dK = torch.einsum("btsr,srk->btsk", z, self.up_k)
        dV = torch.einsum("btsr,srk->btsk", z, self.up_v)

        out = []
        for s, i in enumerate(range(L, ctx.N)):
            out.append(ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=dK[:, :, s].to(h.dtype), dv=dV[:, :, s].to(h.dtype)))
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
def evaluate_module(ctx, module, L, chunks, bs, device, compute_diagnostics=False):
    if module is not None and hasattr(module, "eval"):
        module.eval()
    kl_seq, top1_seq, nll_seq = [], [], []

    # Layer-wise stats over skipped layers L..N-1
    n_skipped = ctx.N - L
    cos_k_sum = torch.zeros(n_skipped, device=device)
    cos_v_sum = torch.zeros(n_skipped, device=device)
    err_v_sum = torch.zeros(n_skipped, device=device)
    total_samples = 0

    for s in range(0, len(chunks), bs):
        ids = chunks[s : s + bs].to(device)
        B, T = ids.shape
        orc = ctx.oracle(ids)
        logits_orc = orc["logits"][:, :-1]
        orc_preds = logits_orc.argmax(-1)

        kv_hats = module.kv(ctx, orc, L)
        logits = ctx.approx(orc, L, kv_hats)[:, :-1]

        # KL & Top-1
        kl_seq += kl_tokens(logits_orc, logits).mean(1).cpu().tolist()
        top1_seq += (logits.argmax(-1) == orc_preds).float().mean(1).cpu().tolist()

        # Cross-entropy / Perplexity against target tokens
        target_tokens = ids[:, 1:].contiguous()
        loss_batch = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_tokens.view(-1), reduction="none")
        nll_seq += loss_batch.view(B, -1).mean(1).cpu().tolist()

        # Compute layer-wise diagnostics if requested
        if compute_diagnostics:
            for idx, lyr in enumerate(range(L, ctx.N)):
                k_hat, v_hat = kv_hats[idx]
                k_orc, v_orc = orc["kv"][lyr]

                # Flatten across heads & hidden dim
                kh_f, ko_f = k_hat.float().reshape(B * T, -1), k_orc.float().reshape(B * T, -1)
                vh_f, vo_f = v_hat.float().reshape(B * T, -1), v_orc.float().reshape(B * T, -1)

                cos_k = F.cosine_similarity(kh_f, ko_f, dim=-1).mean()
                cos_v = F.cosine_similarity(vh_f, vo_f, dim=-1).mean()
                rel_v = (torch.norm(vo_f - vh_f, p=2, dim=-1) / (torch.norm(vo_f, p=2, dim=-1) + 1e-8)).mean()

                cos_k_sum[idx] += cos_k * B
                cos_v_sum[idx] += cos_v * B
                err_v_sum[idx] += rel_v * B
            total_samples += B

    diag = None
    if compute_diagnostics and total_samples > 0:
        diag = {
            "cos_k": (cos_k_sum / total_samples).cpu().tolist(),
            "cos_v": (cos_v_sum / total_samples).cpu().tolist(),
            "rel_err_v": (err_v_sum / total_samples).cpu().tolist(),
        }

    mean_nll = float(np.mean(nll_seq))
    ppl = math.exp(min(20.0, mean_nll))

    return {
        "kl": float(np.mean(kl_seq)),
        "top1": float(np.mean(top1_seq)) * 100.0,
        "ppl": ppl,
        "kl_ci": boot_ci(kl_seq, np.random.RandomState(0)),
        "top1_ci": boot_ci(top1_seq, np.random.RandomState(0)),
        "diag": diag,
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
    p.add_argument("--eval_only", action="store_true", help="Skip training and evaluate checkpoints if present")
    p.add_argument("--retrain", action="store_true", help="Force retrain even if checkpoints exist")
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
        dt = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16 if device.type == "cuda" else torch.float32
        device_map = "cuda" if device.type == "cuda" else None
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, device_map=device_map).eval()
        except TypeError:
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

    data = Data(tok, args, model.config.vocab_size, need_train=not args.eval_only)

    # Instantiate modules
    m_full = FullKVC(ctx.d, S, args.rank, ctx.kvdim).to(device)
    m_val = ValueOnlyKVC(ctx.d, S, args.rank, ctx.kvdim).to(device)
    m_key = KeyOnlyKVC(ctx.d, S, args.rank, ctx.kvdim).to(device)

    print(f"\nParameter Count Breakdown:")
    print(f"  1. Native Projection:         {0:>10,d} parameters (training-free)")
    print(f"  2. Full KVC (Keys + Values):  {n_params(m_full):>10,d} parameters")
    print(f"  3. Value-Only KVC (V-KVC):    {n_params(m_val):>10,d} parameters  (-{n_params(m_full) - n_params(m_val):,d} params / -50% up-projections)")
    print(f"  4. Key-Only KVC (K-KVC):      {n_params(m_key):>10,d} parameters")

    ckpt_full = os.path.join(args.out, f"kvc_full_L{L}.pt")
    ckpt_val = os.path.join(args.out, f"kvc_val_only_L{L}.pt")
    ckpt_key = os.path.join(args.out, f"kvc_key_only_L{L}.pt")

    if args.eval_only:
        for p_name, p_path in [("Full KVC", ckpt_full), ("Value-Only KVC", ckpt_val), ("Key-Only KVC", ckpt_key)]:
            if not os.path.exists(p_path):
                raise FileNotFoundError(f"--eval_only was specified, but checkpoint for {p_name} was not found at '{p_path}'. Please run without --eval_only to train first!")

    # Train or load modules
    if not args.retrain and os.path.exists(ckpt_full):
        print(f"Loading cached checkpoint: {ckpt_full}")
        m_full.load_state_dict(torch.load(ckpt_full, map_location=device))
    elif not args.eval_only:
        print(f"\n--- Training Full KVC (Keys + Values) ---")
        train_corrector(ctx, m_full, L, data, args, device)
        torch.save(m_full.state_dict(), ckpt_full)

    if not args.retrain and os.path.exists(ckpt_val):
        print(f"Loading cached checkpoint: {ckpt_val}")
        m_val.load_state_dict(torch.load(ckpt_val, map_location=device))
    elif not args.eval_only:
        print(f"\n--- Training Value-Only KVC (V-KVC) ---")
        train_corrector(ctx, m_val, L, data, args, device)
        torch.save(m_val.state_dict(), ckpt_val)

    if not args.retrain and os.path.exists(ckpt_key):
        print(f"Loading cached checkpoint: {ckpt_key}")
        m_key.load_state_dict(torch.load(ckpt_key, map_location=device))
    elif not args.eval_only:
        print(f"\n--- Training Key-Only KVC (K-KVC) Diagnostic Control ---")
        train_corrector(ctx, m_key, L, data, args, device)
        torch.save(m_key.state_dict(), ckpt_key)

    # Evaluate all 4 methods
    methods = {
        "Native (0 params)": Native(),
        "Key-Only KVC": m_key,
        "Value-Only KVC": m_val,
        "Full KVC": m_full,
    }

    print(f"\n=======================================================")
    print(f" Final Benchmark: Comprehensive Value-Only vs Full KV Repair")
    print(f"=======================================================")
    eval_cache = {}

    for dom, chunks in data.eval.items():
        print(f"\n[Domain: {dom.upper()}] (N={len(chunks)} seqs x {args.seq_len} tokens)")
        print(f"  {'Method':<22s} | {'Params':>9s} | {'PPL':>7s} | {'Forward KL (nats)':<18s} | {'Top-1 (%)':>9s} | {'KL Red %':>8s}")
        print(f"  {'-'*22}-|-{'-'*9}-|-{'-'*7}-|-{'-'*18}-|-{'-'*9}-|-{'-'*8}")

        # Baseline Native KL for relative reduction calculation
        native_eval = evaluate_module(ctx, methods["Native (0 params)"], L, chunks, args.eval_bs, device, compute_diagnostics=(dom == "wikitext"))
        nat_kl = native_eval["kl"]
        eval_cache[(dom, "Native (0 params)")] = native_eval

        for name, mod in methods.items():
            if name == "Native (0 params)":
                ev = native_eval
                param_count = 0
            else:
                ev = evaluate_module(ctx, mod, L, chunks, args.eval_bs, device, compute_diagnostics=(dom == "wikitext" and name in ["Value-Only KVC", "Full KVC"]))
                eval_cache[(dom, name)] = ev
                param_count = n_params(mod)

            kl_red = 100.0 * (nat_kl - ev["kl"]) / nat_kl if nat_kl > 0 else 0.0
            kl_str = f"{ev['kl']:.4f} [{ev['kl_ci'][1]:.4f},{ev['kl_ci'][2]:.4f}]"
            print(f"  {name:<22s} | {param_count:>9,d} | {ev['ppl']:>7.2f} | {kl_str:<18s} | {ev['top1']:>8.2f}% | {kl_red:>7.1f}%")

    # Layer-wise Mechanistic Diagnostics on WikiText
    print(f"\n=======================================================")
    print(f" Mechanistic Proof: Layer-wise Key vs Value Drift (WikiText)")
    print(f" Layer range: {L} to {ctx.N-1} ({ctx.N-L} skipped layers)")
    print(f"=======================================================")
    print(f"  {'Layer':<6s} | {'cos(K) Nat':<10s} | {'cos(K) V-KVC':<12s} | {'cos(V) Nat':<10s} | {'cos(V) V-KVC':<12s} | {'V-RelErr Nat':<12s} | {'V-RelErr V-KVC':<14s}")
    print(f"  {'-'*6}-|-{'-'*10}-|-{'-'*12}-|-{'-'*10}-|-{'-'*12}-|-{'-'*12}-|-{'-'*14}")

    diag_nat = eval_cache.get(("wikitext", "Native (0 params)"), {}).get("diag")
    diag_val = eval_cache.get(("wikitext", "Value-Only KVC"), {}).get("diag")

    if diag_nat and diag_val:
        for idx, lyr in enumerate(range(L, ctx.N)):
            ck_n = diag_nat["cos_k"][idx]
            ck_v = diag_val["cos_k"][idx]
            cv_n = diag_nat["cos_v"][idx]
            cv_v = diag_val["cos_v"][idx]
            err_n = diag_nat["rel_err_v"][idx]
            err_v = diag_val["rel_err_v"][idx]
            print(f"  L{lyr:<4d} | {ck_n:>10.4f} | {ck_v:>12.4f} | {cv_n:>10.4f} | {cv_v:>12.4f} | {err_n:>12.4f} | {err_v:>14.4f}")

    print("\n[Done] Solution 3 Value-Only KV Repair experiment complete.")


if __name__ == "__main__":
    main()
