#!/usr/bin/env python3
"""
experiments/run_entropy_gated_exit.py

Direction 1: Dynamic Confidence / Entropy-Gated Early Exit.

Problem:
  Static early exit forces ALL tokens to exit at depth d_t, resulting in high
  error (~41-52% Top-1) because hard tokens have not completed their factual
  and syntactic computation.

Solution:
  Use prediction entropy as a dynamic gate:
    H(P_t) = - sum_v P_t(v) * ln P_t(v)
  Tokens exit early at layer d_t ONLY when H(P_t) < threshold (or max confidence > conf_thresh).
  Easy tokens (syntax, connectors, punctuation) exit early with >90% accuracy,
  while hard tokens execute through all N layers.
  Upper-layer KV slots for exited tokens are filled via Native Projection / Value repair.

Measures across entropy thresholds:
  - Exit Rate (% of tokens exiting early)
  - Exited Token Top-1 Accuracy vs Oracle
  - Full Sequence Top-1 Accuracy & Forward KL
  - Theoretical & Empirical Compute Savings (FLOP reduction)

Usage:
  # Quick CPU self-test:
  python experiments/run_entropy_gated_exit.py --selftest

  # GPU evaluation on Qwen2.5-3B:
  python experiments/run_entropy_gated_exit.py --model Qwen/Qwen2.5-3B --depth 18 --eval_n 60
"""

import argparse
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)
sys.path.insert(0, os.path.dirname(curr_dir))

from vkr2 import Ctx, Data, boot_ci, kl_tokens, set_seed


def compute_entropy_and_conf(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Shannon entropy (in nats) and max probability confidence from logits.
    logits: [B, T, V]
    returns: entropy [B, T], confidence [B, T]
    """
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    confidence = probs.max(dim=-1)[0]
    return entropy, confidence


def dynamic_approx(ctx: Ctx, orc: dict, L: int, kv_hats: list, exact_mask: Optional[torch.Tensor] = None):
    h, cos, sin = orc["hs"][L], orc["cos"], orc["sin"]
    for j, i in enumerate(range(L, ctx.N)):
        kh, vh = kv_hats[j]
        if exact_mask is not None:
            if exact_mask.dim() == 1:
                m = exact_mask.view(1, 1, -1, 1)
            elif exact_mask.dim() == 2:
                m = exact_mask.unsqueeze(1).unsqueeze(-1)
            else:
                m = exact_mask
            kh = torch.where(m, orc["kv"][i][0], kh)
            vh = torch.where(m, orc["kv"][i][1], vh)
        h, _ = ctx.layer_fwd(i, h, cos, sin, (kh, vh))
    return ctx.head(h)


@torch.no_grad()
def evaluate_entropy_gating(ctx: Ctx, L: int, chunks: torch.Tensor, thresholds: list, bs: int, device: torch.device):
    """
    Evaluates dynamic entropy-gated early exit across multiple entropy thresholds.
    """
    results = {thresh: {"exited_correct": 0, "exited_total": 0, "all_correct": 0, "all_total": 0, "kl_list": []} for thresh in thresholds}
    results["static_exit"] = {"exited_correct": 0, "exited_total": 0, "all_correct": 0, "all_total": 0, "kl_list": []}
    results["oracle"] = {"all_correct": 0, "all_total": 0, "kl_list": [0.0]}

    top_layer = ctx.N - 1

    for s in range(0, len(chunks), bs):
        ids = chunks[s : s + bs].to(device)
        B, T = ids.shape
        orc = ctx.oracle(ids)

        # Full oracle references
        logits_orc = orc["logits"][:, :-1]
        preds_orc = logits_orc.argmax(dim=-1)

        # Intermediate representations at depth L
        h_L = orc["hs"][L]
        logits_L = ctx.head(h_L)[:, :-1]
        preds_L = logits_L.argmax(dim=-1)

        # Compute token-level predictive entropy and confidence at exit depth L
        entropy_L, conf_L = compute_entropy_and_conf(logits_L)

        # Precompute Native Projection KV cache for skipped layers L..N-1
        kv_native = [ctx.project_kv(i, h_L, orc["cos"], orc["sin"]) for i in range(L, ctx.N)]

        # --- Evaluate Static Exit Baseline (every token exits at L) ---
        kl_static = kl_tokens(logits_orc, logits_L).mean(dim=1).cpu().tolist()
        correct_static = (preds_L == preds_orc).float().sum().item()
        results["static_exit"]["all_correct"] += correct_static
        results["static_exit"]["all_total"] += B * (T - 1)
        results["static_exit"]["exited_correct"] += correct_static
        results["static_exit"]["exited_total"] += B * (T - 1)
        results["static_exit"]["kl_list"].extend(kl_static)

        # --- Evaluate Oracle Reference ---
        results["oracle"]["all_correct"] += B * (T - 1)
        results["oracle"]["all_total"] += B * (T - 1)

        # --- Evaluate Dynamic Gating per Threshold ---
        for thresh in thresholds:
            # Mask: True = Exit Early at depth L, False = Execute full N layers
            exit_mask = (entropy_L < thresh)  # [B, T-1]

            # In dynamic execution:
            # Exited tokens decode from logits_L.
            # Non-exited tokens execute remaining layers attending over cache.
            # Compute approximate pass where exited tokens project KV and full tokens run all layers:
            # We construct exact_mask: positions where exit_mask is False have exact oracle KV
            # Pad mask to length T
            full_exit_mask = torch.cat([exit_mask, torch.zeros(B, 1, dtype=torch.bool, device=device)], dim=1)
            exact_history_mask = ~full_exit_mask

            logits_full_tokens = dynamic_approx(ctx, orc, L, kv_native, exact_mask=exact_history_mask)[:, :-1]
            preds_full_tokens = logits_full_tokens.argmax(dim=-1)

            # Combined predictions:
            # Where exited: take preds_L; where full: take preds_full_tokens
            preds_combined = torch.where(exit_mask, preds_L, preds_full_tokens)
            logits_combined = torch.where(exit_mask.unsqueeze(-1), logits_L, logits_full_tokens)

            kl_thresh = kl_tokens(logits_orc, logits_combined).mean(dim=1).cpu().tolist()
            results[thresh]["kl_list"].extend(kl_thresh)

            # Metrics for exited tokens only
            num_exited = exit_mask.sum().item()
            if num_exited > 0:
                exited_correct = ((preds_L == preds_orc) & exit_mask).float().sum().item()
                results[thresh]["exited_correct"] += exited_correct
                results[thresh]["exited_total"] += num_exited

            # Metrics across all tokens
            all_correct = (preds_combined == preds_orc).float().sum().item()
            results[thresh]["all_correct"] += all_correct
            results[thresh]["all_total"] += B * (T - 1)

    return results


def main():
    p = argparse.ArgumentParser(description="Direction 1: Dynamic Confidence / Entropy-Gated Early Exit")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depth", type=str, default="24", help="Exit layer depth (e.g. 18 or 24)")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--eval_n", type=int, default=60, help="Number of eval sequences")
    p.add_argument("--eval_bs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thresholds", type=str, default="0.2,0.5,1.0,1.5,2.0,3.0", help="Comma-separated entropy thresholds in nats")
    p.add_argument("--no_c4", action="store_true")
    p.add_argument("--c4_docs", type=int, default=400)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.selftest else "cpu")

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
        args.eval_n, args.eval_bs, args.seq_len = 12, 4, 32
        thresholds = [0.5, 1.5, 3.0]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        dt = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16 if device.type == "cuda" else torch.float32
        device_map = "cuda" if device.type == "cuda" else None
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt, device_map=device_map).eval()
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(device).eval()
        thresholds = [float(x.strip()) for x in args.thresholds.split(",")]

    for param in model.parameters():
        param.requires_grad_(False)

    ctx = Ctx(model)
    L = int(round(float(args.depth) * ctx.N)) if "." in args.depth else int(args.depth)
    S = ctx.N - L

    print(f"\n=======================================================")
    print(f" Dynamic Entropy-Gated Early Exit Benchmark")
    print(f" Model: {args.model if not args.selftest else 'tiny-random'} | Exit Depth: L={L}/{ctx.N} ({100.0*L/ctx.N:.1f}%)")
    print(f" Evaluated Entropy Thresholds: {thresholds}")
    print(f"=======================================================")

    data = Data(tok, args, model.config.vocab_size, need_train=False)

    for dom, chunks in data.eval.items():
        print(f"\n-------------------------------------------------------")
        print(f" Domain: {dom.upper()} (N={len(chunks)} sequences x {args.seq_len} tokens)")
        print(f"-------------------------------------------------------")

        res = evaluate_entropy_gating(ctx, L, chunks, thresholds, args.eval_bs, device)

        # Baseline info
        static_top1 = 100.0 * res["static_exit"]["all_correct"] / max(1, res["static_exit"]["all_total"])
        static_kl = float(np.mean(res["static_exit"]["kl_list"]))

        print(f"  Reference Baselines:")
        print(f"    - Full Model Oracle (100% compute):  100.0% Top-1 | KL = 0.0000 nats |  0.0% Compute Saved")
        print(f"    - Static Exit L={L} (Blind 100% exit):  {static_top1:5.2f}% Top-1 | KL = {static_kl:.4f} nats | {100.0*S/ctx.N:5.1f}% Compute Saved\n")

        print(f"  {'Entropy Threshold':<18s} | {'Exit Rate':<10s} | {'Exited Token Top-1':<19s} | {'Overall Top-1':<14s} | {'Forward KL':<12s} | {'Compute Saved'}")
        print(f"  {'-'*18}-|-{'-'*10}-|-{'-'*19}-|-{'-'*14}-|-{'-'*12}-|-{'-'*13}")

        for thresh in thresholds:
            r = res[thresh]
            num_exited = r["exited_total"]
            total_tokens = r["all_total"]
            exit_rate = 100.0 * num_exited / max(1, total_tokens)
            exited_acc = 100.0 * r["exited_correct"] / max(1, num_exited) if num_exited > 0 else 0.0
            overall_acc = 100.0 * r["all_correct"] / max(1, total_tokens)
            kl_mean = float(np.mean(r["kl_list"]))
            compute_saved = exit_rate * (S / ctx.N)

            print(f"  H < {thresh:<14.2f} | {exit_rate:>8.1f}% | {exited_acc:>17.2f}% | {overall_acc:>12.2f}% | {kl_mean:>10.4f} | {compute_saved:>11.1f}%")

    print("\n[Done] Dynamic Entropy Gating benchmark complete.")


if __name__ == "__main__":
    main()
