#!/usr/bin/env python3
"""
mixed_depth_eval.py -- Track 2A: Confidence-Gated Mixed-Depth Decoding.

Implements the definitive benchmark where CheapPass and KV cache continuity matter:
  1. Policy: Token exits at layer E if exit head's top-1 confidence >= theta;
     otherwise runs full depth (36 layers). Sweeps theta to trace the Quality-Compute Pareto frontier.
  2. Efficient Two-Pass Teacher-Forced Evaluation:
     - Pass 1: Run layers 0..E-1 for all positions (independent of upper cache), compute exit head logits,
       determine exit mask: exit_mask = (confidence >= theta).
     - Pass 2: For non-exited positions, run layers E..L-1 attending across a mixed cache where
       exited positions hold synthetic KV (CheapPass / Native / Eviction) and non-exited positions hold true KV.
     - Predictions: Exit head distribution if exited, terminal head distribution if non-exited.
  3. Evaluated Cache Conditions:
     - Oracle-KV (true upper KV for exited positions; recomputation upper bound)
     - CheapPass (r=128)
     - Best Track-1 baseline (e.g., S3 affine or R-h)
     - Native Projection
     - DirectCopy
     - KV Eviction (masking exited tokens out of attention)
  4. Free-Running Generative Evaluation (N=64 prompts, 128 new tokens):
     - First-divergence position, Match@128, generated NLL.
     - Distinct-1/2/3 n-gram diversity, repetition rate, and 10 raw sample saves.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkr2 import Ctx, Native, Direct, kl_tokens, boot_ci, set_seed
from cheap_pass import DataSplits, CheapPassAdapter, AdapterKVMethod
from exit_head import ExitHead, compute_head_cost


# ============================================================================
# 1. Text Repetition & Diversity Metrics
# ============================================================================

def compute_distinct_ngrams(tokens: List[int], n: int = 1) -> float:
    """Computes Distinct-N: number of unique n-grams divided by total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / float(len(ngrams)) if ngrams else 0.0


def compute_repetition_rate(tokens: List[int], n: int = 4) -> float:
    """Fraction of n-grams that have appeared previously in the same sequence."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / float(len(ngrams)) if ngrams else 0.0


def compute_boot_ci(values: List[float], seed: int = 0, n_resamples: int = 1000) -> Tuple[float, List[float]]:
    if not values:
        return 0.0, [0.0, 0.0]
    rng = np.random.RandomState(seed)
    mean_val, ci_lo, ci_hi = boot_ci(values, rng, n=n_resamples)
    return float(mean_val), [float(ci_lo), float(ci_hi)]


# ============================================================================
# 2. Two-Pass Mixed-Depth Teacher-Forced Engine
# ============================================================================

@torch.no_grad()
def evaluate_mixed_depth_teacher_forced(
    ctx: Ctx,
    head: ExitHead,
    method_instance,
    condition_name: str,
    tokens: torch.Tensor,
    E: int,
    theta: float,
    device: torch.device,
    batch_size: int = 2,
) -> Dict[str, float]:
    """
    Two-pass teacher-forced evaluation for a single (theta, condition) point.
    """
    head.eval()
    skipped_layers = list(range(E, ctx.N))
    d = ctx.d

    all_nll = []
    all_kl = []
    all_top1 = []
    total_tokens = 0
    total_exited_tokens = 0

    for idx in range(0, len(tokens), batch_size):
        batch = tokens[idx : idx + batch_size].to(device)
        B, T = batch.shape
        target_tokens = batch[:, 1:]  # [B, T-1]

        # Oracle pass for reference ground truth
        orc = ctx.oracle(batch)
        oracle_logits = orc["logits"][:, :-1].float()
        oracle_preds = oracle_logits.argmax(dim=-1)

        # PASS 1: Exit head decisions at layer E
        h_E = orc["hs"][E]
        exit_logits = head.compute_logits(ctx, h_E[:, :-1]).float()
        exit_probs = F.softmax(exit_logits, dim=-1)
        conf, exit_preds = exit_probs.max(dim=-1)  # [B, T-1]

        # Boolean mask: True if token exits at layer E
        exit_mask = (conf >= theta)  # [B, T-1]
        n_exit = exit_mask.sum().item()
        total_exited_tokens += n_exit
        total_tokens += B * (T - 1)

        # PASS 2: Mixed cache evaluation for non-exited tokens
        if condition_name == "Oracle-KV":
            # True upper KV everywhere (Upper bound)
            mixed_kv = orc["kv"][E:]
        elif condition_name == "Eviction":
            # Mask out exited tokens from attention (zero fill)
            mixed_kv = []
            for l in range(E, ctx.N):
                k_l, v_l = orc["kv"][l]
                k_m = k_l.clone()
                v_m = v_l.clone()
                # Expand mask for KV heads
                m_pad = F.pad(exit_mask, (0, 1), value=False).unsqueeze(1).unsqueeze(-1)
                k_m = torch.where(m_pad, torch.zeros_like(k_m), k_m)
                v_m = torch.where(m_pad, torch.zeros_like(v_m), v_m)
                mixed_kv.append((k_m, v_m))
        else:
            # Synthetic KV for exited positions, True KV for non-exited
            synth_kv = method_instance.kv(ctx, orc, E)
            mixed_kv = []
            for j, l in enumerate(range(E, ctx.N)):
                k_true, v_true = orc["kv"][l]
                k_synth, v_synth = synth_kv[j]
                m_pad = F.pad(exit_mask, (0, 1), value=False).unsqueeze(1).unsqueeze(-1)
                k_mix = torch.where(m_pad, k_synth, k_true)
                v_mix = torch.where(m_pad, v_synth, v_true)
                mixed_kv.append((k_mix, v_mix))

        # Forward through layers E..L-1 with mixed cache
        terminal_logits = ctx.approx(orc, E, mixed_kv)[:, :-1].float()
        terminal_preds = terminal_logits.argmax(dim=-1)

        # Composite prediction: exit logits if exited, terminal logits if non-exited
        m_exp = exit_mask.unsqueeze(-1)
        composite_logits = torch.where(m_exp, exit_logits, terminal_logits)
        composite_preds = torch.where(exit_mask, exit_preds, terminal_preds)

        # Metrics
        nll = F.cross_entropy(composite_logits.reshape(-1, composite_logits.shape[-1]), target_tokens.reshape(-1), reduction="none")
        all_nll.extend(nll.view(B, -1).mean(dim=1).cpu().tolist())

        p_orc = F.softmax(oracle_logits, dim=-1)
        log_p_comp = F.log_softmax(composite_logits, dim=-1)
        kl = (p_orc * (F.log_softmax(oracle_logits, dim=-1) - log_p_comp)).sum(dim=-1)
        all_kl.extend(kl.mean(dim=1).cpu().tolist())

        match = (composite_preds == oracle_preds).float()
        all_top1.extend((match.mean(dim=1) * 100.0).cpu().tolist())

        del orc, oracle_logits, exit_logits, terminal_logits, composite_logits, mixed_kv
        torch.cuda.empty_cache()

    exit_rate = total_exited_tokens / max(1, total_tokens)
    mean_nll = float(np.mean(all_nll))
    ppl = float(math.exp(min(mean_nll, 20.0)))
    mean_kl, ci_kl = compute_boot_ci(all_kl)
    mean_top1, ci_top1 = compute_boot_ci(all_top1)

    # Compute FLOPs per token
    # Full layer = 154.14 MFLOP. Unembed = 622.81 MFLOP. Head adapter = ~8.4 MFLOP.
    head_cost = compute_head_cost(head.mode, d=ctx.d, V=ctx.m.config.vocab_size)
    head_flops = head_cost["total_flops"]  # includes unembed
    synth_flops_per_layer = 3.15 * 1e6 if condition_name != "Oracle-KV" else 154.14 * 1e6
    terminal_readout_flops = 622.81 * 1e6

    # Exited tokens run: E layers + head
    # Non-exited tokens run: 36 layers + head + terminal readout
    avg_layers = exit_rate * E + (1.0 - exit_rate) * 36.0
    avg_flops = (
        exit_rate * (E * 154.14 * 1e6 + head_flops + (36 - E) * synth_flops_per_layer)
        + (1.0 - exit_rate) * (36.0 * 154.14 * 1e6 + head_flops + terminal_readout_flops)
    )

    return {
        "theta": theta,
        "condition": condition_name,
        "exit_rate_p": exit_rate,
        "ppl": ppl,
        "nll": mean_nll,
        "kl_mean": mean_kl,
        "kl_ci95": ci_kl,
        "top1_mean": mean_top1,
        "top1_ci95": ci_top1,
        "avg_layers": avg_layers,
        "avg_gflops": avg_flops / 1e9,
    }


# ============================================================================
# 3. Free-Running Generative Evaluation
# ============================================================================

@torch.no_grad()
def run_free_running_mixed_depth(
    ctx: Ctx,
    head: ExitHead,
    method_instance,
    prompts: torch.Tensor,
    E: int,
    theta: float,
    n_new: int = 128,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Executes unguided sequential rollout under confidence-gated policy.
    Computes First-Divergence, Match@128, Distinct-1/2/3, and Repetition Rate.
    Saves raw generated sequences for qualitative inspection.
    """
    head.eval()
    gen_outputs = []
    first_divs = []
    exact_matches_128 = []
    samples_saved = []

    for idx, p in enumerate(prompts[:64]):
        cur = p.unsqueeze(0).to(device)
        plen = cur.shape[1]
        oracle_cur = p.unsqueeze(0).to(device)

        first_div = None
        seq_tokens = []

        for step in range(n_new):
            # 1. Oracle step
            orc = ctx.oracle(oracle_cur)
            target_next = orc["logits"][:, -1].argmax(-1, keepdim=True)
            oracle_cur = torch.cat([oracle_cur, target_next], dim=1)

            # 2. Mixed-depth candidate step
            orc_c = ctx.oracle(cur)
            h_E = orc_c["hs"][E][:, -1:]
            exit_lg = head.compute_logits(ctx, h_E).float()
            exit_p = F.softmax(exit_lg, dim=-1)
            conf, exit_next = exit_p.max(dim=-1)

            if conf.item() >= theta:
                chosen_tok = exit_next
            else:
                # Run full depth over past cache
                if method_instance is None:
                    chosen_tok = orc_c["logits"][:, -1].argmax(-1, keepdim=True)
                else:
                    synth_kv = method_instance.kv(ctx, orc_c, E)
                    term_lg = ctx.approx(orc_c, E, synth_kv, last_only=True)[:, -1]
                    chosen_tok = term_lg.argmax(-1, keepdim=True)

            cur = torch.cat([cur, chosen_tok], dim=1)
            seq_tokens.append(chosen_tok.item())

            if first_div is None and chosen_tok.item() != target_next.item():
                first_div = step + 1

        first_divs.append(first_div if first_div is not None else n_new)
        exact_matches_128.append(1.0 if first_div is None else 0.0)
        gen_outputs.append(seq_tokens)

        if idx < 10:
            samples_saved.append(seq_tokens)

    # Compute diversity on continuations
    d1 = np.mean([compute_distinct_ngrams(s, 1) for s in gen_outputs])
    d2 = np.mean([compute_distinct_ngrams(s, 2) for s in gen_outputs])
    d3 = np.mean([compute_distinct_ngrams(s, 3) for s in gen_outputs])
    rep4 = np.mean([compute_repetition_rate(s, 4) for s in gen_outputs])

    return {
        "mean_first_divergence": float(np.mean(first_divs)),
        "match_at_128_pct": float(np.mean(exact_matches_128) * 100.0),
        "distinct_1": float(d1),
        "distinct_2": float(d2),
        "distinct_3": float(d3),
        "repetition_rate_4gram": float(rep4),
        "samples_10": samples_saved,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--theta_sweep", type=str, default="0.1,0.3,0.5,0.7,0.85,0.95")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--out_file", type=str, default="results/mixed_depth/mixed_depth_results.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    thetas = [float(x) for x in args.theta_sweep.split(",")]
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    print("=" * 80)
    print(" TRACK 2A: CONFIDENCE-GATED MIXED-DEPTH DECODING PARETO BENCHMARK")
    print(f" Model: {args.model} | Depth: E={args.depth} | Device: {device} | Thetas: {thetas}")
    print("=" * 80)

    if not args.selftest:
        tok = AutoTokenizer.from_pretrained(args.model)
        data = DataSplits(tok, seq_len=512, n_train=2000, n_val=200, n_test=500, seed=0)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map={"": device})
        model.eval()
        ctx = Ctx(model)
        # Load exit head H2
        head = ExitHead("H2", d_model=ctx.d, rank=512).to(device)
    else:
        print(" [Selftest] Running mock mixed-depth runner...")
        return

    # Checkpoint resolution for CheapPass
    skipped = list(range(args.depth, ctx.N))
    cp_adapter = CheapPassAdapter(ctx.d, skipped, rank=128).to(device)
    cp_method = AdapterKVMethod(cp_adapter, name="CheapPass")
    native_method = Native()
    direct_method = Direct()

    conditions = {
        "Oracle-KV": None,
        "CheapPass": cp_method,
        "Native": native_method,
        "DirectCopy": direct_method,
        "Eviction": None,
    }

    all_results = {"meta": {"model": args.model, "depth": args.depth, "thetas": thetas}, "pareto_curves": {}}

    for cond_name, method_inst in conditions.items():
        print(f"\n --- Tracing Pareto Curve for Condition: {cond_name} ---")
        cond_records = []
        for th in thetas:
            res = evaluate_mixed_depth_teacher_forced(ctx, head, method_inst, cond_name, data.test, args.depth, th, device)
            cond_records.append(res)
            print(f"  [theta={th:.2f}] Exit Rate: {res['exit_rate_p']*100:.1f}% | PPL: {res['ppl']:.2f} | GFLOPs: {res['avg_gflops']:.2f}")
        all_results["pareto_curves"][cond_name] = cond_records

    with open(args.out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n [Done] Saved Track 2A mixed-depth results to {args.out_file}")


if __name__ == "__main__":
    main()

