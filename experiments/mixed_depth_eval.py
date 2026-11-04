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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkr2 import Ctx, Native, Direct, kl_tokens, boot_ci, set_seed
from cheap_pass import DataSplits, CheapPassAdapter, AdapterKVMethod, train_adapter
from exit_head import ExitHead, compute_head_cost, train_exit_head
from baselines_linear import PostProjectionScalingMethod, fit_all_group_s


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
def evaluate_mixed_depth_pareto_sweep(
    ctx: Ctx,
    head: ExitHead,
    method_instance,
    condition_name: str,
    tokens: torch.Tensor,
    E: int,
    thetas: List[float],
    device: torch.device,
    batch_size: int = 2,
) -> List[Dict[str, Any]]:
    """
    Two-pass teacher-forced evaluation across all theta thresholds in a single pass per condition.
    Computes Oracle states and synthetic KV once per batch and evaluates all theta masks.
    """
    head.eval()
    accum = {
        th: {
            "nll": [],
            "kl": [],
            "top1": [],
            "exited_tokens": 0,
            "total_tokens": 0,
        }
        for th in thetas
    }

    for idx in range(0, len(tokens), batch_size):
        batch = tokens[idx : idx + batch_size].to(device)
        B, T = batch.shape
        target_tokens = batch[:, 1:]  # [B, T-1]

        # 1. Oracle pass for reference ground truth (once per batch)
        orc = ctx.oracle(batch)
        oracle_logits = orc["logits"][:, :-1].float()
        oracle_preds = oracle_logits.argmax(dim=-1)
        p_orc = F.softmax(oracle_logits, dim=-1)
        log_p_orc = F.log_softmax(oracle_logits, dim=-1)

        # 2. PASS 1: Exit head decisions at layer E (once per batch)
        h_E = orc["hs"][E]
        exit_logits = head.compute_logits(ctx, h_E[:, :-1]).float()
        exit_probs = F.softmax(exit_logits, dim=-1)
        conf, exit_preds = exit_probs.max(dim=-1)  # [B, T-1]

        # Precompute synthetic KV once per batch for non-Oracle/non-Eviction methods
        synth_kv = None
        if condition_name not in ("Oracle-KV", "Eviction"):
            synth_kv = method_instance.kv(ctx, orc, E)

        # 3. PASS 2: Evaluate each threshold theta
        for th in thetas:
            exit_mask = (conf >= th)  # [B, T-1]
            n_exit = int(exit_mask.sum().item())
            accum[th]["exited_tokens"] += n_exit
            accum[th]["total_tokens"] += B * (T - 1)

            if n_exit == 0 or condition_name == "Oracle-KV":
                # If no tokens exited or Oracle-KV upper bound, terminal logits equal oracle logits
                terminal_logits = oracle_logits
                terminal_preds = oracle_preds
            elif condition_name == "Eviction":
                mixed_kv = []
                m_pad = F.pad(exit_mask, (0, 1), value=False).unsqueeze(1).unsqueeze(-1)
                for l in range(E, ctx.N):
                    k_l, v_l = orc["kv"][l]
                    k_m = torch.where(m_pad, torch.zeros_like(k_l), k_l)
                    v_m = torch.where(m_pad, torch.zeros_like(v_l), v_l)
                    mixed_kv.append((k_m, v_m))
                terminal_logits = ctx.approx(orc, E, mixed_kv)[:, :-1].float()
                terminal_preds = terminal_logits.argmax(dim=-1)
                del mixed_kv
            else:
                mixed_kv = []
                m_pad = F.pad(exit_mask, (0, 1), value=False).unsqueeze(1).unsqueeze(-1)
                for j, l in enumerate(range(E, ctx.N)):
                    k_true, v_true = orc["kv"][l]
                    k_synth, v_synth = synth_kv[j]
                    k_mix = torch.where(m_pad, k_synth, k_true)
                    v_mix = torch.where(m_pad, v_synth, v_true)
                    mixed_kv.append((k_mix, v_mix))
                terminal_logits = ctx.approx(orc, E, mixed_kv)[:, :-1].float()
                terminal_preds = terminal_logits.argmax(dim=-1)
                del mixed_kv

            # Composite prediction: exit logits if exited, terminal logits if non-exited
            m_exp = exit_mask.unsqueeze(-1)
            composite_logits = torch.where(m_exp, exit_logits, terminal_logits)
            composite_preds = torch.where(exit_mask, exit_preds, terminal_preds)

            # Metrics
            nll = F.cross_entropy(
                composite_logits.reshape(-1, composite_logits.shape[-1]),
                target_tokens.reshape(-1),
                reduction="none",
            )
            accum[th]["nll"].extend(nll.view(B, -1).mean(dim=1).cpu().tolist())

            log_p_comp = F.log_softmax(composite_logits, dim=-1)
            kl = (p_orc * (log_p_orc - log_p_comp)).sum(dim=-1)
            accum[th]["kl"].extend(kl.mean(dim=1).cpu().tolist())

            match = (composite_preds == oracle_preds).float()
            accum[th]["top1"].extend((match.mean(dim=1) * 100.0).cpu().tolist())

            del composite_logits, composite_preds

        del orc, oracle_logits, exit_logits, synth_kv
        torch.cuda.empty_cache()

    # Compile summary records across all thetas
    head_cost = compute_head_cost(head.mode, d=ctx.d, V=ctx.m.config.vocab_size)
    head_flops = head_cost["total_flops"]  # includes unembed
    synth_flops_per_layer = 3.15 * 1e6 if condition_name != "Oracle-KV" else 154.14 * 1e6
    terminal_readout_flops = 622.81 * 1e6

    records = []
    for th in thetas:
        st = accum[th]
        exit_rate = st["exited_tokens"] / max(1, st["total_tokens"])
        mean_nll = float(np.mean(st["nll"]))
        ppl = float(math.exp(min(mean_nll, 20.0)))
        mean_kl, ci_kl = compute_boot_ci(st["kl"])
        mean_top1, ci_top1 = compute_boot_ci(st["top1"])

        avg_layers = exit_rate * E + (1.0 - exit_rate) * float(ctx.N)
        avg_flops = (
            exit_rate * (E * 154.14 * 1e6 + head_flops + (ctx.N - E) * synth_flops_per_layer)
            + (1.0 - exit_rate) * (float(ctx.N) * 154.14 * 1e6 + head_flops + terminal_readout_flops)
        )

        records.append({
            "theta": th,
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
        })

    return records


# ============================================================================
# 3. Free-Running Generative Evaluation
# ============================================================================

@torch.no_grad()
def precompute_oracle_generations(
    ctx: Ctx,
    prompts: torch.Tensor,
    n_prompts: int = 16,
    n_new: int = 64,
    device: torch.device = torch.device("cpu"),
) -> List[List[int]]:
    """Precomputes greedy full-depth Oracle continuations once across all prompts."""
    oracle_continuations = []
    for idx, p in enumerate(prompts[:n_prompts]):
        oracle_cur = p.unsqueeze(0).to(device)
        seq = []
        for _ in range(n_new):
            orc = ctx.oracle(oracle_cur)
            target_next = orc["logits"][:, -1].argmax(-1, keepdim=True)
            oracle_cur = torch.cat([oracle_cur, target_next], dim=1)
            seq.append(int(target_next.item()))
        oracle_continuations.append(seq)
    return oracle_continuations


@torch.no_grad()
def run_free_running_mixed_depth(
    ctx: Ctx,
    head: ExitHead,
    method_instance,
    condition_name: str,
    prompts: torch.Tensor,
    oracle_continuations: List[List[int]],
    E: int,
    theta: float,
    n_prompts: int = 16,
    n_new: int = 64,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Executes unguided sequential rollout under confidence-gated policy with mixed KV cache.
    Computes First-Divergence, Match@N, Distinct-1/2/3, and Repetition Rate.
    """
    head.eval()
    gen_outputs = []
    first_divs = []
    exact_matches = []
    samples_saved = []
    total_exited = 0
    total_steps = 0

    for idx, p in enumerate(prompts[:n_prompts]):
        cur = p.unsqueeze(0).to(device)
        plen = cur.shape[1]
        orc_seq = oracle_continuations[idx]

        # Track which positions in `cur` exited early (prompt tokens never exit early)
        exited_flags = [False] * plen
        first_div = None
        seq_tokens = []

        for step in range(n_new):
            target_tok = orc_seq[step]
            orc_c = ctx.oracle(cur)
            h_E = orc_c["hs"][E][:, -1:]
            exit_lg = head.compute_logits(ctx, h_E).float()
            exit_p = F.softmax(exit_lg, dim=-1)
            conf, exit_next = exit_p.max(dim=-1)

            total_steps += 1
            if conf.item() >= theta:
                chosen_tok = exit_next
                exited_flags.append(True)
                total_exited += 1
            else:
                exited_flags.append(False)
                any_past_exited = any(exited_flags[:-1])
                if condition_name == "Oracle-KV" or not any_past_exited:
                    chosen_tok = orc_c["logits"][:, -1].argmax(-1, keepdim=True)
                elif condition_name == "Eviction":
                    mask_t = torch.tensor(exited_flags, dtype=torch.bool, device=device).view(1, 1, -1, 1)
                    mixed_kv = []
                    for l in range(E, ctx.N):
                        k_l, v_l = orc_c["kv"][l]
                        k_m = torch.where(mask_t, torch.zeros_like(k_l), k_l)
                        v_m = torch.where(mask_t, torch.zeros_like(v_l), v_l)
                        mixed_kv.append((k_m, v_m))
                    term_lg = ctx.approx(orc_c, E, mixed_kv, last_only=True)[:, -1]
                    chosen_tok = term_lg.argmax(-1, keepdim=True)
                else:
                    synth_kv = method_instance.kv(ctx, orc_c, E)
                    mask_t = torch.tensor(exited_flags, dtype=torch.bool, device=device).view(1, 1, -1, 1)
                    mixed_kv = []
                    for j, l in enumerate(range(E, ctx.N)):
                        k_true, v_true = orc_c["kv"][l]
                        k_synth, v_synth = synth_kv[j]
                        k_mix = torch.where(mask_t, k_synth, k_true)
                        v_mix = torch.where(mask_t, v_synth, v_true)
                        mixed_kv.append((k_mix, v_mix))
                    term_lg = ctx.approx(orc_c, E, mixed_kv, last_only=True)[:, -1]
                    chosen_tok = term_lg.argmax(-1, keepdim=True)

            cur = torch.cat([cur, chosen_tok], dim=1)
            tok_id = int(chosen_tok.item())
            seq_tokens.append(tok_id)

            if first_div is None and tok_id != target_tok:
                first_div = step + 1

        first_divs.append(first_div if first_div is not None else n_new)
        exact_matches.append(1.0 if first_div is None else 0.0)
        gen_outputs.append(seq_tokens)

        if idx < 10:
            samples_saved.append(seq_tokens)

    d1 = np.mean([compute_distinct_ngrams(s, 1) for s in gen_outputs])
    d2 = np.mean([compute_distinct_ngrams(s, 2) for s in gen_outputs])
    d3 = np.mean([compute_distinct_ngrams(s, 3) for s in gen_outputs])
    rep4 = np.mean([compute_repetition_rate(s, 4) for s in gen_outputs])

    return {
        "theta": theta,
        "exit_rate_gen": total_exited / max(1, total_steps),
        "mean_first_divergence": float(np.mean(first_divs)),
        "match_pct": float(np.mean(exact_matches) * 100.0),
        "distinct_1": float(d1),
        "distinct_2": float(d2),
        "distinct_3": float(d3),
        "repetition_rate_4gram": float(rep4),
        "samples_10": samples_saved,
    }


def resolve_adapter_checkpoint(
    E: int,
    rank: int,
    custom_ckpt: Optional[str] = None,
    ckpt_dir: str = "cheap_pass_checkpoints",
) -> Tuple[Optional[str], List[str]]:
    """
    Resolves a trained CheapPass adapter checkpoint from an explicit file path,
    custom directory, or standard candidate directories.
    Returns (resolved_path_or_None, list_of_searched_paths).
    """
    fname = f"adapter_E{E}_r{rank}_primary.pt"
    searched = []

    # 1. Check explicit custom path first (can be a file or a directory)
    if custom_ckpt:
        if os.path.isfile(custom_ckpt):
            return os.path.abspath(custom_ckpt), [os.path.abspath(custom_ckpt)]
        cand_in_dir = os.path.join(custom_ckpt, fname)
        searched.append(os.path.abspath(custom_ckpt))
        searched.append(os.path.abspath(cand_in_dir))
        if os.path.isfile(cand_in_dir):
            return os.path.abspath(cand_in_dir), searched

    # 2. Check candidate directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    candidate_dirs = [
        ckpt_dir,
        os.path.join(repo_root, "cheap_pass_checkpoints"),
        os.path.join(repo_root, "cheap_pass_out"),
        "cheap_pass_checkpoints",
        "cheap_pass_out",
        "../cheap_pass_checkpoints",
        "../cheap_pass_out",
        "/content/kv-cache-continuity/cheap_pass_checkpoints",
        "/content/kv-cache-continuity/cheap_pass_out",
        "/kaggle/working/cheap_pass_checkpoints",
        "/kaggle/working/cheap_pass_out",
    ]
    for cd in candidate_dirs:
        if os.path.isfile(cd):
            searched.append(os.path.abspath(cd))
            return os.path.abspath(cd), searched
        cand = os.path.join(cd, fname)
        abs_cand = os.path.abspath(cand)
        if abs_cand not in searched:
            searched.append(abs_cand)
        if os.path.isfile(cand):
            return abs_cand, searched

    return None, searched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depth", type=int, default=18)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument(
        "--ckpt_path",
        "--adapter_ckpt",
        "--adapter-path",
        dest="ckpt_path",
        type=str,
        default=None,
        help="Explicit file path (or directory) to the CheapPass adapter .pt checkpoint",
    )
    parser.add_argument("--ckpt_dir", type=str, default="cheap_pass_checkpoints")
    parser.add_argument(
        "--head_ckpt",
        "--head-path",
        dest="head_ckpt",
        type=str,
        default=None,
        help="Optional explicit file path to the ExitHead H2 .pt checkpoint",
    )
    parser.add_argument("--no_auto_train", action="store_true", help="Fail instead of training if adapter checkpoint is not found")
    parser.add_argument("--theta_sweep", type=str, default="0.1,0.3,0.5,0.7,0.85,0.95")
    parser.add_argument("--gen_theta", type=float, default=0.7)
    parser.add_argument("--n_gen_prompts", type=int, default=16)
    parser.add_argument("--n_gen_tokens", type=int, default=64)
    parser.add_argument("--skip_free_gen", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--out_file", type=str, default="results/mixed_depth/mixed_depth_results.json")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    thetas = [float(x) for x in args.theta_sweep.split(",")]
    if os.path.dirname(args.out_file):
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    print("=" * 80)
    print(" TRACK 2A: CONFIDENCE-GATED MIXED-DEPTH DECODING PARETO BENCHMARK")
    print(f" Model: {args.model} | Depth: E={args.depth} | Device: {device} | Thetas: {thetas}")
    print("=" * 80)

    # Pre-check adapter checkpoint before loading heavy model weights so the user sees status immediately
    ckpt_path, searched_paths = resolve_adapter_checkpoint(args.depth, args.rank, args.ckpt_path, args.ckpt_dir)
    print("\n" + "-" * 80)
    print(" [CHECKPOINT RESOLUTION STATUS]")
    if ckpt_path is not None:
        print(f"  -> CheapPass Adapter : [FOUND & WILL LOAD] {ckpt_path}")
    else:
        print(f"  -> CheapPass Adapter : [NOT FOUND] Searched {len(searched_paths)} paths:")
        for sp in searched_paths[:6]:
            print(f"       * {sp}")
        if args.no_auto_train:
            raise FileNotFoundError(
                f"CheapPass checkpoint not found and --no_auto_train was set. "
                f"Pass --ckpt_path /path/to/adapter_E{args.depth}_r{args.rank}_primary.pt"
            )
        print("     (Will train adapter automatically on Train split and save checkpoint)")
    print("-" * 80 + "\n")

    if not args.selftest:
        tok = AutoTokenizer.from_pretrained(args.model)
        data = DataSplits(tok, seq_len=512, n_train=2000, n_val=200, n_test=500, seed=0)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map={"": device})
        model.eval()
        ctx = Ctx(model)

        # 1. Load or Train Exit Head H2 (Distilled Bottleneck MLP, r=512)
        head = ExitHead("H2", d_model=ctx.d, rank=512).to(device)
        head_ckpt = args.head_ckpt or os.path.join("checkpoints", f"exit_head_H2_E{args.depth}_r512.pt")
        if os.path.dirname(head_ckpt):
            os.makedirs(os.path.dirname(head_ckpt), exist_ok=True)
        if os.path.isfile(head_ckpt):
            head.load_state_dict(torch.load(head_ckpt, map_location=device))
            head.eval()
            head_up_norm = float(head.up.weight.float().norm().item())
            print(f" [ExitHead Status] LOADED: TRUE | Path: {os.path.abspath(head_ckpt)} | W_up norm: {head_up_norm:.4f}")
        else:
            print(f" [ExitHead Status] LOADED: FALSE (No checkpoint at {head_ckpt}) -> Distilling H2 exit head (500 steps)...")
            head = train_exit_head(ctx, head, data.train, data.val, args.depth, device, steps=500, seed=0)
            torch.save(head.state_dict(), head_ckpt)
            head_up_norm = float(head.up.weight.float().norm().item())
            print(f" [ExitHead Status] TRAINED & SAVED to {os.path.abspath(head_ckpt)} | W_up norm: {head_up_norm:.4f}")
    else:
        print(" [Selftest] Running mock mixed-depth runner...")
        return

    # 2. Load or Train CheapPass Adapter (r=128)
    skipped = list(range(args.depth, ctx.N))
    cp_adapter = CheapPassAdapter(ctx.d, skipped, rank=args.rank).to(device)
    checkpoint_loaded = False
    if ckpt_path is not None:
        cp_adapter.load_state_dict(torch.load(ckpt_path, map_location=device))
        cp_adapter.eval()
        checkpoint_loaded = True
        up_norm = sum(p.float().norm().item() for name, p in cp_adapter.named_parameters() if "up" in name)
        print(f" [CheapPass Status] CHECKPOINT LOADED: TRUE  | Path: {ckpt_path} | W_up norm sum: {up_norm:.4f}")
    else:
        print(f" [CheapPass Status] CHECKPOINT LOADED: FALSE | Training E={args.depth} r={args.rank} adapter...")
        cp_adapter = train_adapter(
            ctx=ctx,
            train_data=data.train,
            val_data=data.val,
            E=args.depth,
            device=device,
            rank=args.rank,
            train_steps=1500,
            batch_size=4,
            out_dir=args.ckpt_dir,
            tag="primary",
        )
        ckpt_path = os.path.abspath(os.path.join(args.ckpt_dir, f"adapter_E{args.depth}_r{args.rank}_primary.pt"))
        checkpoint_loaded = True
        up_norm = sum(p.float().norm().item() for name, p in cp_adapter.named_parameters() if "up" in name)
        print(f" [CheapPass Status] TRAINED & SAVED: TRUE    | Path: {ckpt_path} | W_up norm sum: {up_norm:.4f}")

    if up_norm == 0.0:
        print(" [WARNING] CheapPass W_up norm is 0.0! Adapter is in zero-init state and will equal Native!")

    cp_method = AdapterKVMethod(cp_adapter, name="CheapPass")

    # 3. Fit Closed-Form S3 Affine Baseline
    print(f" [Baseline] Fitting closed-form S3_Affine baseline at E={args.depth}...")
    s3_method = PostProjectionScalingMethod("S3", "S3_affine")
    fit_all_group_s({"S3": s3_method}, ctx, data.train, args.depth, device, n_fit=64, batch_size=2)

    native_method = Native()
    direct_method = Direct()

    conditions = {
        "Oracle-KV": None,
        "CheapPass": cp_method,
        "S3_Affine": s3_method,
        "Native": native_method,
        "DirectCopy": direct_method,
        "Eviction": None,
    }

    all_results = {
        "meta": {
            "model": args.model,
            "depth": args.depth,
            "rank": args.rank,
            "checkpoint_loaded": checkpoint_loaded,
            "adapter_ckpt": ckpt_path,
            "adapter_up_norm": up_norm,
            "head_ckpt": os.path.abspath(head_ckpt),
            "head_up_norm": head_up_norm,
            "thetas": thetas,
            "gen_theta": args.gen_theta,
        },
        "pareto_curves": {},
        "free_running_generation": {},
    }

    # 4. Teacher-Forced Pareto Sweep (Single pass per condition across all thetas)
    for cond_name, method_inst in conditions.items():
        print(f"\n --- Tracing Pareto Curve for Condition: {cond_name} ---")
        cond_records = evaluate_mixed_depth_pareto_sweep(
            ctx, head, method_inst, cond_name, data.test, args.depth, thetas, device
        )
        for res in cond_records:
            print(
                f"  [theta={res['theta']:.2f}] Exit Rate: {res['exit_rate_p']*100:.1f}% | "
                f"PPL: {res['ppl']:.2f} | Top-1: {res['top1_mean']:.2f}% | KL: {res['kl_mean']:.4f} | GFLOPs: {res['avg_gflops']:.2f}"
            )
        all_results["pareto_curves"][cond_name] = cond_records

    # 5. Free-Running Generative Rollout at theta = args.gen_theta
    if not args.skip_free_gen:
        print(f"\n --- Running Free-Running Generation (theta={args.gen_theta}, N={args.n_gen_prompts}, T_new={args.n_gen_tokens}) ---")
        eval_prompts = data.test[: args.n_gen_prompts, :64]
        oracle_conts = precompute_oracle_generations(
            ctx, eval_prompts, n_prompts=args.n_gen_prompts, n_new=args.n_gen_tokens, device=device
        )
        for cond_name, method_inst in conditions.items():
            gen_res = run_free_running_mixed_depth(
                ctx,
                head,
                method_inst,
                cond_name,
                eval_prompts,
                oracle_conts,
                args.depth,
                args.gen_theta,
                n_prompts=args.n_gen_prompts,
                n_new=args.n_gen_tokens,
                device=device,
            )
            all_results["free_running_generation"][cond_name] = gen_res
            print(
                f"  [{cond_name:10s}] Exit: {gen_res['exit_rate_gen']*100:.1f}% | "
                f"FirstDiv: {gen_res['mean_first_divergence']:.1f} | Match: {gen_res['match_pct']:.1f}% | "
                f"Distinct-2: {gen_res['distinct_2']:.3f} | Rep-4: {gen_res['repetition_rate_4gram']:.3f}"
            )

    with open(args.out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n [Done] Saved Track 2A mixed-depth results to {args.out_file}")


if __name__ == "__main__":
    main()

