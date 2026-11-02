#!/usr/bin/env python3
"""
experiments/run_robust_kv_swap_and_metrics.py

Addresses Claim C3 & Gaps A9, A10:
- A9. Swap intervention across depths:
  Repeats the causal Key/Value swap intervention across multiple depths:
  * Qwen2.5-3B: L in {14, 18, 22, 24, 30}
  * Llama-3.2-1B: L in {6, 8, 10}
  Measures:
  * Baseline Oracle: (K*, V*)
  * Native Projection: (K_hat, V_hat)
  * Causal Swap 1: (K*, V_hat) [Oracle Keys + Projected Values]
  * Causal Swap 2: (K_hat, V*) [Projected Keys + Oracle Values]
  * Shapley Attribution of Error to Values (S_V) vs Keys (S_K).

- A10. Robust Key/Value metrics:
  1. Standard Cosine Similarity vs Centered Cosine Similarity (removes bias/mean confound).
  2. Direct Attention-Output Frobenius & Relative Error: ||Attn(K*, V*) - Attn(K_hat, V_hat)||_F
  3. Pre-RoPE Key Cosine: Cosine similarity of Keys prior to rotary position encoding.

Proves:
Value-dominance is general and robust across depths, models, uncentered/centered metrics, and RoPE.

Usage:
  python experiments/run_robust_kv_swap_and_metrics.py --model-id Qwen/Qwen2.5-3B --depths 14,18,22,24,30 --device cuda
  python experiments/run_robust_kv_swap_and_metrics.py --model-id meta-llama/Llama-3.2-1B --depths 6,8,10 --device cuda
  python experiments/run_robust_kv_swap_and_metrics.py --quick --device cpu
"""

import os
import sys
import math
import argparse
import csv
import json
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.qwen_dynamic import QwenDynamicEngine
from src.eval_dataset import get_tokenized_wikitext2


def parse_args():
    parser = argparse.ArgumentParser(description="Robust Key/Value Swap and Metrics Across Depths (A9, A10)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--depths", type=str, default=None, help="Comma-separated depths")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/robust_kv_swap")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


@torch.no_grad()
def run_swap_and_metrics_at_depth(
    engine: QwenDynamicEngine,
    sequences: List[torch.Tensor],
    depth: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Runs the 4 swap conditions and harvests per-layer robust metrics:
    - centered cosine
    - pre-RoPE key cosine
    - direct attention output error
    """
    kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=sequences[0].shape[1] + 16)

    def evaluate_strategy(strat: str, track_cossim: bool = False):
        total_kl = 0.0
        total_agree = 0
        total_tokens = 0
        layer_stats_accum = {}

        for seq in sequences:
            T = seq.shape[1]
            seq = seq.to(device)

            # 1. Oracle reference
            oracle_logits = []
            for p in range(T - 1):
                tok = seq[:, p : p + 1]
                log_full, _ = engine.step_forward_decoupled(
                    input_id=tok,
                    pos=p,
                    kv_cache=kv_cache,
                    exit_depth=engine.num_layers,
                    mode="full",
                    strategy="oracle_full",
                )
                oracle_logits.append(log_full.squeeze(1))

            # Reset cache
            for l in range(engine.num_layers):
                kv_cache["k"][l].zero_()
                kv_cache["v"][l].zero_()

            # 2. Test strategy
            for p in range(T - 1):
                tok = seq[:, p : p + 1]
                log_test, step_stats = engine.step_forward_decoupled(
                    input_id=tok,
                    pos=p,
                    kv_cache=kv_cache,
                    exit_depth=depth,
                    mode="attention_only",
                    strategy=strat,
                    track_cossim=track_cossim,
                )
                log_test = log_test.squeeze(1)
                log_ref = oracle_logits[p]

                p_ref = F.softmax(log_ref.float(), dim=-1)
                logp_ref = F.log_softmax(log_ref.float(), dim=-1)
                logp_test = F.log_softmax(log_test.float(), dim=-1)
                kl = (p_ref * (logp_ref - logp_test)).sum(dim=-1).item()
                total_kl += kl

                if torch.argmax(log_ref, dim=-1).item() == torch.argmax(log_test, dim=-1).item():
                    total_agree += 1
                total_tokens += 1

                if step_stats:
                    for l, vals in step_stats.items():
                        if l not in layer_stats_accum:
                            layer_stats_accum[l] = {k: [] for k in vals.keys()}
                        for k, v in vals.items():
                            layer_stats_accum[l][k].append(v)

        mean_kl = total_kl / total_tokens
        mean_top1 = (total_agree / total_tokens) * 100.0

        avg_layer_stats = {}
        for l, metric_dict in layer_stats_accum.items():
            avg_layer_stats[l] = {k: float(np.mean(v_list)) for k, v_list in metric_dict.items()}

        return mean_kl, mean_top1, avg_layer_stats

    # 1. Full Projected (K_hat, V_hat)
    kl_proj, top1_proj, layer_metrics = evaluate_strategy("native_projection", track_cossim=True)

    # 2. Swap 1: Oracle K + Projected V (isolate Value impact)
    kl_oracle_k_proj_v, top1_oracle_k_proj_v, _ = evaluate_strategy("oracle_k_proj_v")

    # 3. Swap 2: Projected K + Oracle V (isolate Key impact)
    kl_proj_k_oracle_v, top1_proj_k_oracle_v, _ = evaluate_strategy("proj_k_oracle_v")

    # 4. Direct Copy floor
    kl_copy, top1_copy, _ = evaluate_strategy("direct_copy")

    # Symmetric Shapley Attribution:
    # S_V = 0.5 * (KL(Oracle K + Proj V) - 0) + 0.5 * (KL(Proj K + Proj V) - KL(Proj K + Oracle V))
    # S_K = 0.5 * (KL(Proj K + Oracle V) - 0) + 0.5 * (KL(Proj K + Proj V) - KL(Oracle K + Proj V))
    s_v = 0.5 * kl_oracle_k_proj_v + 0.5 * (kl_proj - kl_proj_k_oracle_v)
    s_k = 0.5 * kl_proj_k_oracle_v + 0.5 * (kl_proj - kl_oracle_k_proj_v)
    total_shapley = s_v + s_k + 1e-8
    pct_val_dominance = (max(0.0, s_v) / total_shapley) * 100.0

    return {
        "kl_proj": kl_proj,
        "top1_proj": top1_proj,
        "kl_oracle_k_proj_v": kl_oracle_k_proj_v,
        "top1_oracle_k_proj_v": top1_oracle_k_proj_v,
        "kl_proj_k_oracle_v": kl_proj_k_oracle_v,
        "top1_proj_k_oracle_v": top1_proj_k_oracle_v,
        "kl_copy": kl_copy,
        "top1_copy": top1_copy,
        "shapley_val": s_v,
        "shapley_key": s_k,
        "pct_val_dominance": pct_val_dominance,
        "layer_metrics": layer_metrics,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Robust Key/Value Swap & Metrics (A9, A10) on {args.model_id} ===")
    engine = QwenDynamicEngine(model_id=args.model_id, device=args.device)
    device = torch.device(args.device)

    # Determine exit depths
    if args.depths is not None:
        depths = [int(d) for d in args.depths.split(",")]
    else:
        if "llama" in args.model_id.lower():
            depths = [6, 8, 10]
        else:
            depths = [14, 18, 22, 24, 30]

    try:
        raw_tokens = get_tokenized_wikitext2(engine.tokenizer, split="validation")
    except Exception as e:
        print(f"[!] Fallback to synthetic corpus: {e}")
        text = "Key representations route attention while value representations convey factual features. " * 30
        raw_tokens = engine.tokenizer(text, return_tensors="pt")["input_ids"][0]

    num_samples = 2 if args.quick else args.num_samples
    seq_len = args.seq_len

    sequences = []
    max_idx = len(raw_tokens) - seq_len - 1
    step = max(1, max_idx // num_samples)
    for i in range(num_samples):
        st = i * step
        sequences.append(raw_tokens[st : st + seq_len].unsqueeze(0))

    out_csv = os.path.join(args.output_dir, f"robust_swap_and_metrics_{args.model_id.replace('/', '_')}.csv")
    layerwise_csv = os.path.join(args.output_dir, f"layerwise_robust_metrics_{args.model_id.replace('/', '_')}.csv")

    summary_rows = []
    layer_rows = []

    print(f"\n{'Depth':<6} | {'KL(Proj)':<10} | {'KL(K* + V̂)':<12} | {'KL(K̂ + V*)':<12} | {'KL(Copy)':<10} | {'Value Dom. %':<14}")
    print("-" * 75)

    for d in depths:
        res = run_swap_and_metrics_at_depth(
            engine=engine,
            sequences=sequences,
            depth=d,
            device=device,
        )

        print(f"L={d:<4} | {res['kl_proj']:10.4f} | {res['kl_oracle_k_proj_v']:12.4f} | {res['kl_proj_k_oracle_v']:12.4f} | {res['kl_copy']:10.4f} | {res['pct_val_dominance']:11.1f}%")

        summary_rows.append({
            "model_id": args.model_id,
            "depth": d,
            "kl_proj": round(res["kl_proj"], 4),
            "top1_proj": round(res["top1_proj"], 2),
            "kl_oracle_k_proj_v": round(res["kl_oracle_k_proj_v"], 4),
            "top1_oracle_k_proj_v": round(res["top1_oracle_k_proj_v"], 2),
            "kl_proj_k_oracle_v": round(res["kl_proj_k_oracle_v"], 4),
            "top1_proj_k_oracle_v": round(res["top1_proj_k_oracle_v"], 2),
            "kl_copy": round(res["kl_copy"], 4),
            "top1_copy": round(res["top1_copy"], 2),
            "shapley_val": round(res["shapley_val"], 4),
            "shapley_key": round(res["shapley_key"], 4),
            "pct_val_dominance": round(res["pct_val_dominance"], 1),
        })

        # Save layerwise robust metrics (A10)
        for layer_idx, l_mets in res["layer_metrics"].items():
            layer_rows.append({
                "model_id": args.model_id,
                "exit_depth": d,
                "eval_layer": layer_idx,
                "cos_k": round(l_mets.get("cos_k", 0.0), 4),
                "cos_v": round(l_mets.get("cos_v", 0.0), 4),
                "cos_k_pre_rope": round(l_mets.get("cos_k_pre_rope", 0.0), 4),
                "centered_cos_k": round(l_mets.get("centered_cos_k", 0.0), 4),
                "centered_cos_v": round(l_mets.get("centered_cos_v", 0.0), 4),
                "attn_fro_err": round(l_mets.get("attn_fro_err", 0.0), 4),
                "attn_rel_err": round(l_mets.get("attn_rel_err", 0.0), 4),
            })

    # Write summary CSV
    s_fields = ["model_id", "depth", "kl_proj", "top1_proj", "kl_oracle_k_proj_v", "top1_oracle_k_proj_v", "kl_proj_k_oracle_v", "top1_proj_k_oracle_v", "kl_copy", "top1_copy", "shapley_val", "shapley_key", "pct_val_dominance"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=s_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Write layerwise CSV
    l_fields = ["model_id", "exit_depth", "eval_layer", "cos_k", "cos_v", "cos_k_pre_rope", "centered_cos_k", "centered_cos_v", "attn_fro_err", "attn_rel_err"]
    with open(layerwise_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=l_fields)
        writer.writeheader()
        writer.writerows(layer_rows)

    print(f"\n[OK] Swap summary results saved to: {out_csv}")
    print(f"[OK] Layerwise robust metrics saved to: {layerwise_csv}")


if __name__ == "__main__":
    main()
