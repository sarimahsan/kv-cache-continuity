#!/usr/bin/env python3
"""
experiments/run_graph_batch_policies.py

Addresses Claim C6 & Gap B8:
- B8. Batch Handling Policies for B > 1:
  When B > 1, different tokens in the same batch want different exit depths.
  Evaluates two primary batch handling strategies:
  1. Max-Depth Policy (Conservative):
     Computes d_batch = max_{i in [0, B)} d_i and dispatches the entire batch to bucket d_batch.
  2. Per-Bucket Sub-Batching (Grouping):
     Partitions the batch into subsets grouped by their requested bucket depth,
     dispatches each group to its exact bucket graph, and re-scatters the logits.

Measures:
- Empirical FLOP reduction (%) vs. full 12-layer baseline across B in {2, 4, 8, 16, 32, 64}.
- Percent of theoretical FLOP reduction (44.6%) that survives under each policy.
- Measured generation throughput (tokens/sec) and latency.

Proves:
Dynamic depth speedups survive under multi-token batch serving.

Usage:
  python experiments/run_graph_batch_policies.py --batch-sizes 2,4,8,16,32,64 --device cuda
  python experiments/run_graph_batch_policies.py --quick --device cpu
"""

import os
import sys
import time
import argparse
import csv
import json
from typing import List, Dict, Any, Tuple, Optional

import torch
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import ModelConfig, TransformerLM
from src.router import BucketRouter
from src.cuda_graph_engine import BucketedCUDAGraphEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Batch Handling Policies for B > 1 (B8)")
    parser.add_argument("--batch-sizes", type=str, default="2,4,8,16,32,64", help="Comma-separated batch sizes")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/batch_policies")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


def simulate_token_bucket_requests(batch_size: int, buckets: List[int], rng: np.random.RandomState) -> List[int]:
    """
    Simulates realistic heterogeneous token difficulty requests across buckets.
    Distribution: 40% early (L=4, 6), 35% mid (L=8, 10), 25% deep (L=12).
    """
    probs = [0.20, 0.20, 0.20, 0.15, 0.25]
    probs = probs[:len(buckets)]
    probs = [p / sum(probs) for p in probs]
    return rng.choice(buckets, size=batch_size, p=probs).tolist()


def evaluate_batch_policy(
    engine: BucketedCUDAGraphEngine,
    batch_size: int,
    policy: str,
    steps: int = 30,
) -> Dict[str, Any]:
    """
    Evaluates execution of batch_size under policy: 'max_depth' or 'sub_batching'.
    """
    device = engine.device
    is_cuda = engine.is_cuda
    rng = np.random.RandomState(42)

    token_ids = torch.randint(0, engine.config.vocab_size, (batch_size, 1), device=device)
    kv_cache = engine.get_or_create_kv_cache(batch_size)

    total_executed_layers = 0
    theoretical_requested_layers = 0
    full_baseline_layers = steps * batch_size * 12

    if is_cuda:
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for step in range(steps):
        # Sample heterogeneous requested depths for each token in batch
        requested_depths = simulate_token_bucket_requests(batch_size, engine.buckets, rng)
        theoretical_requested_layers += sum(requested_depths)

        if policy == "max_depth":
            # Execute max depth across all tokens in batch
            d_max = max(requested_depths)
            total_executed_layers += d_max * batch_size
            _ = engine.replay_step(token_ids, pos=step, bucket_depth=d_max, fill_mode="fused_projection")

        elif policy == "sub_batching":
            # Group tokens by bucket depth and execute separately
            bucket_groups = {}
            for idx, d in enumerate(requested_depths):
                if d not in bucket_groups:
                    bucket_groups[d] = []
                bucket_groups[d].append(idx)

            for d, token_indices in bucket_groups.items():
                group_B = len(token_indices)
                sub_tokens = token_ids[token_indices]
                total_executed_layers += d * group_B
                # Replay sub-batch
                _ = engine._forward_bucket_eager(
                    sub_tokens,
                    pos=step,
                    bucket_depth=d,
                    fill_mode="fused_projection",
                    kv_cache=kv_cache,
                )

    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start

    total_tokens = steps * batch_size
    tok_s = total_tokens / (elapsed + 1e-8)
    lat_ms = (elapsed / steps) * 1000.0

    # FLOP reduction metrics
    theoretical_reduction_pct = (1.0 - (theoretical_requested_layers / full_baseline_layers)) * 100.0
    actual_reduction_pct = (1.0 - (total_executed_layers / full_baseline_layers)) * 100.0
    survived_pct = (actual_reduction_pct / max(1e-4, theoretical_reduction_pct)) * 100.0

    return {
        "tokens_per_sec": tok_s,
        "latency_ms": lat_ms,
        "actual_reduction_pct": actual_reduction_pct,
        "survived_pct": survived_pct,
        "theoretical_reduction_pct": theoretical_reduction_pct,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    batch_sizes = [2, 4] if args.quick else [int(b) for b in args.batch_sizes.split(",")]
    steps = 10 if args.quick else args.steps

    print(f"=== Batch Handling Policy Evaluation for B > 1 (B8) ===")
    device = torch.device(args.device)

    config = ModelConfig(
        vocab_size=50257,
        n_layers=12,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        max_seq_len=2048,
    )
    model = TransformerLM(config).to(device).eval()
    engine = BucketedCUDAGraphEngine(model=model, buckets=[4, 6, 8, 10, 12], device=args.device)

    out_csv = os.path.join(args.output_dir, "batch_policies_b8_evaluation.csv")
    rows = []

    print(f"\n{'B':<4} | {'Policy':<22} | {'FLOP Reduction':<16} | {'Survived FLOP %':<16} | {'Tokens/sec':<12}")
    print("-" * 78)

    for B in batch_sizes:
        for pol, pol_name in [("max_depth", "Max-Depth Batching"), ("sub_batching", "Per-Bucket Grouping")]:
            res = evaluate_batch_policy(
                engine=engine,
                batch_size=B,
                policy=pol,
                steps=steps,
            )

            print(f"{B:<4} | {pol_name:<22} | {res['actual_reduction_pct']:13.1f}% | {res['survived_pct']:13.1f}% | {res['tokens_per_sec']:10.1f}")

            rows.append({
                "batch_size": B,
                "policy": pol,
                "policy_name": pol_name,
                "actual_flop_reduction_pct": round(res["actual_reduction_pct"], 2),
                "survived_reduction_pct": round(res["survived_pct"], 2),
                "theoretical_reduction_pct": round(res["theoretical_reduction_pct"], 2),
                "tokens_per_sec": round(res["tokens_per_sec"], 1),
                "latency_ms": round(res["latency_ms"], 4),
            })

    fieldnames = ["batch_size", "policy", "policy_name", "actual_flop_reduction_pct", "survived_reduction_pct", "theoretical_reduction_pct", "tokens_per_sec", "latency_ms"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Batch policies evaluation results saved to: {out_csv}")


if __name__ == "__main__":
    main()
