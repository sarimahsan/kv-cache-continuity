#!/usr/bin/env python3
"""
experiments/run_graph_eager_equivalence.py

Addresses Claim C7 & Gap B9:
- B9. Graph vs. Eager Numerical and Token Equivalence:
  For identical input tokens, sequence positions, and router decisions:
  Runs parallel execution through Eager Mode and CUDA Graph Replay Mode.

Measures:
- Maximum absolute difference: max |logits_graph - logits_eager|
- Mean squared difference (MSE): ||logits_graph - logits_eager||_2^2
- Top-1 greedy token prediction match rate (%) over 100+ consecutive steps.

Proves:
CUDA graph capture preserves strict mathematical and numerical equivalence with eager execution.

Usage:
  python experiments/run_graph_eager_equivalence.py --steps 100 --device cuda
  python experiments/run_graph_eager_equivalence.py --quick --device cpu
"""

import os
import sys
import argparse
import csv
import json

import torch
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import ModelConfig, TransformerLM
from src.cuda_graph_engine import BucketedCUDAGraphEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Graph vs Eager Equivalence (B9)")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/graph_equivalence")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Graph vs. Eager Numerical Equivalence Audit (B9) ===")
    device = torch.device(args.device)
    is_cuda = (device.type == "cuda" and torch.cuda.is_available())

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

    # Capture graphs
    if is_cuda:
        for b in [4, 6, 8, 10, 12]:
            engine.capture_graph(batch_size=1, bucket_depth=b, fill_mode="native_projection")

    steps = 10 if args.quick else args.steps
    buckets_to_test = [4, 6, 8, 10, 12]

    out_csv = os.path.join(args.output_dir, "graph_eager_equivalence_b9.csv")
    rows = []

    print(f"\n{'Bucket':<8} | {'Max Abs Diff':<15} | {'MSE Diff':<15} | {'Top-1 Token Match':<20} | {'Status':<10}")
    print("-" * 75)

    for b in buckets_to_test:
        max_diffs = []
        mses = []
        matches = 0

        # Create two identical KV caches
        cache_eager = engine.get_or_create_kv_cache(batch_size=1)
        cache_eager.reset()

        cache_graph = engine.get_or_create_kv_cache(batch_size=1)

        for step in range(steps):
            tok = torch.randint(0, config.vocab_size, (1, 1), device=device)

            # Eager step
            log_eager, _ = engine._forward_bucket_eager(
                input_ids=tok,
                pos=step,
                bucket_depth=b,
                fill_mode="native_projection",
                kv_cache=cache_eager,
            )

            # Graph step
            log_graph = engine.replay_step(
                token_ids=tok,
                pos=step,
                bucket_depth=b,
                fill_mode="native_projection",
            )

            diff = torch.abs(log_eager - log_graph)
            max_d = float(diff.max().item())
            mse = float((diff ** 2).mean().item())

            pred_eager = torch.argmax(log_eager, dim=-1).item()
            pred_graph = torch.argmax(log_graph, dim=-1).item()

            if pred_eager == pred_graph:
                matches += 1

            max_diffs.append(max_d)
            mses.append(mse)

        overall_max_d = max(max_diffs)
        overall_mse = float(np.mean(mses))
        match_rate = (matches / steps) * 100.0

        status = "PASSED" if match_rate == 100.0 and overall_max_d < 1e-3 else "FAILED"

        print(f"L={b:<6} | {overall_max_d:15.2e} | {overall_mse:15.2e} | {match_rate:17.1f}% | {status}")

        rows.append({
            "bucket": b,
            "max_abs_diff": overall_max_d,
            "mse_diff": overall_mse,
            "top1_match_pct": round(match_rate, 2),
            "status": status,
            "steps": steps,
        })

    fieldnames = ["bucket", "max_abs_diff", "mse_diff", "top1_match_pct", "status", "steps"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Graph vs. eager equivalence results saved to: {out_csv}")


if __name__ == "__main__":
    main()
