#!/usr/bin/env python3
"""
experiments/run_graph_fill_cost.py

Addresses Claim C5 & Gaps B3, B4:
- B3. Fill Cost Inside the Captured Graph:
  For each bucket depth d in {4, 6, 8, 10}:
  Times CUDA Graph replay with and without fill at batch sizes B in {1, 2, 4, 8, 16, 64}.
  Measures:
  * Raw Graph replay latency without fill (t_no_fill)
  * Graph replay latency with Native Projection fill (t_fill)
  * Fill overhead (Delta_fill = t_fill - t_no_fill)
  * Full 12-layer baseline latency (t_full)
  * Time saved by layer skipping (Delta_saved = t_full - t_no_fill)
  * Fill overhead as a percent of time saved: (Delta_fill / Delta_saved) * 100%

- B4. Fused Fill Optimization:
  Stacks skipped layers' W_K and W_V into one single batched GEMM matmul per bucket.
  Directly compares sequential per-layer projections vs. fused batched GEMM.

Proves:
Fill overhead stays a small fraction of compute saved, and fused GEMM eliminates launch overhead.

Usage:
  python experiments/run_graph_fill_cost.py --batch-sizes 1,2,4,8,16,64 --buckets 4,6,8,10 --device cuda
  python experiments/run_graph_fill_cost.py --quick --device cpu
"""

import os
import sys
import time
import argparse
import csv
import json
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.model import ModelConfig, TransformerLM
from src.cuda_graph_engine import BucketedCUDAGraphEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Fill Cost and Fused Fill (B3, B4)")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,64", help="Comma-separated batch sizes")
    parser.add_argument("--buckets", type=str, default="4,6,8,10", help="Comma-separated bucket depths")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=100, help="Timing repetitions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/graph_fill_cost")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


def benchmark_step_latency(
    engine: BucketedCUDAGraphEngine,
    batch_size: int,
    bucket_depth: int,
    fill_mode: str,
    warmup: int = 20,
    repeats: int = 100,
) -> float:
    """
    Measures per-step decode latency in milliseconds using CUDA events or high-res timer.
    """
    token_ids = torch.randint(0, engine.config.vocab_size, (batch_size, 1), device=engine.device)
    is_cuda = engine.is_cuda

    # Capture graph if on CUDA
    if is_cuda:
        engine.capture_graph(batch_size=batch_size, bucket_depth=bucket_depth, fill_mode=fill_mode)

    # Warmup
    for _ in range(warmup):
        _ = engine.replay_step(token_ids=token_ids, pos=0, bucket_depth=bucket_depth, fill_mode=fill_mode)
    if is_cuda:
        torch.cuda.synchronize()

    # Timed runs
    if is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(repeats):
            _ = engine.replay_step(token_ids=token_ids, pos=0, bucket_depth=bucket_depth, fill_mode=fill_mode)
        end_event.record()
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)
        latency_ms = total_ms / repeats
    else:
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = engine.replay_step(token_ids=token_ids, pos=0, bucket_depth=bucket_depth, fill_mode=fill_mode)
        elapsed = time.perf_counter() - t0
        latency_ms = (elapsed / repeats) * 1000.0

    return latency_ms


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    batch_sizes = [1, 2] if args.quick else [int(b) for b in args.batch_sizes.split(",")]
    buckets = [4, 8] if args.quick else [int(d) for d in args.buckets.split(",")]
    warmup = 5 if args.quick else args.warmup
    repeats = 10 if args.quick else args.repeats

    print(f"=== Fill Cost Inside Captured Graph & Fused Fill (B3, B4) ===")
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

    out_csv = os.path.join(args.output_dir, "graph_fill_cost_and_fused_benchmark.csv")
    rows = []

    print(f"\n{'B':<3} | {'Bucket':<7} | {'t_full (ms)':<11} | {'t_no_fill':<10} | {'t_seq_fill':<11} | {'t_fused_fill':<12} | {'Fill % Saved':<13} | {'Fused Speedup':<14}")
    print("-" * 92)

    for B in batch_sizes:
        # Full 12-layer baseline latency
        t_full = benchmark_step_latency(engine, batch_size=B, bucket_depth=12, fill_mode="none", warmup=warmup, repeats=repeats)

        for d in buckets:
            # 1. No fill (raw skipping)
            t_no_fill = benchmark_step_latency(engine, batch_size=B, bucket_depth=d, fill_mode="none", warmup=warmup, repeats=repeats)

            # 2. Sequential Native Projection fill
            t_seq = benchmark_step_latency(engine, batch_size=B, bucket_depth=d, fill_mode="sequential_projection", warmup=warmup, repeats=repeats)

            # 3. Fused batched GEMM fill
            t_fused = benchmark_step_latency(engine, batch_size=B, bucket_depth=d, fill_mode="fused_projection", warmup=warmup, repeats=repeats)

            time_saved = max(1e-4, t_full - t_no_fill)
            fill_overhead_seq = max(0.0, t_seq - t_no_fill)
            fill_overhead_fused = max(0.0, t_fused - t_no_fill)

            pct_saved_seq = (fill_overhead_seq / time_saved) * 100.0
            pct_saved_fused = (fill_overhead_fused / time_saved) * 100.0
            fused_speedup = t_seq / max(1e-4, t_fused)

            print(f"{B:<3} | L={d:<5} | {t_full:8.3f} ms | {t_no_fill:7.3f} ms | {t_seq:8.3f} ms | {t_fused:9.3f} ms | {pct_saved_fused:9.1f}%   | {fused_speedup:9.2f}x")

            rows.append({
                "batch_size": B,
                "bucket_depth": d,
                "t_full_ms": round(t_full, 4),
                "t_no_fill_ms": round(t_no_fill, 4),
                "t_seq_fill_ms": round(t_seq, 4),
                "t_fused_fill_ms": round(t_fused, 4),
                "time_saved_ms": round(time_saved, 4),
                "fill_overhead_seq_ms": round(fill_overhead_seq, 4),
                "fill_overhead_fused_ms": round(fill_overhead_fused, 4),
                "pct_saved_seq": round(pct_saved_seq, 2),
                "pct_saved_fused": round(pct_saved_fused, 2),
                "fused_fill_speedup": round(fused_speedup, 2),
            })

    fieldnames = [
        "batch_size", "bucket_depth", "t_full_ms", "t_no_fill_ms", "t_seq_fill_ms",
        "t_fused_fill_ms", "time_saved_ms", "fill_overhead_seq_ms",
        "fill_overhead_fused_ms", "pct_saved_seq", "pct_saved_fused", "fused_fill_speedup"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Fill cost & fused fill benchmark results saved to: {out_csv}")


if __name__ == "__main__":
    main()
