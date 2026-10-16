#!/usr/bin/env python3
"""
experiments/run_graph_context_scaling.py

Addresses Claim C6 & Gap B12:
- B12. Cache Length Scaling (Attention Read Scaling):
  Measures decode step latency and relative speedup vs. full 12-layer baseline
  as a function of KV cache context length:
    T in {512, 1024, 2048} tokens.

Hypothesis & Scientific Claim:
At short sequence lengths (T < 128), decode latency is dominated by kernel launch overhead.
At long context lengths (T >= 512, 1024, 2048), decode latency is memory-bandwidth bound
by reading cached keys and values from HBM.
Evaluates whether graph-bucketed layer skipping + Native Projection preserves speedup
when memory-bandwidth attention reads dominate.

Usage:
  python experiments/run_graph_context_scaling.py --lengths 512,1024,2048 --device cuda
  python experiments/run_graph_context_scaling.py --quick --device cpu
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
from src.cuda_graph_engine import BucketedCUDAGraphEngine


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Cache Length Scaling (B12)")
    parser.add_argument("--lengths", type=str, default="512,1024,2048", help="Context lengths to benchmark")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/context_scaling")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


def benchmark_context_length(
    engine: BucketedCUDAGraphEngine,
    pos: int,
    batch_size: int,
    bucket_depth: int,
    fill_mode: str,
    repeats: int = 50,
) -> float:
    """Times a single decode step at position `pos` with cache pre-populated up to pos."""
    device = engine.device
    tok = torch.randint(0, engine.config.vocab_size, (batch_size, 1), device=device)
    kv_cache = engine.get_or_create_kv_cache(batch_size)

    # Populate cache up to pos with synthetic representations
    for l in range(engine.config.n_layers):
        kv_cache.k_cache[l][:, :, :pos, :] = torch.randn(batch_size, engine.config.n_heads, pos, engine.config.d_model // engine.config.n_heads, device=device)
        kv_cache.v_cache[l][:, :, :pos, :] = torch.randn(batch_size, engine.config.n_heads, pos, engine.config.d_model // engine.config.n_heads, device=device)

    # Warmup
    for _ in range(10):
        _ = engine._forward_bucket_eager(tok, pos=pos, bucket_depth=bucket_depth, fill_mode=fill_mode, kv_cache=kv_cache)
    if engine.is_cuda:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = engine._forward_bucket_eager(tok, pos=pos, bucket_depth=bucket_depth, fill_mode=fill_mode, kv_cache=kv_cache)
    if engine.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (elapsed / repeats) * 1000.0


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    lengths = [64, 128] if args.quick else [int(t) for t in args.lengths.split(",")]
    repeats = 10 if args.quick else args.repeats

    print(f"=== Cache Context Length Scaling Benchmark (B12) ===")
    print(f"[*] Context lengths: {lengths}")
    device = torch.device(args.device)

    config = ModelConfig(
        vocab_size=50257,
        n_layers=12,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        max_seq_len=max(lengths) + 64,
    )
    model = TransformerLM(config).to(device).eval()
    engine = BucketedCUDAGraphEngine(model=model, buckets=[4, 6, 8, 10, 12], max_seq_len=max(lengths)+64, device=args.device)

    out_csv = os.path.join(args.output_dir, "graph_context_length_scaling_b12.csv")
    rows = []

    print(f"\n{'Context (T)':<12} | {'t_full (12L)':<14} | {'t_bucket (6L)':<14} | {'Speedup':<10} | {'Throughput (tok/s)':<18}")
    print("-" * 75)

    for T in lengths:
        t_full = benchmark_context_length(engine, pos=T, batch_size=args.batch_size, bucket_depth=12, fill_mode="none", repeats=repeats)
        t_bucket = benchmark_context_length(engine, pos=T, batch_size=args.batch_size, bucket_depth=6, fill_mode="fused_projection", repeats=repeats)

        speedup = t_full / max(1e-4, t_bucket)
        tok_s = (1000.0 / t_bucket) * args.batch_size

        print(f"T={T:<10} | {t_full:10.3f} ms | {t_bucket:10.3f} ms | {speedup:8.2f}x | {tok_s:14.1f} tok/s")

        rows.append({
            "context_length": T,
            "t_full_ms": round(t_full, 4),
            "t_bucket_6L_ms": round(t_bucket, 4),
            "speedup": round(speedup, 2),
            "tokens_per_sec": round(tok_s, 1),
        })

    fieldnames = ["context_length", "t_full_ms", "t_bucket_6L_ms", "speedup", "tokens_per_sec"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Context length scaling results saved to: {out_csv}")


if __name__ == "__main__":
    main()
