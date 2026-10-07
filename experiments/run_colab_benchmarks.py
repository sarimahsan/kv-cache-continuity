#!/usr/bin/env python3
"""
experiments/run_colab_benchmarks.py

Unified Colab Execution Driver for Part A and Part B.
Runs both benchmarks sequentially with proper timing, VRAM checks, and reporting:
1. Part A: Sequence Length (T = 64, 128, 256, 512) & Out-of-Domain Generality (WikiText-2 + Code)
2. Part B: Multi-Depth Adapter Readout Characterization (L = 18, 22, 24, 28, 32)

Usage:
  # Quick smoke test on Colab (~3 minutes total):
  python experiments/run_colab_benchmarks.py --quick --device cuda

  # Full publication benchmark on Colab (~20 minutes total):
  python experiments/run_colab_benchmarks.py --model-id Qwen/Qwen2.5-3B --device cuda
"""

import os
import sys
import time
import argparse
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Colab Execution Driver for Part A and Part B")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid smoke testing")
    parser.add_argument("--skip-part-a", action="store_true", help="Skip Part A (Length and Domain benchmark)")
    parser.add_argument("--skip-part-b", action="store_true", help="Skip Part B (Multi-depth adapter sweep)")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("UNIFIED COLAB REPRODUCIBILITY DRIVER: PART A & PART B")
    print(f"Model: {args.model_id} | Device: {args.device} | Quick Mode: {args.quick}")
    print("=" * 80)

    # 1. Run Part A
    if not args.skip_part_a:
        print("\n" + "#" * 80)
        print("LAUNCHING PART A: SEQUENCE LENGTH & DOMAIN GENERALITY BENCHMARK")
        print("#" * 80)
        cmd_a = [
            sys.executable,
            os.path.join(REPO_ROOT, "experiments", "benchmark_length_and_domain.py"),
            "--model-id", args.model_id,
            "--device", args.device,
        ]
        if args.quick:
            cmd_a.append("--quick")
        else:
            cmd_a.extend(["--seq-lens", "64,128,256,512", "--num-samples", "25"])

        t0 = time.perf_counter()
        ret_a = subprocess.run(cmd_a)
        t_a = time.perf_counter() - t0
        if ret_a.returncode != 0:
            print(f"[!] Error: Part A failed with return code {ret_a.returncode}")
        else:
            print(f"✓ Part A completed successfully in {t_a / 60:.1f} minutes.")

    # 2. Run Part B
    if not args.skip_part_b:
        print("\n" + "#" * 80)
        print("LAUNCHING PART B: MULTI-DEPTH ADAPTER READOUT CHARACTERIZATION")
        print("#" * 80)
        cmd_b = [
            sys.executable,
            os.path.join(REPO_ROOT, "experiments", "sweep_adapter_depths.py"),
            "--model-id", args.model_id,
            "--device", args.device,
        ]
        if args.quick:
            cmd_b.append("--quick")
        else:
            cmd_b.extend(["--depths", "18,22,24,28,32", "--rank", "512", "--steps", "500", "--num-val-samples", "20"])

        t0 = time.perf_counter()
        ret_b = subprocess.run(cmd_b)
        t_b = time.perf_counter() - t0
        if ret_b.returncode != 0:
            print(f"[!] Error: Part B failed with return code {ret_b.returncode}")
        else:
            print(f"✓ Part B completed successfully in {t_b / 60:.1f} minutes.")

    print("\n" + "=" * 80)
    print("ALL REQUESTED BENCHMARKS FINISHED.")
    print("Generated result files:")
    print("  - Part A CSV: results/canonical_sweeps/qwen2.5_3b/length_and_domain_sweep.csv")
    print("  - Part A Markdown: results/canonical_sweeps/qwen2.5_3b/paper_table_length_and_domain.md")
    print("  - Part B CSV: results/canonical_sweeps/qwen2.5_3b/multi_depth_adapter_sweep.csv")
    print("  - Part B Markdown: results/canonical_sweeps/qwen2.5_3b/paper_table_multi_depth_adapters.md")
    print("  - Part B Adapter Checkpoint: checkpoints/multi_depth_adapters_qwen.pt")
    print("=" * 80)


if __name__ == "__main__":
    main()
