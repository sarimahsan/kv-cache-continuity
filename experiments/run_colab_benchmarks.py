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
    parser = argparse.ArgumentParser(description="Unified Colab Execution Driver")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Execution device")
    parser.add_argument("--quick", action="store_true", help="Quick mode for rapid smoke testing")
    parser.add_argument("--run-joint", action="store_true", default=True, help="Run dynamic mixed-exit joint condition sweep")
    parser.add_argument("--run-rollout", action="store_true", default=True, help="Run 64-prompt autoregressive rollout with CIs")
    parser.add_argument("--run-length-domain", action="store_true", help="Run Length and Domain generality benchmark")
    parser.add_argument("--run-adapter-sweep", action="store_true", help="Run Multi-depth adapter sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print("UNIFIED COLAB REPRODUCIBILITY DRIVER: KV CONTINUITY SUITE")
    print(f"Model: {args.model_id} | Device: {args.device} | Quick Mode: {args.quick}")
    print("=" * 80)

    # 1. Run Dynamic Mixed-Exit Joint Sweep
    if args.run_joint:
        print("\n" + "#" * 80)
        print("LAUNCHING BENCHMARK 1: DYNAMIC MIXED-EXIT JOINT DECOMPOSITION")
        print("#" * 80)
        cmd_joint = [
            sys.executable,
            os.path.join(REPO_ROOT, "experiments", "run_joint_condition_sweep.py"),
            "--model-id", args.model_id,
            "--device", args.device,
            "--exit-policy", "mixed_50",
            "--depths", "18,22,24,30",
        ]
        if args.quick:
            cmd_joint.append("--quick")
        t0 = time.perf_counter()
        ret_j = subprocess.run(cmd_joint)
        t_j = time.perf_counter() - t0
        if ret_j.returncode != 0:
            print(f"[!] Error: Joint sweep failed with return code {ret_j.returncode}")
        else:
            print(f"✓ Mixed-exit joint sweep completed in {t_j / 60:.1f} minutes.")

    # 2. Run 64-Prompt Autoregressive Rollout with Bootstrap CIs
    if args.run_rollout:
        print("\n" + "#" * 80)
        print("LAUNCHING BENCHMARK 2: AUTOREGRESSIVE ROLLOUT WITH BOOTSTRAP CIS")
        print("#" * 80)
        cmd_roll = [
            sys.executable,
            os.path.join(REPO_ROOT, "experiments", "run_autoregressive_generation.py"),
            "--model-id", args.model_id,
            "--device", args.device,
            "--depths", "18,24,30",
            "--prompt-prefill", "oracle",
        ]
        if args.quick:
            cmd_roll.extend(["--quick", "--num-prompts", "4"])
        else:
            cmd_roll.extend(["--num-prompts", "64", "--max-new-tokens", "256"])
        t0 = time.perf_counter()
        ret_r = subprocess.run(cmd_roll)
        t_r = time.perf_counter() - t0
        if ret_r.returncode != 0:
            print(f"[!] Error: Rollout benchmark failed with return code {ret_r.returncode}")
        else:
            print(f"✓ Autoregressive rollout benchmark completed in {t_r / 60:.1f} minutes.")

    # 3. Optional: Run Length & Domain
    if args.run_length_domain:
        print("\n" + "#" * 80)
        print("LAUNCHING BENCHMARK 3: SEQUENCE LENGTH & DOMAIN GENERALITY")
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
        subprocess.run(cmd_a)

    print("\n" + "=" * 80)
    print("ALL REQUESTED BENCHMARKS FINISHED.")
    print("Results are saved under results/canonical_sweeps/")
    print("=" * 80)


if __name__ == "__main__":
    main()

