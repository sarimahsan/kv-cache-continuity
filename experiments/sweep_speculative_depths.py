#!/usr/bin/env python3
"""
experiments/sweep_speculative_depths.py

Systematic Depth vs. Acceptance Rate vs. Speedup Sweep for Self-Speculative Decoding.
Resolves the operational trade-off between draft cost ratio c = d_t / L and acceptance rate alpha(d_t).

Evaluates across candidate depths d_t in {12, 14, 18, 22, 24, 28, 30}:
1. Empirical Draft Acceptance Rate: alpha(d_t)
2. Theoretical Naive Speedup: S_naive(d_t) = (1 + alpha) / (c + 1)
3. Theoretical Partial-Verification Speedup: S_partial(d_t) = (1 + alpha) / (c + (1 - c)) = 1 + alpha
4. Real-world Wall-clock Throughput (tok/s) and Measured Speedup
5. Output Equivalence Rate (% exact token match with oracle)

Usage:
    python experiments/sweep_speculative_depths.py \\
        --model-id Qwen/Qwen2.5-3B \\
        --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \\
        --depths 14,18,22,24,28,30 \\
        --gamma 1 \\
        --num-prompts 12 \\
        --max-tokens 30 \\
        --device cuda
"""

import os
import sys
import time
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.exit_adapter import ExitCalibrationModule
from src.speculative_engine import QwenSelfSpeculativeEngine

EVAL_PROMPTS = [
    "If a train travels at 60 miles per hour and leaves at 2:00 PM, the time it takes to travel 180 miles is",
    "The quadratic formula to solve equations of the form ax^2 + bx + c = 0 is given by",
    "To prove that the square root of 2 is irrational, we begin by assuming that",
    "Write a Python function to compute the longest common subsequence of two strings using dynamic programming:",
    "Implement a binary search tree in C++ with methods for insertion, search, and deletion:",
    "Explain the difference between TCP and UDP protocols in computer networking:",
    "Photosynthesis is a biological process used by plants and other organisms to convert light energy into",
    "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915 and",
    "The central dogma of molecular biology states that genetic information flows from DNA to RNA and",
    "The Industrial Revolution marked a major turning point in human history, leading to unprecedented changes in",
    "In philosophy, the mind-body problem examines the relationship between consciousness and the physical brain, specifically",
    "The primary cause of the fall of the Western Roman Empire remains a subject of intense debate among historians, including",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep speculative draft depths vs. acceptance and speedup.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Model ID")
    parser.add_argument("--adapter-path", type=str, default="checkpoints/exit_adapters_qwen_mlp.pt", help="Path to adapter checkpoint")
    parser.add_argument("--depths", type=str, default="14,18,22,24,28,30", help="Comma-separated draft depths")
    parser.add_argument("--gamma", type=int, default=1, help="Speculative drafting horizon")
    parser.add_argument("--num-prompts", type=int, default=12, help="Number of benchmark prompts")
    parser.add_argument("--max-tokens", type=int, default=30, help="Tokens to generate per prompt")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/qwen2.5_3b", help="Output directory")
    return parser.parse_args()


def run_baseline(engine: QwenSelfSpeculativeEngine, prompt_ids: torch.Tensor, max_new_tokens: int) -> Dict[str, Any]:
    t_start = time.perf_counter()
    curr_logits = engine.prefill(prompt_ids)
    next_tok = torch.argmax(curr_logits[:, -1, :], dim=-1, keepdim=True)
    generated = [next_tok.item()]

    curr_pos = engine.kv_cache["current_len"]
    curr_input = next_tok

    for _ in range(max_new_tokens - 1):
        logits = engine.forward_full_step(curr_input, pos=curr_pos)
        next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_tok.item())
        curr_input = next_tok
        curr_pos += 1

    elapsed = time.perf_counter() - t_start
    return {
        "tokens": generated,
        "elapsed_sec": elapsed,
        "tok_per_sec": len(generated) / elapsed
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "float16" else (torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)
    os.makedirs(args.output_dir, exist_ok=True)

    candidate_depths = [int(d.strip()) for d in args.depths.split(",") if d.strip()]

    print("=" * 80)
    print("SPECULATIVE DRAFT DEPTH vs. ACCEPTANCE & SPEEDUP TRADE-OFF SWEEP")
    print(f"Model ID: {args.model_id} | Device: {device} | Dtype: {torch_dtype}")
    print(f"Target Depths to Sweep: {candidate_depths}")
    print(f"Draft Horizon: gamma = {args.gamma} | Max Tokens: {args.max_tokens}")
    print("=" * 80)

    print(f"[*] Loading model '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    total_layers = model.config.num_hidden_layers  # 36 for Qwen2.5-3B

    # Load Adapter Module if provided
    adapter_module = None
    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"[*] Loading exit calibration adapter from '{args.adapter_path}'...")
        adapter_module = ExitCalibrationModule.load_checkpoint(args.adapter_path, device=device)
        adapter_module.eval()
        print(f"✓ Loaded adapter for depths: {adapter_module.candidate_depths}")
    else:
        print("[!] No exit adapter found. Falling back to native projection readout.")

    selected_prompts = EVAL_PROMPTS[:args.num_prompts]
    encoded_prompts = [tokenizer.encode(p, return_tensors="pt").to(device) for p in selected_prompts]

    # Measure baseline throughput across the prompts
    print("\n[*] Benchmarking Baseline Full-Model Throughput (L = 36)...")
    temp_engine = QwenSelfSpeculativeEngine(
        model=model,
        tokenizer=tokenizer,
        draft_depth=candidate_depths[0],
        adapter_module=None,
        device=args.device,
        max_seq_len=2048,
    )

    baseline_runs = []
    baseline_speeds = []
    for p_idx, p_ids in enumerate(encoded_prompts):
        res = run_baseline(temp_engine, p_ids, max_new_tokens=args.max_tokens)
        baseline_runs.append(res)
        baseline_speeds.append(res["tok_per_sec"])

    mean_base_speed = sum(baseline_speeds) / len(baseline_speeds)
    print(f"✓ Mean Baseline Throughput: {mean_base_speed:.2f} tokens/sec")

    sweep_results = []

    print("\n" + "=" * 90)
    print(f"{'Depth':<6} {'Rel %':<8} {'Cost c':<8} {'Alpha (%)':<12} {'S_naive':<10} {'S_partial':<11} {'Speed (tok/s)':<14} {'S_meas':<8} {'Exact %'}")
    print("=" * 90)

    for depth in candidate_depths:
        c = depth / total_layers
        rel_pct = c * 100.0

        engine = QwenSelfSpeculativeEngine(
            model=model,
            tokenizer=tokenizer,
            draft_depth=depth,
            adapter_module=adapter_module,
            device=args.device,
            max_seq_len=2048,
        )

        acc_rates = []
        spec_speeds = []
        matches = []

        for p_idx, p_ids in enumerate(encoded_prompts):
            spec_res = engine.generate_speculative(
                p_ids,
                max_new_tokens=args.max_tokens,
                gamma=args.gamma,
                temperature=0.0
            )
            acc_rates.append(spec_res["acceptance_rate"])
            spec_speeds.append(spec_res["tokens_per_sec"])
            is_match = (spec_res["tokens"] == baseline_runs[p_idx]["tokens"])
            matches.append(1 if is_match else 0)

        mean_alpha = sum(acc_rates) / len(acc_rates)
        mean_spec_speed = sum(spec_speeds) / len(spec_speeds)
        exact_pct = (sum(matches) / len(matches)) * 100.0

        # Theoretical Speedup Calculations
        alpha_frac = mean_alpha / 100.0
        s_naive = (1.0 + alpha_frac) / (c + 1.0)
        s_partial = 1.0 + alpha_frac
        s_measured = mean_spec_speed / mean_base_speed

        sweep_results.append({
            "depth": depth,
            "rel_depth_pct": rel_pct,
            "cost_ratio_c": c,
            "alpha_pct": mean_alpha,
            "s_naive_theoretical": s_naive,
            "s_partial_theoretical": s_partial,
            "mean_baseline_speed": mean_base_speed,
            "mean_speculative_speed": mean_spec_speed,
            "s_measured": s_measured,
            "exact_match_pct": exact_pct
        })

        print(f"L = {depth:<3} {rel_pct:>5.1f}%  {c:>6.3f}  {mean_alpha:>9.2f}%   {s_naive:>8.3f}x   {s_partial:>9.3f}x   {mean_spec_speed:>10.2f}    {s_measured:>6.3f}x  {exact_pct:>6.1f}%")

    print("=" * 90)

    # Save to CSV
    csv_path = os.path.join(args.output_dir, "speculative_depth_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("depth,rel_depth_pct,cost_ratio_c,alpha_pct,s_naive_theoretical,s_partial_theoretical,mean_baseline_speed,mean_speculative_speed,s_measured,exact_match_pct\n")
        for r in sweep_results:
            f.write(f"{r['depth']},{r['rel_depth_pct']:.2f},{r['cost_ratio_c']:.4f},{r['alpha_pct']:.2f},{r['s_naive_theoretical']:.4f},{r['s_partial_theoretical']:.4f},{r['mean_baseline_speed']:.2f},{r['mean_speculative_speed']:.2f},{r['s_measured']:.4f},{r['exact_match_pct']:.2f}\n")

    print(f"\n✓ Saved depth sweep CSV to: {csv_path}")

    # Generate Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        depths = [r["depth"] for r in sweep_results]
        rel_depths = [r["rel_depth_pct"] for r in sweep_results]
        alphas = [r["alpha_pct"] for r in sweep_results]
        s_naives = [r["s_naive_theoretical"] for r in sweep_results]
        s_partials = [r["s_partial_theoretical"] for r in sweep_results]
        s_meas = [r["s_measured"] for r in sweep_results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: Acceptance Rate vs Depth
        ax1.plot(rel_depths, alphas, "o-", color="#2563EB", linewidth=2, markersize=7, label=r"Draft Acceptance Rate $\alpha(d)$")
        ax1.set_xlabel("Relative Draft Depth (%)", fontsize=12)
        ax1.set_ylabel(r"Acceptance Rate $\alpha$ (%)", fontsize=12, color="#2563EB")
        ax1.tick_params(axis="y", labelcolor="#2563EB")
        ax1.set_title("(a) Empirical Acceptance Scaling", fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)

        for d, rd, a in zip(depths, rel_depths, alphas):
            ax1.annotate(f"L={d}\n({a:.1f}%)", (rd, a), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

        # Panel 2: Speedup Comparison: Naive vs Partial vs Measured
        ax2.plot(rel_depths, s_partials, "^-", color="#16A34A", linewidth=2, markersize=7, label=r"Partial Verification $\mathcal{S}_{\text{partial}} = 1+\alpha$")
        ax2.plot(rel_depths, s_meas, "s-", color="#7C3AED", linewidth=2, markersize=7, label=r"Measured GPU Speedup $\mathcal{S}_{\text{meas}}$")
        ax2.plot(rel_depths, s_naives, "v--", color="#DC2626", linewidth=2, markersize=7, label=r"Naive Verification $\mathcal{S}_{\text{naive}} = \frac{1+\alpha}{c+1}$")

        ax2.axhline(1.0, color="#6B7280", linestyle=":", linewidth=1.5, label="Break-Even ($1.0\\times$)")
        ax2.set_xlabel("Relative Draft Depth (%)", fontsize=12)
        ax2.set_ylabel("Speedup Ratio", fontsize=12)
        ax2.set_title("(b) Speedup: Naive vs. Partial Verification", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", framealpha=0.9, fontsize=9)

        fig.suptitle("Self-Speculative Decoding Operational Trade-Off (Qwen2.5-3B)", fontsize=14, fontweight="bold", y=1.02)
        fig.tight_layout()

        plot_path_pdf = os.path.join(args.output_dir, "speculative_depth_tradeoff.pdf")
        plot_path_png = os.path.join(args.output_dir, "speculative_depth_tradeoff.png")
        fig.savefig(plot_path_pdf, bbox_inches="tight")
        fig.savefig(plot_path_png, bbox_inches="tight")
        plt.close(fig)
        print(f"✓ Saved operational trade-off plot to: {plot_path_pdf} and {plot_path_png}")
    except Exception as e:
        print(f"[!] Plot generation skipped: {e}")


if __name__ == "__main__":
    main()
