#!/usr/bin/env python3
"""
experiments/benchmark_speculative_decoding.py

End-to-End Self-Speculative Decoding Benchmark on Qwen-2.5-3B.
Validates:
1. 100% Lossless Output Equivalence (Greedy Verification).
2. Empirical Acceptance Rate (alpha).
3. Wall-Clock Speedup vs. Full 36-Layer Baseline.
4. Exact GPU VRAM Peak Allocation (<8 GB on 15 GB GPU).

Usage:
  python experiments/benchmark_speculative_decoding.py \
      --model-id Qwen/Qwen2.5-3B \
      --adapter-path checkpoints/exit_adapters_qwen.pt \
      --draft-depth 24 \
      --gamma 2 \
      --max-tokens 50 \
      --device cuda
"""

import os
import sys
import time
import argparse
import torch

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.exit_adapter import ExitCalibrationModule
from src.speculative_engine import QwenSelfSpeculativeEngine


BENCHMARK_PROMPTS = [
    "The Apollo program was the third United States human spaceflight program carried out by NASA,",
    "In computer science, an attention mechanism is a technique that mimics cognitive attention by",
    "Photosynthesis is a biological process used by plants and other organisms to convert light energy into",
    "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915 and",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Self-Speculative Decoding on Qwen-2.5")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Hugging Face model ID")
    default_adapter = "checkpoints/exit_adapters_qwen_mlp.pt" if os.path.exists("checkpoints/exit_adapters_qwen_mlp.pt") else "checkpoints/exit_adapters_qwen.pt"
    parser.add_argument("--adapter-path", type=str, default=default_adapter, help="Path to ExitCalibrationModule checkpoint")
    parser.add_argument("--draft-depth", type=int, default=24, help="Exit layer for drafting (e.g., 24 or 29)")
    parser.add_argument("--gamma", type=int, default=2, help="Number of draft tokens per cycle")
    parser.add_argument("--max-tokens", type=int, default=40, help="Tokens to generate per prompt")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Model dtype")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def run_baseline_generation(engine: QwenSelfSpeculativeEngine, prompt_ids: torch.Tensor, max_new_tokens: int):
    """Generates standard autoregressive tokens using full 36 layers."""
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
    peak_vram = torch.cuda.max_memory_allocated(engine.device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

    return {
        "tokens": generated,
        "text": engine.tokenizer.decode(generated, skip_special_tokens=True),
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(generated) / elapsed,
        "peak_vram_mb": peak_vram,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "float16" else (torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)

    print("=" * 80)
    print("SELF-SPECULATIVE DECODING BENCHMARK (Continuous Projected KV Cache)")
    print(f"Model: {args.model_id} | Device: {device} | Dtype: {torch_dtype}")
    print(f"Draft Depth: L = {args.draft_depth} | Speculative Horizon: gamma = {args.gamma}")
    print("=" * 80)

    # 1. Load Tokenizer & Model
    print(f"[*] Loading model '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    # 2. Load Exit Adapters if provided
    adapter_module = None
    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"[*] Loading ExitCalibrationModule from '{args.adapter_path}'...")
        adapter_module = ExitCalibrationModule.load_checkpoint(args.adapter_path, device=device)
        adapter_module.eval()
        print(f"✓ Loaded adapters for depths: {adapter_module.candidate_depths}")
    else:
        print("[!] No exit adapter found. Falling back to native projection readout.")

    # 3. Initialize Speculative Engine
    engine = QwenSelfSpeculativeEngine(
        model=model,
        tokenizer=tokenizer,
        draft_depth=args.draft_depth,
        adapter_module=adapter_module,
        device=args.device,
        max_seq_len=2048,
    )

    baseline_speeds = []
    spec_speeds = []
    acceptance_rates = []
    match_count = 0

    print("\n" + "-" * 80)
    print(f"{'Prompt #':<10} {'Baseline (tok/s)':<18} {'Speculative (tok/s)':<22} {'Speedup':<10} {'Acceptance %':<14} {'Exact Match?'}")
    print("-" * 80)

    for p_idx, prompt_text in enumerate(BENCHMARK_PROMPTS):
        torch.cuda.reset_peak_memory_stats(device)
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        # A. Baseline Full-Depth Run
        base_res = run_baseline_generation(engine, prompt_ids, max_new_tokens=args.max_tokens)
        baseline_speeds.append(base_res["tokens_per_sec"])

        # B. Self-Speculative Run
        torch.cuda.reset_peak_memory_stats(device)
        spec_res = engine.generate_speculative(
            prompt_ids,
            max_new_tokens=args.max_tokens,
            gamma=args.gamma,
            temperature=0.0  # Greedy decoding for lossless verification
        )
        spec_speeds.append(spec_res["tokens_per_sec"])
        acceptance_rates.append(spec_res["acceptance_rate"])

        # C. Verification of Exact Token Equivalence
        is_identical = (base_res["tokens"] == spec_res["tokens"])
        if is_identical:
            match_count += 1

        speedup = spec_res["tokens_per_sec"] / base_res["tokens_per_sec"]
        print(
            f"{p_idx + 1:<10} {base_res['tokens_per_sec']:<18.2f} {spec_res['tokens_per_sec']:<22.2f} "
            f"{speedup:<10.2f}x {spec_res['acceptance_rate']:<14.1f}% {'✓ YES' if is_identical else '✗ NO'}"
        )

    avg_base_speed = sum(baseline_speeds) / len(baseline_speeds)
    avg_spec_speed = sum(spec_speeds) / len(spec_speeds)
    avg_speedup = avg_spec_speed / avg_base_speed
    avg_acceptance = sum(acceptance_rates) / len(acceptance_rates)
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0

    print("-" * 80)
    print(f"AVERAGE SPEEDUP:         {avg_speedup:.2f}x")
    print(f"AVERAGE ACCEPTANCE RATE: {avg_acceptance:.1f}%")
    print(f"LOSSLESS EQUIVALENCE:    {match_count}/{len(BENCHMARK_PROMPTS)} prompts matched 100% token-for-token")
    print(f"PEAK GPU VRAM CONSUMED:  {peak_vram_gb:.2f} GB / 15.0 GB limit")
    print("=" * 80)


if __name__ == "__main__":
    main()
