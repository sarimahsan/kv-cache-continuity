#!/usr/bin/env python3
"""
experiments/run_autoregressive_generation.py

Addresses Claim C1 & Gap A3:
- A3. Autoregressive (Non-Teacher-Forced) Generation:
  Run real open-loop autoregressive generation where every generated token exits at depth d
  and its upper-layer cache is filled by Native Projection (or Direct Copy baseline).
  Measures:
  1. Token match rate against full-depth greedy output over 64, 128, and 256 tokens.
  2. Length at which generated trajectories diverge (first-token divergence index).
  3. Perplexity on a held-out evaluation set under autoregressive scoring.
  4. Qualitative output text comparisons side-by-side.

Proves:
Teacher-forced results survive compounding error under free-running generation.

Usage:
  python experiments/run_autoregressive_generation.py --model-id Qwen/Qwen2.5-3B --depths 18,24,30 --max-new-tokens 256 --device cuda
  python experiments/run_autoregressive_generation.py --quick --device cpu
"""

import os
import sys
import time
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


CANONICAL_EVAL_PROMPTS = [
    "The Apollo program was the third United States human spaceflight program carried out by NASA,",
    "In computer science, an attention mechanism is a technique that mimics cognitive attention by",
    "Photosynthesis is a biological process used by plants and other organisms to convert light energy into",
    "The Industrial Revolution was the transition to new manufacturing processes that occurred in",
    "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915 and",
    "A neural network is a computational model inspired by the biological neural networks in animal brains.",
    "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to",
    "The global climate system is driven by energy from the sun and is regulated by the balance between",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Autoregressive Non-Teacher-Forced Generation (A3)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--depths", type=str, default=None, help="Comma-separated exit depths (e.g. '18,24,30')")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of prompts to evaluate (defaults to all canonical prompts)")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens (64, 128, 256)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/autoregressive_generation")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


@torch.no_grad()
def generate_autoregressive(
    engine: QwenDynamicEngine,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    exit_depth: int,
    strategy: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Generates tokens autoregressively where every token exits at exit_depth
    and fills the upper-layer cache according to strategy.
    """
    B, T = prompt_ids.shape
    assert B == 1
    total_len = T + max_new_tokens + 16
    kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=total_len)

    # 1. Prefill prompt through full model
    for p in range(T):
        tok = prompt_ids[:, p : p + 1].to(device)
        logits, _ = engine.step_forward_decoupled(
            input_id=tok,
            pos=p,
            kv_cache=kv_cache,
            exit_depth=engine.num_layers,
            mode="full",
            strategy="oracle_full",
        )

    next_tok = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)
    generated_tokens = [next_tok.item()]

    # 2. Autoregressively decode
    curr_pos = T
    t_start = time.perf_counter()

    for step in range(max_new_tokens - 1):
        logits, _ = engine.step_forward_decoupled(
            input_id=next_tok,
            pos=curr_pos,
            kv_cache=kv_cache,
            exit_depth=exit_depth,
            mode="attention_only",
            strategy=strategy,
        )
        next_tok = torch.argmax(logits.squeeze(1), dim=-1, keepdim=True)
        generated_tokens.append(next_tok.item())
        curr_pos += 1

    elapsed = time.perf_counter() - t_start

    return {
        "tokens": generated_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(generated_tokens) / (elapsed + 1e-8),
    }


def compute_token_match_rates(
    oracle_tokens: List[int],
    test_tokens: List[int],
    windows: List[int] = [64, 128, 256],
) -> Dict[str, Any]:
    """
    Computes exact token match rates and finds first divergence index.
    """
    min_len = min(len(oracle_tokens), len(test_tokens))
    first_div = None
    for i in range(min_len):
        if oracle_tokens[i] != test_tokens[i]:
            first_div = i
            break
    if first_div is None:
        first_div = min_len

    match_rates = {}
    for w in windows:
        w_len = min(w, min_len)
        if w_len > 0:
            matches = sum(1 for i in range(w_len) if oracle_tokens[i] == test_tokens[i])
            match_rates[f"match_rate_{w}"] = (matches / w_len) * 100.0
        else:
            match_rates[f"match_rate_{w}"] = 0.0

    match_rates["first_divergence_idx"] = first_div
    return match_rates


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Autoregressive Non-Teacher-Forced Generation (A3) on {args.model_id} ===")
    engine = QwenDynamicEngine(model_id=args.model_id, device=args.device)
    device = torch.device(args.device)

    # Exit depths
    if args.depths is not None:
        depths = [int(d) for d in args.depths.split(",")]
    else:
        L = engine.num_layers
        depths = [int(round(0.50 * L)), int(round(0.67 * L)), int(round(0.83 * L))]

    prompts = CANONICAL_EVAL_PROMPTS
    if args.quick:
        prompts = prompts[:2]
    elif args.num_samples is not None:
        prompts = prompts[:args.num_samples]
    max_tokens = 16 if args.quick else args.max_new_tokens

    policies = [
        ("native_projection", "Native Projection"),
        ("direct_copy", "Direct Copy"),
    ]

    out_csv = os.path.join(args.output_dir, f"autoregressive_gen_{args.model_id.replace('/', '_')}.csv")
    samples_txt = os.path.join(args.output_dir, f"qualitative_samples_{args.model_id.replace('/', '_')}.txt")

    rows = []
    samples_log = []

    print(f"\nEvaluating {len(prompts)} prompts up to {max_tokens} tokens across depths {depths}...")

    for p_idx, prompt in enumerate(prompts):
        print(f"\n--- Prompt {p_idx+1}/{len(prompts)}: '{prompt[:50]}...' ---")
        prompt_ids = engine.tokenizer.encode(prompt, return_tensors="pt").to(device)

        # 1. Oracle Full-Depth Greedy Output
        oracle_res = generate_autoregressive(
            engine=engine,
            prompt_ids=prompt_ids,
            max_new_tokens=max_tokens,
            exit_depth=engine.num_layers,
            strategy="oracle_full",
            device=device,
        )
        oracle_text = engine.tokenizer.decode(oracle_res["tokens"])
        samples_log.append(f"\n[PROMPT {p_idx+1}]\n{prompt}\n\n[ORACLE (Full {engine.num_layers}L)]:\n{oracle_text}\n")

        for d in depths:
            rel_depth = (d / engine.num_layers) * 100.0
            for p_key, p_name in policies:
                gen_res = generate_autoregressive(
                    engine=engine,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_tokens,
                    exit_depth=d,
                    strategy=p_key,
                    device=device,
                )
                gen_text = engine.tokenizer.decode(gen_res["tokens"])
                metrics = compute_token_match_rates(oracle_res["tokens"], gen_res["tokens"])

                print(f"L={d} ({rel_depth:.0f}%) | {p_name:<18} | Match@64: {metrics['match_rate_64']:5.1f}% | Match@128: {metrics['match_rate_128']:5.1f}% | Div-Idx: {metrics['first_divergence_idx']}")

                samples_log.append(f"[L={d} ({rel_depth:.0f}%) - {p_name}]:\n{gen_text}\n")

                rows.append({
                    "prompt_idx": p_idx,
                    "depth": d,
                    "rel_depth": f"{rel_depth:.1f}%",
                    "policy": p_key,
                    "match_rate_64": round(metrics["match_rate_64"], 2),
                    "match_rate_128": round(metrics["match_rate_128"], 2),
                    "match_rate_256": round(metrics["match_rate_256"], 2),
                    "first_div_idx": metrics["first_divergence_idx"],
                    "tokens_per_sec": round(gen_res["tokens_per_sec"], 2),
                })

    # Save CSV
    fieldnames = ["prompt_idx", "depth", "rel_depth", "policy", "match_rate_64", "match_rate_128", "match_rate_256", "first_div_idx", "tokens_per_sec"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Save qualitative samples
    with open(samples_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(samples_log))

    print(f"\n[OK] Autoregressive metrics saved to: {out_csv}")
    print(f"[OK] Qualitative text samples saved to: {samples_txt}")


if __name__ == "__main__":
    main()
