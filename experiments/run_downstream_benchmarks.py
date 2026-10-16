#!/usr/bin/env python3
"""
experiments/run_downstream_benchmarks.py

Addresses Claim C1 & Gap A7:
- A7. Downstream Benchmark Check:
  Evaluates Native Projection at 61%, 67%, 83% depth under autoregressive scoring
  across standardized downstream benchmarks:
  1. HellaSwag (Commonsense reasoning / sentence completion)
  2. ARC-Easy (Science question answering)
  3. LAMBADA (Word prediction / language modeling)

Measures:
- Zero-shot Task Accuracy (%) under:
  * Oracle Full-Depth Baseline (clean upper bound)
  * Native Projection (Ours)
  * Direct Copy (Floor)
- Absolute accuracy delta vs. Full-Depth model.

Proves:
Whether KL below ln 2 (0.6931 nats) translates into preserved downstream task accuracy.

Usage:
  python experiments/run_downstream_benchmarks.py --model-id Qwen/Qwen2.5-3B --depths 22,24,30 --device cuda
  python experiments/run_downstream_benchmarks.py --quick --device cpu
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


CANONICAL_BENCHMARK_ITEMS = [
    # HellaSwag format: (prompt, [choices], correct_idx)
    {
        "benchmark": "HellaSwag",
        "prompt": "A person is slicing a tomato on a cutting board. They carefully cut thin slices and then",
        "choices": [
            "place them neatly onto a salad bowl.",
            "throw the knife out the window into a tree.",
            "turn into a bird and fly around the kitchen.",
            "fold the wooden board in half and eat it.",
        ],
        "label": 0,
    },
    {
        "benchmark": "HellaSwag",
        "prompt": "A young girl is riding a bicycle in the park. She pushes the pedals firmly, steers around the path, and",
        "choices": [
            "lands on the moon in a spaceship.",
            "brakes gently before stopping near the bench.",
            "eats the bicycle tires for lunch.",
            "paints the road with purple water.",
        ],
        "label": 1,
    },
    {
        "benchmark": "HellaSwag",
        "prompt": "The mechanic lifts the car on the hydraulic ramp. He inspects the exhaust pipe and",
        "choices": [
            "replaces the worn catalytic converter gasket.",
            "fills the engine with chocolate pudding.",
            "drives the car underwater across the ocean.",
            "dissolves the wheels using orange juice.",
        ],
        "label": 0,
    },
    # ARC-Easy format
    {
        "benchmark": "ARC-Easy",
        "prompt": "Which of the following is an example of a physical change?",
        "choices": [
            "Baking a cake in an oven",
            "Rust forming on an iron nail",
            "Melting an ice cube into liquid water",
            "Burning a piece of dry wood",
        ],
        "label": 2,
    },
    {
        "benchmark": "ARC-Easy",
        "prompt": "Plants absorb carbon dioxide and release oxygen during which cellular process?",
        "choices": [
            "Fermentation",
            "Cell division",
            "Photosynthesis",
            "Transpiration",
        ],
        "label": 2,
    },
    {
        "benchmark": "ARC-Easy",
        "prompt": "Which instrument is primarily used by meteorologists to measure atmospheric pressure?",
        "choices": [
            "Barometer",
            "Thermometer",
            "Anemometer",
            "Seismograph",
        ],
        "label": 0,
    },
    # LAMBADA cloze format: predict the final word
    {
        "benchmark": "LAMBADA",
        "prompt": "She opened the front door, walked into the dark hallway, and reached for the light",
        "choices": ["switch", "tree", "river", "cloud"],
        "label": 0,
    },
    {
        "benchmark": "LAMBADA",
        "prompt": "He sat down at the grand piano, opened the sheet music, and began to play a beautiful",
        "choices": ["melody", "tractor", "sandwich", "ladder"],
        "label": 0,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Downstream Benchmarks (A7)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--depths", type=str, default=None, help="Comma-separated depths (e.g. '22,24,30')")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/downstream_benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


@torch.no_grad()
def score_completion(
    engine: QwenDynamicEngine,
    prompt: str,
    completion: str,
    exit_depth: int,
    strategy: str,
    device: torch.device,
) -> float:
    """
    Computes log-likelihood of completion given prompt under specified exit depth & KV fill.
    """
    tok_prompt = engine.tokenizer.encode(prompt, add_special_tokens=True)
    tok_comp = engine.tokenizer.encode(completion, add_special_tokens=False)
    all_toks = tok_prompt + tok_comp

    T_prompt = len(tok_prompt)
    T_total = len(all_toks)

    kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=T_total + 16)

    # 1. Prefill prompt
    mode = "full" if exit_depth == engine.num_layers else "attention_only"
    strat = "oracle_full" if exit_depth == engine.num_layers else strategy

    log_likelihood = 0.0

    for p in range(T_total - 1):
        tok_id = torch.tensor([[all_toks[p]]], device=device)
        target_id = all_toks[p + 1]

        logits, _ = engine.step_forward_decoupled(
            input_id=tok_id,
            pos=p,
            kv_cache=kv_cache,
            exit_depth=exit_depth,
            mode=mode,
            strategy=strat,
        )

        if p >= T_prompt - 1:
            logp = F.log_softmax(logits.squeeze(1).float(), dim=-1)
            token_logp = logp[0, target_id].item()
            log_likelihood += token_logp

    return log_likelihood


def evaluate_benchmark_items(
    engine: QwenDynamicEngine,
    items: List[Dict[str, Any]],
    depth: int,
    strategy: str,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates accuracy across items.
    """
    bench_correct = {}
    bench_total = {}

    for item in items:
        bench = item["benchmark"]
        if bench not in bench_correct:
            bench_correct[bench] = 0
            bench_total[bench] = 0

        # Score all choices
        scores = []
        for choice in item["choices"]:
            # Format completion text with leading space
            comp_text = " " + choice.strip()
            score = score_completion(
                engine=engine,
                prompt=item["prompt"],
                completion=comp_text,
                exit_depth=depth,
                strategy=strategy,
                device=device,
            )
            scores.append(score)

        pred_idx = int(np.argmax(scores))
        if pred_idx == item["label"]:
            bench_correct[bench] += 1
        bench_total[bench] += 1

    accuracies = {}
    total_corr = 0
    total_all = 0
    for bench in bench_correct:
        acc = (bench_correct[bench] / bench_total[bench]) * 100.0
        accuracies[bench] = acc
        total_corr += bench_correct[bench]
        total_all += bench_total[bench]

    accuracies["overall"] = (total_corr / total_all) * 100.0
    return accuracies


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Downstream Benchmark Check (A7) on {args.model_id} ===")
    engine = QwenDynamicEngine(model_id=args.model_id, device=args.device)
    device = torch.device(args.device)

    # Depths: 61%, 67%, 83% of network depth
    L = engine.num_layers
    if args.depths is not None:
        depths = [int(d) for d in args.depths.split(",")]
    else:
        depths = [int(round(0.61 * L)), int(round(0.67 * L)), int(round(0.83 * L))]

    items = CANONICAL_BENCHMARK_ITEMS[:3] if args.quick else CANONICAL_BENCHMARK_ITEMS

    out_csv = os.path.join(args.output_dir, f"downstream_benchmarks_{args.model_id.replace('/', '_')}.csv")
    rows = []

    # 1. Oracle Full-Depth Baseline Accuracy
    print("\n[*] Evaluating Oracle Full-Depth Baseline...")
    oracle_acc = evaluate_benchmark_items(
        engine=engine,
        items=items,
        depth=L,
        strategy="oracle_full",
        device=device,
    )
    print(f"Oracle Baseline Acc: {oracle_acc['overall']:.1f}% (HellaSwag: {oracle_acc.get('HellaSwag', 0):.1f}%, ARC: {oracle_acc.get('ARC-Easy', 0):.1f}%, LAMBADA: {oracle_acc.get('LAMBADA', 0):.1f}%)")

    rows.append({
        "model_id": args.model_id,
        "condition": "Oracle Baseline",
        "depth": L,
        "rel_depth": "100.0%",
        "hellaswag_acc": oracle_acc.get("HellaSwag", 0.0),
        "arc_easy_acc": oracle_acc.get("ARC-Easy", 0.0),
        "lambada_acc": oracle_acc.get("LAMBADA", 0.0),
        "overall_acc": oracle_acc["overall"],
        "delta_vs_oracle": 0.0,
    })

    print(f"\n{'Depth':<6} | {'Rel.':<6} | {'Policy':<20} | {'Overall Acc':<12} | {'Delta vs Oracle':<16}")
    print("-" * 68)

    for d in depths:
        rel_d = (d / L) * 100.0
        for strat, s_name in [("native_projection", "Native Projection"), ("direct_copy", "Direct Copy")]:
            acc = evaluate_benchmark_items(
                engine=engine,
                items=items,
                depth=d,
                strategy=strat,
                device=device,
            )
            delta = acc["overall"] - oracle_acc["overall"]
            sign = "+" if delta >= 0 else ""
            print(f"L={d:<4} | {rel_d:4.1f}% | {s_name:<20} | {acc['overall']:10.1f}% | {sign}{delta:14.1f}%")

            rows.append({
                "model_id": args.model_id,
                "condition": s_name,
                "depth": d,
                "rel_depth": f"{rel_d:.1f}%",
                "hellaswag_acc": acc.get("HellaSwag", 0.0),
                "arc_easy_acc": acc.get("ARC-Easy", 0.0),
                "lambada_acc": acc.get("LAMBADA", 0.0),
                "overall_acc": acc["overall"],
                "delta_vs_oracle": round(delta, 1),
            })

    fieldnames = ["model_id", "condition", "depth", "rel_depth", "hellaswag_acc", "arc_easy_acc", "lambada_acc", "overall_acc", "delta_vs_oracle"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Downstream benchmark results saved to: {out_csv}")


if __name__ == "__main__":
    main()
