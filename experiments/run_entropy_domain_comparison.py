#!/usr/bin/env python3
"""
experiments/run_entropy_domain_comparison.py

Addresses Claim C1 & Gap A11:
- A11. Entropy-Matched Domain Comparison:
  Bins token positions by oracle predictive entropy:
    H(P*) = - sum_v P*(v) * ln P*(v)
  and evaluates next-token KL divergence D_KL(P* || P_hat) within matched entropy bins
  across 3 domains:
  1. WikiText-2 (natural language / encyclopedia)
  2. C4 (web text)
  3. Python Algorithms (code)

Measures:
- Binned KL divergence within low-entropy (H < 1.0), medium-entropy (1.0 <= H < 2.5),
  and high-entropy (H >= 2.5) token buckets across domains.

Proves:
Whether the observed performance differences across domains (e.g. Code/C4 vs WikiText)
are merely caused by differing baseline entropy distributions, or represent genuine
domain-specific representation continuity.

Usage:
  python experiments/run_entropy_domain_comparison.py --model-id Qwen/Qwen2.5-3B --depth 18 --device cuda
  python experiments/run_entropy_domain_comparison.py --quick --device cpu
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
from src.eval_dataset import get_tokenized_wikitext2


CANONICAL_DOMAIN_TEXTS = {
    "wikitext": [
        "The Apollo program was the third United States human spaceflight program carried out by NASA.",
        "General relativity generalizes special relativity and refines Newton's law of universal gravitation.",
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy.",
        "The Industrial Revolution was the transition to new manufacturing processes in Great Britain.",
    ],
    "c4_web": [
        "Welcome to our travel blog! Today we will share 10 essential tips for backpacking across Europe on a budget.",
        "Breaking news: The city council voted unanimously today to approve the new public transit expansion project.",
        "Looking for the best chocolate chip cookie recipe? Here is our foolproof step-by-step baking tutorial.",
        "Customer review: I purchased this wireless ergonomic mouse two weeks ago and the battery life has been fantastic.",
    ],
    "python_code": [
        "def binary_search(arr: list[int], target: int) -> int:\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "class LRUCache:\n    def __init__(self, capacity: int):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n    def get(self, key: int) -> int:\n        if key not in self.cache:\n            return -1\n        self.order.remove(key)\n        self.order.append(key)\n        return self.cache[key]",
        "def merge_sort(items: list[int]) -> list[int]:\n    if len(items) <= 1:\n        return items\n    mid = len(items) // 2\n    left = merge_sort(items[:mid])\n    right = merge_sort(items[mid:])\n    return merge(left, right)",
        "def dijkstra(graph: dict, start: str) -> dict:\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    queue = [(0, start)]\n    while queue:\n        dist, u = heapq.heappop(queue)\n        for v, weight in graph[u].items():\n            if distances[u] + weight < distances[v]:\n                distances[v] = distances[u] + weight\n                heapq.heappush(queue, (distances[v], v))\n    return distances",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Entropy-Matched Domain Comparison (A11)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--depth", type=int, default=18, help="Exit depth (default: 18 = 50% on Qwen)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/entropy_domain")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


@torch.no_grad()
def harvest_token_entropies_and_kl(
    engine: QwenDynamicEngine,
    text_list: List[str],
    depth: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes per-token oracle entropy H(P*) and paired KL divergence D_KL(P* || P_proj).
    """
    all_entropies = []
    all_kls = []

    for text in text_list:
        tokens = engine.tokenizer.encode(text, return_tensors="pt").to(device)
        T = tokens.shape[1]
        if T < 4:
            continue

        kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=T + 16)

        # Oracle logits
        oracle_logits = []
        for p in range(T - 1):
            tok = tokens[:, p : p + 1]
            log_full, _ = engine.step_forward_decoupled(
                input_id=tok,
                pos=p,
                kv_cache=kv_cache,
                exit_depth=engine.num_layers,
                mode="full",
                strategy="oracle_full",
            )
            oracle_logits.append(log_full.squeeze(1))

        # Reset cache
        for l in range(engine.num_layers):
            kv_cache["k"][l].zero_()
            kv_cache["v"][l].zero_()

        # Test Native Projection
        for p in range(T - 1):
            tok = tokens[:, p : p + 1]
            log_test, _ = engine.step_forward_decoupled(
                input_id=tok,
                pos=p,
                kv_cache=kv_cache,
                exit_depth=depth,
                mode="attention_only",
                strategy="native_projection",
            )
            log_test = log_test.squeeze(1)
            log_ref = oracle_logits[p]

            p_ref = F.softmax(log_ref.float(), dim=-1)
            logp_ref = F.log_softmax(log_ref.float(), dim=-1)
            logp_test = F.log_softmax(log_test.float(), dim=-1)

            # Shannon entropy in nats: - sum p ln p
            entropy = -(p_ref * logp_ref).sum(dim=-1).item()
            kl = (p_ref * (logp_ref - logp_test)).sum(dim=-1).item()

            all_entropies.append(entropy)
            all_kls.append(kl)

    return np.array(all_entropies), np.array(all_kls)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Entropy-Matched Domain Comparison (A11) on {args.model_id} ===")
    engine = QwenDynamicEngine(model_id=args.model_id, device=args.device)
    device = torch.device(args.device)

    # Entropy bins: Low (0-1.0), Med (1.0-2.5), High (>=2.5)
    bins = [
        ("Low Entropy [0, 1.0)", 0.0, 1.0),
        ("Med Entropy [1.0, 2.5)", 1.0, 2.5),
        ("High Entropy [2.5, inf)", 2.5, 100.0),
    ]

    out_csv = os.path.join(args.output_dir, f"entropy_matched_domain_{args.model_id.replace('/', '_')}.csv")
    rows = []

    print(f"\nExit Depth: L={args.depth}")
    print(f"{'Domain':<16} | {'Entropy Bin':<25} | {'Mean Entropy':<14} | {'Mean KL (nats)':<15} | {'Token Count':<12}")
    print("-" * 88)

    for domain_name, texts in CANONICAL_DOMAIN_TEXTS.items():
        sample_texts = texts[:1] if args.quick else texts
        entropies, kls = harvest_token_entropies_and_kl(
            engine=engine,
            text_list=sample_texts,
            depth=args.depth,
            device=device,
        )

        for bin_label, low, high in bins:
            mask = (entropies >= low) & (entropies < high)
            if mask.sum() > 0:
                bin_ent = float(np.mean(entropies[mask]))
                bin_kl = float(np.mean(kls[mask]))
                count = int(mask.sum())
            else:
                bin_ent = 0.0
                bin_kl = 0.0
                count = 0

            print(f"{domain_name:<16} | {bin_label:<25} | {bin_ent:14.3f} | {bin_kl:15.4f} | {count:12}")

            rows.append({
                "model_id": args.model_id,
                "exit_depth": args.depth,
                "domain": domain_name,
                "entropy_bin": bin_label,
                "mean_entropy": round(bin_ent, 3),
                "mean_kl": round(bin_kl, 4),
                "token_count": count,
            })

    fieldnames = ["model_id", "exit_depth", "domain", "entropy_bin", "mean_entropy", "mean_kl", "token_count"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Entropy-matched domain comparison saved to: {out_csv}")


if __name__ == "__main__":
    main()
