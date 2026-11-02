#!/usr/bin/env python3
"""
experiments/benchmark_large_scale.py

Publication-Grade Multi-Domain Speculative Decoding Benchmark (N = 60..100 Prompts).
Evaluates Qwen-2.5-3B across 4 distinct domains:
1. Mathematics & Quantitative Reasoning (GSM8K style)
2. Code Generation & Algorithms (HumanEval / Python style)
3. Factual Knowledge & Physical Sciences (MMLU / TriviaQA style)
4. Multi-turn Reasoning & Instruction (MT-Bench / Alpaca style)

Outputs:
- Per-domain and aggregate speedup with 95% Confidence Intervals (CI_95).
- Per-domain empirical acceptance rate (alpha) with standard deviations.
- 100% Token-for-Token Lossless Equivalence verification.
- Peak VRAM footprint tracking (< 15.0 GB limit).
- Automated Markdown and JSON summary generation.

Usage:
    python experiments/benchmark_large_scale.py \
        --model-id Qwen/Qwen2.5-3B \
        --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \
        --draft-depth 24 \
        --gamma 1 \
        --num-prompts-per-domain 15 \
        --max-tokens 30 \
        --device cuda
"""

import os
import sys
import time
import json
import math
import argparse
from typing import List, Dict, Any, Tuple, Optional
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.exit_adapter import ExitCalibrationModule
from src.speculative_engine import QwenSelfSpeculativeEngine


# -----------------------------------------------------------------------------
# Curated Multi-Domain Benchmark Prompts (20 per domain = 80 total)
# -----------------------------------------------------------------------------
MULTI_DOMAIN_PROMPTS = {
    "Mathematics & Reasoning": [
        "If a train travels at 60 miles per hour and leaves at 2:00 PM, the time it takes to travel 180 miles is",
        "The quadratic formula to solve equations of the form ax^2 + bx + c = 0 is given by",
        "To prove that the square root of 2 is irrational, we begin by assuming that",
        "In number theory, Euler's totient function phi(n) counts the positive integers up to n that are",
        "Let X be a normally distributed random variable with mean mu and variance sigma squared. The probability density function is",
        "A sequence is defined by a_1 = 1 and a_{n+1} = 2*a_n + 1. The closed-form expression for a_n is",
        "The divergence theorem in vector calculus relates the flux of a vector field through a closed surface to",
        "Bayes' theorem states that the conditional probability of an event A given event B is calculated as",
        "In linear algebra, an n x n matrix A is invertible if and only if its determinant is",
        "The Fibonacci sequence begins with 0 and 1, and each subsequent number is formed by",
        "A prime number is a natural number strictly greater than 1 that has no positive divisors other than",
        "The central limit theorem states that the normalized sum of independent and identically distributed random variables tends toward",
        "In graph theory, an Eulerian path in a finite graph is a trail that visits every",
        "The fundamental theorem of arithmetic states that every integer greater than 1 either is a prime itself or",
        "To find the eigenvalues of a matrix M, we must solve the characteristic equation given by",
        "The derivative of f(x) = x * ln(x) with respect to x using the product rule is",
        "A fair coin is flipped 5 times. The probability of obtaining exactly 3 heads is",
        "The volume of a sphere of radius r is given by the mathematical formula",
        "The Riemann zeta function is defined for complex numbers s with real part greater than 1 by the infinite series",
        "In modular arithmetic, Fermat's Little Theorem states that if p is a prime number and a is not divisible by p, then",
    ],
    "Code Generation & Algorithms": [
        "def binary_search(arr, target):\n    \"\"\"Find target in sorted array using binary search.\"\"\"\n    left, right = 0, len(arr) - 1\n",
        "def quicksort(arr):\n    \"\"\"Sort array using recursive quicksort algorithm.\"\"\"\n    if len(arr) <= 1:\n        return arr\n",
        "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if string is a valid palindrome ignoring non-alphanumerics.\"\"\"\n",
        "def fibonacci(n: int) -> int:\n    \"\"\"Compute the n-th Fibonacci number using dynamic programming.\"\"\"\n",
        "def merge_sorted_lists(l1, l2):\n    \"\"\"Merge two sorted linked lists into a single sorted list.\"\"\"\n",
        "def depth_first_search(graph, start, visited=None):\n    \"\"\"Perform DFS traversal on an adjacency list graph.\"\"\"\n",
        "def breadth_first_search(graph, start_node):\n    \"\"\"Perform BFS on a graph and return visited order.\"\"\"\n",
        "def reverse_linked_list(head):\n    \"\"\"Iteratively reverse a singly linked list.\"\"\"\n",
        "def two_sum(nums: list, target: int) -> list:\n    \"\"\"Return indices of two numbers that add up to target using a hash table.\"\"\"\n",
        "def find_max_subarray(nums: list) -> int:\n    \"\"\"Kadane's algorithm to find maximum contiguous subarray sum.\"\"\"\n",
        "def valid_parentheses(s: str) -> bool:\n    \"\"\"Determine if string containing '()[]{}' has valid brackets using a stack.\"\"\"\n",
        "def top_k_frequent(nums: list, k: int) -> list:\n    \"\"\"Find the k most frequent elements in nums using a min-heap.\"\"\"\n",
        "def climb_stairs(n: int) -> int:\n    \"\"\"Number of distinct ways to climb n steps taking 1 or 2 steps at a time.\"\"\"\n",
        "def coin_change(coins: list, amount: int) -> int:\n    \"\"\"Fewest number of coins needed to make up amount using dynamic programming.\"\"\"\n",
        "def invert_binary_tree(root):\n    \"\"\"Invert a binary tree recursively by swapping left and right children.\"\"\"\n",
        "def longest_common_subsequence(text1: str, text2: str) -> int:\n    \"\"\"Compute LCS length between two strings using 2D DP.\"\"\"\n",
        "def dijkstra(graph, start_vertex):\n    \"\"\"Compute shortest path distances using Dijkstra's algorithm with priority queue.\"\"\"\n",
        "def serialize_binary_tree(root):\n    \"\"\"Serialize a binary tree into a string format.\"\"\"\n",
        "def lowest_common_ancestor(root, p, q):\n    \"\"\"Find the lowest common ancestor node in a binary search tree.\"\"\"\n",
        "def trie_insert(root, word: str):\n    \"\"\"Insert word into a prefix Trie data structure.\"\"\"\n",
    ],
    "Factual Knowledge & Science": [
        "In computer science, an attention mechanism is a technique that mimics cognitive attention by",
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy into",
        "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915 and",
        "The Apollo program was the third United States human spaceflight program carried out by NASA,",
        "DNA is a molecule composed of two polynucleotide chains that coil around each other to form a",
        "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell,",
        "The periodic table is a tabular display of chemical elements organized by atomic number and",
        "Plate tectonics is the scientific theory describing the large-scale motion of seven large plates and",
        "Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of",
        "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and",
        "The Doppler effect is the change in frequency of a wave in relation to an observer who is",
        "Platelets, also called thrombocytes, are a component of blood whose primary function is to",
        "Superconductivity is a set of physical properties observed in certain materials where electrical resistance vanishes and",
        "The French Revolution was a period of fundamental political and societal change in France that began with",
        "The ozone layer is a region of Earth's stratosphere that absorbs most of the Sun's ultraviolet radiation and",
        "Thermodynamics is the branch of physics that deals with heat, work, temperature, and their relation to",
        "Newton's third law of motion states that for every action in an isolated system, there is",
        "The Great Barrier Reef is the world's largest coral reef system composed of over 2,900 individual reefs and",
        "Semiconductors are materials whose electrical conductivity value falls between that of a conductor and an",
        "Nuclear fusion is a reaction in which two or more atomic nuclei are combined to form one or more different",
    ],
    "Language & Open-Ended Reasoning": [
        "The main ethical implications of deploying autonomous artificial intelligence systems in healthcare include",
        "To reduce carbon emissions in global supply chains, international manufacturing organizations should focus on",
        "The difference between inductive reasoning and deductive reasoning lies primarily in how conclusions are",
        "When designing scalable microservices architectures, software engineers must balance modular decoupling against",
        "The economic concept of opportunity cost refers to the potential benefits that an individual or business misses out on when",
        "A central advantage of containerization over traditional virtualization using hypervisors is that containers share",
        "In linguistics, the Sapir-Whorf hypothesis suggests that the structure of a human language influences",
        "The primary function of monetary policy implemented by central banks is to regulate money supply in order to",
        "When evaluating the trade-off between bias and variance in machine learning models, high variance is often characterized by",
        "The principle of separation of powers in democratic governance divides political responsibilities into three distinct branches:",
        "To ensure psychological safety and high productivity within agile software engineering teams, engineering leaders should",
        "The tragedy of the commons is an economic theory that describes a situation in which individuals with access to a shared resource",
        "A major challenge in training deep generative diffusion models for high-resolution image synthesis is",
        "In cognitive psychology, working memory is the system responsible for temporarily holding and manipulating information during",
        "To mitigate flash crowd traffic spikes on high-throughput web applications, caching tiers typically utilize",
        "The social contract theory formulated by Thomas Hobbes and John Locke posits that legitimate political authority arises from",
        "In natural language processing, the vanishing gradient problem in recurrent neural networks was traditionally mitigated by",
        "The distinction between correlation and causation is critical in scientific research because observing two variables moving together does not imply",
        "When implementing fault-tolerant distributed consensus protocols such as Raft or Paxos, the system must guarantee safety under",
        "To build resilient supply chains against geopolitical disruptions, modern multinational corporations are adopting strategies like",
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description="Large-Scale Multi-Domain Speculative Decoding Benchmark")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Model ID")
    default_adapter = "checkpoints/exit_adapters_qwen_mlp.pt" if os.path.exists("checkpoints/exit_adapters_qwen_mlp.pt") else "checkpoints/exit_adapters_qwen.pt"
    parser.add_argument("--adapter-path", type=str, default=default_adapter, help="Adapter checkpoint")
    parser.add_argument("--draft-depth", type=int, default=24, help="Exit layer for drafting (e.g. 24 or 29)")
    parser.add_argument("--gamma", type=int, default=1, help="Speculative horizon (default: 1)")
    parser.add_argument("--num-prompts-per-domain", type=int, default=15, help="Number of prompts to sample per domain (max 20)")
    parser.add_argument("--max-tokens", type=int, default=30, help="Tokens to generate per prompt")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", type=str, default="experiments/benchmark_large_scale_results.json")
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


def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """Computes mean, standard error, and margin of error for a 95% CI."""
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(data) / n
    if n == 1:
        return mean, 0.0, 0.0
    variance = sum((x - mean) ** 2 for x in data) / (n - 1)
    std_err = math.sqrt(variance / n)
    # Z-value for 95% CI ~ 1.96
    margin = 1.96 * std_err
    return mean, std_err, margin


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "float16" else (torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)

    total_prompts = args.num_prompts_per_domain * len(MULTI_DOMAIN_PROMPTS)

    print("=" * 80)
    print("LARGE-SCALE MULTI-DOMAIN SELF-SPECULATIVE BENCHMARK")
    print(f"Model: {args.model_id} | Device: {device} | Dtype: {torch_dtype}")
    print(f"Draft Depth: L = {args.draft_depth} | Horizon: gamma = {args.gamma}")
    print(f"Sample Size: N = {total_prompts} prompts ({args.num_prompts_per_domain} per domain across 4 domains)")
    print(f"Max Generation Tokens: {args.max_tokens} tokens per prompt")
    print("=" * 80)

    # 1. Load Model & Tokenizer
    print(f"[*] Loading model '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    # 2. Load Exit Calibration Adapter
    adapter_module = None
    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"[*] Loading ExitCalibrationModule from '{args.adapter_path}'...")
        adapter_module = ExitCalibrationModule.load_checkpoint(args.adapter_path, device=device)
        adapter_module.eval()
        print(f"✓ Loaded adapters for depths: {adapter_module.candidate_depths}")
    else:
        print("[!] No exit adapter found. Using native projection readout.")

    # 3. Initialize Speculative Engine
    engine = QwenSelfSpeculativeEngine(
        model=model,
        tokenizer=tokenizer,
        draft_depth=args.draft_depth,
        adapter_module=adapter_module,
        device=args.device,
        max_seq_len=2048,
    )

    domain_results = {}
    all_speedups = []
    all_acceptances = []
    all_base_speeds = []
    all_spec_speeds = []
    all_exact_matches = []

    prompt_counter = 0

    for domain_name, prompt_list in MULTI_DOMAIN_PROMPTS.items():
        selected_prompts = prompt_list[:args.num_prompts_per_domain]
        domain_results[domain_name] = {
            "baseline_speeds": [],
            "spec_speeds": [],
            "speedups": [],
            "acceptance_rates": [],
            "exact_matches": []
        }

        print(f"\nEvaluating Domain: {domain_name} ({len(selected_prompts)} prompts)")
        print("-" * 80)
        print(f"{'#':<4} {'Baseline (tok/s)':<18} {'Speculative (tok/s)':<22} {'Speedup':<10} {'Acceptance %':<14} {'Exact Match?'}")
        print("-" * 80)

        for p_idx, prompt_text in enumerate(selected_prompts):
            prompt_counter += 1
            torch.cuda.reset_peak_memory_stats(device)
            prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

            # Baseline Pass
            base_res = run_baseline(engine, prompt_ids, max_new_tokens=args.max_tokens)
            base_speed = base_res["tok_per_sec"]

            # Speculative Pass
            torch.cuda.reset_peak_memory_stats(device)
            spec_res = engine.generate_speculative(
                prompt_ids,
                max_new_tokens=args.max_tokens,
                gamma=args.gamma,
                temperature=0.0
            )
            spec_speed = spec_res["tokens_per_sec"]
            acc_rate = spec_res["acceptance_rate"]

            is_match = (base_res["tokens"] == spec_res["tokens"])
            speedup = spec_speed / base_speed

            # Record
            domain_results[domain_name]["baseline_speeds"].append(base_speed)
            domain_results[domain_name]["spec_speeds"].append(spec_speed)
            domain_results[domain_name]["speedups"].append(speedup)
            domain_results[domain_name]["acceptance_rates"].append(acc_rate)
            domain_results[domain_name]["exact_matches"].append(1 if is_match else 0)

            all_speedups.append(speedup)
            all_acceptances.append(acc_rate)
            all_base_speeds.append(base_speed)
            all_spec_speeds.append(spec_speed)
            all_exact_matches.append(1 if is_match else 0)

            print(f"{prompt_counter:<4} {base_speed:<18.2f} {spec_speed:<22.2f} {speedup:<10.2f}x {acc_rate:<14.1f}% {'✓ YES' if is_match else '✗ NO'}")

    # Summary Statistics
    peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0

    mean_speedup, _, ci_speedup = compute_confidence_interval(all_speedups)
    mean_acc, _, ci_acc = compute_confidence_interval(all_acceptances)
    mean_base_speed, _, _ = compute_confidence_interval(all_base_speeds)
    mean_spec_speed, _, _ = compute_confidence_interval(all_spec_speeds)
    lossless_pct = (sum(all_exact_matches) / len(all_exact_matches)) * 100.0

    print("\n" + "=" * 80)
    print("STATISTICAL BENCHMARK SUMMARY (N = %d Prompts)" % len(all_speedups))
    print("=" * 80)
    print(f"Overall Mean Speedup           : {mean_speedup:.3f}x ± {ci_speedup:.3f} (95% CI: [{mean_speedup - ci_speedup:.3f}x, {mean_speedup + ci_speedup:.3f}x])")
    print(f"Overall Mean Acceptance Rate   : {mean_acc:.1f}% ± {ci_acc:.1f}% (95% CI: [{mean_acc - ci_acc:.1f}%, {mean_acc + ci_acc:.1f}%])")
    print(f"Mean Baseline Throughput       : {mean_base_speed:.2f} tok/s")
    print(f"Mean Speculative Throughput    : {mean_spec_speed:.2f} tok/s")
    print(f"Lossless Equivalence Rate      : {sum(all_exact_matches)}/{len(all_exact_matches)} ({lossless_pct:.1f}%)")
    print(f"Peak GPU VRAM Allocated        : {peak_vram_gb:.2f} GB / 15.0 GB limit")
    print("-" * 80)

    # Per-Domain Breakdown Table
    print(f"{'Domain':<32} {'Count':<8} {'Base tok/s':<14} {'Spec tok/s':<14} {'Speedup':<12} {'Acceptance %':<14} {'Lossless %'}")
    print("-" * 80)
    for domain_name, data in domain_results.items():
        d_mean_base = sum(data["baseline_speeds"]) / len(data["baseline_speeds"])
        d_mean_spec = sum(data["spec_speeds"]) / len(data["spec_speeds"])
        d_mean_speedup, _, d_ci_speedup = compute_confidence_interval(data["speedups"])
        d_mean_acc, _, d_ci_acc = compute_confidence_interval(data["acceptance_rates"])
        d_lossless = (sum(data["exact_matches"]) / len(data["exact_matches"])) * 100.0
        print(f"{domain_name:<32} {len(data['speedups']):<8} {d_mean_base:<14.2f} {d_mean_spec:<14.2f} {d_mean_speedup:<6.2f}±{d_ci_speedup:<4.2f} {d_mean_acc:<6.1f}±{d_ci_acc:<4.1f}% {d_lossless:.1f}%")
    print("=" * 80)

    # Save to JSON
    summary_data = {
        "model_id": args.model_id,
        "draft_depth": args.draft_depth,
        "gamma": args.gamma,
        "total_prompts": len(all_speedups),
        "mean_speedup": mean_speedup,
        "ci_speedup_95": ci_speedup,
        "mean_acceptance_rate": mean_acc,
        "ci_acceptance_95": ci_acc,
        "lossless_equivalence_pct": lossless_pct,
        "peak_vram_gb": peak_vram_gb,
        "domain_results": domain_results
    }
    with open(args.output_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[*] Complete structured results saved to '{args.output_json}'.")


if __name__ == "__main__":
    main()
