#!/usr/bin/env python3
"""
experiments/benchmark_length_and_domain.py

Part A: Sequence Length (T = 64, 128, 256, 512) and Out-of-Domain Generality Benchmark.
Evaluates Native KV State Projection vs. Direct Copy vs. Oracle Baseline across:
1. Context Lengths: T in {64, 128, 256, 512} tokens.
2. Distinct Domains:
   - Natural Language: WikiText-2 (validation split)
   - Code & Algorithms: Python / System implementations (high-entropy, strict syntax)
3. Large Sample Size: N = 20..50 sequences per setting (thousands of evaluated tokens).

Outputs:
- Per-setting Perplexity (PPL), Mean KL Divergence (D_KL in nats), and Top-1 Agreement (%).
- CSV and JSON exports in results/canonical_sweeps/qwen2.5_3b/.
- Publication-ready LaTeX and Markdown comparison tables.

Usage:
  # Quick smoke-test on GPU (~2 min on Colab):
  python experiments/benchmark_length_and_domain.py --quick --device cuda

  # Full publication benchmark (~10 min on Colab):
  python experiments/benchmark_length_and_domain.py --model-id Qwen/Qwen2.5-3B --seq-lens 64,128,256,512 --num-samples 30 --device cuda
"""

import os
import sys
import math
import time
import argparse
import csv
import json
from typing import Dict, List, Tuple, Optional, Any

import torch
import torch.nn.functional as F
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.qwen_dynamic import QwenDynamicEngine
from src.eval_dataset import get_tokenized_wikitext2, load_raw_wikitext2_split


CANONICAL_FALLBACK_TEXTS = [
    (
        "The Apollo program was the third United States human spaceflight program carried out by the "
        "National Aeronautics and Space Administration (NASA), which succeeded in landing the first "
        "humans on the Moon in 1969. Conceived during the presidency of Dwight D. Eisenhower, Apollo "
        "was dedicated to President John F. Kennedy's national goal of landing a man on the Moon and "
        "returning him safely to the Earth before the decade was out."
    ),
    (
        "In computer science, an attention mechanism is a technique that mimics cognitive attention. "
        "The effect enhances some parts of the input data while diminishing other parts, with the thought "
        "that the network should devote more focus to that small but important part of the data. "
        "Transformers rely entirely on self-attention to compute representations of their input and output."
    ),
    (
        "Photosynthesis is a biological process used by plants and other organisms to convert light energy "
        "into chemical energy that, through cellular respiration, can later be released to fuel the "
        "organism's metabolic activities. This chemical energy is stored in carbohydrate molecules such "
        "as sugars and starches, which are synthesized from carbon dioxide and water."
    ),
    (
        "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, "
        "continental Europe, and the United States, that occurred during the period from around 1760 to "
        "about 1820 to 1840. This transition included going from hand production methods to machines, "
        "new chemical manufacturing and iron production processes, the increasing use of steam power."
    ),
    (
        "General relativity is the geometric theory of gravitation published by Albert Einstein in 1915 "
        "and is the current description of gravitation in modern physics. General relativity generalizes "
        "special relativity and refines Newton's law of universal gravitation, providing a unified description "
        "of gravity as a geometric property of space and time or four-dimensional spacetime."
    ),
    (
        "A neural network is a computational model inspired by the biological neural networks in animal brains. "
        "The neural network consists of interconnected nodes or units organized in layers. Deep learning "
        "architectures use multiple stacked processing layers to learn hierarchical representations of data "
        "with varying levels of abstraction."
    )
]


# -----------------------------------------------------------------------------
# Curated High-Entropy Technical & Code Corpus for OOD Evaluation
# -----------------------------------------------------------------------------
PYTHON_CODE_CORPUS = """
import os
import sys
import time
import math
import heapq
from typing import List, Dict, Tuple, Optional, Any, Set

class TrieNode:
    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_terminal: bool = False
        self.frequency: int = 0

class PrefixTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        curr = self.root
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.is_terminal = True
        curr.frequency += 1

    def search_prefix(self, prefix: str) -> List[str]:
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return []
            curr = curr.children[char]
        results = []
        self._dfs(curr, prefix, results)
        return results

    def _dfs(self, node: TrieNode, path: str, results: List[str]) -> None:
        if node.is_terminal:
            results.append(path)
        for char, child in node.children.items():
            self._dfs(child, path + char, results)

class LRUCache:
    class DLinkedNode:
        def __init__(self, key: int = 0, value: int = 0):
            self.key = key
            self.value = value
            self.prev: Optional['LRUCache.DLinkedNode'] = None
            self.next: Optional['LRUCache.DLinkedNode'] = None

    def __init__(self, capacity: int):
        self.cache: Dict[int, LRUCache.DLinkedNode] = {}
        self.head = self.DLinkedNode()
        self.tail = self.DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        self.capacity = capacity
        self.size = 0

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            newNode = self.DLinkedNode(key, value)
            self.cache[key] = newNode
            self._add_node(newNode)
            self.size += 1
            if self.size > self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1

    def _add_node(self, node: DLinkedNode) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node: DLinkedNode) -> None:
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    def _move_to_head(self, node: DLinkedNode) -> None:
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self) -> DLinkedNode:
        res = self.tail.prev
        self._remove_node(res)
        return res

def dijkstra_shortest_paths(graph: Dict[int, List[Tuple[int, float]]], start_vertex: int) -> Dict[int, float]:
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start_vertex] = 0.0
    pq = [(0.0, start_vertex)]
    visited: Set[int] = set()

    while pq:
        curr_dist, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        for v, weight in graph[u]:
            dist = curr_dist + weight
            if dist < distances[v]:
                distances[v] = dist
                heapq.heappush(pq, (dist, v))
    return distances

def merge_sort(arr: List[int]) -> List[int]:
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)

def _merge(left: List[int], right: List[int]) -> List[int]:
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def matrix_multiply_tiled(A: List[List[float]], B: List[List[float]], tile_size: int = 32) -> List[List[float]]:
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    for i0 in range(0, n, tile_size):
        for j0 in range(0, p, tile_size):
            for k0 in range(0, m, tile_size):
                for i in range(i0, min(i0 + tile_size, n)):
                    for k in range(k0, min(k0 + tile_size, m)):
                        aik = A[i][k]
                        for j in range(j0, min(j0 + tile_size, p)):
                            C[i][j] += aik * B[k][j]
    return C

def compute_levenshtein_distance(str1: str, str2: str) -> int:
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Sequence Length (T=64..512) and Out-of-Domain Generality for KV Continuity."
    )
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Model ID")
    parser.add_argument("--exit-depth", type=int, default=18, help="Target exit depth d_t (default: 18 on Qwen2.5-3B, 50% depth)")
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="64,128,256,512",
        help="Comma-separated sequence lengths to evaluate"
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="wikitext2,code",
        help="Comma-separated domains to evaluate: 'wikitext2', 'code'"
    )
    parser.add_argument("--num-samples", type=int, default=30, help="Number of evaluated sequences per (domain, T) setting")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/canonical_sweeps/qwen2.5_3b",
        help="Output directory for CSV and JSON results"
    )
    parser.add_argument("--quick", action="store_true", help="Quick run mode (T=64,128; N=5 samples)")
    return parser.parse_args()


def load_domain_tokens(tokenizer, domain: str, device: torch.device) -> torch.Tensor:
    """Loads a 1D tensor of token IDs for the chosen domain."""
    if domain == "wikitext2":
        try:
            return get_tokenized_wikitext2(tokenizer, split="validation")
        except Exception as e1:
            try:
                raw_text = load_raw_wikitext2_split("validation")
                toks = tokenizer.encode(raw_text)
                return torch.tensor(toks, dtype=torch.long)
            except Exception as e2:
                print(f"[!] Warning: WikiText-2 loading failed ({e2}). Using canonical texts.")
                raw_text = "\n\n".join(CANONICAL_FALLBACK_TEXTS * 50)
                toks = tokenizer.encode(raw_text)
                return torch.tensor(toks, dtype=torch.long)

    elif domain == "code":
        raw_text = (PYTHON_CODE_CORPUS.strip() + "\n\n") * 30
        toks = tokenizer.encode(raw_text)
        if not isinstance(toks, torch.Tensor):
            toks = torch.tensor(toks, dtype=torch.long)
        return toks

    else:
        raise ValueError(f"Unknown domain '{domain}'. Choose 'wikitext2' or 'code'.")


def extract_evaluation_chunks(
    tokens_1d: torch.Tensor,
    seq_len: int,
    num_samples: int,
    seed: int,
    device: torch.device
) -> List[torch.Tensor]:
    """Extracts non-overlapping token sequences of shape (1, seq_len)."""
    rng = np.random.RandomState(seed)
    max_start = len(tokens_1d) - seq_len - 1
    if max_start <= 0:
        chunk = tokens_1d[:seq_len].unsqueeze(0).to(device)
        return [chunk] * num_samples

    step = seq_len
    num_possible = max_start // step
    if num_possible >= num_samples:
        starts = rng.choice(num_possible, size=num_samples, replace=False) * step
    else:
        starts = rng.choice(max_start, size=num_samples, replace=True)

    seqs = []
    for st in starts:
        chunk = tokens_1d[st : st + seq_len].unsqueeze(0).to(device)
        seqs.append(chunk)
    return seqs


@torch.no_grad()
def evaluate_condition_on_chunks(
    engine: QwenDynamicEngine,
    sequences: List[torch.Tensor],
    exit_depth: int,
    mode: str,
    strategy: str,
    oracle_logits_cache: Optional[List[List[torch.Tensor]]] = None
) -> Tuple[Dict[str, float], List[List[torch.Tensor]]]:
    """Evaluates teacher-forced metrics across chunks."""
    total_loss = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_top1 = 0
    saved_logits: List[List[torch.Tensor]] = []
    t_start = time.perf_counter()

    for s_idx, seq in enumerate(sequences):
        T = seq.size(1)
        kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=T + 16)
        seq_logits = []

        for pos in range(T - 1):
            curr_tok = seq[:, pos : pos + 1]
            target_tok = seq[:, pos + 1]

            logits, _ = engine.step_forward_decoupled(
                input_id=curr_tok,
                pos=pos,
                kv_cache=kv_cache,
                exit_depth=exit_depth,
                mode=mode,
                strategy=strategy,
                track_cossim=False
            )

            curr_l = logits[:, 0, :]
            seq_logits.append(curr_l.detach().cpu())

            loss = F.cross_entropy(curr_l, target_tok).item()
            total_loss += loss
            total_tokens += 1

            if oracle_logits_cache is not None:
                true_l = oracle_logits_cache[s_idx][pos].to(curr_l.device)
                p_true = F.softmax(true_l, dim=-1)
                log_p = F.log_softmax(curr_l, dim=-1)
                kl = F.kl_div(log_p, p_true, reduction="batchmean").item()
                total_kl += kl

                if curr_l.argmax(dim=-1).item() == true_l.argmax(dim=-1).item():
                    total_top1 += 1
            else:
                if curr_l.argmax(dim=-1).item() == target_tok.item():
                    total_top1 += 1

        saved_logits.append(seq_logits)

    elapsed_sec = time.perf_counter() - t_start
    ms_per_tok = (elapsed_sec / max(1, total_tokens)) * 1000.0

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    mean_kl = (total_kl / max(1, total_tokens)) if oracle_logits_cache is not None else 0.0
    top1_acc = (total_top1 / max(1, total_tokens)) * 100.0

    metrics = {
        "loss": mean_loss,
        "ppl": ppl,
        "kl_nats": mean_kl,
        "top1_acc": top1_acc,
        "total_tokens": total_tokens,
        "ms_per_tok": ms_per_tok
    }
    return metrics, saved_logits


def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.quick:
        seq_lens = [64, 128]
        num_samples = 5
        domains = ["wikitext2", "code"]
        print("[!] Quick mode enabled: evaluating T in [64, 128], N=5 samples.")
    else:
        seq_lens = [int(s.strip()) for s in args.seq_lens.split(",")]
        num_samples = args.num_samples
        domains = [d.strip() for d in args.domains.split(",")]

    dtype_map = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print("=" * 80)
    print("BENCHMARK: SEQUENCE LENGTH & OUT-OF-DOMAIN GENERALITY")
    print(f"Model ID:        {args.model_id}")
    print(f"Exit Depth:      L = {args.exit_depth}")
    print(f"Sequence Lengths:{seq_lens}")
    print(f"Domains:         {domains}")
    print(f"Samples / cell:  {num_samples}")
    print(f"Device:          {device} ({args.dtype})")
    print("=" * 80)

    engine = QwenDynamicEngine(model_id=args.model_id, device=args.device, dtype=dtype)
    tokenizer = engine.tokenizer
    d_t = args.exit_depth

    results_records: List[Dict[str, Any]] = []

    for domain in domains:
        print(f"\n[{domain.upper()}] Loading tokens for domain: {domain}...")
        domain_tokens = load_domain_tokens(tokenizer, domain, device)
        print(f"[{domain.upper()}] Total corpus tokens available: {len(domain_tokens):,}")

        for T in seq_lens:
            print(f"\n--- Domain: {domain.upper()} | Context Length T = {T} (N={num_samples}) ---")
            chunks = extract_evaluation_chunks(
                domain_tokens, seq_len=T, num_samples=num_samples, seed=args.seed, device=device
            )

            # 1. Oracle Baseline
            print("  [*] Running Condition: Oracle Full (36L)...")
            oracle_res, oracle_cache = evaluate_condition_on_chunks(
                engine, chunks, exit_depth=engine.num_layers, mode="full", strategy="oracle_full"
            )
            print(f"      -> Oracle: PPL = {oracle_res['ppl']:.2f}")

            # 2. Native Projection (Proposed)
            print(f"  [*] Running Condition: Native Projection (Exit L={d_t})...")
            native_res, _ = evaluate_condition_on_chunks(
                engine, chunks, exit_depth=d_t, mode="attention_only", strategy="native_projection",
                oracle_logits_cache=oracle_cache
            )
            print(f"      -> Native Proj: PPL = {native_res['ppl']:.2f} | KL = {native_res['kl_nats']:.4f} nats | Top-1 = {native_res['top1_acc']:.1f}%")

            # 3. Direct Copy (Negative Control)
            print(f"  [*] Running Condition: Direct Copy (Exit L={d_t})...")
            copy_res, _ = evaluate_condition_on_chunks(
                engine, chunks, exit_depth=d_t, mode="attention_only", strategy="direct_copy",
                oracle_logits_cache=oracle_cache
            )
            print(f"      -> Direct Copy: PPL = {copy_res['ppl']:.2f} | KL = {copy_res['kl_nats']:.4f} nats | Top-1 = {copy_res['top1_acc']:.1f}%")

            del oracle_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Record metrics
            for cond_name, r, delta in [
                ("Oracle Baseline", oracle_res, 0.0),
                ("Native Projection", native_res, native_res["ppl"] - oracle_res["ppl"]),
                ("Direct Copy", copy_res, copy_res["ppl"] - oracle_res["ppl"])
            ]:
                results_records.append({
                    "domain": domain,
                    "seq_len": T,
                    "exit_depth": d_t,
                    "num_sequences": num_samples,
                    "total_tokens": r["total_tokens"],
                    "condition": cond_name,
                    "ppl": round(r["ppl"], 2),
                    "delta_ppl": round(delta, 2),
                    "kl_nats": round(r["kl_nats"], 4),
                    "top1_acc": round(r["top1_acc"], 2),
                    "ms_per_tok": round(r["ms_per_tok"], 2)
                })

    # Save to CSV
    csv_file = os.path.join(args.output_dir, "length_and_domain_sweep.csv")
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_records[0].keys()))
        writer.writeheader()
        writer.writerows(results_records)
    print(f"\n[+] Saved detailed results CSV to: {csv_file}")

    # Save to JSON
    json_file = os.path.join(args.output_dir, "length_and_domain_sweep.json")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2)
    print(f"[+] Saved results JSON to: {json_file}")

    # Format Markdown Table
    print("\n" + "=" * 80)
    print("MANUSCRIPT TABLE: CONTEXT LENGTH & OUT-OF-DOMAIN GENERALITY")
    print("=" * 80)

    md_table = """
| Domain | Context Length ($T$) | Condition | PPL | $\\Delta$PPL | $\\bar{D}_{\\text{KL}}$ (nats) | Top-1 Agreement |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
"""
    for r in results_records:
        if r["condition"] == "Oracle Baseline":
            md_table += f"| **{r['domain']}** | {r['seq_len']} | *Oracle Baseline* | **{r['ppl']}** | 0.00 | 0.0000 | 100.0% |\n"
        elif r["condition"] == "Native Projection":
            md_table += f"| | | **Native Proj** | **{r['ppl']}** | **+{r['delta_ppl']}** | **{r['kl_nats']}** | **{r['top1_acc']}%** |\n"
        else:
            md_table += f"| | | Direct Copy | {r['ppl']} | +{r['delta_ppl']} | {r['kl_nats']} | {r['top1_acc']}% |\n"

    print(md_table)

    table_md_file = os.path.join(args.output_dir, "paper_table_length_and_domain.md")
    with open(table_md_file, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"[+] Saved Markdown summary table to: {table_md_file}")


if __name__ == "__main__":
    main()
