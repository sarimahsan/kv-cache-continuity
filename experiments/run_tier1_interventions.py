#!/usr/bin/env python3
"""
experiments/run_tier1_interventions.py

Tier 1 Experiments Runner for Manuscript Revision:
1. Unified Slice Protocol for Table I & Table II:
   - Standardizes on N=30 sequences (1,890 tokens per seed) across 3 seeds (42, 123, 7) on WikiText-2.
   - Eliminates slice-size discrepancies across Table I (previously 378 tokens), Table II (1,890 tokens),
     and context/domain benchmarks.
   - Computes non-parametric 95% bootstrap confidence intervals for D_KL and Top-1 agreement.
   - Incorporates the Cache Omission baseline (skipped slots absent/masked out) alongside Direct Copy.
   - Evaluates the Joint Condition: Native Projection KV cache + trained exit adapter readout.

2. Expanded Causal Key/Value Swap Intervention (Table VII):
   - Evaluates the same large-slice protocol on Qwen2.5-3B across multiple exit depths: L=18, L=24, and L=30.
   - Cross-architecture check on Llama-3.2-1B at L=8.
   - Computes marginal and Shapley attribution for Value dominance (% S_V vs % S_K) across depths and models.

3. Centered Cosine Similarity & Direct Attention-Output Error:
   - Centered cosine removes the Qwen QKV-bias / constant channel mean confound.
   - Direct attention error ||Attn(K*, V*) - Attn(K_hat, V_hat)||_F per layer isolates the true impact on attention.

4. Llama-3.2-1B Shallower Depths Sweep:
   - Sweeps L in {4, 5, 6, 7, 8, 10, 12, 14} across 3 seeds to pinpoint the exact ln 2 crossing depth.

Usage:
  # Full revision suite on Qwen2.5-3B (GPU):
  python experiments/run_tier1_interventions.py --model-type qwen --model-id Qwen/Qwen2.5-3B --device cuda

  # Full revision suite on Llama-3.2-1B (GPU):
  python experiments/run_tier1_interventions.py --model-type llama --model-id meta-llama/Llama-3.2-1B --device cuda

  # Quick smoke-test:
  python experiments/run_tier1_interventions.py --quick --device cpu
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
from src.exit_adapter import ExitCalibrationModule
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


def parse_args():
    parser = argparse.ArgumentParser(description="Tier 1 Manuscript Revision Experiments Runner")
    parser.add_argument("--model-type", type=str, choices=["qwen", "llama", "gemma"], default="qwen")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for teacher-forced evaluation")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of sequences per seed (default: 30 = 1,890 tokens)")
    parser.add_argument("--seeds", type=str, default="42,123,7", help="Comma-separated seeds for multi-seed eval")
    parser.add_argument("--swap-depths", type=str, default=None, help="Comma-separated exit depths for K/V swap test (e.g. '18,24,30' for Qwen, '8' for Llama)")
    parser.add_argument("--sweep-depths", type=str, default=None, help="Comma-separated candidate depths for depth sweep")
    parser.add_argument("--adapter-checkpoint", type=str, default=None, help="Optional path to ExitCalibrationModule checkpoint (.pt)")
    parser.add_argument("--output-dir", type=str, default="results/tier1_revision", help="Directory for output CSVs and JSON")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap iterations for confidence intervals")
    parser.add_argument("--quick", action="store_true", help="Quick run with 5 samples and 1 seed")
    parser.add_argument("--skip-table1", action="store_true", help="Skip Table I standardized disentanglement matrix")
    parser.add_argument("--skip-swap", action="store_true", help="Skip expanded K/V swap test")
    parser.add_argument("--skip-multiseed", action="store_true", help="Skip multi-seed sweep")
    parser.add_argument("--skip-cosine", action="store_true", help="Skip centered cosine & direct attention error analysis")
    return parser.parse_args()


# ==============================================================================
# STATISTICAL HELPERS (BOOTSTRAP CONFIDENCE INTERVALS)
# ==============================================================================
def compute_bootstrap_ci(
    values: List[float],
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes non-parametric bootstrap mean, standard error, and empirical percentile CI.
    """
    if len(values) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_err": 0.0}
    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    if len(arr) == 1:
        return {"mean": mean_val, "ci_lower": mean_val, "ci_upper": mean_val, "std_err": 0.0}

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for b in range(n_boot):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[b] = np.mean(sample)

    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100.0))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    std_err = float(np.std(boot_means, ddof=1))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err
    }


def load_eval_tokens(tokenizer, dataset_name: str = "wikitext2") -> torch.Tensor:
    """Loads 1D token tensor for evaluation, with robust fallback."""
    if dataset_name == "wikitext2":
        try:
            return get_tokenized_wikitext2(tokenizer, split="validation")
        except Exception as e:
            print(f"[!] Warning: WikiText-2 loading failed: {e}. Falling back to canonical texts.")

    # Fallback to concatenated canonical texts
    full_text = "\n\n".join(CANONICAL_FALLBACK_TEXTS * 6)
    if hasattr(tokenizer, "encode"):
        toks = tokenizer.encode(full_text)
        if not isinstance(toks, torch.Tensor):
            toks = torch.tensor(toks, dtype=torch.long)
    else:
        toks = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
    return toks


def sample_sequences(all_tokens: torch.Tensor, seq_len: int, num_samples: int, seed: int, device: torch.device) -> List[torch.Tensor]:
    """Extracts deterministically seeded random non-overlapping chunks of length seq_len."""
    rng = np.random.RandomState(seed)
    max_start = len(all_tokens) - seq_len - 1
    if max_start <= 0:
        chunk = all_tokens[:seq_len].unsqueeze(0).to(device)
        return [chunk] * num_samples

    starts = rng.choice(max_start, size=num_samples, replace=(num_samples > (max_start // seq_len)))
    seqs = []
    for st in starts:
        chunk = all_tokens[st : st + seq_len].unsqueeze(0).to(device)
        seqs.append(chunk)
    return seqs


# ==============================================================================
# 1. CORE EVALUATION LOOP WITH TOKEN-LEVEL METRIC HARVESTING
# ==============================================================================
@torch.no_grad()
def evaluate_condition(
    engine: QwenDynamicEngine,
    sequences: List[torch.Tensor],
    exit_depth: int,
    mode: str,
    strategy: str,
    oracle_logits_cache: Optional[List[List[torch.Tensor]]] = None,
    track_cossim: bool = False,
    adapter: Optional[Any] = None,
    n_boot: int = 1000
) -> Tuple[Dict[str, Any], List[List[torch.Tensor]], Optional[Dict[int, Dict[str, float]]]]:
    """
    Evaluates a specific condition on a list of (1, seq_len) token tensors.
    If oracle_logits_cache is provided, computes exact KL divergence against oracle.
    Returns (metrics_dict, saved_logits, cossim_summary).
    """
    total_loss = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_top1 = 0

    per_token_kl: List[float] = []
    per_token_top1: List[float] = []
    per_token_loss: List[float] = []

    all_layer_stats: Dict[int, Dict[str, List[float]]] = {}
    saved_logits: List[List[torch.Tensor]] = []

    for s_idx, seq in enumerate(sequences):
        T = seq.size(1)
        kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=T + 16)
        seq_logits = []

        for pos in range(T - 1):
            curr_tok = seq[:, pos : pos + 1]
            target_tok = seq[:, pos + 1]

            logits, stats = engine.step_forward_decoupled(
                input_id=curr_tok,
                pos=pos,
                kv_cache=kv_cache,
                exit_depth=exit_depth,
                mode=mode,
                strategy=strategy,
                track_cossim=track_cossim,
                adapter=adapter
            )

            curr_l = logits[:, 0, :]
            seq_logits.append(curr_l.detach().cpu())

            loss = F.cross_entropy(curr_l, target_tok).item()
            total_loss += loss
            total_tokens += 1
            per_token_loss.append(loss)

            if oracle_logits_cache is not None:
                true_l = oracle_logits_cache[s_idx][pos].to(curr_l.device)
                p_true = F.softmax(true_l, dim=-1)
                log_p = F.log_softmax(curr_l, dim=-1)
                kl = F.kl_div(log_p, p_true, reduction="batchmean").item()
                total_kl += kl
                per_token_kl.append(kl)

                match = 1.0 if (curr_l.argmax(dim=-1).item() == true_l.argmax(dim=-1).item()) else 0.0
                total_top1 += int(match)
                per_token_top1.append(match)
            else:
                match = 1.0 if (curr_l.argmax(dim=-1).item() == target_tok.item()) else 0.0
                total_top1 += int(match)
                per_token_top1.append(match)

            if track_cossim and stats is not None:
                for l, st in stats.items():
                    if l not in all_layer_stats:
                        all_layer_stats[l] = {k: [] for k in st.keys()}
                    for k, val in st.items():
                        all_layer_stats[l][k].append(val)

        saved_logits.append(seq_logits)

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    mean_kl = (total_kl / max(1, total_tokens)) if oracle_logits_cache is not None else 0.0
    top1_acc = (total_top1 / max(1, total_tokens)) * 100.0

    # Compute bootstrap confidence intervals
    kl_boot = compute_bootstrap_ci(per_token_kl, n_boot=n_boot) if per_token_kl else {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_err": 0.0}
    top1_boot = compute_bootstrap_ci([x * 100.0 for x in per_token_top1], n_boot=n_boot) if per_token_top1 else {"mean": top1_acc, "ci_lower": top1_acc, "ci_upper": top1_acc, "std_err": 0.0}

    metrics = {
        "loss": mean_loss,
        "ppl": ppl,
        "kl_nats": mean_kl,
        "kl_ci_lower": kl_boot["ci_lower"],
        "kl_ci_upper": kl_boot["ci_upper"],
        "kl_std_err": kl_boot["std_err"],
        "top1_acc": top1_acc,
        "top1_ci_lower": top1_boot["ci_lower"],
        "top1_ci_upper": top1_boot["ci_upper"],
        "top1_std_err": top1_boot["std_err"],
        "total_tokens": total_tokens
    }

    cossim_summary = None
    if track_cossim and all_layer_stats:
        cossim_summary = {}
        for l in sorted(all_layer_stats.keys()):
            cossim_summary[l] = {k: float(np.mean(vals)) for k, vals in all_layer_stats[l].items()}

    return metrics, saved_logits, cossim_summary


# ==============================================================================
# 2. EXPERIMENT 1: STANDARDIZED TABLE I WITH BOOTSTRAP CIs & CACHE OMISSION
# ==============================================================================
def run_table1_standardized(
    engine: QwenDynamicEngine,
    all_tokens: torch.Tensor,
    depths: List[int],
    seeds: List[int],
    num_samples_per_seed: int,
    seq_len: int,
    output_dir: str,
    adapter: Optional[Any] = None,
    n_boot: int = 1000
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("EXPERIMENT 1: STANDARDIZED HEADLINE TABLE I DISENTANGLEMENT MATRIX")
    print(f"Protocol: N={num_samples_per_seed} sequences across {len(seeds)} seeds ({seeds}) on WikiText-2 (Total: {num_samples_per_seed * (seq_len - 1) * len(seeds):,} tokens)")
    print(f"Candidate Exit Depths: {depths}")
    print("=" * 90)

    # Base Oracle full pass per seed
    conditions = [
        ("readout_only", "oracle_full", "Readout-Only (Zero-Shot)", False),
        ("attention_only", "cache_omission", "Attn-Only (Cache Omission)", False),
        ("attention_only", "direct_copy", "Attn-Only (Direct Copy)", False),
        ("attention_only", "native_projection", "Attn-Only (Native Proj)", False),
    ]
    if adapter is not None:
        conditions.append(("readout_only", "oracle_full", "Readout-Only (Adapter)", True))
        conditions.append(("combined", "native_projection", "Joint (Native Proj + Adapter)", True))

    table_records = []
    oracle_ppls = []

    for s_idx, seed in enumerate(seeds):
        print(f"\n--- [Seed {seed} ({s_idx + 1}/{len(seeds)})] Generating Oracle Logits ---")
        seqs = sample_sequences(all_tokens, seq_len=seq_len, num_samples=num_samples_per_seed, seed=seed, device=engine.device)

        oracle_m, oracle_logits, _ = evaluate_condition(
            engine=engine,
            sequences=seqs,
            exit_depth=engine.num_layers,
            mode="full",
            strategy="oracle_full",
            oracle_logits_cache=None,
            n_boot=n_boot
        )
        oracle_ppls.append(oracle_m["ppl"])
        print(f"  Oracle Baseline (L={engine.num_layers}): PPL = {oracle_m['ppl']:.4f}")

        for d in depths:
            for mode, strat, label, use_adapter in conditions:
                ad_obj = adapter if use_adapter else None
                m, _, _ = evaluate_condition(
                    engine=engine,
                    sequences=seqs,
                    exit_depth=d,
                    mode=mode,
                    strategy=strat,
                    oracle_logits_cache=oracle_logits,
                    adapter=ad_obj,
                    n_boot=n_boot
                )
                delta_ppl = m["ppl"] - oracle_m["ppl"]
                row = {
                    "seed": seed,
                    "exit_depth": d,
                    "rel_depth": (d / engine.num_layers) * 100.0,
                    "mode": mode,
                    "strategy": strat,
                    "condition_label": label,
                    "ppl": m["ppl"],
                    "delta_ppl": delta_ppl,
                    "kl_nats": m["kl_nats"],
                    "kl_ci_lower": m["kl_ci_lower"],
                    "kl_ci_upper": m["kl_ci_upper"],
                    "top1_acc": m["top1_acc"],
                    "top1_ci_lower": m["top1_ci_lower"],
                    "top1_ci_upper": m["top1_ci_upper"]
                }
                table_records.append(row)
                print(f"  [L={d:2d} | {label:<32}] PPL = {m['ppl']:>9.2f} | ΔPPL = {delta_ppl:>+9.2f} | KL = {m['kl_nats']:>6.4f} [{m['kl_ci_lower']:.4f}, {m['kl_ci_upper']:.4f}] | Top-1 = {m['top1_acc']:>5.1f}% [{m['top1_ci_lower']:.1f}%, {m['top1_ci_upper']:.1f}%]")

    # Aggregate across seeds
    aggregated = []
    mean_oracle_ppl = float(np.mean(oracle_ppls))

    print("\n" + "=" * 105)
    print(f"TABLE I: UNIFIED STANDARDIZED DISENTANGLEMENT MATRIX (Aggregated across {len(seeds)} Seeds, Oracle PPL* = {mean_oracle_ppl:.2f})")
    print("=" * 105)
    print(f"{'Exit L':<8} | {'Condition':<32} | {'PPL':<10} | {'ΔPPL':<10} | {'KL (nats) [95% CI]':<26} | {'Top-1 [95% CI]':<22}")
    print("-" * 105)

    for d in depths:
        for mode, strat, label, use_adapter in conditions:
            sub = [r for r in table_records if r["exit_depth"] == d and r["condition_label"] == label]
            ppl_mu = float(np.mean([r["ppl"] for r in sub]))
            delta_mu = ppl_mu - mean_oracle_ppl
            kl_mu = float(np.mean([r["kl_nats"] for r in sub]))
            kl_lo = float(np.mean([r["kl_ci_lower"] for r in sub]))
            kl_hi = float(np.mean([r["kl_ci_upper"] for r in sub]))
            top1_mu = float(np.mean([r["top1_acc"] for r in sub]))
            top1_lo = float(np.mean([r["top1_ci_lower"] for r in sub]))
            top1_hi = float(np.mean([r["top1_ci_upper"] for r in sub]))

            agg_row = {
                "exit_depth": d,
                "rel_depth": (d / engine.num_layers) * 100.0,
                "condition_label": label,
                "ppl": ppl_mu,
                "delta_ppl": delta_mu,
                "kl_nats": kl_mu,
                "kl_ci_lower": kl_lo,
                "kl_ci_upper": kl_hi,
                "top1_acc": top1_mu,
                "top1_ci_lower": top1_lo,
                "top1_ci_upper": top1_hi,
                "num_seeds": len(seeds)
            }
            aggregated.append(agg_row)
            print(f"{d:<8} | {label:<32} | {ppl_mu:>10.2f} | {delta_mu:>+10.2f} | {kl_mu:>6.4f} [{kl_lo:.4f}, {kl_hi:.4f}]     | {top1_mu:>5.1f}% [{top1_lo:.1f}%, {top1_hi:.1f}%]")

    # Save CSV and JSON
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "standardized_table1_headline.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit_depth", "rel_depth", "condition_label", "ppl", "delta_ppl", "kl_nats", "kl_ci_lower", "kl_ci_upper", "top1_acc", "top1_ci_lower", "top1_ci_upper", "num_seeds"])
        writer.writeheader()
        writer.writerows(aggregated)

    json_path = os.path.join(output_dir, "standardized_table1_headline.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"aggregated": aggregated, "oracle_ppl": mean_oracle_ppl, "raw_runs": table_records}, f, indent=2)

    print(f"\n✓ Saved Standardized Table I results to {csv_path}")
    return {"aggregated": aggregated, "oracle_ppl": mean_oracle_ppl}


# ==============================================================================
# 3. EXPERIMENT 2: EXPANDED CAUSAL KEY/VALUE SWAP (TABLE VII) WITH SHAPLEY
# ==============================================================================
def run_expanded_causal_kv_swap(
    engine: QwenDynamicEngine,
    all_tokens: torch.Tensor,
    swap_depths: List[int],
    seeds: List[int],
    num_samples_per_seed: int,
    seq_len: int,
    output_dir: str,
    n_boot: int = 1000
) -> Dict[str, Any]:
    print("\n" + "=" * 90)
    print("EXPERIMENT 2: EXPANDED CAUSAL KEY/VALUE SWAP INTERVENTION (TABLE VII)")
    print(f"Evaluating across exit depths: {swap_depths} on larger slice (N={num_samples_per_seed}, seeds={seeds})")
    print("Isolates Key routing vs. Value semantic tolerance with Marginal & Shapley Attributions")
    print("=" * 90)

    swap_conditions = [
        ("oracle_full", "Oracle Baseline (K*, V*)", "full"),
        ("proj_k_oracle_v", "Projected Keys + Oracle Values", "attention_only"),
        ("oracle_k_proj_v", "Oracle Keys + Projected Values", "attention_only"),
        ("native_projection", "Native Projection (K̂, V̂)", "attention_only"),
    ]

    depth_summaries = []

    for d in swap_depths:
        rel_d = (d / engine.num_layers) * 100.0
        print(f"\n>>> Running Causal Swap at L = {d} / {engine.num_layers} ({rel_d:.1f}% Relative Depth) <<<")

        depth_records = []
        for s_idx, seed in enumerate(seeds):
            seqs = sample_sequences(all_tokens, seq_len=seq_len, num_samples=num_samples_per_seed, seed=seed, device=engine.device)

            # Oracle reference pass
            oracle_m, oracle_logits, _ = evaluate_condition(
                engine=engine,
                sequences=seqs,
                exit_depth=engine.num_layers,
                mode="full",
                strategy="oracle_full",
                n_boot=n_boot
            )

            for strat, name, mode in swap_conditions:
                m, _, _ = evaluate_condition(
                    engine=engine,
                    sequences=seqs,
                    exit_depth=d if mode != "full" else engine.num_layers,
                    mode=mode,
                    strategy=strat,
                    oracle_logits_cache=oracle_logits,
                    n_boot=n_boot
                )
                depth_records.append({
                    "exit_depth": d,
                    "seed": seed,
                    "strategy": strat,
                    "condition_name": name,
                    "ppl": m["ppl"],
                    "kl_nats": m["kl_nats"],
                    "kl_ci_lower": m["kl_ci_lower"],
                    "kl_ci_upper": m["kl_ci_upper"],
                    "top1_acc": m["top1_acc"],
                    "top1_ci_lower": m["top1_ci_lower"],
                    "top1_ci_upper": m["top1_ci_upper"]
                })

        # Aggregate across seeds for this depth
        cond_means = {}
        for strat, name, _ in swap_conditions:
            sub = [r for r in depth_records if r["strategy"] == strat]
            cond_means[strat] = {
                "name": name,
                "ppl": float(np.mean([r["ppl"] for r in sub])),
                "kl": float(np.mean([r["kl_nats"] for r in sub])),
                "kl_lo": float(np.mean([r["kl_ci_lower"] for r in sub])),
                "kl_hi": float(np.mean([r["kl_ci_upper"] for r in sub])),
                "top1": float(np.mean([r["top1_acc"] for r in sub])),
                "top1_lo": float(np.mean([r["top1_ci_lower"] for r in sub])),
                "top1_hi": float(np.mean([r["top1_ci_upper"] for r in sub])),
            }

        kl_oracle = cond_means["oracle_full"]["kl"]
        kl_proj_k_oracle_v = cond_means["proj_k_oracle_v"]["kl"]
        kl_oracle_k_proj_v = cond_means["oracle_k_proj_v"]["kl"]
        kl_full_proj = cond_means["native_projection"]["kl"]

        # Marginal & Shapley Attributions
        delta_v1 = kl_oracle_k_proj_v - kl_oracle
        delta_v2 = kl_full_proj - kl_proj_k_oracle_v
        shapley_v = 0.5 * (delta_v1 + delta_v2)

        delta_k1 = kl_proj_k_oracle_v - kl_oracle
        delta_k2 = kl_full_proj - kl_oracle_k_proj_v
        shapley_k = 0.5 * (delta_k1 + delta_k2)

        total_shapley = max(1e-8, shapley_v + shapley_k)
        pct_value_dominance = (shapley_v / total_shapley) * 100.0

        print(f"\nResults for L = {d}:")
        print(f"{'Condition':<32} | {'Val PPL':<10} | {'KL (nats) [95% CI]':<26} | {'Top-1 [95% CI]':<22}")
        print("-" * 96)
        for strat, name, _ in swap_conditions:
            cm = cond_means[strat]
            print(f"{name:<32} | {cm['ppl']:>10.2f} | {cm['kl']:>6.4f} [{cm['kl_lo']:.4f}, {cm['kl_hi']:.4f}]     | {cm['top1']:>5.1f}% [{cm['top1_lo']:.1f}%, {cm['top1_hi']:.1f}%]")

        print(f"--> Shapley Attribution at L={d}: Value Dominance = {pct_value_dominance:.1f}% | Key Routing = {100.0 - pct_value_dominance:.1f}%")

        depth_summaries.append({
            "exit_depth": d,
            "rel_depth": rel_d,
            "conditions": cond_means,
            "shapley_value_nats": shapley_v,
            "shapley_key_nats": shapley_k,
            "pct_value_dominance": pct_value_dominance
        })

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "expanded_table7_causal_kv_swap.csv")
    flat_rows = []
    for ds in depth_summaries:
        d = ds["exit_depth"]
        for strat, cm in ds["conditions"].items():
            flat_rows.append({
                "exit_depth": d,
                "rel_depth": ds["rel_depth"],
                "strategy": strat,
                "condition_name": cm["name"],
                "ppl": cm["ppl"],
                "kl_nats": cm["kl"],
                "kl_ci_lower": cm["kl_lo"],
                "kl_ci_upper": cm["kl_hi"],
                "top1_acc": cm["top1"],
                "top1_ci_lower": cm["top1_lo"],
                "top1_ci_upper": cm["top1_hi"],
                "pct_value_dominance": ds["pct_value_dominance"]
            })
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit_depth", "rel_depth", "strategy", "condition_name", "ppl", "kl_nats", "kl_ci_lower", "kl_ci_upper", "top1_acc", "top1_ci_lower", "top1_ci_upper", "pct_value_dominance"])
        writer.writeheader()
        writer.writerows(flat_rows)

    json_path = os.path.join(output_dir, "expanded_table7_causal_kv_swap.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(depth_summaries, f, indent=2)

    print(f"\n✓ Saved Expanded Table VII Causal Swap results to {csv_path}")

    # Print LaTeX snippet for revised Table VII
    print("\n--- LaTeX Snippet for Revised Table VII (Multi-Depth & Model Causal Swap) ---")
    print("\\begin{table}[!t]")
    print("\\centering")
    print("\\caption{Expanded Causal Key/Value Swap Intervention on WikiText-2 ($N=30$ slice per seed, $\\mu$ across 3 seeds)}")
    print("\\label{tab:kv_swap_expanded}")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{@{}llcccc@{}}")
    print("\\toprule")
    print("\\textbf{Exit Depth} & \\textbf{Attention Cache State} & \\textbf{Val PPL} & $\\boldsymbol{\\kl}$ \\textbf{(nats)} & \\textbf{Top-1} & \\textbf{Val Share} \\\\")
    print("\\midrule")
    for ds in depth_summaries:
        d = ds["exit_depth"]
        cms = ds["conditions"]
        print(f"\\multicolumn{{6}}{{l}}{{\\textit{{Exit Depth $L = {d}$ ({ds['rel_depth']:.1f}\\% relative depth)}}}} \\\\")
        print(f" & Oracle Baseline ($K^*, V^*$) & {cms['oracle_full']['ppl']:.2f} & {cms['oracle_full']['kl']:.4f} & {cms['oracle_full']['top1']:.1f}\\% & -- \\\\")
        print(f" & Projected Keys + Oracle Values & {cms['proj_k_oracle_v']['ppl']:.2f} & {cms['proj_k_oracle_v']['kl']:.4f} & {cms['proj_k_oracle_v']['top1']:.1f}\\% & -- \\\\")
        print(f" & Oracle Keys + Projected Values & {cms['oracle_k_proj_v']['ppl']:.2f} & {cms['oracle_k_proj_v']['kl']:.4f} & {cms['oracle_k_proj_v']['top1']:.1f}\\% & -- \\\\")
        print(f" & Native Projection ($\\hat{{K}}, \\hat{{V}}$) & {cms['native_projection']['ppl']:.2f} & {cms['native_projection']['kl']:.4f} & {cms['native_projection']['top1']:.1f}\\% & \\textbf{{{ds['pct_value_dominance']:.1f}\\%}} \\\\")
        print("\\midrule")
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")

    return {"depth_summaries": depth_summaries}


# ==============================================================================
# 4. EXPERIMENT 3: CENTERED COSINE & DIRECT ATTENTION-OUTPUT ERROR DIAGNOSTIC
# ==============================================================================
def run_centered_cosine_diagnostic(
    engine: QwenDynamicEngine,
    all_tokens: torch.Tensor,
    exit_depth: int,
    num_samples: int,
    seq_len: int,
    output_dir: str
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 90)
    print("EXPERIMENT 3: CENTERED COSINE & DIRECT ATTENTION-OUTPUT ERROR")
    print(f"Model ID: {engine.model_id} | Exit Depth: L = {exit_depth}")
    print("Computes standard vs. centered cosine for (K, V) and ||Attn(K*,V*) - Attn(K̂,V̂)|| per layer")
    print("=" * 90)

    seqs = sample_sequences(all_tokens, seq_len=seq_len, num_samples=num_samples, seed=42, device=engine.device)

    # First execute oracle pass to populate oracle KV buffers
    _, oracle_logits, _ = evaluate_condition(
        engine=engine,
        sequences=seqs,
        exit_depth=engine.num_layers,
        mode="full",
        strategy="oracle_full"
    )

    # Now evaluate Native Projection while tracking per-layer cossim and attention error
    _, _, cossim_summary = evaluate_condition(
        engine=engine,
        sequences=seqs,
        exit_depth=exit_depth,
        mode="attention_only",
        strategy="native_projection",
        oracle_logits_cache=oracle_logits,
        track_cossim=True
    )

    rows = []
    print(f"{'Layer':<6} | {'Key Cos (Std)':<14} | {'Key Cent Cos':<14} | {'Val Cos (Std)':<14} | {'Val Cent Cos':<14} | {'Attn Error ||Δ||_F':<20} | {'Rel Attn Err':<14}")
    print("-" * 105)

    for l in sorted(cossim_summary.keys()):
        s = cossim_summary[l]
        row = {
            "exit_depth": exit_depth,
            "layer_1_indexed": l + 1,
            "cos_k_std": s.get("cos_k", 0.0),
            "cos_k_centered": s.get("centered_cos_k", 0.0),
            "cos_v_std": s.get("cos_v", 0.0),
            "cos_v_centered": s.get("centered_cos_v", 0.0),
            "attn_fro_err": s.get("attn_fro_err", 0.0),
            "attn_rel_err": s.get("attn_rel_err", 0.0),
            "attn_cos": s.get("attn_cos", 0.0),
            "attn_centered_cos": s.get("attn_centered_cos", 0.0),
            "rel_k": s.get("rel_k", 0.0),
            "rel_v": s.get("rel_v", 0.0),
        }
        rows.append(row)
        print(f"{l+1:<6} | {s['cos_k']:<14.4f} | {s['centered_cos_k']:<14.4f} | {s['cos_v']:<14.4f} | {s['centered_cos_v']:<14.4f} | {s['attn_fro_err']:<20.4f} | {s['attn_rel_err']:<14.4f}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"centered_cosine_and_attn_err_L{exit_depth}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Saved Centered Cosine & Attention Output Error diagnostic to {csv_path}")
    return rows


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.model_id is None:
        if args.model_type == "qwen":
            args.model_id = "Qwen/Qwen2.5-3B"
        elif args.model_type == "llama":
            args.model_id = "meta-llama/Llama-3.2-1B"
        elif args.model_type == "gemma":
            args.model_id = "google/gemma-2-2b"

    # Default swap depths
    if args.swap_depths is not None:
        swap_depths = [int(d.strip()) for d in args.swap_depths.split(",") if d.strip()]
    else:
        if args.model_type == "qwen":
            swap_depths = [18, 24, 30]
        elif args.model_type == "llama":
            swap_depths = [8]
        else:
            swap_depths = [13]

    # Default sweep depths (includes Llama shallower depths L=4,5,6,7 to locate exact ln 2 crossing!)
    if args.sweep_depths is not None:
        sweep_depths = [int(d.strip()) for d in args.sweep_depths.split(",") if d.strip()]
    else:
        if args.model_type == "qwen":
            sweep_depths = [18, 22, 24, 28, 30, 32, 34]
        elif args.model_type == "llama":
            sweep_depths = [4, 5, 6, 7, 8, 10, 12, 14]
        else:
            sweep_depths = [8, 13, 17, 21, 24]

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.quick:
        seeds = [42]
        args.num_samples = 5
        sweep_depths = sweep_depths[:2]
        swap_depths = swap_depths[:1]

    print("=" * 90)
    print("STARTING TIER 1 EXPERIMENT SUITE (MANUSCRIPT REVISION)")
    print(f"Model ID: {args.model_id} | Type: {args.model_type} | Device: {device}")
    print(f"Unified Slice Protocol: N={args.num_samples} sequences, SeqLen={args.seq_len}, Seeds={seeds}")
    print(f"Causal Swap Depths: {swap_depths} | Depth Sweep: {sweep_depths}")
    print("=" * 90)

    # 1. Initialize Engine
    dtype = torch.bfloat16 if args.dtype == "bfloat16" or (args.dtype == "auto" and torch.cuda.is_bf16_supported()) else torch.float16
    engine = QwenDynamicEngine(
        model_id=args.model_id,
        device=args.device,
        dtype=dtype if device.type == "cuda" else torch.float32,
    )

    # Load optional adapter checkpoint if provided
    adapter = None
    if args.adapter_checkpoint and os.path.exists(args.adapter_checkpoint):
        print(f"[*] Loading ExitCalibrationModule checkpoint from: {args.adapter_checkpoint}")
        try:
            adapter = ExitCalibrationModule.load_checkpoint(args.adapter_checkpoint, device=device)
            print(f"[OK] Loaded adapter successfully (candidate depths: {adapter.candidate_depths}, rank: {adapter.rank})")
        except Exception as e:
            print(f"[!] Warning: Failed to load adapter checkpoint: {e}")

    # 2. Load Evaluation Token Stream
    print("[*] Preparing evaluation tokens (WikiText-2)...")
    all_tokens = load_eval_tokens(engine.tokenizer, dataset_name="wikitext2")
    print(f"[OK] Loaded {len(all_tokens):,} evaluation tokens.")

    # 3. Task 1: Standardized Table I Disentanglement Matrix with Bootstrap CIs
    if not args.skip_table1:
        run_table1_standardized(
            engine=engine,
            all_tokens=all_tokens,
            depths=sweep_depths,
            seeds=seeds,
            num_samples_per_seed=args.num_samples,
            seq_len=args.seq_len,
            output_dir=args.output_dir,
            adapter=adapter,
            n_boot=args.n_bootstrap
        )

    # 4. Task 2: Expanded Causal Key/Value Swap Intervention (Table VII)
    if not args.skip_swap:
        run_expanded_causal_kv_swap(
            engine=engine,
            all_tokens=all_tokens,
            swap_depths=swap_depths,
            seeds=seeds,
            num_samples_per_seed=args.num_samples,
            seq_len=args.seq_len,
            output_dir=args.output_dir,
            n_boot=args.n_bootstrap
        )

    # 5. Task 3: Centered Cosine & Direct Attention Error Diagnostic
    if not args.skip_cosine:
        target_diag_depth = swap_depths[0]
        run_centered_cosine_diagnostic(
            engine=engine,
            all_tokens=all_tokens,
            exit_depth=target_diag_depth,
            num_samples=min(20, args.num_samples),
            seq_len=args.seq_len,
            output_dir=args.output_dir
        )

    print("\n" + "=" * 90)
    print("ALL TIER 1 REVISION EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results stored under: {os.path.abspath(args.output_dir)}")
    print("=" * 90)


if __name__ == "__main__":
    main()
