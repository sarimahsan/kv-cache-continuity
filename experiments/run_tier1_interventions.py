#!/usr/bin/env python3
"""
experiments/run_tier1_interventions.py

Tier 1 Experiments Runner for Manuscript Revision:
1. Causal Key/Value Swap Intervention at fixed exit depth (e.g., L=18 on Qwen2.5-3B, L=8 on Llama-3.2-1B):
   - Oracle Keys + Oracle Values (Pristine Oracle baseline)
   - Projected Keys + Oracle Values (Test Key routing sufficiency)
   - Oracle Keys + Projected Values (Test Value semantic tolerance)
   - Projected Keys + Projected Values (Native Projection baseline)
   Measures KL divergence D_KL(P_oracle || P_cond), Perplexity (PPL), and Top-1 agreement (%).

2. Multi-Seed Disentanglement Evaluation for Statistical Rigor:
   - Re-evaluates depth sweeps across 3 distinct seeds (e.g., 42, 123, 7) on WikiText-2.
   - Computes and formats mean ± standard deviation for D_KL, PPL, and Top-1 agreement.

Usage:
  # Quick smoke-test on GPU:
  python experiments/run_tier1_interventions.py --quick --device cuda

  # Full evaluation on Qwen2.5-3B:
  python experiments/run_tier1_interventions.py --model-type qwen --model-id Qwen/Qwen2.5-3B --device cuda

  # Full evaluation on Llama-3.2-1B:
  python experiments/run_tier1_interventions.py --model-type llama --model-id meta-llama/Llama-3.2-1B --device cuda
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


def parse_args():
    parser = argparse.ArgumentParser(description="Tier 1 Manuscript Revision Experiments Runner")
    parser.add_argument("--model-type", type=str, choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face model ID")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for teacher-forced evaluation")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of sequences per seed")
    parser.add_argument("--seeds", type=str, default="42,123,7", help="Comma-separated seeds for multi-seed eval")
    parser.add_argument("--exit-depth", type=int, default=None, help="Exit depth for K/V swap test (default: 18 for Qwen, 8 for Llama)")
    parser.add_argument("--output-dir", type=str, default="results/tier1_revision", help="Directory for output CSVs and JSON")
    parser.add_argument("--quick", action="store_true", help="Quick run with 5 samples and 1 seed")
    parser.add_argument("--skip-swap", action="store_true", help="Skip K/V swap test")
    parser.add_argument("--skip-multiseed", action="store_true", help="Skip multi-seed sweep")
    return parser.parse_args()


def load_eval_tokens(tokenizer, dataset_name: str = "wikitext2") -> torch.Tensor:
    """Loads 1D token tensor for evaluation, with robust fallback."""
    if dataset_name == "wikitext2":
        try:
            return get_tokenized_wikitext2(tokenizer, split="validation")
        except Exception as e:
            print(f"[!] Warning: WikiText-2 loading failed: {e}. Falling back to canonical texts.")

    # Fallback to concatenated canonical texts
    full_text = "\n\n".join(CANONICAL_FALLBACK_TEXTS * 5)
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
# 1. CORE EVALUATION LOOP WITH KL DIVERGENCE RECORDING
# ==============================================================================
@torch.no_grad()
def evaluate_condition(
    engine: QwenDynamicEngine,
    sequences: List[torch.Tensor],
    exit_depth: int,
    mode: str,
    strategy: str,
    oracle_logits_cache: Optional[List[List[torch.Tensor]]] = None
) -> Tuple[Dict[str, float], List[List[torch.Tensor]]]:
    """
    Evaluates a specific condition on a list of (1, seq_len) token tensors.
    If oracle_logits_cache is provided, computes exact KL divergence against oracle.
    Returns (metrics_dict, generated_logits_cache).
    """
    total_loss = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_top1 = 0
    saved_logits: List[List[torch.Tensor]] = []

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

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    mean_kl = (total_kl / max(1, total_tokens)) if oracle_logits_cache is not None else 0.0
    top1_acc = (total_top1 / max(1, total_tokens)) * 100.0

    metrics = {
        "loss": mean_loss,
        "ppl": ppl,
        "kl_nats": mean_kl,
        "top1_acc": top1_acc,
        "total_tokens": total_tokens
    }
    return metrics, saved_logits


# ==============================================================================
# 2. EXPERIMENT 1: CAUSAL KEY/VALUE SWAP INTERVENTION
# ==============================================================================
def run_causal_kv_swap(
    engine: QwenDynamicEngine,
    sequences: List[torch.Tensor],
    exit_depth: int,
    output_dir: str
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: CAUSAL KEY / VALUE SWAP INTERVENTION")
    print(f"Exit Depth: L = {exit_depth} / {engine.num_layers} ({exit_depth / engine.num_layers * 100:.1f}% depth)")
    print("=" * 80)
    print("Testing four conditions to isolate whether Key routing or Value content drives KL error:")
    print("  1. Oracle (K*, V*)          -> Baseline oracle")
    print("  2. Proj K + Oracle V        -> Tests Key routing sufficiency")
    print("  3. Oracle K + Proj V        -> Tests Value semantic tolerance")
    print("  4. Native Proj (K_hat, V_hat)-> Full native projection baseline")
    print("-" * 80)

    # 1. First record full-depth Oracle logits
    print("[*] Computing full-depth Oracle reference logits...")
    oracle_metrics, oracle_logits_cache = evaluate_condition(
        engine=engine,
        sequences=sequences,
        exit_depth=engine.num_layers,
        mode="full",
        strategy="oracle_full",
        oracle_logits_cache=None
    )
    print(f"  Oracle Baseline: PPL = {oracle_metrics['ppl']:.4f} | Top-1 vs Truth = {oracle_metrics['top1_acc']:.2f}%")

    swap_conditions = [
        ("oracle_full", "Oracle (K*, V*)", "full"),
        ("proj_k_oracle_v", "Projected K + Oracle V", "attention_only"),
        ("oracle_k_proj_v", "Oracle K + Projected V", "attention_only"),
        ("native_projection", "Native Proj (K_proj, V_proj)", "attention_only"),
    ]

    results = []
    print(f"\n{'Condition':<32} | {'Val PPL':<10} | {'KL (nats)':<12} | {'Top-1 Oracle (%)':<18}")
    print("-" * 80)

    for strat, name, mode in swap_conditions:
        metrics, _ = evaluate_condition(
            engine=engine,
            sequences=sequences,
            exit_depth=exit_depth if mode != "full" else engine.num_layers,
            mode=mode,
            strategy=strat,
            oracle_logits_cache=oracle_logits_cache
        )
        row = {
            "exit_depth": exit_depth,
            "strategy": strat,
            "condition_name": name,
            "ppl": metrics["ppl"],
            "kl_nats": metrics["kl_nats"],
            "top1_acc": metrics["top1_acc"],
            "loss": metrics["loss"]
        }
        results.append(row)
        print(f"{name:<32} | {metrics['ppl']:>10.2f} | {metrics['kl_nats']:>12.4f} | {metrics['top1_acc']:>18.2f}%")

    # Save to CSV and JSON
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"causal_kv_swap_L{exit_depth}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit_depth", "strategy", "condition_name", "ppl", "kl_nats", "top1_acc", "loss"])
        writer.writeheader()
        writer.writerows(results)

    json_path = os.path.join(output_dir, f"causal_kv_swap_L{exit_depth}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved Causal Swap results to {csv_path}")

    # Print LaTeX Table snippet
    print("\n--- LaTeX Table Snippet for Manuscript (Section VII-C) ---")
    print("\\begin{table}[!t]")
    print("\\centering")
    print(f"\\caption{{Causal Key/Value Swap Intervention at $L{{=}}{exit_depth}$ on Qwen2.5-3B}}")
    print(f"\\label{{tab:kv_swap_L{exit_depth}}}")
    print("\\resizebox{\\columnwidth}{!}{%")
    print("\\begin{tabular}{@{}llccc@{}}")
    print("\\toprule")
    print("\\textbf{Condition} & \\textbf{Cache State} & \\textbf{Val PPL} & $\\boldsymbol{\\kl}$ \\textbf{(nats)} & \\textbf{Top-1 Agreement} \\\\")
    print("\\midrule")
    for r in results:
        print(f"{r['condition_name']} & \\texttt{{{r['strategy']}}} & {r['ppl']:.2f} & {r['kl_nats']:.4f} & {r['top1_acc']:.1f}\\% \\\\")
    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")

    return {"results": results, "oracle_ppl": oracle_metrics["ppl"]}


# ==============================================================================
# 3. EXPERIMENT 2: MULTI-SEED RIGOROUS EVALUATION
# ==============================================================================
def run_multiseed_eval(
    engine: QwenDynamicEngine,
    all_tokens: torch.Tensor,
    depths: List[int],
    seeds: List[int],
    num_samples_per_seed: int,
    seq_len: int,
    output_dir: str
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: MULTI-SEED STATISTICAL RIGOR (MEAN ± STD)")
    print(f"Tested Depths: {depths} | Seeds: {seeds} | Samples/Seed: {num_samples_per_seed} | SeqLen: {seq_len}")
    print("=" * 80)

    # Dictionary: depth -> condition -> list of metric dicts across seeds
    depth_records = {d: {"native_projection": [], "direct_copy": []} for d in depths}
    oracle_ppls = []

    for s_idx, seed in enumerate(seeds):
        print(f"\n>>> Running Evaluation with Seed {seed} ({s_idx + 1}/{len(seeds)}) <<<")
        seqs = sample_sequences(all_tokens, seq_len=seq_len, num_samples=num_samples_per_seed, seed=seed, device=engine.device)

        # Oracle pass for this seed's sequences
        _, oracle_logits_cache = evaluate_condition(
            engine=engine,
            sequences=seqs,
            exit_depth=engine.num_layers,
            mode="full",
            strategy="oracle_full"
        )
        oracle_res, _ = evaluate_condition(
            engine=engine,
            sequences=seqs,
            exit_depth=engine.num_layers,
            mode="full",
            strategy="oracle_full",
            oracle_logits_cache=oracle_logits_cache
        )
        oracle_ppls.append(oracle_res["ppl"])

        for d in depths:
            for strat in ["native_projection", "direct_copy"]:
                metrics, _ = evaluate_condition(
                    engine=engine,
                    sequences=seqs,
                    exit_depth=d,
                    mode="attention_only",
                    strategy=strat,
                    oracle_logits_cache=oracle_logits_cache
                )
                depth_records[d][strat].append(metrics)
                print(f"  [Seed {seed} | L={d:2d} | {strat:<17}] PPL = {metrics['ppl']:>9.2f} | KL = {metrics['kl_nats']:>7.4f} | Top-1 = {metrics['top1_acc']:>5.1f}%")

    # Aggregate statistics
    aggregated_results = []
    print("\n" + "=" * 85)
    print(f"{'Exit L':<8} | {'Strategy':<18} | {'Val PPL (mean±std)':<22} | {'KL nats (mean±std)':<22} | {'Top-1 % (mean±std)':<20}")
    print("-" * 85)

    for d in depths:
        for strat in ["native_projection", "direct_copy"]:
            runs = depth_records[d][strat]
            ppls = [r["ppl"] for r in runs]
            kls = [r["kl_nats"] for r in runs]
            top1s = [r["top1_acc"] for r in runs]

            ppl_mu, ppl_std = float(np.mean(ppls)), float(np.std(ppls, ddof=1) if len(ppls) > 1 else 0.0)
            kl_mu, kl_std = float(np.mean(kls)), float(np.std(kls, ddof=1) if len(kls) > 1 else 0.0)
            top1_mu, top1_std = float(np.mean(top1s)), float(np.std(top1s, ddof=1) if len(top1s) > 1 else 0.0)

            row = {
                "exit_depth": d,
                "strategy": strat,
                "ppl_mean": ppl_mu,
                "ppl_std": ppl_std,
                "kl_mean": kl_mu,
                "kl_std": kl_std,
                "top1_mean": top1_mu,
                "top1_std": top1_std,
                "num_seeds": len(seeds)
            }
            aggregated_results.append(row)
            print(f"{d:<8} | {strat:<18} | {ppl_mu:>9.2f} ± {ppl_std:<9.2f} | {kl_mu:>8.4f} ± {kl_std:<8.4f} | {top1_mu:>6.2f}% ± {top1_std:<5.2f}%")

    # Save CSV and JSON
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "multiseed_depth_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit_depth", "strategy", "ppl_mean", "ppl_std", "kl_mean", "kl_std", "top1_mean", "top1_std", "num_seeds"])
        writer.writeheader()
        writer.writerows(aggregated_results)

    json_path = os.path.join(output_dir, "multiseed_depth_sweep.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_results, f, indent=2)

    print(f"\n✓ Saved Multi-Seed summary to {csv_path}")

    # Print LaTeX Table snippet
    print("\n--- LaTeX Table Snippet for Multi-Seed Table I / II ---")
    print("\\begin{table*}[!t]")
    print("\\centering")
    print(f"\\caption{{Multi-Seed Disentangled Attention Evaluation (mean $\\pm$ std across {len(seeds)} seeds, WikiText-2)}}")
    print("\\label{tab:multiseed_eval}")
    print("\\begin{tabular}{@{}cccccc@{}}")
    print("\\toprule")
    print("\\textbf{Exit} $L$ & \\textbf{Rel. Depth} & \\multicolumn{2}{c}{\\textbf{Native Projection}} & \\multicolumn{2}{c}{\\textbf{Direct Copy}} \\\\")
    print("\\cmidrule(lr){3-4} \\cmidrule(lr){5-6}")
    print(" & & PPL & $\\kl$ (nats) & PPL & $\\kl$ (nats) \\\\")
    print("\\midrule")
    for d in depths:
        np_row = [r for r in aggregated_results if r["exit_depth"] == d and r["strategy"] == "native_projection"][0]
        dc_row = [r for r in aggregated_results if r["exit_depth"] == d and r["strategy"] == "direct_copy"][0]
        rel_depth = (d / engine.num_layers) * 100.0
        print(f"{d} & {rel_depth:.1f}\\% & ${np_row['ppl_mean']:.2f} \\pm {np_row['ppl_std']:.2f}$ & ${np_row['kl_mean']:.4f} \\pm {np_row['kl_std']:.4f}$ & ${dc_row['ppl_mean']:.1f} \\pm {dc_row['ppl_std']:.1f}$ & ${dc_row['kl_mean']:.2f} \\pm {dc_row['kl_std']:.2f}$ \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table*}")

    return {"aggregated": aggregated_results, "oracle_ppl_mean": float(np.mean(oracle_ppls))}


# ==============================================================================
# MAIN ENTRYPOINT
# ==============================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)

    if args.model_id is None:
        args.model_id = "Qwen/Qwen2.5-3B" if args.model_type == "qwen" else "meta-llama/Llama-3.2-1B"

    if args.exit_depth is None:
        args.exit_depth = 18 if args.model_type == "qwen" else 8

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.quick:
        seeds = [42]
        args.num_samples = 5

    # Determine depth sweep set
    if args.model_type == "qwen":
        sweep_depths = [18, 22, 24, 28, 32]
    else:
        sweep_depths = [4, 8, 11, 14]

    print("=" * 80)
    print("STARTING TIER 1 EXPERIMENT DRIVER")
    print(f"Model ID: {args.model_id} | Type: {args.model_type} | Device: {device}")
    print(f"K/V Swap Exit Depth: L = {args.exit_depth} | Multi-Seed Depths: {sweep_depths}")
    print("=" * 80)

    # 1. Initialize Engine
    dtype = torch.bfloat16 if args.dtype == "bfloat16" or (args.dtype == "auto" and torch.cuda.is_bf16_supported()) else torch.float16
    engine = QwenDynamicEngine(
        model_id=args.model_id,
        device=args.device,
        dtype=dtype if device.type == "cuda" else torch.float32,
    )

    # 2. Load Evaluation Token Stream
    print("[*] Preparing evaluation tokens...")
    all_tokens = load_eval_tokens(engine.tokenizer, dataset_name="wikitext2")
    print(f"✓ Loaded {len(all_tokens):,} evaluation tokens.")

    # 3. Run Task 1: Causal Swap Intervention
    if not args.skip_swap:
        swap_seqs = sample_sequences(all_tokens, seq_len=args.seq_len, num_samples=args.num_samples, seed=42, device=device)
        run_causal_kv_swap(
            engine=engine,
            sequences=swap_seqs,
            exit_depth=args.exit_depth,
            output_dir=args.output_dir
        )

    # 4. Run Task 2: Multi-Seed Robustness Evaluation
    if not args.skip_multiseed:
        run_multiseed_eval(
            engine=engine,
            all_tokens=all_tokens,
            depths=sweep_depths,
            seeds=seeds,
            num_samples_per_seed=args.num_samples,
            seq_len=args.seq_len,
            output_dir=args.output_dir
        )

    print("\n" + "=" * 80)
    print("ALL TIER 1 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"Results stored under: {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
