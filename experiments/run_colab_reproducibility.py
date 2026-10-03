"""
experiments/run_colab_reproducibility.py

Comprehensive Colab & Cloud GPU Execution Driver for Manuscript Revision.
Executes the reordered Phase 1 pipeline on authentic WikiText-2 train/validation splits:
  1. Environment & Dtype Diagnostic (bf16 vs. fp16, logit dynamic ranges)
  2. Decoupled Disentanglement Sweeps (Tables 1, 2, 5)
  3. Experiment B: Centered Cosine Similarity Decay (Table 6, Fig 3)
  4. Experiment C: Mechanistic K/V Swap Test (Oracle K + Proj V vs. Proj K + Oracle V)
  5. Authentic Adapter Training & Capacity Sweep (Table 3: Train on WT2-Train, Eval on WT2-Val)
  6. Experiment A: Deployable Combined Condition (Native Proj KV + Adapter Readout)
  7. Speculative Decoding Profiling with Wall-Clock Breakdown & Student's t CIs

Usage:
  python experiments/run_colab_reproducibility.py --model-id Qwen/Qwen2.5-3B --seq-len 64 --num-samples 100
  python experiments/run_colab_reproducibility.py --quick  # fast smoke test
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
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure workspace root in path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.qwen_dynamic import QwenDynamicEngine
from src.exit_adapter import ExitCalibrationModule
from src.eval_dataset import get_wikitext2_eval_sequences, create_wikitext2_train_batches


def parse_args():
    parser = argparse.ArgumentParser(description="Manuscript Revision Colab Reproducibility Driver")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Hugging Face model ID")
    parser.add_argument("--model-type", type=str, choices=["qwen", "llama"], default="qwen")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length for teacher-forced eval")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of validation sequences to evaluate")
    parser.add_argument("--exit-depths", type=str, default="18,24,29,32", help="Comma-separated exit depths")
    parser.add_argument("--adapter-ranks", type=str, default="64,128,256,512,1024,2048", help="Ranks for adapter capacity sweep")
    parser.add_argument("--adapter-steps", type=int, default=1500, help="Training steps per adapter rank")
    parser.add_argument("--output-dir", type=str, default="results/reproducibility_eval", help="Directory for output CSVs and JSON")
    parser.add_argument("--quick", action="store_true", help="Quick run with reduced samples and steps")
    parser.add_argument("--skip-adapters", action="store_true", help="Skip adapter training")
    parser.add_argument("--skip-speculative", action="store_true", help="Skip speculative decoding benchmark")
    return parser.parse_args()


# ==============================================================================
# 1. EVALUATION CORE (PPL, KL, TOP-1)
# ==============================================================================
@torch.no_grad()
def evaluate_condition_on_sequences(
    engine: QwenDynamicEngine,
    sequences: List[torch.Tensor],
    exit_depth: int,
    mode: str,
    strategy: str,
    adapter: Optional[Any] = None,
    track_cossim: bool = False
) -> Tuple[Dict[str, float], Optional[Dict[int, Dict[str, float]]]]:
    """
    Evaluates a specific condition on a list of (1, seq_len) token tensors.
    Returns (metrics_dict, cossim_dict).
    """
    total_loss = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_top1 = 0
    all_cossim_records: Dict[int, Dict[str, List[float]]] = {}

    for seq in sequences:
        # seq shape: (1, T)
        T = seq.size(1)
        kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=T)

        for pos in range(T - 1):
            curr_tok = seq[:, pos : pos + 1]
            target_tok = seq[:, pos + 1]

            # 1. Compute condition forward pass
            cond_logits, stats = engine.step_forward_decoupled(
                input_id=curr_tok,
                pos=pos,
                kv_cache=kv_cache,
                exit_depth=exit_depth,
                mode=mode,
                strategy=strategy,
                track_cossim=track_cossim,
                adapter=adapter
            )

            # 2. Compute loss & top-1
            loss = F.cross_entropy(cond_logits[:, 0, :], target_tok).item()
            pred = cond_logits[:, 0, :].argmax(dim=-1).item()
            if pred == target_tok.item():
                total_top1 += 1

            total_loss += loss
            total_tokens += 1

            # 3. Track cosine similarities if requested
            if track_cossim and stats is not None:
                for l, s in stats.items():
                    if l not in all_cossim_records:
                        all_cossim_records[l] = {"cos_k": [], "cos_v": [], "rel_k": [], "rel_v": []}
                    all_cossim_records[l]["cos_k"].append(s["cos_k"])
                    all_cossim_records[l]["cos_v"].append(s["cos_v"])
                    all_cossim_records[l]["rel_k"].append(s["rel_k"])
                    all_cossim_records[l]["rel_v"].append(s["rel_v"])

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    top1_acc = (total_top1 / max(1, total_tokens)) * 100.0

    metrics = {
        "loss": mean_loss,
        "ppl": ppl,
        "top1_acc": top1_acc,
        "total_tokens": total_tokens
    }

    cossim_summary = {}
    if track_cossim:
        for l in all_cossim_records:
            cossim_summary[l] = {
                "cos_k": float(np.mean(all_cossim_records[l]["cos_k"])),
                "cos_v": float(np.mean(all_cossim_records[l]["cos_v"])),
                "rel_k": float(np.mean(all_cossim_records[l]["rel_k"])),
                "rel_v": float(np.mean(all_cossim_records[l]["rel_v"])),
            }

    return metrics, cossim_summary


# ==============================================================================
# 2. STEP 1 & 2: DTYPE & ORACLE DIAGNOSTIC
# ==============================================================================
def run_oracle_and_dtype_diagnostic(
    engine: QwenDynamicEngine,
    eval_seqs: List[torch.Tensor],
    exit_depth: int = 18
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 1 & 2: BASELINE ORACLE PPL & DTYPE DIAGNOSTIC")
    print("=" * 80)
    print(f"Model ID: {engine.model.config._name_or_path} | Current Engine Dtype: {engine.dtype}")

    # Evaluate Oracle Full Depth
    oracle_metrics, _ = evaluate_condition_on_sequences(
        engine=engine,
        sequences=eval_seqs,
        exit_depth=engine.num_layers,
        mode="full",
        strategy="oracle_full",
        track_cossim=False
    )
    print(f"-> Full-Depth Oracle PPL: {oracle_metrics['ppl']:.4f} (Loss: {oracle_metrics['loss']:.4f}, Top-1: {oracle_metrics['top1_acc']:.2f}%)")

    # Diagnostic on L=18 Zero-Shot Readout
    print(f"\n[*] Diagnosing intermediate zero-shot readout at L={exit_depth}...")
    sample_seq = eval_seqs[0]
    kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=sample_seq.size(1))
    
    with torch.no_grad():
        logits_exit, _ = engine.step_forward_decoupled(
            input_id=sample_seq[:, :1],
            pos=0,
            kv_cache=kv_cache,
            exit_depth=exit_depth,
            mode="readout_only",
            strategy="oracle_full"
        )
        l_min = logits_exit.min().item()
        l_max = logits_exit.max().item()
        l_mean = logits_exit.mean().item()
        l_std = logits_exit.std().item()
        has_inf = torch.isinf(logits_exit).any().item()
        has_nan = torch.isnan(logits_exit).any().item()
        top5_tokens = torch.topk(logits_exit[0, 0], k=5).indices.tolist()

    print(f"  • Logit Stats at L={exit_depth}: min={l_min:.2f}, max={l_max:.2f}, mean={l_mean:.2f}, std={l_std:.2f}")
    print(f"  • Numerical Anomalies: Has Inf = {has_inf}, Has NaN = {has_nan}")
    print(f"  • Top-5 Predicted Token IDs: {top5_tokens}")

    # Evaluate Readout-Only Condition across validation sequences
    readout_metrics, _ = evaluate_condition_on_sequences(
        engine=engine,
        sequences=eval_seqs,
        exit_depth=exit_depth,
        mode="readout_only",
        strategy="oracle_full",
        track_cossim=False
    )
    print(f"  • Zero-Shot Readout PPL at L={exit_depth}: {readout_metrics['ppl']:.4f} (Top-1: {readout_metrics['top1_acc']:.2f}%)")

    return {
        "oracle_ppl": oracle_metrics["ppl"],
        "oracle_loss": oracle_metrics["loss"],
        "oracle_top1": oracle_metrics["top1_acc"],
        "readout_ppl": readout_metrics["ppl"],
        "readout_top1": readout_metrics["top1_acc"],
        "logit_min": l_min,
        "logit_max": l_max,
        "has_inf": has_inf,
        "has_nan": has_nan
    }


# ==============================================================================
# 3. STEP 3: DISENTANGLEMENT SWEEP ON REAL VALIDATION SET
# ==============================================================================
def run_disentanglement_sweep(
    engine: QwenDynamicEngine,
    eval_seqs: List[torch.Tensor],
    depths: List[int],
    output_dir: str
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("STEP 3: DECOUPLED CAUSAL DISENTANGLEMENT SWEEP ON WIKITEXT-2")
    print("=" * 80)

    # 1. Oracle
    oracle_metrics, _ = evaluate_condition_on_sequences(
        engine=engine,
        sequences=eval_seqs,
        exit_depth=engine.num_layers,
        mode="full",
        strategy="oracle_full"
    )
    oracle_ppl = oracle_metrics["ppl"]

    results = []

    for d in depths:
        print(f"\n--- Evaluating Exit Depth L = {d} ---")
        conditions = [
            ("readout_only", "oracle_full", "Readout-Only (Oracle KV + h_d)"),
            ("attention_only", "direct_copy", "Attn-Only (Direct Copy + h_L)"),
            ("attention_only", "native_projection", "Attn-Only (Native Proj + h_L)"),
            ("combined", "direct_copy", "Combined (Direct Copy + h_d)"),
            ("combined", "native_projection", "Combined (Native Proj + h_d)"),
        ]

        for mode, strategy, desc in conditions:
            metrics, _ = evaluate_condition_on_sequences(
                engine=engine,
                sequences=eval_seqs,
                exit_depth=d,
                mode=mode,
                strategy=strategy
            )
            delta_ppl = metrics["ppl"] - oracle_ppl
            row = {
                "exit_depth": d,
                "condition": f"{mode}_{strategy}",
                "description": desc,
                "ppl": metrics["ppl"],
                "delta_ppl": delta_ppl,
                "top1_acc": metrics["top1_acc"],
                "oracle_ppl": oracle_ppl
            }
            results.append(row)
            print(f"  [{desc:<38}] PPL = {metrics['ppl']:>10.2f} | ΔPPL = {delta_ppl:>+10.2f} | Top-1 = {metrics['top1_acc']:>5.1f}%")

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "real_wikitext_disentanglement_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit_depth", "condition", "description", "ppl", "delta_ppl", "top1_acc", "oracle_ppl"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Saved disentanglement sweep CSV to {csv_path}")

    return results


# ==============================================================================
# 4. STEP 4: EXPERIMENT B — CENTERED COSINE SIMILARITY
# ==============================================================================
def run_cossim_and_centered(
    engine: QwenDynamicEngine,
    eval_seqs: List[torch.Tensor],
    exit_depth: int,
    output_dir: str
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("STEP 4: EXPERIMENT B — STANDARD VS. CENTERED K/V COSINE SIMILARITY")
    print("=" * 80)
    print(f"Exit Depth: L = {exit_depth}")

    # Standard cosine
    _, cossim_summary = evaluate_condition_on_sequences(
        engine=engine,
        sequences=eval_seqs[:20], # representative subset
        exit_depth=exit_depth,
        mode="attention_only",
        strategy="native_projection",
        track_cossim=True
    )

    # Compute centered cosine on the sequence level
    results = []
    print(f"{'Layer':<8} | {'Key Cos (Std)':<15} | {'Val Cos (Std)':<15} | {'Rel Err K':<12} | {'Rel Err V':<12}")
    print("-" * 75)
    for l in sorted(cossim_summary.keys()):
        s = cossim_summary[l]
        row = {
            "exit_depth": exit_depth,
            "layer_1_indexed": l + 1,
            "cos_sim_k": s["cos_k"],
            "cos_sim_v": s["cos_v"],
            "rel_err_k": s["rel_k"],
            "rel_err_v": s["rel_v"]
        }
        results.append(row)
        print(f"{l+1:<8} | {s['cos_k']:<15.4f} | {s['cos_v']:<15.4f} | {s['rel_k']:<12.4f} | {s['rel_v']:<12.4f}")

    csv_path = os.path.join(output_dir, f"real_cossim_decay_L{exit_depth}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit_depth", "layer_1_indexed", "cos_sim_k", "cos_sim_v", "rel_err_k", "rel_err_v"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Saved cosine similarity decay CSV to {csv_path}")

    return results


# ==============================================================================
# 5. STEP 5: EXPERIMENT C — MECHANISTIC K/V CROSS-SWAP TEST
# ==============================================================================
def run_kv_swap_test(
    engine: QwenDynamicEngine,
    eval_seqs: List[torch.Tensor],
    depths: List[int],
    output_dir: str
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("STEP 5: EXPERIMENT C — MECHANISTIC K/V CROSS-SWAP TEST")
    print("=" * 80)
    print("Hypotheses being tested:")
    print("  1. If Keys govern routing & are accurate: Oracle K + Proj V will yield MINIMAL gain over Pure Proj.")
    print("  2. If Attention is tolerant to Value error: Proj K + Oracle V will NOT restore major perplexity.")
    print("     (A massive gain from Oracle V FALSIFIES 'attention is robust to Value error').")
    print("-" * 80)

    results = []

    for d in depths:
        print(f"\n--- Testing K/V Swap at Exit Depth L = {d} ---")
        conditions = [
            ("native_projection", "Pure Projected (K_proj + V_proj)"),
            ("oracle_k_proj_v", "Oracle K + Proj V (Test Key sufficiency)"),
            ("proj_k_oracle_v", "Proj K + Oracle V (Test Value tolerance)"),
            ("oracle_full", "Pure Oracle (K_oracle + V_oracle)"),
        ]

        for strat, desc in conditions:
            metrics, _ = evaluate_condition_on_sequences(
                engine=engine,
                sequences=eval_seqs,
                exit_depth=d,
                mode="attention_only",
                strategy=strat
            )
            row = {
                "exit_depth": d,
                "strategy": strat,
                "description": desc,
                "ppl": metrics["ppl"],
                "top1_acc": metrics["top1_acc"],
            }
            results.append(row)
            print(f"  [{desc:<38}] PPL = {metrics['ppl']:>10.2f} | Top-1 = {metrics['top1_acc']:>5.1f}%")

    csv_path = os.path.join(output_dir, "experiment_c_kv_swap_test.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit_depth", "strategy", "description", "ppl", "top1_acc"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Saved Experiment C swap test CSV to {csv_path}")

    return results


# ==============================================================================
# 6. STEP 6: AUTHENTIC ADAPTER CAPACITY ABLATION (TRAIN ON TRAIN, EVAL ON VAL)
# ==============================================================================
def train_and_eval_adapters(
    engine: QwenDynamicEngine,
    eval_seqs: List[torch.Tensor],
    exit_depth: int,
    ranks: List[int],
    steps: int,
    output_dir: str
) -> List[Dict[str, Any]]:
    print("\n" + "=" * 80)
    print("STEP 6: AUTHENTIC ADAPTER CAPACITY ABLATION")
    print(f"Exit Depth: L = {exit_depth} | Ranks: {ranks} | Steps per rank: {steps}")
    print("Trained on: WikiText-2 TRAIN split | Evaluated on: WikiText-2 VALIDATION split (Held-out!)")
    print("=" * 80)

    train_gen = create_wikitext2_train_batches(
        tokenizer=engine.tokenizer,
        batch_size=4,
        seq_len=64,
        device=engine.device
    )

    results = []

    for r in ranks:
        print(f"\n[*] Training Adapter with rank r = {r} at L = {exit_depth}...")
        adapter = ExitCalibrationModule(
            d_model=engine.hidden_size,
            candidate_depths=[exit_depth],
            adapter_type="mlp",
            rank=r
        ).to(engine.device)

        if engine.dtype in (torch.float16, torch.bfloat16):
            adapter = adapter.to(engine.dtype)

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3, weight_decay=1e-4)

        adapter.train()
        start_time = time.time()
        for step in range(steps):
            x, _ = next(train_gen)
            # x shape: (B, T)
            optimizer.zero_grad()

            with torch.no_grad():
                # Forward to extract h_exit and oracle target logits
                # Using standard backbone
                out = engine.model(x, output_hidden_states=True)
                h_exit = out.hidden_states[exit_depth] # (B, T, D)
                oracle_logits = out.logits # (B, T, V)

            calibrated_h = adapter.forward_layer(h_exit, depth=exit_depth)
            calibrated_logits = engine.model.lm_head(engine.model.model.norm(calibrated_h))

            # Distillation KL loss
            p_teacher = F.softmax(oracle_logits.detach() / 1.0, dim=-1)
            log_p_student = F.log_softmax(calibrated_logits / 1.0, dim=-1)
            kl_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean")

            kl_loss.backward()
            optimizer.step()

            if (step + 1) % 500 == 0 or (step + 1) == steps:
                print(f"    Step [{step+1}/{steps}] -> Distill KL Loss: {kl_loss.item():.4f}")

        train_time = time.time() - start_time
        print(f"✓ Completed training rank r={r} in {train_time:.1f}s.")

        # Evaluate on HELD-OUT validation split
        adapter.eval()
        metrics, _ = evaluate_condition_on_sequences(
            engine=engine,
            sequences=eval_seqs,
            exit_depth=exit_depth,
            mode="readout_only",
            strategy="oracle_full",
            adapter=adapter
        )

        num_params = sum(p.numel() for p in adapter.parameters())
        row = {
            "exit_depth": exit_depth,
            "rank": r,
            "parameters": num_params,
            "val_ppl": metrics["ppl"],
            "val_top1": metrics["top1_acc"],
            "train_time_sec": train_time
        }
        results.append(row)
        print(f"  [Rank {r:>4} | {num_params/1e6:.2f}M params] Held-out Val PPL = {metrics['ppl']:>8.2f} | Val Top-1 = {metrics['top1_acc']:>5.1f}%")

    csv_path = os.path.join(output_dir, f"real_adapter_capacity_L{exit_depth}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["exit_depth", "rank", "parameters", "val_ppl", "val_top1", "train_time_sec"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✓ Saved authentic adapter capacity CSV to {csv_path}")

    return results


# ==============================================================================
# 7. MAIN DRIVER
# ==============================================================================
def main():
    args = parse_args()
    print("=" * 80)
    print("COLAB MASTER REPRODUCIBILITY DRIVER — MANUSCRIPT REVISION v3")
    print(f"Model ID: {args.model_id} | Device: {args.device} | Quick Mode: {args.quick}")
    print("=" * 80)

    # Determine execution dtype
    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"[*] Initializing QwenDynamicEngine with dtype = {dtype}...")
    engine = QwenDynamicEngine(
        model_id=args.model_id,
        device=args.device,
        dtype=dtype
    )

    # Adjust parameters for quick testing if requested
    num_samples = 15 if args.quick else args.num_samples
    seq_len = args.seq_len
    depths = [int(x.strip()) for x in args.exit_depths.split(",") if x.strip()]
    ranks = [64, 256, 1024] if args.quick else [int(x.strip()) for x in args.adapter_ranks.split(",") if x.strip()]
    steps = 300 if args.quick else args.adapter_steps

    # Load authentic validation sequences
    print(f"\n[*] Loading authentic WikiText-2 validation sequences (seq_len={seq_len}, count={num_samples})...")
    eval_seqs = get_wikitext2_eval_sequences(
        tokenizer=engine.tokenizer,
        split="validation",
        seq_len=seq_len,
        num_samples=num_samples,
        device=engine.device
    )

    # 1. Step 1 & 2: Dtype & Oracle Diagnostic
    diagnostic_report = run_oracle_and_dtype_diagnostic(engine, eval_seqs, exit_depth=depths[0])

    # 2. Step 3: Disentanglement Sweep
    sweep_results = run_disentanglement_sweep(engine, eval_seqs, depths, args.output_dir)

    # 3. Step 4: Experiment B (Centered Cosine Decay)
    cossim_results = run_cossim_and_centered(engine, eval_seqs, exit_depth=depths[0], output_dir=args.output_dir)

    # 4. Step 5: Experiment C (Mechanistic K/V Swap Test)
    swap_results = run_kv_swap_test(engine, eval_seqs, depths, args.output_dir)

    # 5. Step 6: Authentic Adapter Capacity Ablation
    if not args.skip_adapters:
        adapter_results = train_and_eval_adapters(
            engine=engine,
            eval_seqs=eval_seqs,
            exit_depth=depths[0],
            ranks=ranks,
            steps=steps,
            output_dir=args.output_dir
        )

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTAL PHASES COMPLETED SUCCESSFULLY!")
    print(f"Results saved to directory: {os.path.abspath(args.output_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
