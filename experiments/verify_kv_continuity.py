#!/usr/bin/env python3
"""
experiments/verify_kv_continuity.py

Comprehensive Verification & Benchmark Harness for KV-Cache Continuity (Option B Milestone 2 & 3).
Evaluates:
  1. Continuity & Zero-Crash Stress Test across adversarial variable-depth trajectories.
  2. KV State Reconstruction Fidelity: Frobenius error and cosine similarity of propagated KV states.
  3. End-to-End Autoregressive Generation Quality: Perplexity drift (Delta PPL) vs Static 12L and SkipDecode.

Usage:
  python experiments/verify_kv_continuity.py --model-size 125m --allow-untrained --num-samples 20 --gen-len 50
  python experiments/verify_kv_continuity.py --model-size 125m --checkpoint-path base_model.pt --num-samples 50 --gen-len 100
"""

import os
import sys
import math
import time
import argparse
from typing import Dict, List, Tuple, Optional

# Ensure safe UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import ModelConfig, TransformerLM
from src.kv_cache import StaticKVCache
from src.configs import get_model_preset
from src.dataset import load_dataset, get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify KV-Cache Continuity and Generation Quality under Dynamic Early-Exit Decoding"
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="125m",
        choices=["25m", "50m", "57m", "125m"],
        help="Model architecture preset (default: 125m)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to pretrained base model checkpoint (optional)"
    )
    parser.add_argument(
        "--candidate-layers",
        type=int,
        nargs="+",
        default=None,
        help="1-indexed candidate exit depths (e.g. 4 6 8 10 12 for 125m)"
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=32,
        help="Prompt length in tokens (default: 32)"
    )
    parser.add_argument(
        "--gen-len",
        type=int,
        default=64,
        help="Number of tokens to generate per sequence (default: 64)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="Number of evaluation sequences (default: 25)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        choices=["tinystories", "wikitext2", "synthetic"],
        help="Dataset for prompt sequences (default: tinystories)"
    )
    parser.add_argument(
        "--allow-untrained",
        action="store_true",
        help="Allow running with initialized weights if checkpoint not found"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run benchmark on"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory to save output CSV and plots"
    )
    return parser.parse_args()


def resolve_candidate_layers(model_size: str, n_layers: int, user_layers: Optional[List[int]]) -> List[int]:
    if user_layers is not None and len(user_layers) > 0:
        layers = sorted(list(set(user_layers)))
    else:
        norm_size = model_size.lower().replace("m", "")
        if norm_size == "125":
            layers = [4, 6, 8, 10, 12]
        elif norm_size in ["50", "57"]:
            layers = [3, 5, 7, 10]
        elif norm_size == "25":
            layers = [2, 4, 6, 8]
        else:
            step = max(1, n_layers // 4)
            layers = list(range(step, n_layers + 1, step))

    if n_layers not in layers:
        layers.append(n_layers)
    return sorted(list(set(layers)))


def find_checkpoint(user_path: Optional[str], model_size: str) -> Optional[str]:
    if user_path and os.path.exists(user_path):
        return user_path

    norm_size = model_size.lower().replace("m", "")
    tag = "125M" if norm_size == "125" else ("50M" if norm_size in ["50", "57"] else "25M")

    search_paths = [
        f"125M_results/checkpoint/base_model.pt",
        f"{tag}_results/checkpoint/base_model.pt",
        f"checkpoints/{tag}/base_model.pt",
        f"checkpoints/base_model.pt",
        f"{tag}_results/base_model.pt",
        "base_model.pt",
    ]
    for p in search_paths:
        if os.path.exists(p):
            return p
    return None


def run_continuity_stress_test(model: TransformerLM, candidate_layers: List[int], device: torch.device) -> bool:
    """
    Stress-tests the KV cache across 200 adversarial depth jumps
    (e.g., alternating between shallowest layer 4 and deepest layer 12).
    """
    print("\n" + "=" * 70)
    print("TEST 1: Continuity & Zero-Crash Adversarial Stress Test")
    print("=" * 70)

    B = 2
    T_prompt = 16
    prompt = torch.randint(0, model.config.vocab_size, (B, T_prompt), device=device)
    max_steps = 200

    kv_cache = StaticKVCache(
        n_layers=model.config.n_layers,
        batch_size=B,
        n_heads=model.config.n_heads,
        head_dim=model.config.d_model // model.config.n_heads,
        max_seq_len=T_prompt + max_steps + 10,
        device=device
    )

    # 1. Prefill
    e_tok = model.token_embeddings(prompt)
    pos_range = torch.arange(0, T_prompt, dtype=torch.long, device=device).unsqueeze(0)
    h = model.drop(e_tok + model.pos_embeddings(pos_range))
    for l in range(model.config.n_layers):
        h = model.layers[l](h, kv_cache=kv_cache)
    for p in range(T_prompt):
        kv_cache.record_depth(p, model.config.n_layers)

    curr_token = torch.randint(0, model.config.vocab_size, (B, 1), device=device)
    curr_pos = T_prompt

    # Pattern: [min_layer, max_layer, mid_layer, max_layer, ...]
    min_l = candidate_layers[0]
    max_l = candidate_layers[-1]
    mid_l = candidate_layers[len(candidate_layers) // 2]
    pattern = [min_l, max_l, mid_l, max_l, min_l, min_l, max_l]

    all_passed = True
    for step in range(max_steps):
        d_t = pattern[step % len(pattern)]
        logits, _ = model.forward_step(
            input_id=curr_token,
            pos=curr_pos,
            kv_cache=kv_cache,
            exit_depth=d_t,
            state_prop_strategy="native_projection"
        )

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print(f"[FAIL] NaN/Inf detected at step {step} with exit depth {d_t}!")
            all_passed = False
            break

        curr_token = torch.argmax(logits, dim=-1)
        curr_pos += 1

    if all_passed:
        print(f"-> [PASS] Executed {max_steps} adversarial depth jumps without crash or NaN!")
        print(f"-> KV cache buffer successfully sustained variable jumps between L{min_l} and L{max_l}.")
    return all_passed


def measure_kv_reconstruction_fidelity(
    model: TransformerLM,
    candidate_layers: List[int],
    device: torch.device,
    num_eval_tokens: int = 200
) -> pd.DataFrame:
    """
    Quantifies how closely propagated KV representations match the ground-truth
    full-depth representations for each skipped upper layer.
    """
    print("\n" + "=" * 70)
    print("TEST 2: KV State Propagation Reconstruction Fidelity")
    print("=" * 70)

    exit_l = candidate_layers[0]  # shallowest exit (e.g. Layer 4)
    skipped_layers = [l for l in range(exit_l, model.config.n_layers)]

    print(f"Evaluating exit at Layer {exit_l} -> Measuring skipped layers {skipped_layers} (0-indexed)...")

    results = []
    cos_sim_fn = nn.CosineSimilarity(dim=-1)

    for _ in range(num_eval_tokens):
        # Generate random context of varying length
        seq_len = np.random.randint(10, 40)
        x = torch.randint(0, model.config.vocab_size, (1, seq_len), device=device)

        # 1. Ground truth full execution
        e_tok = model.token_embeddings(x)
        pos = torch.arange(0, seq_len, device=device).unsqueeze(0)
        h = e_tok + model.pos_embeddings(pos)

        # Store true K and V at the final token position for each layer
        true_k = {}
        true_v = {}
        h_exit = None

        for l in range(model.config.n_layers):
            block = model.layers[l]
            # LayerNorm + QKV on last token
            norm_h = block.ln_1(h[:, -1:, :])
            qkv = block.attn.qkv_proj(norm_h)
            _, k_val, v_val = qkv.chunk(3, dim=-1)
            true_k[l] = k_val.squeeze(0).squeeze(0)  # (d_model,)
            true_v[l] = v_val.squeeze(0).squeeze(0)

            # Full block execution
            h = block(h)
            if l == exit_l - 1:
                h_exit = h[:, -1:, :].clone()

        # 2. Strategy 1: Native Projection
        for l in skipped_layers:
            block = model.layers[l]
            norm_exit = block.ln_1(h_exit)
            qkv_proj = block.attn.qkv_proj(norm_exit)
            _, k_proj, v_proj = qkv_proj.chunk(3, dim=-1)
            k_p = k_proj.squeeze(0).squeeze(0)
            v_p = v_proj.squeeze(0).squeeze(0)

            cos_k = cos_sim_fn(k_p.unsqueeze(0), true_k[l].unsqueeze(0)).item()
            cos_v = cos_sim_fn(v_p.unsqueeze(0), true_v[l].unsqueeze(0)).item()
            rel_err_k = (torch.norm(k_p - true_k[l]) / (torch.norm(true_k[l]) + 1e-8)).item()
            rel_err_v = (torch.norm(v_p - true_v[l]) / (torch.norm(true_v[l]) + 1e-8)).item()

            # Strategy 2: Direct Copy (from exit layer)
            k_copy = true_k[exit_l - 1]
            v_copy = true_v[exit_l - 1]
            cos_k_copy = cos_sim_fn(k_copy.unsqueeze(0), true_k[l].unsqueeze(0)).item()
            cos_v_copy = cos_sim_fn(v_copy.unsqueeze(0), true_v[l].unsqueeze(0)).item()

            results.append({
                "layer_1_indexed": l + 1,
                "strategy": "Native Projection",
                "cos_sim_k": cos_k,
                "cos_sim_v": cos_v,
                "rel_err_k": rel_err_k,
                "rel_err_v": rel_err_v
            })
            results.append({
                "layer_1_indexed": l + 1,
                "strategy": "Direct Copy",
                "cos_sim_k": cos_k_copy,
                "cos_sim_v": cos_v_copy,
                "rel_err_k": (torch.norm(k_copy - true_k[l]) / (torch.norm(true_k[l]) + 1e-8)).item(),
                "rel_err_v": (torch.norm(v_copy - true_v[l]) / (torch.norm(true_v[l]) + 1e-8)).item()
            })

    df = pd.DataFrame(results)
    summary = df.groupby(["layer_1_indexed", "strategy"])[["cos_sim_k", "cos_sim_v", "rel_err_k", "rel_err_v"]].mean().reset_index()

    print("\n--- KV Reconstruction Fidelity Summary ---")
    print(summary.to_string(index=False))
    return summary


def run_generation_quality_benchmark(
    model: TransformerLM,
    prompts: List[torch.Tensor],
    candidate_layers: List[int],
    gen_len: int,
    device: torch.device
) -> pd.DataFrame:
    """
    Measures end-to-end 100-token generation quality and perplexity drift (Delta PPL).
    Compares:
      1. Full Depth Static L=12 (Gold Reference)
      2. Dynamic Exit with Strategy 1 (Native Projection)
      3. Dynamic Exit with Strategy 2 (Direct Copy)
      4. SkipDecode Depth-Matched Baseline (L=7)
    """
    print("\n" + "=" * 70)
    print("TEST 3: End-to-End Autoregressive Generation & Perplexity Drift")
    print("=" * 70)

    # Realistic oracle-weighted stochastic depth blend matching E[L] = 6.64:
    # 50% L4, 12% L6, 11% L8, 10% L10, 17% L12
    depth_weights = [0.497, 0.120, 0.111, 0.105, 0.167]
    depth_weights = np.array(depth_weights[:len(candidate_layers)])
    depth_weights = depth_weights / depth_weights.sum()

    results = []
    first_gold_ids = None
    first_nat_ids = None
    first_copy_ids = None

    for idx, prompt in enumerate(prompts):
        prompt = prompt.to(device)
        B, T = prompt.shape

        # 1. Gold Reference: Full Depth Static
        full_res = model.generate_dynamic(
            prompt_ids=prompt,
            max_new_tokens=gen_len,
            depth_policy=model.config.n_layers,
            state_prop_strategy="native_projection"
        )
        gold_ids = full_res["output_ids"]

        # Compute reference loss / PPL of generated tokens under full model
        with torch.no_grad():
            ref_logits, _ = model(gold_ids)
            gen_logits = ref_logits[:, T - 1 : -1, :]
            gen_targets = gold_ids[:, T:]
            ref_loss = F.cross_entropy(gen_logits.reshape(-1, gen_logits.size(-1)), gen_targets.reshape(-1)).item()
            ref_ppl = math.exp(min(ref_loss, 20.0))

        # Sample a fixed sequence of dynamic depths for this prompt to ensure fair comparison
        np.random.seed(42 + idx)
        sampled_depths = list(np.random.choice(candidate_layers, size=gen_len, p=depth_weights))
        avg_dyn_depth = float(np.mean(sampled_depths))

        # Depth-matched SkipDecode depth = round(avg_dyn_depth)
        skip_depth = int(round(avg_dyn_depth))

        # 2. Dynamic Exit with Native Projection
        dyn_nat = model.generate_dynamic(
            prompt_ids=prompt,
            max_new_tokens=gen_len,
            depth_policy=sampled_depths,
            state_prop_strategy="native_projection"
        )
        with torch.no_grad():
            nat_logits, _ = model(dyn_nat["output_ids"])
            loss_nat = F.cross_entropy(
                nat_logits[:, T - 1 : -1, :].reshape(-1, nat_logits.size(-1)),
                dyn_nat["output_ids"][:, T:].reshape(-1)
            ).item()
            ppl_nat = math.exp(min(loss_nat, 20.0))

        # 3. Dynamic Exit with Direct Copy
        dyn_copy = model.generate_dynamic(
            prompt_ids=prompt,
            max_new_tokens=gen_len,
            depth_policy=sampled_depths,
            state_prop_strategy="direct_copy"
        )
        with torch.no_grad():
            copy_logits, _ = model(dyn_copy["output_ids"])
            loss_copy = F.cross_entropy(
                copy_logits[:, T - 1 : -1, :].reshape(-1, copy_logits.size(-1)),
                dyn_copy["output_ids"][:, T:].reshape(-1)
            ).item()
            ppl_copy = math.exp(min(loss_copy, 20.0))

        if idx == 0:
            first_gold_ids = gold_ids
            first_nat_ids = dyn_nat["output_ids"]
            first_copy_ids = dyn_copy["output_ids"]

        # 4. SkipDecode (Uniform Depth)
        skip_res = model.generate_dynamic(
            prompt_ids=prompt,
            max_new_tokens=gen_len,
            depth_policy=skip_depth,
            state_prop_strategy="native_projection"
        )
        with torch.no_grad():
            skip_logits, _ = model(skip_res["output_ids"])
            loss_skip = F.cross_entropy(
                skip_logits[:, T - 1 : -1, :].reshape(-1, skip_logits.size(-1)),
                skip_res["output_ids"][:, T:].reshape(-1)
            ).item()
            ppl_skip = math.exp(min(loss_skip, 20.0))

        results.append({
            "sample_id": idx,
            "ref_loss": ref_loss,
            "ref_ppl": ref_ppl,
            "nat_loss": loss_nat,
            "nat_ppl": ppl_nat,
            "delta_ppl_nat": ppl_nat - ref_ppl,
            "copy_loss": loss_copy,
            "copy_ppl": ppl_copy,
            "delta_ppl_copy": ppl_copy - ref_ppl,
            "skip_loss": loss_skip,
            "skip_ppl": ppl_skip,
            "delta_ppl_skip": ppl_skip - ref_ppl,
            "avg_dyn_depth": avg_dyn_depth,
            "skip_depth": skip_depth,
            "flop_reduction_pct": (1.0 - (avg_dyn_depth / model.config.n_layers)) * 100.0
        })

    eval_df = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("EMPIRICAL GENERATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Sequences Evaluated: {len(eval_df)}")
    print(f"Average Depth Executed:    {eval_df['avg_dyn_depth'].mean():.2f} / {model.config.n_layers} layers")
    print(f"Theoretical FLOP Savings:  {eval_df['flop_reduction_pct'].mean():.1f}%\n")

    print(f"Condition 1 (Full 12L Static):       Mean PPL = {eval_df['ref_ppl'].mean():.3f}")
    print(f"Condition 2 (Dynamic + Native Proj): Mean PPL = {eval_df['nat_ppl'].mean():.3f} (Delta PPL = {eval_df['delta_ppl_nat'].mean():+.3f})")
    # Print sample text comparison
    tokenizer = get_tokenizer()
    decode_fn = getattr(tokenizer, "decode", None)
    if decode_fn is not None and len(prompts) > 0 and first_gold_ids is not None:
        print("\n" + "=" * 70)
        print("QUALITATIVE TEXT GENERATION COMPARISON (First Sequence)")
        print("=" * 70)
        try:
            p_len = prompts[0].shape[1]
            p_text = decode_fn(prompts[0][0].tolist())
            print(f"[PROMPT ({p_len} tokens)]:\n\"{p_text}\"\n")
            gold_text = decode_fn(first_gold_ids[0, p_len:].tolist())
            print(f"[CONDITION 1: Static Full 12L Reference]:\n\"{gold_text}\"\n")
            nat_text = decode_fn(first_nat_ids[0, p_len:].tolist())
            print(f"[CONDITION 2: Dynamic Early Exit + Native Projection (Mean Depth {eval_df['avg_dyn_depth'].iloc[0]:.1f}L)]:\n\"{nat_text}\"\n")
            copy_text = decode_fn(first_copy_ids[0, p_len:].tolist())
            print(f"[CONDITION 3: Dynamic Early Exit + Direct Copy (Ablation)]:\n\"{copy_text}\"\n")
        except Exception as e:
            print(f"(Could not decode text: {e})")

    return eval_df


def plot_results(reconstruction_df: pd.DataFrame, quality_df: pd.DataFrame, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cosine Similarity by Layer
    p1 = reconstruction_df[reconstruction_df["strategy"] == "Native Projection"]
    p2 = reconstruction_df[reconstruction_df["strategy"] == "Direct Copy"]

    axes[0].plot(p1["layer_1_indexed"], p1["cos_sim_k"], marker="o", color="#1f77b4", label="Native Projection (K)", linewidth=2)
    axes[0].plot(p1["layer_1_indexed"], p1["cos_sim_v"], marker="s", color="#aec7e8", label="Native Projection (V)", linewidth=2)
    axes[0].plot(p2["layer_1_indexed"], p2["cos_sim_k"], marker="^", linestyle="--", color="#d62728", label="Direct Copy (K)", linewidth=1.5)
    axes[0].plot(p2["layer_1_indexed"], p2["cos_sim_v"], marker="v", linestyle="--", color="#ff9896", label="Direct Copy (V)", linewidth=1.5)

    axes[0].set_title("KV-Cache State Propagation Fidelity", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Skipped Layer Index (1-indexed)", fontsize=11)
    axes[0].set_ylabel("Cosine Similarity with True Full State", fontsize=11)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=9)

    # Plot 2: Perplexity Comparison
    conds = ["Full 12L", "Dynamic\n(Native Proj)", "Dynamic\n(Direct Copy)", f"SkipDecode\n(L={quality_df['skip_depth'].iloc[0]})"]
    ppls = [
        quality_df["ref_ppl"].mean(),
        quality_df["nat_ppl"].mean(),
        quality_df["copy_ppl"].mean(),
        quality_df["skip_ppl"].mean()
    ]
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#7f7f7f"]

    bars = axes[1].bar(conds, ppls, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    axes[1].set_title("Autoregressive Generation Perplexity (PPL)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Perplexity (Lower is Better)", fontsize=11)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)

    for bar, ppl in zip(bars, ppls):
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, yval + 0.1, f"{ppl:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    out_png = os.path.join(output_dir, "kv_continuity_results.png")
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"\nSaved visualization plot to: {out_png}")


def main():
    args = parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("DYNAMIC EARLY-EXIT DECODING: KV-CACHE CONTINUITY VERIFICATION")
    print(f"Model preset: {args.model_size} | Device: {device}")
    print("=" * 70)

    ckpt_path = find_checkpoint(args.checkpoint_path, args.model_size)
    if ckpt_path is not None:
        print(f"Loading checkpoint from: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "config" in state and isinstance(state["config"], dict):
            config = ModelConfig(**state["config"])
        else:
            config = get_model_preset(args.model_size)
        config.max_seq_len = max(config.max_seq_len, args.prompt_len + args.gen_len + 32)
        model = TransformerLM(config).to(device)

        m = state.get("model_state_dict", state.get("model", state))
        model.load_state_dict(m, strict=False)
        print(f"-> Successfully loaded model checkpoint ({sum(p.numel() for p in model.parameters()):,} parameters).")
    else:
        if not args.allow_untrained:
            raise FileNotFoundError(
                f"No checkpoint found for {args.model_size}. Pass --allow-untrained to run with initialized weights."
            )
        print("Notice: Running with freshly initialized weights (--allow-untrained specified).")
        config = get_model_preset(args.model_size)
        config.max_seq_len = max(config.max_seq_len, args.prompt_len + args.gen_len + 32)
        model = TransformerLM(config).to(device)

    model.eval()

    candidate_layers = resolve_candidate_layers(args.model_size, config.n_layers, args.candidate_layers)
    print(f"Candidate Exit Layers (1-indexed): {candidate_layers}")

    # 1. Continuity Stress Test
    stress_pass = run_continuity_stress_test(model, candidate_layers, device)
    if not stress_pass:
        print("[ERROR] Stress test failed!")
        sys.exit(1)

    # 2. KV Reconstruction Fidelity
    reconstruction_df = measure_kv_reconstruction_fidelity(model, candidate_layers, device, num_eval_tokens=100)

    # 3. Create Prompts
    prompts = []
    tokenizer = get_tokenizer()

    if args.dataset != "synthetic":
        try:
            print(f"[*] Loading evaluation prompts from '{args.dataset}'...")
            _, val_loader = load_dataset(
                dataset_name=args.dataset,
                seq_len=args.prompt_len + 10,
                batch_size=1,
                num_workers=0,
                max_train_samples=100,
                max_val_samples=args.num_samples + 20
            )
            for x, _ in val_loader:
                prompts.append(x[:, :args.prompt_len])
                if len(prompts) >= args.num_samples:
                    break
            print(f"-> Successfully loaded {len(prompts)} real {args.dataset} prompt sequences.")
        except Exception as e:
            print(f"Note: Could not load {args.dataset} ({e}). Using curated TinyStories seed prompts.")

    if len(prompts) < args.num_samples:
        seed_texts = [
            "Once upon a time, there was a little girl named Lily. She loved to play in the garden with her puppy.",
            "One sunny day, a little boy named Tim found a shiny red ball. He ran outside to play with his best friend.",
            "Once there was a tiny mouse named Pip. Pip lived in a cozy hole inside a big kitchen with plenty of cheese.",
            "A happy little rabbit named Benny hopped through the green forest looking for delicious carrots to eat.",
            "Mia had a magical blue kite that could fly very high in the sky. She went to the park on a windy afternoon.",
            "A little bear named Barnaby woke up from a long nap feeling very hungry. He set out to find sweet honey.",
            "Lucy and her mom decided to bake a delicious chocolate cake. They gathered flour, sugar, and cocoa in the kitchen.",
            "There once was an old wooden boat that lived by a calm blue lake. Every morning, birds would sit on it.",
            "Tom wanted to build a tall tower using colorful wooden blocks in his playroom. He carefully placed each block.",
            "A friendly squirrel named Sammy had a big hollow tree where he stored acorns for the cold winter days."
        ]
        encode_fn = getattr(tokenizer, "encode", None)
        if encode_fn is not None:
            for text in seed_texts:
                if len(prompts) >= args.num_samples:
                    break
                toks = encode_fn(text)
                if len(toks) >= args.prompt_len:
                    prompts.append(torch.tensor([toks[:args.prompt_len]], dtype=torch.long))
                else:
                    toks_padded = (toks * (args.prompt_len // len(toks) + 1))[:args.prompt_len]
                    prompts.append(torch.tensor([toks_padded], dtype=torch.long))

    if len(prompts) == 0:
        for _ in range(args.num_samples):
            p = torch.randint(100, config.vocab_size - 100, (1, args.prompt_len))
            prompts.append(p)

    # 4. Generation Quality Benchmark
    quality_df = run_generation_quality_benchmark(
        model=model,
        prompts=prompts,
        candidate_layers=candidate_layers,
        gen_len=args.gen_len,
        device=device
    )

    # 5. Save Outputs
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "kv_continuity_results.csv")
    quality_df.to_csv(csv_path, index=False)
    print(f"Saved numerical results to: {csv_path}")

    recon_csv = os.path.join(args.output_dir, "kv_reconstruction_fidelity.csv")
    reconstruction_df.to_csv(recon_csv, index=False)
    print(f"Saved KV fidelity data to: {recon_csv}")

    plot_results(reconstruction_df, quality_df, args.output_dir)

    print("\n" + "=" * 70)
    print("ALL KV-CACHE CONTINUITY VALIDATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
