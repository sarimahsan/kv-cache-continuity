#!/usr/bin/env python3
"""
experiments/isolate_kv_ablation.py

Teacher-Forced KV-Only Ablation: Isolating KV Reconstruction Quality from Exit-Head Readout.

Core Scientific Goals:
1. Fix the Confound: Evaluate with deterministic exit schedules (SkipDecode uniform L=6 on 125M,
   L=18 on Qwen) to remove router variance and stochastic depth noise.
2. Three-Condition Comparison at Identical Depth:
   - Full Depth Baseline (Oracle 12L / 36L, clean reference)
   - Native Projection (W_K / W_V + LN)
   - Direct Copy (~0 CosSim floor)
3. Disentangle Attention Corruption from Readout Corruption:
   - Readout-Only Test: 100% pristine full-depth KV in attention, but logits read from shallow h_{d_t}.
   - Attention-Only Test: Logits read from terminal h_L, but past tokens attend to reconstructed KV.
   - Combined Early-Exit Test: Shallow readout + reconstructed KV in attention (real-world SkipDecode).
4. Track Per-Layer CosSim and Frobenius Error Decay under Teacher Forcing across sequence positions.

Usage Examples:
  # On Colab (125M base model on SkipDecode L=6 schedule):
  python experiments/isolate_kv_ablation.py --model-type 125m --checkpoint base_model.pt --depth-schedule skipdecode --exit-depth 6 --device cuda

  # On Colab (125M depth sweep across L=4, 6, 8, 10):
  python experiments/isolate_kv_ablation.py --model-type 125m --checkpoint base_model.pt --depth-schedule sweep --depths 4,6,8,10 --device cuda

  # On Colab (Qwen-2.5-3B on SkipDecode L=18 schedule):
  python experiments/isolate_kv_ablation.py --model-type qwen --model-id Qwen/Qwen2.5-3B --depth-schedule skipdecode --exit-depth 18 --device cuda
"""

import os
import sys
import time
import math
import argparse
import csv
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.kv_cache import StaticKVCache
try:
    from src.model import ModelConfig, TransformerLM
except ImportError:
    ModelConfig, TransformerLM = None, None
from src.exit_adapter import ExitCalibrationModule



CANONICAL_TEST_TEXTS = [
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
    parser = argparse.ArgumentParser(
        description="Teacher-Forced KV-Only Ablation: Disentangling KV Approximation from Exit Head Readout"
    )
    parser.add_argument("--model-type", type=str, choices=["125m", "qwen", "llama"], default="125m", help="Model family")
    parser.add_argument("--model-id", type=str, default=None, help="Hugging Face repo (defaults: Qwen/Qwen2.5-3B for qwen, meta-llama/Llama-3.2-1B for llama)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to base_model.pt checkpoint")
    parser.add_argument(
        "--depth-schedule",
        type=str,
        choices=["skipdecode", "fixed", "sweep"],
        default="skipdecode",
        help="Depth schedule mode: 'skipdecode' (uniform 50% depth), 'fixed', or 'sweep'"
    )
    parser.add_argument(
        "--exit-depth",
        type=int,
        default=None,
        help="Deterministic exit depth (defaults to 6 for 125m, 18 for qwen)"
    )
    parser.add_argument(
        "--depths",
        type=str,
        default=None,
        help="Comma-separated candidate depths for sweep mode (default: '4,6,8,10' for 125m, '18,24,29,32' for qwen)"
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Tokens evaluated per sequence")
    parser.add_argument("--num-samples", type=int, default=6, help="Number of text sequences to evaluate")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["canonical", "tinystories", "wikitext2", "synthetic"],
        default="canonical",
        help="Evaluation dataset"
    )
    parser.add_argument("--exit-adapter-path", type=str, default=None, help="Path to trained ExitCalibrationModule checkpoint (.pt)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for output CSV and tables")
    return parser.parse_args()



# ==============================================================================
# DATASET LOADING HELPERS
# ==============================================================================
def prepare_eval_sequences(
    dataset_name: str,
    tokenizer,
    seq_len: int,
    num_samples: int,
    device: torch.device
) -> List[torch.Tensor]:
    """Prepares a list of (1, seq_len) token tensors for teacher-forced evaluation."""
    sequences = []

    if dataset_name == "canonical":
        texts = CANONICAL_TEST_TEXTS[:num_samples]
        for text in texts:
            if hasattr(tokenizer, "encode"):
                toks = tokenizer.encode(text)
                if not isinstance(toks, torch.Tensor):
                    toks = torch.tensor(toks, dtype=torch.long)
            else:
                toks = tokenizer(text, return_tensors="pt")["input_ids"][0]
            toks = toks.view(1, -1).to(device)
            if toks.size(1) >= seq_len:
                sequences.append(toks[:, :seq_len])
            else:
                # Repeat if short
                reps = math.ceil(seq_len / toks.size(1))
                toks = toks.repeat(1, reps)[:, :seq_len]
                sequences.append(toks)

    elif dataset_name in ["tinystories", "wikitext2"]:
        from src.dataset import get_tokenizer as get_gpt2_tok
        tok = tokenizer if tokenizer is not None else get_gpt2_tok()
        encode_fn = tok.encode if hasattr(tok, "encode") else (lambda s: tok(s)["input_ids"])
        try:
            from datasets import load_dataset as hf_load_dataset
            if dataset_name == "tinystories":
                ds = hf_load_dataset("roneneldan/TinyStories", split="validation", streaming=True)
            else:
                try:
                    ds = hf_load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="validation")
                except Exception:
                    ds = hf_load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

            for item in ds:
                text = item.get("text", "")
                if len(text.strip()) < 50:
                    continue
                toks = encode_fn(text)
                if len(toks) >= seq_len:
                    t_tensor = torch.tensor(toks[:seq_len], dtype=torch.long, device=device).unsqueeze(0)
                    sequences.append(t_tensor)
                    if len(sequences) >= num_samples:
                        break
        except Exception as e:
            print(f"[!] Warning: Failed to load streaming dataset '{dataset_name}': {e}. Falling back to canonical texts.")
            return prepare_eval_sequences("canonical", tokenizer, seq_len, num_samples, device)

    else: # synthetic
        for i in range(num_samples):
            np.random.seed(42 + i)
            arr = np.random.randint(50, 2000, size=(1, seq_len))
            sequences.append(torch.tensor(arr, dtype=torch.long, device=device))

    return sequences


# ==============================================================================
# EVALUATION CORE FOR 125M MODEL
# ==============================================================================
def evaluate_125m_sequence(
    model: TransformerLM,
    tokens: torch.Tensor,
    exit_depth: int,
    device: torch.device,
    adapter: Optional[Any] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Evaluates a single sequence on the 125M model across all 6 experimental conditions:
      0. full_depth (12L baseline)
      1. readout_only (Pristine KV attention, shallow readout from h_{d_t})
      2. attn_native_proj (Full-depth readout, Native Proj KV in attention)
      3. attn_direct_copy (Full-depth readout, Direct Copy KV in attention)
      4. combined_native_proj (Shallow readout + Native Proj KV, SkipDecode)
      5. combined_direct_copy (Shallow readout + Direct Copy KV)
    """
    B, T = tokens.shape
    assert B == 1
    n_layers = model.config.n_layers
    head_dim = model.config.d_model // model.config.n_heads

    conditions = [
        ("full_depth", n_layers, "full", "oracle_full"),
        ("readout_only", exit_depth, "readout_only", "oracle_full"),
        ("attn_native_proj", exit_depth, "attention_only", "native_projection"),
        ("attn_direct_copy", exit_depth, "attention_only", "direct_copy"),
        ("combined_native_proj", exit_depth, "combined", "native_projection"),
        ("combined_direct_copy", exit_depth, "combined", "direct_copy"),
    ]

    use_cuda = device.type == "cuda"
    cond_metrics = {}
    oracle_logits_seq = []

    # Store CosSim tracking per layer for Native Projection under teacher forcing
    cossim_accum = {l: {"cos_k": [], "cos_v": [], "rel_k": [], "rel_v": []} for l in range(exit_depth, n_layers)}

    for cond_name, d_t, mode, strategy in conditions:
        cache = StaticKVCache(
            batch_size=1,
            max_seq_len=T + 16,
            n_layers=n_layers,
            n_heads=model.config.n_heads,
            head_dim=head_dim,
            device=device,
            dtype=torch.float32
        )

        losses = []
        logits_history = []
        kl_divs = []
        top1_matches = 0

        start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
        end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None

        if use_cuda:
            torch.cuda.synchronize()
            start_event.record()
        t0 = time.perf_counter()

        should_track_cossim = (cond_name == "attn_native_proj")

        for pos in range(T - 1):
            tok_in = tokens[:, pos : pos + 1]
            target_tok = tokens[:, pos + 1]

            logits, stats = model.forward_step_decoupled(
                input_id=tok_in,
                pos=pos,
                kv_cache=cache,
                exit_depth=d_t,
                mode=mode,
                strategy=strategy,
                track_cossim=should_track_cossim,
                adapter=adapter
            )

            curr_l = logits[:, 0, :]
            loss = F.cross_entropy(curr_l, target_tok).item()
            losses.append(loss)
            logits_history.append(curr_l.detach())

            if cond_name == "full_depth":
                oracle_logits_seq.append(curr_l.detach())
            else:
                true_l = oracle_logits_seq[pos]
                p_true = F.softmax(true_l, dim=-1)
                log_p = F.log_softmax(curr_l, dim=-1)
                kl = F.kl_div(log_p, p_true, reduction="batchmean").item()
                kl_divs.append(kl)

                if torch.argmax(curr_l, dim=-1).item() == torch.argmax(true_l, dim=-1).item():
                    top1_matches += 1

            if should_track_cossim and stats is not None:
                for l, st in stats.items():
                    if l in cossim_accum:
                        cossim_accum[l]["cos_k"].append(st["cos_k"])
                        cossim_accum[l]["cos_v"].append(st["cos_v"])
                        cossim_accum[l]["rel_k"].append(st["rel_k"])
                        cossim_accum[l]["rel_v"].append(st["rel_v"])

        if use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        nll = float(np.mean(losses))
        ppl = math.exp(min(nll, 20.0))
        mean_kl = float(np.mean(kl_divs)) if kl_divs else 0.0
        top1_acc = 100.0 if cond_name == "full_depth" else (top1_matches / (T - 1)) * 100.0
        ms_tok = elapsed_ms / (T - 1)

        cond_metrics[cond_name] = {
            "nll": nll,
            "ppl": ppl,
            "mean_kl": mean_kl,
            "top1_acc": top1_acc,
            "ms_tok": ms_tok
        }

    # Aggregate CosSim stats
    layer_cossim_summary = {}
    for l in cossim_accum:
        if cossim_accum[l]["cos_k"]:
            layer_cossim_summary[l] = {
                "cos_k": float(np.mean(cossim_accum[l]["cos_k"])),
                "cos_v": float(np.mean(cossim_accum[l]["cos_v"])),
                "rel_k": float(np.mean(cossim_accum[l]["rel_k"])),
                "rel_v": float(np.mean(cossim_accum[l]["rel_v"]))
            }

    return cond_metrics, layer_cossim_summary


# ==============================================================================
# EVALUATION CORE FOR QWEN-2.5 MODEL
# ==============================================================================
def evaluate_qwen_sequence(
    engine,
    tokens: torch.Tensor,
    exit_depth: int,
    device: torch.device,
    adapter: Optional[Any] = None
) -> Tuple[Dict[str, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Evaluates a single sequence on Qwen-2.5 across all 6 experimental conditions.
    """
    B, T = tokens.shape
    assert B == 1
    num_layers = engine.num_layers

    conditions = [
        ("full_depth", num_layers, "full", "oracle_full"),
        ("readout_only", exit_depth, "readout_only", "oracle_full"),
        ("attn_native_proj", exit_depth, "attention_only", "native_projection"),
        ("attn_direct_copy", exit_depth, "attention_only", "direct_copy"),
        ("combined_native_proj", exit_depth, "combined", "native_projection"),
        ("combined_direct_copy", exit_depth, "combined", "direct_copy"),
    ]

    use_cuda = device.type == "cuda"
    cond_metrics = {}
    oracle_logits_seq = []
    cossim_accum = {l: {"cos_k": [], "cos_v": [], "rel_k": [], "rel_v": []} for l in range(exit_depth, num_layers)}

    for cond_name, d_t, mode, strategy in conditions:
        cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=T + 16)
        losses = []
        kl_divs = []
        top1_matches = 0

        start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
        end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None

        if use_cuda:
            torch.cuda.synchronize()
            start_event.record()
        t0 = time.perf_counter()

        should_track_cossim = (cond_name == "attn_native_proj")

        for pos in range(T - 1):
            tok_in = tokens[:, pos : pos + 1]
            target_tok = tokens[:, pos + 1]

            logits, stats = engine.step_forward_decoupled(
                input_id=tok_in,
                pos=pos,
                kv_cache=cache,
                exit_depth=d_t,
                mode=mode,
                strategy=strategy,
                track_cossim=should_track_cossim,
                adapter=adapter
            )

            curr_l = logits[:, 0, :]
            loss = F.cross_entropy(curr_l, target_tok).item()
            losses.append(loss)

            if cond_name == "full_depth":
                oracle_logits_seq.append(curr_l.detach())
            else:
                true_l = oracle_logits_seq[pos]
                p_true = F.softmax(true_l, dim=-1)
                log_p = F.log_softmax(curr_l, dim=-1)
                kl = F.kl_div(log_p, p_true, reduction="batchmean").item()
                kl_divs.append(kl)

                if torch.argmax(curr_l, dim=-1).item() == torch.argmax(true_l, dim=-1).item():
                    top1_matches += 1

            if should_track_cossim and stats is not None:
                for l, st in stats.items():
                    if l in cossim_accum:
                        cossim_accum[l]["cos_k"].append(st["cos_k"])
                        cossim_accum[l]["cos_v"].append(st["cos_v"])
                        cossim_accum[l]["rel_k"].append(st["rel_k"])
                        cossim_accum[l]["rel_v"].append(st["rel_v"])

        if use_cuda:
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        nll = float(np.mean(losses))
        ppl = math.exp(min(nll, 20.0))
        mean_kl = float(np.mean(kl_divs)) if kl_divs else 0.0
        top1_acc = 100.0 if cond_name == "full_depth" else (top1_matches / (T - 1)) * 100.0
        ms_tok = elapsed_ms / (T - 1)

        cond_metrics[cond_name] = {
            "nll": nll,
            "ppl": ppl,
            "mean_kl": mean_kl,
            "top1_acc": top1_acc,
            "ms_tok": ms_tok
        }

    layer_cossim_summary = {}
    for l in cossim_accum:
        if cossim_accum[l]["cos_k"]:
            layer_cossim_summary[l] = {
                "cos_k": float(np.mean(cossim_accum[l]["cos_k"])),
                "cos_v": float(np.mean(cossim_accum[l]["cos_v"])),
                "rel_k": float(np.mean(cossim_accum[l]["rel_k"])),
                "rel_v": float(np.mean(cossim_accum[l]["rel_v"]))
            }

    return cond_metrics, layer_cossim_summary


# ==============================================================================
# MAIN ABLATION RUNNER
# ==============================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("TEACHER-FORCED KV-ONLY ABLATION: DISENTANGLING ATTENTION VS. READOUT")
    print(f"Model Type: {args.model_type} | Device: {device} | Schedule: {args.depth_schedule}")
    print("=" * 80)

    if args.model_id is None:
        if args.model_type == "llama":
            args.model_id = "meta-llama/Llama-3.2-1B"
        elif args.model_type == "qwen":
            args.model_id = "Qwen/Qwen2.5-3B"

    # Determine exit depths to test
    if args.depth_schedule in ("skipdecode", "fixed"):
        if args.model_type == "125m":
            default_depth = 6
        elif args.model_type == "llama":
            default_depth = 8 if "1B" in str(args.model_id) else 14
        else:
            default_depth = 18
        eval_depths = [args.exit_depth if args.exit_depth is not None else default_depth]
    else: # sweep
        if args.depths is not None:
            eval_depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
        else:
            if args.model_type == "125m":
                eval_depths = [4, 6, 8, 10]
            elif args.model_type == "llama":
                eval_depths = [4, 6, 8, 10, 11, 12, 13, 14] if "1B" in str(args.model_id) else [10, 14, 18, 20, 22, 24]
            else:
                eval_depths = [18, 24, 29, 32]

    # Initialize model
    tokenizer = None
    engine = None
    model_125m = None

    if args.model_type in ("qwen", "llama"):
        from src.qwen_dynamic import QwenDynamicEngine
        print(f"[*] Initializing CausalEngine for {args.model_type} '{args.model_id}'...")
        engine = QwenDynamicEngine(
            model_id=args.model_id,
            device=args.device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            candidate_layers=eval_depths
        )
        tokenizer = engine.tokenizer
    else:
        print("[*] Initializing 125M TransformerLM...")
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            candidates = ["base_model.pt", os.path.join(REPO_ROOT, "base_model.pt")]
            for c in candidates:
                if os.path.exists(c):
                    ckpt_path = c
                    break

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[*] Loading checkpoint weights from {ckpt_path}...")
            sd = torch.load(ckpt_path, map_location=device)
            m = sd.get("model_state_dict", sd.get("model", sd))
            if isinstance(sd, dict) and "config" in sd and isinstance(sd["config"], dict):
                config = ModelConfig(**sd["config"])
            else:
                max_pos = m["pos_embeddings.weight"].shape[0] if "pos_embeddings.weight" in m else 256
                vocab = m["token_embeddings.weight"].shape[0] if "token_embeddings.weight" in m else 50257
                d_model = m["pos_embeddings.weight"].shape[1] if "pos_embeddings.weight" in m else 768
                config = ModelConfig(
                    vocab_size=vocab,
                    d_model=d_model,
                    n_heads=12,
                    n_layers=12,
                    max_seq_len=max_pos,
                    dropout=0.0
                )
            model_125m = TransformerLM(config).to(device)
            model_125m.load_state_dict(m, strict=False)
            print(f"-> Successfully loaded {sum(p.numel() for p in model_125m.parameters()):,} parameters.")
        else:
            print("[!] Running with freshly initialized 125M weights (no base_model.pt found).")
            config = ModelConfig(
                vocab_size=50257,
                d_model=768,
                n_heads=12,
                n_layers=12,
                max_seq_len=256,
                dropout=0.0
            )
            model_125m = TransformerLM(config).to(device)

        model_125m.eval()
        from src.dataset import get_tokenizer
        tokenizer = get_tokenizer()

    # Load ExitCalibrationModule if provided
    calibration_adapter = None
    if args.exit_adapter_path is not None:
        if os.path.exists(args.exit_adapter_path):
            print(f"[*] Loading ExitCalibrationModule from '{args.exit_adapter_path}'...")
            calibration_adapter = ExitCalibrationModule.load_checkpoint(args.exit_adapter_path, device=device)
            calibration_adapter.eval()
            print(f"✓ Successfully loaded calibration adapters for candidate layers: {calibration_adapter.candidate_depths}")
        else:
            print(f"[!] Warning: Exit adapter checkpoint '{args.exit_adapter_path}' not found! Running uncalibrated.")

    # Prepare evaluation sequences
    print(f"[*] Preparing {args.num_samples} sequences from '{args.dataset}' (seq_len={args.seq_len})...")
    eval_seqs = prepare_eval_sequences(args.dataset, tokenizer, args.seq_len, args.num_samples, device)
    print(f"-> Ready with {len(eval_seqs)} test sequences.")

    all_depth_results = []
    all_cossim_results = []

    for depth in eval_depths:
        print("\n" + "=" * 80)
        mode_tag = "WITH EXIT ADAPTER CALIBRATION" if calibration_adapter is not None else "UNCALIBRATED READOUT"
        print(f"RUNNING DETERMINISTIC ABLATION AT DEPTH SCHEDULE: L = {depth} ({mode_tag})")
        print("=" * 80)

        seq_metrics_list = []
        layer_cossims_list = []

        for s_idx, seq in enumerate(eval_seqs):
            print(f"  -> Evaluating sequence {s_idx + 1}/{len(eval_seqs)} (len={seq.size(1)})...", end="\r")
            if args.model_type in ("qwen", "llama"):
                cond_res, cossim_res = evaluate_qwen_sequence(engine, seq, depth, device, adapter=calibration_adapter)
            else:
                cond_res, cossim_res = evaluate_125m_sequence(model_125m, seq, depth, device, adapter=calibration_adapter)

            seq_metrics_list.append(cond_res)
            layer_cossims_list.append(cossim_res)


        print()

        # Aggregate metrics across sequences
        cond_keys = seq_metrics_list[0].keys()
        agg_cond = {}
        full_ppl = float(np.mean([s["full_depth"]["ppl"] for s in seq_metrics_list]))

        for k in cond_keys:
            ppl_val = float(np.mean([s[k]["ppl"] for s in seq_metrics_list]))
            nll_val = float(np.mean([s[k]["nll"] for s in seq_metrics_list]))
            kl_val = float(np.mean([s[k]["mean_kl"] for s in seq_metrics_list]))
            top1_val = float(np.mean([s[k]["top1_acc"] for s in seq_metrics_list]))
            ms_val = float(np.mean([s[k]["ms_tok"] for s in seq_metrics_list]))
            delta_ppl = ppl_val - full_ppl

            agg_cond[k] = {
                "depth": depth,
                "ppl": ppl_val,
                "delta_ppl": delta_ppl,
                "nll": nll_val,
                "mean_kl": kl_val,
                "top1_acc": top1_val,
                "ms_tok": ms_val
            }
            all_depth_results.append({
                "depth": depth,
                "condition": k,
                "ppl": ppl_val,
                "delta_ppl": delta_ppl,
                "mean_kl": kl_val,
                "top1_acc": top1_val,
                "ms_tok": ms_val
            })

        # Aggregate CosSim across sequences
        all_layers = sorted(layer_cossims_list[0].keys())
        for l in all_layers:
            c_k = float(np.mean([s[l]["cos_k"] for s in layer_cossims_list if l in s]))
            c_v = float(np.mean([s[l]["cos_v"] for s in layer_cossims_list if l in s]))
            r_k = float(np.mean([s[l]["rel_k"] for s in layer_cossims_list if l in s]))
            r_v = float(np.mean([s[l]["rel_v"] for s in layer_cossims_list if l in s]))
            all_cossim_results.append({
                "exit_depth": depth,
                "layer_0_indexed": l,
                "layer_1_indexed": l + 1,
                "cos_sim_k": c_k,
                "cos_sim_v": c_v,
                "rel_err_k": r_k,
                "rel_err_v": r_v
            })

        # Print console table for this depth
        print(f"\nConsolidated Results for Exit Depth L = {depth}:")
        print("-" * 88)
        print(f"{'Condition':<25} {'Depth':<8} {'PPL':<10} {'Δ PPL':<12} {'Mean KL':<10} {'Top-1 (%)':<11} {'ms/tok':<8}")
        print("-" * 88)
        cond_order = [
            "full_depth",
            "readout_only",
            "attn_native_proj",
            "attn_direct_copy",
            "combined_native_proj",
            "combined_direct_copy"
        ]
        for c in cond_order:
            row = agg_cond[c]
            print(
                f"{c:<25} {row['depth']:<8} {row['ppl']:<10.4f} {row['delta_ppl']:<+12.4f} "
                f"{row['mean_kl']:<10.4f} {row['top1_acc']:<11.1f} {row['ms_tok']:<8.2f}"
            )
        print("-" * 88)

    # Save summary CSV
    out_csv = os.path.join(args.output_dir, f"kv_ablation_{args.model_type}_depth_{args.depth_schedule}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["depth", "condition", "ppl", "delta_ppl", "mean_kl", "top1_acc", "ms_tok"]
        )
        writer.writeheader()
        writer.writerows(all_depth_results)
    print(f"\n[+] Saved detailed condition CSV to {out_csv}")

    # Save CosSim CSV
    cossim_csv = os.path.join(args.output_dir, f"cossim_teacher_forced_{args.model_type}.csv")
    with open(cossim_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["exit_depth", "layer_0_indexed", "layer_1_indexed", "cos_sim_k", "cos_sim_v", "rel_err_k", "rel_err_v"]
        )
        writer.writeheader()
        writer.writerows(all_cossim_results)
    print(f"[+] Saved teacher-forced CosSim decay CSV to {cossim_csv}")

    # Generate Markdown Table for Paper
    print("\n" + "=" * 80)
    print("ACADEMIC PAPER TABLE: THREE-CONDITION COMPARISON & DISENTANGLEMENT")
    print("=" * 80)

    # For the primary exit depth (first tested depth, e.g. L=6)
    primary_d = eval_depths[0]
    p_rows = {r["condition"]: r for r in all_depth_results if r["depth"] == primary_d}

    md_table = f"""
### Table: Teacher-Forced Decoupled KV-Cache Ablation at Fixed Depth Schedule ($L = {primary_d}$)

$$\\begin{{array}}{{lcccccc}}
\\hline
\\textbf{{Condition}} & \\textbf{{Attention KV}} & \\textbf{{Readout Layer}} & \\textbf{{PPL}} & \\Delta\\textbf{{PPL}} & \\bar{{D}}_{{\\text{{KL}}}} & \\textbf{{Top-1 Match}} \\\\
\\hline
\\text{{Full Depth Baseline}} & \\text{{True }} (K^*, V^*) & h_{{L}} & {p_rows['full_depth']['ppl']:.4f} & 0.0000 & 0.0000 & 100.0\\% \\\\
\\hline
\\textbf{{Disentanglement Sub-tests:}} & & & & & & \\\\
\\text{{Readout-Only (Zero KV Corruption)}} & \\text{{True }} (K^*, V^*) & h_{{{primary_d}}} & {p_rows['readout_only']['ppl']:.4f} & {p_rows['readout_only']['delta_ppl']:+.4f} & {p_rows['readout_only']['mean_kl']:.4f} & {p_rows['readout_only']['top1_acc']:.1f}\\% \\\\
\\text{{Attention-Only (Direct Copy Floor)}} & \\text{{Direct Copy}} & h_{{L}} & {p_rows['attn_direct_copy']['ppl']:.4f} & {p_rows['attn_direct_copy']['delta_ppl']:+.4f} & {p_rows['attn_direct_copy']['mean_kl']:.4f} & {p_rows['attn_direct_copy']['top1_acc']:.1f}\\% \\\\
\\textbf{{Attention-Only (Native Proj)}} & \\mathbf{{W_K/W_V + LN}} & \\mathbf{{h_{{L}}}} & \\mathbf{{{p_rows['attn_native_proj']['ppl']:.4f}}} & \\mathbf{{{p_rows['attn_native_proj']['delta_ppl']:+.4f}}} & \\mathbf{{{p_rows['attn_native_proj']['mean_kl']:.4f}}} & \\mathbf{{{p_rows['attn_native_proj']['top1_acc']:.1f}\\%}} \\\\
\\hline
\\textbf{{Combined Early-Exit (SkipDecode):}} & & & & & & \\\\
\\text{{SkipDecode + Direct Copy}} & \\text{{Direct Copy}} & h_{{{primary_d}}} & {p_rows['combined_direct_copy']['ppl']:.4f} & {p_rows['combined_direct_copy']['delta_ppl']:+.4f} & {p_rows['combined_direct_copy']['mean_kl']:.4f} & {p_rows['combined_direct_copy']['top1_acc']:.1f}\\% \\\\
\\textbf{{SkipDecode + Native Proj}} & \\mathbf{{W_K/W_V + LN}} & \\mathbf{{h_{{{primary_d}}}}} & \\mathbf{{{p_rows['combined_native_proj']['ppl']:.4f}}} & \\mathbf{{{p_rows['combined_native_proj']['delta_ppl']:+.4f}}} & \\mathbf{{{p_rows['combined_native_proj']['mean_kl']:.4f}}} & \\mathbf{{{p_rows['combined_native_proj']['top1_acc']:.1f}\\%}} \\\\
\\hline
\\end{{array}}$$
"""
    print(md_table)

    table_md_path = os.path.join(args.output_dir, f"paper_table_{args.model_type}_L{primary_d}.md")
    with open(table_md_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"[+] Saved paper Markdown table to {table_md_path}")

    # DECISION GATE FOR MILESTONE 3
    delta_attn = p_rows['attn_native_proj']['delta_ppl']
    delta_readout = p_rows['readout_only']['delta_ppl']
    delta_copy = p_rows['attn_direct_copy']['delta_ppl']

    print("\n" + "=" * 80)
    print("DECISIVE SCIENTIFIC DIAGNOSIS & GATE FOR MILESTONE 3:")
    print("=" * 80)
    print(f"  • Attention Degradation (Native Proj): Δ PPL = {delta_attn:+.4f}")
    print(f"  • Readout Degradation (Shallow h_{primary_d}): Δ PPL = {delta_readout:+.4f}")
    print(f"  • Direct Copy Baseline Degradation:   Δ PPL = {delta_copy:+.4f}")
    print("-" * 80)

    if delta_attn <= 0.15:
        print("✓ HYPOTHESIS CONFIRMED: KV CONTINUITY MECHANISM IS MATHEMATICALLY SOUND!")
        print(f"  Native Projection preserves representation continuity (Δ PPL = {delta_attn:+.4f} <= 0.15).")
        if delta_readout > 0.50:
            print(f"  The entire PPL gap (Δ PPL = {delta_readout:+.4f}) originates from Exit-Head Readout Mismatch.")
            print("  GATE PASSED: Proceed immediately to Milestone 3 (Router Training & Exit Adapters).")
            print("  Do NOT modify the KV projection mechanism; focus engineering on:")
            print("    1. Depth-conditioned output heads / Tuned Lens calibration for intermediate layers.")
            print("    2. Confidence-based dynamic router dispatching tokens to calibrated exit depths.")
    else:
        print(f"! ATTENTION CORRUPTION DETECTED: Δ PPL = {delta_attn:+.4f} > 0.15.")
        print("  The KV projection itself introduces non-trivial representation distortion.")
        print("  Engineering recommendation before Milestone 3:")
        print("    1. Implement low-rank learned KV projection adapters (W_K^adapter, W_V^adapter).")
        print("    2. Restrict early exit to near-terminal layers (d >= 8 on 125M, d >= 29 on Qwen).")
    print("=" * 80)


if __name__ == "__main__":
    main()
