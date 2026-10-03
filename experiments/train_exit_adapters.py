#!/usr/bin/env python3
"""
experiments/train_exit_adapters.py

Train Lightweight Exit-Head Calibration Adapters (Tuned Lens / LoRA).
Milestone 3 - Step 1: Mapping Intermediate Manifolds M_l -> M_L.

Scientific Principle:
The base Transformer model remains 100% FROZEN.
We train a lightweight linear or low-rank affine adapter:
    P_exit^(l): M_l -> M_L
such that logits emitted from intermediate layer l:
    logits_l = LMHead( LN_f( P_exit^(l)( h_l ) ) )
match the ground-truth next-token targets and terminal teacher distribution.

VRAM & Compute Efficiency:
- Since the base model is frozen, gradients are NEVER backpropagated through
  the deep Transformer layers.
- We only compute gradients for the tiny adapter parameters (<0.5% model size).
- Runs cleanly on a single consumer GPU or free Colab T4 runtime.

Usage Examples:
  # On Colab (125M base model on candidate depth L=6 or sweep):
  python experiments/train_exit_adapters.py --model-type 125m --checkpoint base_model.pt --candidate-depths 4,6,8,10 --steps 500 --device cuda

  # On Colab (Qwen-2.5-3B on candidate depths 18, 24, 29):
  python experiments/train_exit_adapters.py --model-type qwen --model-id Qwen/Qwen2.5-3B --candidate-depths 18,24,29 --steps 500 --device cuda
"""

import os
import sys
import time
import math
import argparse
from typing import List, Dict, Any, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.exit_adapter import ExitCalibrationModule
from src.model import ModelConfig, TransformerLM


def parse_args():
    parser = argparse.ArgumentParser(description="Train Exit-Head Calibration Adapters (Milestone 3)")
    parser.add_argument("--model-type", type=str, choices=["125m", "qwen"], default="125m", help="Model family")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Hugging Face repo for Qwen")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to base_model.pt checkpoint")
    parser.add_argument(
        "--candidate-depths",
        type=str,
        default=None,
        help="Comma-separated candidate depths (e.g. '4,6,8,10' for 125m, '18,24,29' for qwen)"
    )
    parser.add_argument(
        "--adapter-type",
        type=str,
        choices=["linear", "mlp", "lora"],
        default="linear",
        help="Adapter architecture: 'linear' (Tuned Lens), 'mlp' (Non-linear MLP), or 'lora' (Low-Rank Residual)"
    )

    parser.add_argument("--rank", type=int, default=128, help="Rank for low-rank adapter")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length in tokens")
    parser.add_argument("--steps", type=int, default=600, help="Number of training steps")
    parser.add_argument("--eval-every", type=int, default=100, help="Evaluation interval")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for adapter")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Weight for teacher KD loss (0.0 to 1.0)")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["tinystories", "wikitext2", "synthetic"],
        default="tinystories",
        help="Training dataset"
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save adapter weights")
    return parser.parse_args()


# ==============================================================================
# DATASTREAM GENERATOR
# ==============================================================================
def create_batch_stream(
    dataset_name: str,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device
):
    """Yields batches of shape (B, seq_len) indefinitely."""
    if dataset_name in ["tinystories", "wikitext2"]:
        from src.dataset import get_tokenizer as get_gpt2_tok
        tok = tokenizer if tokenizer is not None else get_gpt2_tok()
        encode_fn = tok.encode if hasattr(tok, "encode") else (lambda s: tok(s)["input_ids"])
        try:
            from datasets import load_dataset as hf_load_dataset
            if dataset_name == "tinystories":
                ds = hf_load_dataset("roneneldan/TinyStories", split="train", streaming=True)
            else:
                try:
                    ds = hf_load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
                except Exception:
                    ds = hf_load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

            buffer = []
            for item in ds:
                text = item.get("text", "")
                if len(text.strip()) < 50:
                    continue
                toks = encode_fn(text)
                buffer.extend(toks)
                buffer.append(50256)

                while len(buffer) >= batch_size * (seq_len + 1):
                    chunk = buffer[: batch_size * (seq_len + 1)]
                    buffer = buffer[batch_size * (seq_len + 1) :]
                    t_tensor = torch.tensor(chunk, dtype=torch.long, device=device).reshape(batch_size, seq_len + 1)
                    yield t_tensor[:, :-1].contiguous(), t_tensor[:, 1:].contiguous()
        except Exception as e:
            print(f"[!] Warning: Streaming dataset '{dataset_name}' failed: {e}. Falling back to synthetic batches.")

    # Synthetic fallback
    while True:
        toks = torch.randint(100, 2000, (batch_size, seq_len + 1), device=device)
        yield toks[:, :-1].contiguous(), toks[:, 1:].contiguous()



# ==============================================================================
# 125M FORWARD FEATURE EXTRACTOR
# ==============================================================================
@torch.no_grad()
def extract_125m_intermediate_states(
    model: TransformerLM,
    x: torch.Tensor,
    candidate_depths: List[int]
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Runs a forward pass through the frozen 125M model.
    Returns:
      intermediate_states: {depth: h_depth of shape (B, T, d_model)}
      terminal_h: h_L of shape (B, T, d_model)
      teacher_logits: (B, T, vocab_size)
    """
    device = x.device
    B, T = x.shape
    pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
    h = model.drop(model.token_embeddings(x) + model.pos_embeddings(pos))

    saved_states = {}
    for l in range(model.config.n_layers):
        h = model.layers[l](h)
        depth_1_indexed = l + 1
        if depth_1_indexed in candidate_depths:
            saved_states[depth_1_indexed] = h.clone()

    terminal_h = h
    teacher_norm = model.ln_f(terminal_h)
    teacher_logits = model.lm_head(teacher_norm)

    return saved_states, terminal_h, teacher_logits


# ==============================================================================
# QWEN-2.5 FORWARD FEATURE EXTRACTOR
# ==============================================================================
@torch.no_grad()
def extract_qwen_intermediate_states(
    model,
    x: torch.Tensor,
    candidate_depths: List[int]
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Runs a forward pass through the frozen Qwen model.
    Extracts intermediate states at candidate depths.
    """
    outputs = model(
        input_ids=x,
        output_hidden_states=True,
        return_dict=True
    )
    all_hidden = outputs.hidden_states  # Tuple of (B, T, d_model), length num_layers + 1
    # all_hidden[0] is embedding, all_hidden[l] is after layer l (1-indexed)
    saved_states = {}
    for d in candidate_depths:
        if d < len(all_hidden):
            saved_states[d] = all_hidden[d].clone()

    terminal_h = all_hidden[-1]
    teacher_logits = outputs.logits

    return saved_states, terminal_h, teacher_logits


# ==============================================================================
# MAIN TRAINING ROUTINE
# ==============================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("MILESTONE 3: TRAINING EXIT-HEAD CALIBRATION ADAPTERS (TUNED LENS)")
    print(f"Model: {args.model_type} | Adapter: {args.adapter_type} | Device: {device}")
    print("=" * 80)

    # 1. Parse candidate depths
    if args.candidate_depths is not None:
        candidate_depths = [int(x.strip()) for x in args.candidate_depths.split(",") if x.strip()]
    else:
        candidate_depths = [4, 6, 8, 10] if args.model_type == "125m" else [18, 24, 29]

    # 2. Load and freeze base model
    tokenizer = None
    if args.model_type == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[*] Loading pretrained Qwen model '{args.model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device.type == "cuda" else torch.float32
        )
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                dtype=model_dtype,
                device_map=args.device,
                trust_remote_code=True
            )
        except TypeError:
            base_model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                torch_dtype=model_dtype,
                device_map=args.device,
                trust_remote_code=True
            )
        d_model = base_model.config.hidden_size
        ln_f = base_model.model.norm
        lm_head = base_model.lm_head
    else:
        print("[*] Loading 125M TransformerLM...")
        ckpt_path = args.checkpoint
        if ckpt_path is None:
            for c in ["base_model.pt", os.path.join(REPO_ROOT, "base_model.pt")]:
                if os.path.exists(c):
                    ckpt_path = c
                    break

        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[*] Loading weights from {ckpt_path}...")
            sd = torch.load(ckpt_path, map_location=device)
            m = sd.get("model_state_dict", sd.get("model", sd))
            config = ModelConfig(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=256, dropout=0.0)
            base_model = TransformerLM(config).to(device)
            base_model.load_state_dict(m, strict=False)
        else:
            print("[!] Running with initialized 125M architecture.")
            config = ModelConfig(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=256, dropout=0.0)
            base_model = TransformerLM(config).to(device)

        d_model = base_model.config.d_model
        ln_f = base_model.ln_f
        lm_head = base_model.lm_head
        from src.dataset import get_tokenizer
        tokenizer = get_tokenizer()

    # CRITICAL: Freeze ALL base model weights
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    print("✓ Base model frozen successfully (0 parameters updated).")

    # Match adapter dtype to the base model's parameter dtype (e.g. bfloat16 on GPU)
    model_dtype = next(base_model.parameters()).dtype

    # 3. Instantiate ExitCalibrationModule with matching device and dtype
    print(f"[*] Initializing ExitCalibrationModule for layers {candidate_depths} (d_model={d_model}, dtype={model_dtype})...")
    calibration_module = ExitCalibrationModule(
        d_model=d_model,
        candidate_depths=candidate_depths,
        adapter_type=args.adapter_type,
        rank=args.rank
    ).to(device=device, dtype=model_dtype)

    trainable_params = sum(p.numel() for p in calibration_module.parameters() if p.requires_grad)
    print(f"-> Trainable adapter footprint: {trainable_params:,} parameters (< 0.5% model size).")

    optimizer = torch.optim.AdamW(
        calibration_module.parameters(),
        lr=args.lr,

        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )

    # 4. Prepare data stream
    data_stream = create_batch_stream(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device
    )

    print(f"\n[*] Starting calibration training for {args.steps} steps...")
    print("-" * 85)
    print(f"{'Step':<8} {'Total Loss':<12} {'Distill Loss':<14} {'Mean CE':<10} {'Avg PPL':<10} {'Top-1 (%)':<10}")
    print("-" * 85)

    step_losses = []
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        x_batch, y_batch = next(data_stream)

        # Forward pass through frozen base model to get intermediate representations
        if args.model_type == "qwen":
            intermediate_states, terminal_h, teacher_logits = extract_qwen_intermediate_states(
                base_model, x_batch, candidate_depths
            )
        else:
            intermediate_states, terminal_h, teacher_logits = extract_125m_intermediate_states(
                base_model, x_batch, candidate_depths
            )

        optimizer.zero_grad()
        total_loss = 0.0
        ce_losses = []
        kd_losses = []
        top1_matches_all = []

        # Train each candidate exit depth adapter concurrently
        for d in candidate_depths:
            if d not in intermediate_states:
                continue

            h_d = intermediate_states[d]  # (B, T, d_model)
            # Calibrate through adapter
            h_calibrated = calibration_module.forward_layer(h_d, depth=d)

            # Project to vocabulary via terminal head
            norm_calibrated = ln_f(h_calibrated)
            student_logits = lm_head(norm_calibrated)  # (B, T, vocab_size)

            # Next-token Cross Entropy Loss (computed in float32 for numerical stability)
            loss_ce = F.cross_entropy(
                student_logits.reshape(-1, student_logits.size(-1)).float(),
                y_batch.reshape(-1)
            )


            # Distillation loss against terminal teacher
            if args.distill_alpha > 0.0:
                t_temp = 2.0
                p_teacher = F.softmax((teacher_logits / t_temp).float(), dim=-1)
                log_p_student = F.log_softmax((student_logits / t_temp).float(), dim=-1)
                loss_kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (t_temp ** 2)
                loss_layer = (1.0 - args.distill_alpha) * loss_ce + args.distill_alpha * loss_kd
            else:
                loss_kd = torch.tensor(0.0, device=device)
                loss_layer = loss_ce


            total_loss = total_loss + loss_layer
            ce_losses.append(loss_ce.item())
            kd_losses.append(loss_kd.item())

            # Top-1 accuracy against ground truth targets
            preds = torch.argmax(student_logits, dim=-1)
            top1_matches_all.append((preds == y_batch).float().mean().item())

        total_loss.backward()
        nn.utils.clip_grad_norm_(calibration_module.parameters(), max_norm=1.0)
        optimizer.step()

        step_losses.append(total_loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            mean_ce = float(np.mean(ce_losses))
            mean_kd = float(np.mean(kd_losses))
            mean_ppl = math.exp(min(mean_ce, 20.0))
            avg_top1 = float(np.mean(top1_matches_all)) * 100.0

            print(
                f"{step:<8} {total_loss.item():<12.4f} {mean_kd:<14.4f} {mean_ce:<10.4f} "
                f"{mean_ppl:<10.2f} {avg_top1:<10.1f}%"
            )

    elapsed_time = time.perf_counter() - t_start
    print("-" * 85)
    print(f"✓ Training finished in {elapsed_time:.1f}s ({args.steps / elapsed_time:.1f} steps/s).")

    # 5. Save trained adapters
    save_path = os.path.join(args.output_dir, f"exit_adapters_{args.model_type}.pt")
    calibration_module.save_checkpoint(save_path)
    print(f"\n[+] Saved calibrated exit adapters to: {save_path}")

    print("\n" + "=" * 80)
    print("NEXT STEP: EVALUATE POST-CALIBRATION ABLATION")
    print("=" * 80)
    print("Run the teacher-forced decoupled ablation script with --exit-adapter-path:")
    print(f"  python experiments/isolate_kv_ablation.py --model-type {args.model_type} --exit-adapter-path {save_path} --device {args.device}")
    print("=" * 80)


if __name__ == "__main__":
    main()
