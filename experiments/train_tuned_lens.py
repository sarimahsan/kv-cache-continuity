#!/usr/bin/env python3
"""
experiments/train_tuned_lens.py

Fast, Ultra-Lightweight Tuned-Lens / Logit-Direction Adapter Training.
Exploits the Key-stable / Value-evolving asymmetry by training an ultra-cheap
affine or low-rank readout correction (0.2M - 0.5M parameters) in < 3 minutes.

Target:
    Calibrate intermediate hidden states h_{d_t} -> h_L before unembedding,
    lifting empirical draft acceptance rate alpha without heavy MLP overhead.

Usage:
    python experiments/train_tuned_lens.py \\
        --model-id Qwen/Qwen2.5-3B \\
        --target-depths 18,22,24,28 \\
        --adapter-type lora \\
        --rank 64 \\
        --steps 400 \\
        --output-path checkpoints/exit_adapters_qwen_tuned_lens.pt \\
        --device cuda
"""

import os
import sys
import time
import math
import argparse
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.exit_adapter import ExitCalibrationModule


def parse_args():
    parser = argparse.ArgumentParser(description="Fast Tuned-Lens / Readout Head Calibration")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Hugging Face repo")
    parser.add_argument("--target-depths", type=str, default="18,22,24,28", help="Comma-separated candidate depths")
    parser.add_argument("--adapter-type", type=str, choices=["lora", "linear", "mlp"], default="lora", help="Adapter type")
    parser.add_argument("--rank", type=int, default=64, help="Bottleneck rank (e.g. 64 for 0.26M params)")
    parser.add_argument("--steps", type=int, default=400, help="Training distillation steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Token length per sequence")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--distill-temp", type=float, default=2.0, help="Distillation temperature")
    parser.add_argument("--output-path", type=str, default="checkpoints/exit_adapters_qwen_tuned_lens.pt", help="Checkpoint save path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    target_depths = [int(d.strip()) for d in args.target_depths.split(",") if d.strip()]

    print("=" * 80)
    print("FAST TUNED-LENS READOUT HEAD CALIBRATION")
    print(f"Model ID: {args.model_id} | Depths: {target_depths} | Type: {args.adapter_type} | Rank: {args.rank}")
    print(f"Steps: {args.steps} | Batch Size: {args.batch_size} | Seq Len: {args.seq_len}")
    print("=" * 80)

    print(f"[*] Loading model '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    d_model = model.config.hidden_size

    # Initialize ExitCalibrationModule
    adapter_module = ExitCalibrationModule(
        d_model=d_model,
        candidate_depths=target_depths,
        adapter_type=args.adapter_type,
        rank=args.rank
    ).to(device)

    total_params = sum(p.numel() for p in adapter_module.parameters())
    print(f"✓ Initialized ExitCalibrationModule with {total_params:,} parameters ({total_params / 1e6:.2f}M params).")
    print(f"  Cost per token: ~{total_params * 2 / 1e6:.2f} MFLOPs (negligible next to 6,000 MFLOPs full pass).")

    optimizer = torch.optim.AdamW(adapter_module.parameters(), lr=args.lr, weight_decay=0.01)

    # Simple diverse synthetic/domain text generator for fast distillation
    SAMPLE_TEXTS = [
        "In artificial intelligence, transformer models have fundamentally changed natural language processing by using self-attention mechanisms to model relationships between tokens in parallel.",
        "The mathematical foundation of general relativity relies on differential geometry, where spacetime is modeled as a pseudo-Riemannian manifold whose curvature is directly related to the energy-momentum tensor.",
        "To implement an efficient sorting algorithm, quicksort selects a pivot element and partitions the array into sub-arrays according to whether elements are less than or greater than the pivot.",
        "Photosynthesis in C3 plants involves the Calvin cycle where the enzyme RuBisCO fixes carbon dioxide from the atmosphere into 3-phosphoglycerate molecules.",
        "Quantum mechanics describes physical systems using state vectors in a complex Hilbert space, where observable quantities correspond to self-adjoint linear operators.",
        "In modern operating systems, virtual memory provides an abstraction of storage that decouples the logical address space from physical memory using page tables and translation lookaside buffers."
    ]

    print("\n[*] Starting distillation training...")
    t_start = time.perf_counter()

    for step in range(1, args.steps + 1):
        # Sample texts
        batch_texts = [SAMPLE_TEXTS[(step * args.batch_size + b) % len(SAMPLE_TEXTS)] for b in range(args.batch_size)]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.seq_len).to(device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            teacher_logits = outputs.logits
            hidden_states = outputs.hidden_states  # tuple of length num_layers + 1

        total_loss = 0.0

        for depth in target_depths:
            # Intermediate state h_{d_t}
            h_d = hidden_states[depth]
            calibrated_h = adapter_module.forward_layer(h_d, depth=depth)

            # Pass through final norm and lm_head
            norm_cal = model.model.norm(calibrated_h)
            student_logits = model.lm_head(norm_cal)

            # KL divergence distillation loss against full-model teacher logits
            p_teacher = F.softmax(teacher_logits / args.distill_temp, dim=-1)
            log_p_student = F.log_softmax(student_logits / args.distill_temp, dim=-1)

            kl_loss = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (args.distill_temp ** 2)

            # Next-token cross-entropy loss against ground truth
            shift_logits = student_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss_d = 0.7 * kl_loss + 0.3 * ce_loss
            total_loss = total_loss + loss_d

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_module.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 50 == 0 or step == args.steps:
            elapsed = time.perf_counter() - t_start
            print(f"  Step [{step:4d}/{args.steps}] | Total Loss: {total_loss.item():.4f} | Time: {elapsed:.1f}s")

    # Save checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    adapter_module.save_checkpoint(args.output_path)
    print(f"\n✓ Successfully trained and saved tuned-lens calibration module to: {args.output_path}")


if __name__ == "__main__":
    main()
