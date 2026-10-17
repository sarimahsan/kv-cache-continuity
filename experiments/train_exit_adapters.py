#!/usr/bin/env python3
"""
experiments/train_exit_adapters.py

Train Lightweight Exit-Head Calibration Adapters (Tuned Lens / MLP / LoRA).
Milestone 3: Mapping Intermediate Manifolds M_l -> M_L for Fast Early-Exit Readout.

Extended Training Suite Addressing Reviewer Critique:
1. Long-Horizon Training on Tens of Millions of Tokens:
   - Configurable for 20,000 to 30,000+ steps (e.g., 25k steps x 8 batch x 128 seq = 25.6M tokens).
   - Cosine learning rate schedule with linear warmup.
   - Scalable to bottleneck ranks r in {512, 2048}.

2. Disjoint Validation Curve & Plateau Diagnostics:
   - Periodically evaluates on the disjoint WikiText-2 validation split.
   - Tracks Train Loss, Val Loss, Val Perplexity, Val KL Divergence, and Val Top-1 Agreement.
   - Logs continuous curve to CSV/JSON to test whether the ~82% readout agreement at L=18
     plateaus (proving intermediate readout is fundamentally hard) or climbs toward ~95%.

3. Multi-Depth Calibration:
   - Trains adapters across multiple exit depths: L in {18, 22, 24, 30}.
   - Automatically saves best validation checkpoint and final checkpoint.

Usage:
  # Long training of r=2048 adapter at L=18 (25,000 steps, ~25M tokens):
  python experiments/train_exit_adapters.py --model-type qwen --model-id Qwen/Qwen2.5-3B --candidate-depths 18 --adapter-type mlp --rank 2048 --steps 25000 --batch-size 8 --seq-len 128 --device cuda

  # Multi-depth training at L=18, 24, 30 (r=512):
  python experiments/train_exit_adapters.py --model-type qwen --model-id Qwen/Qwen2.5-3B --candidate-depths 18,24,30 --adapter-type mlp --rank 512 --steps 20000 --device cuda
"""

import os
import sys
import time
import math
import argparse
import csv
import json
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
try:
    from src.model import ModelConfig, TransformerLM
except ImportError:
    ModelConfig, TransformerLM = None, None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM, AutoTokenizer = None, None

from src.eval_dataset import get_tokenized_wikitext2


def parse_args():
    parser = argparse.ArgumentParser(description="Train Exit-Head Calibration Adapters with Full Validation Curves")
    parser.add_argument("--model-type", type=str, choices=["125m", "qwen", "llama"], default="qwen")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to base_model.pt (for 125m)")
    parser.add_argument(
        "--candidate-depths",
        type=str,
        default="18,24,30",
        help="Comma-separated candidate depths (e.g. '18,24,30' for Qwen, '4,6,8' for 125m)"
    )
    parser.add_argument(
        "--adapter-type",
        type=str,
        choices=["linear", "mlp", "lora"],
        default="mlp",
        help="Adapter architecture: 'mlp' (Non-linear MLP bottleneck), 'linear' (Tuned Lens), or 'lora'"
    )
    parser.add_argument("--rank", type=int, default=2048, help="Rank / bottleneck dim (e.g. 512 or 2048)")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length in tokens")
    parser.add_argument("--steps", type=int, default=25000, help="Total training steps (e.g. 20000-30000)")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Linear warmup steps")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluation and checkpoint interval")
    parser.add_argument("--val-samples", type=int, default=20, help="Number of sequences for validation evaluation")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate for AdamW")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Minimum learning rate after cosine decay")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--distill-alpha", type=float, default=0.5, help="Weight for teacher KD loss (0.0 to 1.0)")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "c4", "tinystories", "synthetic"],
        default="wikitext2",
        help="Training dataset"
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save adapter weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--resume-step", type=int, default=None, help="Step number to resume from (overrides step saved in checkpoint)")
    return parser.parse_args()


# ==============================================================================
# DATASTREAM GENERATOR WITH STREAMING REPEAT
# ==============================================================================
def create_batch_stream(
    dataset_name: str,
    tokenizer,
    batch_size: int,
    seq_len: int,
    device: torch.device
):
    """Yields batches of shape (B, seq_len) indefinitely from training split."""
    encode_fn = tokenizer.encode if hasattr(tokenizer, "encode") else (lambda s: tokenizer(s)["input_ids"])
    try:
        from datasets import load_dataset as hf_load_dataset
        if dataset_name == "wikitext2":
            try:
                ds = hf_load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
            except Exception:
                ds = hf_load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        elif dataset_name == "c4":
            ds = hf_load_dataset("stas/c4-en-10k", split="train")
        elif dataset_name == "tinystories":
            ds = hf_load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        else:
            raise ValueError("Unknown dataset")

        buffer = []
        while True:
            for item in ds:
                text = item.get("text", "")
                if len(text.strip()) < 30:
                    continue
                toks = encode_fn(text)
                if isinstance(toks, torch.Tensor):
                    toks = toks.tolist()
                buffer.extend(toks)

                while len(buffer) >= batch_size * (seq_len + 1):
                    chunk = buffer[: batch_size * (seq_len + 1)]
                    buffer = buffer[batch_size * (seq_len + 1) :]
                    t_tensor = torch.tensor(chunk, dtype=torch.long, device=device).reshape(batch_size, seq_len + 1)
                    yield t_tensor[:, :-1].contiguous(), t_tensor[:, 1:].contiguous()
    except Exception as e:
        print(f"[!] Warning: Streaming dataset '{dataset_name}' failed ({e}). Falling back to synthetic batches.")

    # Synthetic fallback
    while True:
        toks = torch.randint(100, 2000, (batch_size, seq_len + 1), device=device)
        yield toks[:, :-1].contiguous(), toks[:, 1:].contiguous()


# ==============================================================================
# DISJOINT VALIDATION DATASET PREPARATION
# ==============================================================================
def prepare_validation_sequences(tokenizer, num_samples: int, seq_len: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Loads disjoint validation split tokens and extracts evaluation pairs (x, y)."""
    val_toks = None
    try:
        val_toks = get_tokenized_wikitext2(tokenizer, split="validation")
    except Exception as e:
        print(f"[!] Could not load raw WikiText-2 val split: {e}")

    if val_toks is None or len(val_toks) < seq_len + 1:
        # Fallback synthetic
        val_toks = torch.randint(100, 2000, (num_samples * (seq_len + 1),), dtype=torch.long)

    rng = np.random.RandomState(42)
    max_st = len(val_toks) - seq_len - 1
    pairs = []
    starts = rng.choice(max_st, size=num_samples, replace=False) if max_st > num_samples else range(0, max_st, seq_len)
    for st in starts:
        chunk = val_toks[st : st + seq_len + 1].unsqueeze(0).to(device)
        pairs.append((chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()))
    return pairs


# ==============================================================================
# FEATURE EXTRACTORS
# ==============================================================================
@torch.no_grad()
def extract_intermediate_states(
    base_model,
    model_type: str,
    x: torch.Tensor,
    candidate_depths: List[int]
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Runs forward pass through frozen backbone and extracts intermediate representations."""
    if model_type in ("qwen", "llama"):
        outputs = base_model(input_ids=x, output_hidden_states=True, return_dict=True)
        all_hidden = outputs.hidden_states  # Tuple of length num_layers + 1
        saved_states = {}
        for d in candidate_depths:
            if d < len(all_hidden):
                saved_states[d] = all_hidden[d].clone()
        terminal_h = all_hidden[-1]
        teacher_logits = outputs.logits
        return saved_states, terminal_h, teacher_logits
    else:
        # 125M TransformerLM
        B, T = x.shape
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0)
        h = base_model.drop(base_model.token_embeddings(x) + base_model.pos_embeddings(pos))
        saved_states = {}
        for l in range(base_model.config.n_layers):
            h = base_model.layers[l](h)
            d = l + 1
            if d in candidate_depths:
                saved_states[d] = h.clone()
        terminal_h = h
        teacher_logits = base_model.lm_head(base_model.ln_f(terminal_h))
        return saved_states, terminal_h, teacher_logits


# ==============================================================================
# DISJOINT VALIDATION EVALUATION ROUTINE
# ==============================================================================
@torch.no_grad()
def evaluate_validation(
    base_model,
    model_type: str,
    calibration_module: ExitCalibrationModule,
    ln_f,
    lm_head,
    val_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    candidate_depths: List[int]
) -> Dict[str, Any]:
    """Evaluates validation loss, PPL, KL divergence, and top-1 agreement across candidate depths."""
    calibration_module.eval()
    results = {}

    for d in candidate_depths:
        total_loss = 0.0
        total_kl = 0.0
        total_top1 = 0
        total_tokens = 0

        for x_val, y_val in val_pairs:
            saved_states, _, teacher_logits = extract_intermediate_states(base_model, model_type, x_val, [d])
            if d not in saved_states:
                continue

            h_calibrated = calibration_module.forward_layer(saved_states[d], depth=d)
            student_logits = lm_head(ln_f(h_calibrated))

            B, T, V = student_logits.shape
            s_flat = student_logits.reshape(-1, V)
            t_flat = teacher_logits.reshape(-1, V)
            y_flat = y_val.reshape(-1)

            loss = F.cross_entropy(s_flat.float(), y_flat).item()
            total_loss += loss * y_flat.size(0)

            p_teach = F.softmax(t_flat.float(), dim=-1)
            log_p_stud = F.log_softmax(s_flat.float(), dim=-1)
            kl = F.kl_div(log_p_stud, p_teach, reduction="batchmean").item()
            total_kl += kl * y_flat.size(0)

            matches = (s_flat.argmax(dim=-1) == t_flat.argmax(dim=-1)).sum().item()
            total_top1 += matches
            total_tokens += y_flat.size(0)

        mean_loss = total_loss / max(1, total_tokens)
        ppl = math.exp(min(mean_loss, 20.0))
        mean_kl = total_kl / max(1, total_tokens)
        top1_pct = (total_top1 / max(1, total_tokens)) * 100.0

        results[d] = {
            "val_loss": mean_loss,
            "val_ppl": ppl,
            "val_kl": mean_kl,
            "val_top1": top1_pct
        }

    calibration_module.train()
    return results


# ==============================================================================
# MAIN TRAINING DRIVER
# ==============================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    candidate_depths = [int(x.strip()) for x in args.candidate_depths.split(",") if x.strip()]

    print("=" * 90)
    print("EXIT-HEAD CALIBRATION ADAPTER LONG-HORIZON TRAINING")
    print(f"Model: {args.model_type} ({args.model_id}) | Adapter: {args.adapter_type} (rank={args.rank})")
    print(f"Candidate Exit Depths: {candidate_depths} | Total Steps: {args.steps:,}")
    print(f"Batch Size: {args.batch_size} | Seq Len: {args.seq_len} tokens | Tokens Seen: {args.steps * args.batch_size * args.seq_len:,}")
    print(f"Learning Rate: {args.lr} (warmup: {args.warmup_steps} steps, min_lr: {args.min_lr})")
    print("=" * 90)

    # 1. Load and freeze base model
    tokenizer = None
    if args.model_type in ("qwen", "llama"):
        assert AutoModelForCausalLM is not None, "transformers library is required for Qwen / Llama training."
        print(f"[*] Loading pretrained {args.model_type} from '{args.model_id}'...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        model_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else (
            torch.float16 if device.type == "cuda" else torch.float32
        )
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
        print("[*] Initializing 125M TransformerLM...")
        config = ModelConfig(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=256, dropout=0.0)
        base_model = TransformerLM(config).to(device)
        d_model = base_model.config.d_model
        ln_f = base_model.ln_f
        lm_head = base_model.lm_head
        from src.dataset import get_tokenizer
        tokenizer = get_tokenizer()

    # Freeze base model completely
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    print("✓ Base model frozen successfully (0 backbone parameters updated).")

    # Match adapter dtype
    model_dtype = next(base_model.parameters()).dtype

    # 2. Instantiate ExitCalibrationModule
    print(f"[*] Initializing ExitCalibrationModule for depths {candidate_depths} (rank={args.rank}, dtype={model_dtype})...")
    calibration_module = ExitCalibrationModule(
        d_model=d_model,
        candidate_depths=candidate_depths,
        adapter_type=args.adapter_type,
        rank=args.rank
    ).to(device=device, dtype=model_dtype)

    trainable_params = sum(p.numel() for p in calibration_module.parameters() if p.requires_grad)
    print(f"-> Trainable adapter parameter count: {trainable_params:,} parameters.")

    # 3. Optimizer & Cosine Annealing LR Scheduler with Warmup
    optimizer = torch.optim.AdamW(
        calibration_module.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98)
    )

    start_step = 0
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {args.resume}")
        print(f"[*] Resuming from checkpoint: {args.resume}")
        ckpt_data = torch.load(args.resume, map_location=device)
        if isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
            calibration_module.load_state_dict(ckpt_data["state_dict"])
            if "step" in ckpt_data and args.resume_step is None:
                start_step = int(ckpt_data["step"])
            if "optimizer_state_dict" in ckpt_data:
                try:
                    optimizer.load_state_dict(ckpt_data["optimizer_state_dict"])
                    print("✓ Optimizer state successfully restored.")
                except Exception as e:
                    print(f"[!] Warning: Could not restore optimizer state ({e}). Using fresh optimizer.")
        elif isinstance(ckpt_data, dict):
            # Might be raw state dict
            calibration_module.load_state_dict(ckpt_data)
        else:
            raise ValueError(f"Unrecognized checkpoint format in {args.resume}")

        if args.resume_step is not None:
            start_step = args.resume_step

        print(f"✓ Resuming training starting at step {start_step + 1} (target total steps: {args.steps}).")

    def get_lr(step: int) -> float:
        if step < args.warmup_steps:
            return args.lr * (step / max(1, args.warmup_steps))
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr + (args.lr - args.min_lr) * cosine_decay

    # 4. Prepare data streams
    print("[*] Preparing training and disjoint validation data...")
    train_stream = create_batch_stream(
        dataset_name=args.dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device=device
    )
    val_pairs = prepare_validation_sequences(tokenizer, num_samples=args.val_samples, seq_len=args.seq_len, device=device)
    print(f"✓ Disjoint validation set prepared ({len(val_pairs)} sequence chunks).")

    # 5. Training Loop
    training_log = []
    best_val_top1 = 0.0
    t_start = time.perf_counter()

    csv_log_path = os.path.join(args.output_dir, f"training_log_{args.model_type}_{args.adapter_type}_r{args.rank}.csv")
    print(f"\n[*] Starting training loop. Logging progress to {csv_log_path}...")
    print("-" * 115)
    print(f"{'Step':<7} | {'LR':<9} | {'Train Loss':<11} | {'Mean KD':<10} | {'Val PPL (L18)':<14} | {'Val KL':<10} | {'Val Top-1':<11} | {'Elapsed':<8}")
    print("-" * 115)

    for step in range(start_step + 1, args.steps + 1):
        # Update learning rate
        curr_lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = curr_lr

        x_batch, y_batch = next(train_stream)

        # Forward through frozen backbone
        saved_states, terminal_h, teacher_logits = extract_intermediate_states(
            base_model, args.model_type, x_batch, candidate_depths
        )

        optimizer.zero_grad()
        kd_losses = []
        ce_losses = []

        for d in candidate_depths:
            if d not in saved_states:
                continue
            h_d = saved_states[d]
            h_calibrated = calibration_module.forward_layer(h_d, depth=d)
            student_logits = lm_head(ln_f(h_calibrated))

            loss_ce = F.cross_entropy(student_logits.reshape(-1, student_logits.size(-1)).float(), y_batch.reshape(-1))

            if args.distill_alpha > 0.0:
                t_temp = 2.0
                p_teach = F.softmax((teacher_logits / t_temp).float(), dim=-1)
                log_p_stud = F.log_softmax((student_logits / t_temp).float(), dim=-1)
                loss_kd = F.kl_div(log_p_stud, p_teach, reduction="batchmean") * (t_temp ** 2)
                loss_layer = (1.0 - args.distill_alpha) * loss_ce + args.distill_alpha * loss_kd
                del p_teach, log_p_stud
            else:
                loss_kd = torch.tensor(0.0, device=device)
                loss_layer = loss_ce

            ce_losses.append(loss_ce.item())
            kd_losses.append(loss_kd.item())

            # Backward immediately per depth to release logits & computation graph from VRAM
            (loss_layer / len(candidate_depths)).backward()
            del student_logits, h_calibrated, loss_layer

        nn.utils.clip_grad_norm_(calibration_module.parameters(), max_norm=1.0)
        optimizer.step()
        del saved_states, terminal_h, teacher_logits
        torch.cuda.empty_cache()

        step_loss = float(np.mean(ce_losses)) + float(np.mean(kd_losses) if kd_losses else 0.0)

        # Print training progress every 10 steps so output is never stuck
        if step % 10 == 0 or step == 1:
            elapsed_m = (time.perf_counter() - t_start) / 60.0
            steps_per_sec = step / max(1e-4, time.perf_counter() - t_start)
            print(
                f"Step {step:>5}/{args.steps} | LR: {curr_lr:.2e} | Loss: {step_loss:.4f} | "
                f"CE: {np.mean(ce_losses):.4f} | KD: {np.mean(kd_losses):.4f} | {steps_per_sec:.2f} it/s | {elapsed_m:.1f}m elapsed",
                flush=True
            )

        # Validation evaluation at intervals
        if step % args.eval_every == 0 or step == args.steps:
            val_results = evaluate_validation(
                base_model=base_model,
                model_type=args.model_type,
                calibration_module=calibration_module,
                ln_f=ln_f,
                lm_head=lm_head,
                val_pairs=val_pairs,
                candidate_depths=candidate_depths
            )

            primary_d = candidate_depths[0]
            primary_val = val_results.get(primary_d, {"val_ppl": 0.0, "val_kl": 0.0, "val_top1": 0.0})

            elapsed_str = f"{(time.perf_counter() - t_start) / 60.0:.1f}m"
            print(
                f"\n>>> [EVAL @ Step {step}] PPL (L={primary_d}): {primary_val['val_ppl']:.2f} | "
                f"KL: {primary_val['val_kl']:.4f} | Top-1: {primary_val['val_top1']:.2f}% | Elapsed: {elapsed_str}\n",
                flush=True
            )

            log_entry = {
                "step": step,
                "lr": curr_lr,
                "train_loss": step_loss,
                "mean_ce": float(np.mean(ce_losses)),
                "mean_kd": float(np.mean(kd_losses)),
                "elapsed_sec": time.perf_counter() - t_start,
                "val_metrics": val_results
            }
            training_log.append(log_entry)

            # Checkpoint best validation model
            if primary_val["val_top1"] > best_val_top1:
                best_val_top1 = primary_val["val_top1"]
                best_ckpt_path = os.path.join(args.output_dir, f"best_exit_adapters_{args.model_type}_{args.adapter_type}_r{args.rank}.pt")
                torch.save({
                    "d_model": calibration_module.d_model,
                    "candidate_depths": calibration_module.candidate_depths,
                    "adapter_type": calibration_module.adapter_type,
                    "rank": calibration_module.rank,
                    "state_dict": calibration_module.state_dict(),
                    "step": step,
                    "optimizer_state_dict": optimizer.state_dict()
                }, best_ckpt_path)
                print(f"[+] Saved new best validation checkpoint ({best_val_top1:.2f}% top-1) to: {best_ckpt_path}", flush=True)

    # 6. Save final checkpoint and training curve CSV
    final_ckpt_path = os.path.join(args.output_dir, f"final_exit_adapters_{args.model_type}_{args.adapter_type}_r{args.rank}.pt")
    torch.save({
        "d_model": calibration_module.d_model,
        "candidate_depths": calibration_module.candidate_depths,
        "adapter_type": calibration_module.adapter_type,
        "rank": calibration_module.rank,
        "state_dict": calibration_module.state_dict(),
        "step": args.steps,
        "optimizer_state_dict": optimizer.state_dict()
    }, final_ckpt_path)
    print(f"\n[+] Saved final adapter checkpoint to: {final_ckpt_path}")
    if best_val_top1 > 0:
        print(f"[+] Best validation checkpoint achieved: {best_val_top1:.2f}% top-1 agreement.")

    # Write training log CSV
    flat_log = []
    for entry in training_log:
        row = {
            "step": entry["step"],
            "lr": entry["lr"],
            "train_loss": entry["train_loss"],
            "mean_ce": entry["mean_ce"],
            "mean_kd": entry["mean_kd"],
            "elapsed_sec": entry["elapsed_sec"]
        }
        for d, v in entry["val_metrics"].items():
            row[f"val_ppl_L{d}"] = v["val_ppl"]
            row[f"val_kl_L{d}"] = v["val_kl"]
            row[f"val_top1_L{d}"] = v["val_top1"]
        flat_log.append(row)

    if flat_log:
        with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_log[0].keys()))
            writer.writeheader()
            writer.writerows(flat_log)
        print(f"✓ Saved detailed training curve to {csv_log_path}")

    print("\n" + "=" * 90)
    print("ADAPTER TRAINING COMPLETE!")
    print("=" * 90)


if __name__ == "__main__":
    main()
