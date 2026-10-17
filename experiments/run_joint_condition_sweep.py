#!/usr/bin/env python3
"""
experiments/run_joint_condition_sweep.py

Addresses Claim C2 & Gap A4:
- A4. Joint condition (Native Projection + adapter readout):
  Evaluates the complete, real-world early exit system at multiple sweep depths:
  1. Attention-Only Error (E_attn): Terminal readout, but past tokens attend to reconstructed KV cache.
  2. Readout-Only Error (E_readout): Pristine full-depth KV in attention, but read out from intermediate state h_d.
  3. Joint Condition (E_joint): Intermediate readout from h_d + reconstructed KV cache in attention.

Proves:
Whether the two-error decomposition predicts the real combined system:
Tests whether D_KL(joint) roughly equals D_KL(attn) + D_KL(readout) or exhibits sub/super-additive interaction:
  Interaction Term: Delta = D_KL(joint) - [D_KL(attn) + D_KL(readout)]

Usage:
  python experiments/run_joint_condition_sweep.py --model-id Qwen/Qwen2.5-3B --adapter-path checkpoints/exit_adapters_qwen_mlp.pt --depths 18,24,30 --device cuda
  python experiments/run_joint_condition_sweep.py --quick --device cpu
"""

import os
import sys
import math
import argparse
import csv
import json
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.qwen_dynamic import QwenDynamicEngine
from src.exit_adapter import ExitCalibrationModule
from src.eval_dataset import get_tokenized_wikitext2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Joint Condition and Error Decomposition (A4)")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--adapter-path", type=str, default="checkpoints/exit_adapters_qwen_mlp.pt")
    parser.add_argument("--depths", type=str, default=None, help="Comma-separated exit depths (e.g. '18,24,30')")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=str, default="results/canonical_sweeps/joint_condition")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    return parser.parse_args()


@torch.no_grad()
def evaluate_decomposition_depth(
    engine: QwenDynamicEngine,
    adapter: Optional[ExitCalibrationModule],
    sequences: List[torch.Tensor],
    depth: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluates:
      1. Oracle reference (mode="full", strategy="oracle_full")
      2. Attention-Only (mode="attention_only", strategy="native_projection") -> E_attn
      3. Readout-Only (mode="readout_only", strategy="oracle_full") -> E_readout
      4. Joint Early Exit (mode="combined", strategy="native_projection") -> E_joint
    """
    kv_cache = engine.init_static_kv_cache(batch_size=1, max_seq_len=sequences[0].shape[1] + 16)

    def run_mode(mode: str, strategy: str) -> Tuple[float, float, float]:
        total_kl = 0.0
        total_agree = 0
        total_tokens = 0
        total_nll = 0.0

        for seq in sequences:
            T = seq.shape[1]
            seq = seq.to(device)

            # First compute oracle reference logits
            oracle_logits = []
            for p in range(T - 1):
                tok = seq[:, p : p + 1]
                log_full, _ = engine.step_forward_decoupled(
                    input_id=tok,
                    pos=p,
                    kv_cache=kv_cache,
                    exit_depth=engine.num_layers,
                    mode="full",
                    strategy="oracle_full",
                )
                oracle_logits.append(log_full.squeeze(1))

            # Reset cache
            for l in range(engine.num_layers):
                kv_cache["k"][l].zero_()
                kv_cache["v"][l].zero_()

            # Run test mode
            for p in range(T - 1):
                tok = seq[:, p : p + 1]
                target_tok = seq[0, p + 1].item()

                log_test, _ = engine.step_forward_decoupled(
                    input_id=tok,
                    pos=p,
                    kv_cache=kv_cache,
                    exit_depth=depth,
                    mode=mode,
                    strategy=strategy,
                    adapter=adapter,
                )
                log_test = log_test.squeeze(1)
                log_ref = oracle_logits[p]

                p_ref = F.softmax(log_ref.float(), dim=-1)
                logp_ref = F.log_softmax(log_ref.float(), dim=-1)
                logp_test = F.log_softmax(log_test.float(), dim=-1)
                kl = (p_ref * (logp_ref - logp_test)).sum(dim=-1).item()
                total_kl += kl

                if torch.argmax(log_ref, dim=-1).item() == torch.argmax(log_test, dim=-1).item():
                    total_agree += 1

                nll = F.cross_entropy(log_test.float(), torch.tensor([target_tok], device=device)).item()
                total_nll += nll
                total_tokens += 1

        m_kl = total_kl / total_tokens
        m_top1 = (total_agree / total_tokens) * 100.0
        m_ppl = math.exp(min(total_nll / total_tokens, 15.0))
        return m_kl, m_top1, m_ppl

    # Run the 3 conditions
    kl_attn, top1_attn, ppl_attn = run_mode("attention_only", "native_projection")
    kl_readout, top1_readout, ppl_readout = run_mode("readout_only", "oracle_full")
    kl_joint, top1_joint, ppl_joint = run_mode("combined", "native_projection")

    # Additive prediction & interaction delta
    kl_sum_pred = kl_attn + kl_readout
    interaction_delta = kl_joint - kl_sum_pred

    return {
        "kl_attn": kl_attn,
        "top1_attn": top1_attn,
        "ppl_attn": ppl_attn,
        "kl_readout": kl_readout,
        "top1_readout": top1_readout,
        "ppl_readout": ppl_readout,
        "kl_joint": kl_joint,
        "top1_joint": top1_joint,
        "ppl_joint": ppl_joint,
        "kl_sum_pred": kl_sum_pred,
        "interaction_delta": interaction_delta,
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Joint Condition and Error Decomposition Sweep (A4) on {args.model_id} ===")
    engine = QwenDynamicEngine(model_id=args.model_id, device=args.device)
    device = torch.device(args.device)

    # Load adapter if available
    adapter = None
    if os.path.exists(args.adapter_path):
        print(f"[*] Loading ExitCalibrationModule checkpoint from: {args.adapter_path}")
        try:
            ckpt = torch.load(args.adapter_path, map_location=device)
            cand_depths = ckpt.get("candidate_depths", [18, 22, 24, 30])
            rank = ckpt.get("rank", 512)
            adapter_type = ckpt.get("adapter_type", "mlp")
            adapter = ExitCalibrationModule(
                d_model=engine.config.hidden_size,
                candidate_depths=cand_depths,
                adapter_type=adapter_type,
                rank=rank,
            ).to(device)
            state_dict = ckpt["state_dict"] if "state_dict" in ckpt else (ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
            adapter.load_state_dict(state_dict)
            adapter.eval()
            print(f"[OK] Exit adapter loaded successfully for depths {cand_depths}.")
        except Exception as e:
            print(f"[!] Warning: Could not instantiate adapter: {e}. Evaluating with uncalibrated intermediate state.")
            adapter = None
    else:
        print(f"[i] No adapter found at {args.adapter_path}. Running with zero-shot intermediate readout.")

    # Determine candidate exit depths
    if args.depths is not None:
        depths = [int(d) for d in args.depths.split(",")]
    else:
        L = engine.num_layers
        depths = [int(round(0.50 * L)), int(round(0.67 * L)), int(round(0.83 * L))]

    # Load tokens
    try:
        raw_tokens = get_tokenized_wikitext2(engine.tokenizer, split="validation")
    except Exception as e:
        print(f"[!] Fallback to synthetic corpus: {e}")
        text = "Transformers utilize self-attention mechanisms to perform natural language processing tasks. " * 30
        raw_tokens = engine.tokenizer(text, return_tensors="pt")["input_ids"][0]

    num_samples = 2 if args.quick else args.num_samples
    seq_len = args.seq_len

    sequences = []
    max_idx = len(raw_tokens) - seq_len - 1
    step = max(1, max_idx // num_samples)
    for i in range(num_samples):
        st = i * step
        sequences.append(raw_tokens[st : st + seq_len].unsqueeze(0))

    out_csv = os.path.join(args.output_dir, f"joint_condition_decomposition_{args.model_id.replace('/', '_')}.csv")
    rows = []

    print(f"\n{'Depth':<6} | {'Rel.':<6} | {'KL_attn':<9} | {'KL_readout':<11} | {'Sum(Pred)':<10} | {'KL_joint':<9} | {'Interaction Delta':<18}")
    print("-" * 84)

    for d in depths:
        rel_d = (d / engine.num_layers) * 100.0
        decomp = evaluate_decomposition_depth(
            engine=engine,
            adapter=adapter,
            sequences=sequences,
            depth=d,
            device=device,
        )

        sign = "+" if decomp["interaction_delta"] >= 0 else ""
        print(f"L={d:<4} | {rel_d:4.0f}% | {decomp['kl_attn']:9.4f} | {decomp['kl_readout']:11.4f} | {decomp['kl_sum_pred']:10.4f} | {decomp['kl_joint']:9.4f} | {sign}{decomp['interaction_delta']:12.4f}")

        rows.append({
            "model_id": args.model_id,
            "depth": d,
            "rel_depth": f"{rel_d:.1f}%",
            "kl_attn": round(decomp["kl_attn"], 4),
            "kl_readout": round(decomp["kl_readout"], 4),
            "kl_sum_pred": round(decomp["kl_sum_pred"], 4),
            "kl_joint": round(decomp["kl_joint"], 4),
            "interaction_delta": round(decomp["interaction_delta"], 4),
            "top1_joint": round(decomp["top1_joint"], 2),
            "ppl_joint": round(decomp["ppl_joint"], 2),
        })

    fieldnames = ["model_id", "depth", "rel_depth", "kl_attn", "kl_readout", "kl_sum_pred", "kl_joint", "interaction_delta", "top1_joint", "ppl_joint"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Joint condition decomposition results saved to: {out_csv}")


if __name__ == "__main__":
    main()
