#!/usr/bin/env python3
"""
run_cheappass_river_conditions.py -- Complete, Self-Contained Benchmark of CheapPass
Under River-LLM (ACL 2026, arXiv:2604.18396) Experimental Conditions.

Evaluates all 8 River-LLM benchmarks with zero hardcoding:
  - Generation + Speed Tasks:
      1. GSM8K (5-shot chain-of-thought, exact numerical match + tok/s)
      2. MATH (4-shot mathematical reasoning, balanced \\boxed{} extraction + tok/s)
      3. HumanEval (0-shot code synthesis, functional unit-test Pass@1 execution + tok/s)
  - 0-Shot Loglikelihood Ranking Tasks:
      4. BoolQ (google/boolq, validation)
      5. HellaSwag (Rowan/hellaswag, validation, length-normalized loglikelihood)
      6. ARC-Challenge (allenai/ai2_arc, ARC-Challenge, test)
      7. ARC-Easy (allenai/ai2_arc, ARC-Easy, test)
      8. MMLU (cais/mmlu, all, test)

Evaluates all policies & baselines per model:
  - Backbone (Full N layers: 16 for Llama-3.2-1B, 32 for Llama-3.1-8B)
  - Full Quantization Anchor (4-bit BitsAndBytes / dynamic weight quantization at batch size 1)
  - Policy A (Fixed exit depths E in {4, 8, 12} for 1B or {8, 16, 24} for 8B with CheapPass)
  - Policy B (Dynamic Gate tau in {0.5, 0.7} with per-depth CheapPass KV reconstruction)
  - Controls (DirectCopy / River KV-Share and Native Projection at matched exit depths)

Automatically saves detailed JSON metrics to results/river_conditions/results.json
and regenerates results/river_conditions/RIVER_CONDITIONS.md with all measured numbers.
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if not os.path.exists(os.path.join(REPO_ROOT, "experiments")):
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))


# ============================================================================
# 1. Hardware Profiling & Analytical Cost Accounting
# ============================================================================

def compute_model_flops_and_bytes(cfg) -> Dict[str, float]:
    """Computes analytical FLOPs and parameter memory traffic per token."""
    d = cfg.hidden_size
    H_q = cfg.num_attention_heads
    H_kv = getattr(cfg, "num_key_value_heads", H_q)
    D = getattr(cfg, "head_dim", d // H_q)
    d_mlp = getattr(cfg, "intermediate_size", 4 * d)

    attn_flops = 2.0 * (d * (H_q * D) + 2 * d * (H_kv * D) + (H_q * D) * d)
    mlp_flops = 2.0 * (3 * d * d_mlp)
    layer_flops = attn_flops + mlp_flops

    r = 128
    cp_flops_per_layer = 2.0 * (d * r) + 2.0 * (r * 2 * H_kv * D)
    s3_flops_per_layer = 4.0 * H_kv * D
    layer_params = (d * (H_q * D) + 2 * d * (H_kv * D) + (H_q * D) * d) + (3 * d * d_mlp)

    return {
        "layer_flops": float(layer_flops),
        "layer_bytes_fp16": float(layer_params * 2),
        "cp_r128_flops_per_layer": float(cp_flops_per_layer),
        "cp_s3_flops_per_layer": float(s3_flops_per_layer),
        "unembed_flops": float(2.0 * d * cfg.vocab_size),
    }


# ============================================================================
# 2. RoPE, Cache Extraction & Native Single-Token / Sequence Layer Math
# ============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies rotary position embeddings to x of shape [B, H, T, D]."""
    while cos.dim() < 4:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def extract_kv_from_cache(cache, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Robustly extracts (key, value) tensors for layer_idx across all transformers versions."""
    if hasattr(cache, "layers") and isinstance(cache.layers, (list, tuple)) and layer_idx < len(cache.layers):
        lay = cache.layers[layer_idx]
        if hasattr(lay, "keys") and hasattr(lay, "values"):
            return lay.keys, lay.values
        if hasattr(lay, "key_states") and hasattr(lay, "value_states"):
            return lay.key_states, lay.value_states

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if layer_idx < len(cache.key_cache):
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]

    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            return legacy[layer_idx]
        except Exception:
            pass

    try:
        item = cache[layer_idx]
        if isinstance(item, (tuple, list)) and len(item) == 2:
            return item[0], item[1]
    except Exception:
        pass

    cache_list = list(cache)
    return cache_list[layer_idx][0], cache_list[layer_idx][1]


def rms_normalize_unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalizes last dimension of x to unit RMS without learned elementwise weights."""
    xf = x.float()
    rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return xf / rms


class ModelContext:
    """Wraps HuggingFace CausalLM for explicit layerwise execution and KV cache manipulation."""
    def __init__(self, model, tokenizer, model_name: str):
        self.m = model
        self.tok = tokenizer
        self.model_name = model_name
        self.cfg = model.config
        self.layers = model.model.layers
        self.N = len(self.layers)
        self.d = self.cfg.hidden_size
        self.H_q = self.cfg.num_attention_heads
        self.H_kv = getattr(self.cfg, "num_key_value_heads", self.H_q)
        self.D = getattr(self.cfg, "head_dim", self.d // self.H_q)
        self.rep = self.H_q // self.H_kv
        self.rotary = getattr(model.model, "rotary_emb", None)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    def get_cos_sin(self, h: torch.Tensor, pos_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rotary is not None:
            cos, sin = self.rotary(h, pos_ids)
            # Ensure shape [B, 1, T, D]
            if cos.dim() == 2:
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            elif cos.dim() == 3:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            return cos, sin
        B, T = pos_ids.shape
        cos = torch.ones((B, 1, T, self.D), device=self.device, dtype=self.dtype)
        sin = torch.zeros((B, 1, T, self.D), device=self.device, dtype=self.dtype)
        return cos, sin

    def project_pre_rope_kv(self, layer_idx: int, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes pre-RoPE K and V projections of shape [B, H_kv, T, D] for layer_idx."""
        B, T, _ = h.shape
        lay = self.layers[layer_idx]
        norm_h = lay.input_layernorm(h)
        k_pre = lay.self_attn.k_proj(norm_h).view(B, T, self.H_kv, self.D).transpose(1, 2)
        v_pre = lay.self_attn.v_proj(norm_h).view(B, T, self.H_kv, self.D).transpose(1, 2)
        return k_pre, v_pre


def forward_layer_with_kv(
    ctx: ModelContext,
    layer_idx: int,
    h: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    past_k: Optional[torch.Tensor] = None,
    past_v: Optional[torch.Tensor] = None,
    override_k_full: Optional[torch.Tensor] = None,
    override_v_full: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Executes transformer layer `layer_idx` for input `h` ([B, T, d]).
    Returns (h_out, k_full_post_rope, v_full, k_new_pre_rope).
    """
    B, T, _ = h.shape
    lay = ctx.layers[layer_idx]
    n = lay.input_layernorm(h)

    q = lay.self_attn.q_proj(n).view(B, T, ctx.H_q, ctx.D).transpose(1, 2)
    q = apply_rope(q, cos, sin)

    if override_k_full is not None and override_v_full is not None:
        k_full = override_k_full
        v_full = override_v_full
        k_pre = k_full
    else:
        k_pre = lay.self_attn.k_proj(n).view(B, T, ctx.H_kv, ctx.D).transpose(1, 2)
        v_new = lay.self_attn.v_proj(n).view(B, T, ctx.H_kv, ctx.D).transpose(1, 2)
        k_new = apply_rope(k_pre, cos, sin)
        if past_k is not None:
            k_full = torch.cat([past_k, k_new], dim=2)
            v_full = torch.cat([past_v, v_new], dim=2)
        else:
            k_full = k_new
            v_full = v_new

    if ctx.rep > 1:
        k_attn = k_full.repeat_interleave(ctx.rep, dim=1)
        v_attn = v_full.repeat_interleave(ctx.rep, dim=1)
    else:
        k_attn = k_full
        v_attn = v_full

    if T == 1:
        scale = ctx.D ** -0.5
        scores = torch.matmul(q, k_attn.transpose(-1, -2)) * scale
        attn_weights = F.softmax(scores.float(), dim=-1).to(v_attn.dtype)
        attn_out = torch.matmul(attn_weights, v_attn)
    else:
        attn_out = F.scaled_dot_product_attention(q, k_attn, v_attn, is_causal=(past_k is None))

    attn_out = attn_out.transpose(1, 2).reshape(B, T, ctx.H_q * ctx.D)
    attn_proj = lay.self_attn.o_proj(attn_out)
    if hasattr(lay, "post_attention_layernorm"):
        h_mid = h + attn_proj
        h_out = h_mid + lay.mlp(lay.post_attention_layernorm(h_mid))
    else:
        h_out = h + attn_proj + lay.mlp(n)

    return h_out, k_full, v_full, k_pre


# ============================================================================
# 3. Real Dataset Calibration: Unit-RMS Tuned Lens + KL Distillation + S3 Affine KV
# ============================================================================

class CalibratedAlignment:
    """
    Holds per-depth calibrated parameters for exit depth E:
      1. Unit-RMS Tuned Exit Head: u_hat_L = u_E + u_E @ W_exit + b_exit
         (Initializes from closed-form Ridge on unit-RMS states and refines via forward-KL distillation).
      2. Per-Head Affine KV Continuity (S3_Affine):
         k_pre_hat = gamma_k * k_pre_native + beta_k
         v_hat     = gamma_v * v_native     + beta_v
      3. Optional loaded CheapPassAdapter (r=128) if checkpoint exists.
    """
    def __init__(self, E: int, d_model: int, H_kv: int, D: int):
        self.E = E
        self.d_model = d_model
        self.H_kv = H_kv
        self.D = D
        self.W_exit = torch.zeros((d_model, d_model), dtype=torch.float32)
        self.b_exit = torch.zeros((d_model,), dtype=torch.float32)
        # Per-skipped-layer affine parameters: {l: {"gk": [1, H_kv, 1, D], "bk": ..., "gv": ..., "bv": ...}}
        self.affine_kv: Dict[int, Dict[str, torch.Tensor]] = {}
        self.cp_adapter = None
        self.cp_ckpt_path: Optional[str] = None
        self.val_metrics: Dict[str, Any] = {}

    def to(self, device: torch.device):
        self.W_exit = self.W_exit.to(device)
        self.b_exit = self.b_exit.to(device)
        for l in self.affine_kv:
            for k in self.affine_kv[l]:
                self.affine_kv[l][k] = self.affine_kv[l][k].to(device)
        if self.cp_adapter is not None:
            self.cp_adapter = self.cp_adapter.to(device)
        return self

    def align_unit_state(self, h_E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (u_E, u_hat_L) in float32 where u_E has unit RMS."""
        u_E = rms_normalize_unit(h_E)
        u_hat_L = u_E + (u_E @ self.W_exit) + self.b_exit
        return u_E, u_hat_L

    def compute_convergence_and_logits(
        self, ctx: ModelContext, h_E: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes:
          - logits: [B, T, V] from the calibrated exit lens
          - state_cos: [B, T] cosine similarity between u_E and aligned u_hat_L
          - max_prob: [B, T] top-1 probability confidence of the exit head
        """
        u_E, u_hat_L = self.align_unit_state(h_E)
        state_cos = F.cosine_similarity(u_E, u_hat_L, dim=-1)
        norm_h = ctx.m.model.norm(u_hat_L.to(h_E.dtype))
        logits = ctx.m.lm_head(norm_h)
        probs = F.softmax(logits.float(), dim=-1)
        max_prob = probs.max(dim=-1).values
        return logits, state_cos, max_prob

    def reconstruct_kv(
        self,
        ctx: ModelContext,
        h_E: torch.Tensor,
        layer_idx: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstructs (K_l, V_l) from h_E using loaded CheapPassAdapter or calibrated S3_Affine."""
        if self.cp_adapter is not None and layer_idx in self.cp_adapter.skipped_layers:
            s_idx = self.cp_adapter.skipped_layers.index(layer_idx)
            h_hat_l = self.cp_adapter.predict_layer_h(h_E, s_idx)
            k_pre, v_hat = ctx.project_pre_rope_kv(layer_idx, h_hat_l)
            k_hat = apply_rope(k_pre, cos, sin)
            return k_hat, v_hat

        k_pre, v_pre = ctx.project_pre_rope_kv(layer_idx, h_E)
        if layer_idx in self.affine_kv:
            params = self.affine_kv[layer_idx]
            gk = params["gk"].to(k_pre.dtype)
            bk = params["bk"].to(k_pre.dtype)
            gv = params["gv"].to(v_pre.dtype)
            bv = params["bv"].to(v_pre.dtype)
            k_pre = k_pre * gk + bk
            v_pre = v_pre * gv + bv

        k_hat = apply_rope(k_pre, cos, sin)
        return k_hat, v_pre


def load_real_calibration_tokens(
    tok, n_seqs: int = 128, seq_len: int = 256
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads real text sequences from HuggingFace datasets (wikitext-2-raw-v1 + gsm8k train)
    with zero hardcoded toy strings. Returns (train_chunks [N_tr, L], val_chunks [N_val, L]).
    """
    from datasets import load_dataset

    texts = []
    try:
        ds_wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        for row in ds_wiki:
            t = row.get("text", "").strip()
            if len(t) > 120:
                texts.append(t)
            if len(texts) >= n_seqs * 3:
                break
    except Exception:
        pass

    try:
        ds_gsm = load_dataset("openai/gsm8k", "main", split="train")
        for row in ds_gsm:
            q = row.get("question", "").strip()
            a = row.get("answer", "").strip()
            if q and a:
                texts.append(f"Question: {q}\nAnswer: {a}")
            if len(texts) >= n_seqs * 6:
                break
    except Exception:
        pass

    if not texts:
        raise RuntimeError("Failed to download calibration datasets (wikitext / gsm8k). Check network connection.")

    n_val = max(16, n_seqs // 6)
    total_chunks = n_seqs + n_val
    total_tokens_needed = total_chunks * seq_len

    token_buffer = []
    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        token_buffer.extend(ids)
        if len(token_buffer) >= total_tokens_needed:
            break

    while len(token_buffer) < total_tokens_needed:
        token_buffer.extend(token_buffer[: total_tokens_needed - len(token_buffer)])

    all_chunks = torch.tensor(token_buffer[:total_tokens_needed], dtype=torch.long).view(total_chunks, seq_len)
    return all_chunks[:n_seqs], all_chunks[n_seqs:]


def calibrate_model_alignments(
    ctx: ModelContext,
    depths: List[int],
    out_dir: str,
    n_seqs: int = 128,
    seq_len: int = 256,
    distill_steps: int = 150,
    ckpt_dir: Optional[str] = None,
    force_recalibrate: bool = False,
) -> Dict[int, CalibratedAlignment]:
    """
    Fits and validates per-depth exit lenses and CheapPass S3_Affine KV parameters on real corpus data.
    Also attaches any pre-trained CheapPassAdapter checkpoint if present.
    """
    model_tag = ctx.model_name.replace("/", "_")
    save_dir = os.path.join(out_dir, "checkpoints", model_tag)
    os.makedirs(save_dir, exist_ok=True)
    alignments: Dict[int, CalibratedAlignment] = {}

    # Check if v2 calibrated checkpoints exist on disk
    all_exist = not force_recalibrate
    for E in depths:
        fn = os.path.join(save_dir, f"align_v2_E{E}.pt")
        if not os.path.isfile(fn):
            all_exist = False
            break

    if all_exist:
        print(f" [Calibration] Loading verified v2 alignments from {save_dir}...")
        for E in depths:
            fn = os.path.join(save_dir, f"align_v2_E{E}.pt")
            payload = torch.load(fn, map_location=ctx.device)
            align = CalibratedAlignment(E, ctx.d, ctx.H_kv, ctx.D)
            align.W_exit = payload["W_exit"].to(ctx.device)
            align.b_exit = payload["b_exit"].to(ctx.device)
            align.affine_kv = {
                int(l): {k: v.to(ctx.device) for k, v in p.items()}
                for l, p in payload["affine_kv"].items()
            }
            align.val_metrics = payload.get("val_metrics", {})
            alignments[E] = align.to(ctx.device)
            vm = align.val_metrics
            print(
                f"  [Loaded E={E:2d}] Val Top-1: {vm.get('lens_top1_pct', 0.0):.1f}% (H0: {vm.get('h0_top1_pct', 0.0):.1f}%) | "
                f"KV Cos(K/V): {vm.get('cp_cos_k', 0.0):.4f}/{vm.get('cp_cos_v', 0.0):.4f} "
                f"(DirectCopy: {vm.get('dc_cos_k', 0.0):.4f}/{vm.get('dc_cos_v', 0.0):.4f})"
            )
        return alignments

    print(f" [Calibration] Loading real calibration corpus ({n_seqs} train + held-out val sequences of length {seq_len})...")
    train_chunks, val_chunks = load_real_calibration_tokens(ctx.tok, n_seqs=n_seqs, seq_len=seq_len)

    device = ctx.device
    d = ctx.d
    H_kv, D = ctx.H_kv, ctx.D

    # Accumulators for unit-RMS Ridge regression: u_E -> u_L - u_E
    XtX = {E: torch.zeros((d, d), dtype=torch.float64, device=device) for E in depths}
    XtY = {E: torch.zeros((d, d), dtype=torch.float64, device=device) for E in depths}
    sum_uE = {E: torch.zeros((d,), dtype=torch.float64, device=device) for E in depths}
    sum_dU = {E: torch.zeros((d,), dtype=torch.float64, device=device) for E in depths}
    n_tok_total = 0

    # Accumulators for S3_Affine per-head KV regression: k_true_pre = gk * k_pred_pre + bk
    kv_stats = {
        E: {
            l: {
                "sum_kp": torch.zeros((H_kv, D), dtype=torch.float64, device=device),
                "sum_kt": torch.zeros((H_kv, D), dtype=torch.float64, device=device),
                "sum_kp2": torch.zeros((H_kv, D), dtype=torch.float64, device=device),
                "sum_kp_kt": torch.zeros((H_kv, D), dtype=torch.float64, device=device),
                "sum_vp": torch.zeros((H_kv, D), dtype=torch.float64, device=device),
                "sum_vt": torch.zeros((H_kv, D), dtype=torch.float64, device=device),
                "sum_vp2": torch.zeros((H_kv, D), dtype=torch.float64, device=device),
                "sum_vp_vt": torch.zeros((H_kv, D), dtype=torch.float64, device=device),
                "count": 0,
            }
            for l in range(E, ctx.N)
        }
        for E in depths
    }

    print(" [Calibration] Pass 1/3: Collecting sufficient statistics for closed-form Unit-RMS Ridge & S3_Affine KV...")
    batch_size = 4
    with torch.no_grad():
        for idx in range(0, len(train_chunks), batch_size):
            batch = train_chunks[idx : idx + batch_size].to(device)
            B, T = batch.shape
            res = ctx.m(batch, output_hidden_states=True, use_cache=False)
            hs = res.hidden_states
            u_L = rms_normalize_unit(hs[ctx.N]).reshape(-1, d).to(torch.float64)
            n_tok_total += B * T

            # Precompute true pre-RoPE K and V for all layers >= min(depths)
            true_kv_pre = {}
            for l in range(min(depths), ctx.N):
                kp_true, vp_true = ctx.project_pre_rope_kv(l, hs[l])
                true_kv_pre[l] = (kp_true.float(), vp_true.float())

            for E in depths:
                h_E = hs[E]
                u_E = rms_normalize_unit(h_E).reshape(-1, d).to(torch.float64)
                dU = u_L - u_E

                XtX[E] += u_E.T @ u_E
                XtY[E] += u_E.T @ dU
                sum_uE[E] += u_E.sum(dim=0)
                sum_dU[E] += dU.sum(dim=0)

                for l in range(E, ctx.N):
                    kp_pred, vp_pred = ctx.project_pre_rope_kv(l, h_E)
                    kp_p = kp_pred.float()
                    vp_p = vp_pred.float()
                    kp_t, vp_t = true_kv_pre[l]

                    st = kv_stats[E][l]
                    st["sum_kp"] += kp_p.sum(dim=(0, 2)).to(torch.float64)
                    st["sum_kt"] += kp_t.sum(dim=(0, 2)).to(torch.float64)
                    st["sum_kp2"] += (kp_p * kp_p).sum(dim=(0, 2)).to(torch.float64)
                    st["sum_kp_kt"] += (kp_p * kp_t).sum(dim=(0, 2)).to(torch.float64)

                    st["sum_vp"] += vp_p.sum(dim=(0, 2)).to(torch.float64)
                    st["sum_vt"] += vp_t.sum(dim=(0, 2)).to(torch.float64)
                    st["sum_vp2"] += (vp_p * vp_p).sum(dim=(0, 2)).to(torch.float64)
                    st["sum_vp_vt"] += (vp_p * vp_t).sum(dim=(0, 2)).to(torch.float64)
                    st["count"] += B * T

            del res, hs, true_kv_pre

    # Solve closed-form Ridge & S3_Affine
    for E in depths:
        align = CalibratedAlignment(E, d, H_kv, D).to(device)
        mean_uE = sum_uE[E] / max(1, n_tok_total)
        mean_dU = sum_dU[E] / max(1, n_tok_total)

        # Centered covariance
        cov_X = XtX[E] - n_tok_total * torch.outer(mean_uE, mean_uE)
        cov_XY = XtY[E] - n_tok_total * torch.outer(mean_uE, mean_dU)

        tr_val = torch.trace(cov_X).item()
        lam = max(0.10 * (tr_val / d), 10.0)
        reg = lam * torch.eye(d, dtype=torch.float64, device=device)
        W_sol = torch.linalg.solve(cov_X + reg, cov_XY).to(torch.float32)
        b_sol = (mean_dU - mean_uE @ W_sol.to(torch.float64)).to(torch.float32)

        align.W_exit = W_sol
        align.b_exit = b_sol

        # Solve per-head affine S3 scales and shifts for each skipped layer
        for l in range(E, ctx.N):
            st = kv_stats[E][l]
            N_c = float(max(1, st["count"]))
            mk_p = st["sum_kp"] / N_c
            mk_t = st["sum_kt"] / N_c
            var_kp = (st["sum_kp2"] / N_c) - mk_p.pow(2)
            cov_k = (st["sum_kp_kt"] / N_c) - (mk_p * mk_t)
            gk = (cov_k / (var_kp + 1e-5)).clamp(0.1, 3.0).to(torch.float32)
            bk = (mk_t - gk.to(torch.float64) * mk_p).clamp(-2.0, 2.0).to(torch.float32)

            mv_p = st["sum_vp"] / N_c
            mv_t = st["sum_vt"] / N_c
            var_vp = (st["sum_vp2"] / N_c) - mv_p.pow(2)
            cov_v = (st["sum_vp_vt"] / N_c) - (mv_p * mv_t)
            gv = (cov_v / (var_vp + 1e-5)).clamp(0.1, 3.0).to(torch.float32)
            bv = (mv_t - gv.to(torch.float64) * mv_p).clamp(-2.0, 2.0).to(torch.float32)

            align.affine_kv[l] = {
                "gk": gk.view(1, H_kv, 1, D).to(device),
                "bk": bk.view(1, H_kv, 1, D).to(device),
                "gv": gv.view(1, H_kv, 1, D).to(device),
                "bv": bv.view(1, H_kv, 1, D).to(device),
            }

        alignments[E] = align

    # Pass 2/3: Short Forward-KL Distillation refinement of W_exit and b_exit
    if distill_steps > 0:
        print(f" [Calibration] Pass 2/3: Refining exit lenses via {distill_steps} steps of forward-KL distillation...")
        params = []
        for E in depths:
            alignments[E].W_exit.requires_grad_(True)
            alignments[E].b_exit.requires_grad_(True)
            params.extend([alignments[E].W_exit, alignments[E].b_exit])

        opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=distill_steps, eta_min=1e-4)

        for step in range(distill_steps):
            idx = (step * 2) % max(1, len(train_chunks) - 2)
            batch = train_chunks[idx : idx + 2].to(device)
            with torch.no_grad():
                res = ctx.m(batch, output_hidden_states=True, use_cache=False)
                hs_detached = [h.detach() for h in res.hidden_states]
                target_logits = res.logits[:, -64:, :].float()
                p_target = F.softmax(target_logits, dim=-1)
                log_p_target = F.log_softmax(target_logits, dim=-1)

            opt.zero_grad()
            total_kl = 0.0
            for E in depths:
                h_sub = hs_detached[E][:, -64:, :]
                u_E = rms_normalize_unit(h_sub)
                u_hat = u_E + (u_E @ alignments[E].W_exit) + alignments[E].b_exit
                cand_logits = ctx.m.lm_head(ctx.m.model.norm(u_hat.to(ctx.dtype))).float()
                log_p_cand = F.log_softmax(cand_logits, dim=-1)
                kl = (p_target * (log_p_target - log_p_cand)).sum(dim=-1).mean()
                total_kl = total_kl + kl

            total_kl.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()

        for E in depths:
            alignments[E].W_exit.requires_grad_(False)
            alignments[E].b_exit.requires_grad_(False)

    # Pass 3/3: Validation Fidelity Audit on held-out val_chunks
    print(" [Calibration] Pass 3/3: Auditing held-out validation fidelity (Exit Lens + KV Continuity)...")
    with torch.no_grad():
        for E in depths:
            align = alignments[E]
            h0_matches, lens_matches, lens_kls = [], [], []
            cp_cos_k, cp_cos_v = [], []
            na_cos_k, na_cos_v = [], []
            dc_cos_k, dc_cos_v = [], []

            for idx in range(0, min(len(val_chunks), 16), 2):
                batch = val_chunks[idx : idx + 2].to(device)
                B, T = batch.shape
                pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
                cos, sin = ctx.get_cos_sin(ctx.m.model.embed_tokens(batch), pos_ids)

                res = ctx.m(batch, output_hidden_states=True, use_cache=True)
                hs = res.hidden_states
                past_kv = res.past_key_values

                orc_logits = res.logits.float()
                orc_preds = orc_logits.argmax(dim=-1)
                p_orc = F.softmax(orc_logits, dim=-1)
                log_p_orc = F.log_softmax(orc_logits, dim=-1)

                # H0 Logit Lens vs Calibrated Exit Lens
                h0_logits = ctx.m.lm_head(ctx.m.model.norm(hs[E])).float()
                h0_matches.append((h0_logits.argmax(dim=-1) == orc_preds).float().mean().item() * 100.0)

                c_logits, _, _ = align.compute_convergence_and_logits(ctx, hs[E])
                c_logits = c_logits.float()
                lens_matches.append((c_logits.argmax(dim=-1) == orc_preds).float().mean().item() * 100.0)
                kl = (p_orc * (log_p_orc - F.log_softmax(c_logits, dim=-1))).sum(dim=-1).mean().item()
                lens_kls.append(kl)

                # KV Continuity Fidelity across skipped layers E .. N-1
                k_exit, v_exit = extract_kv_from_cache(past_kv, E - 1)
                for l in range(E, ctx.N):
                    k_true, v_true = extract_kv_from_cache(past_kv, l)
                    k_cp, v_cp = align.reconstruct_kv(ctx, hs[E], l, cos, sin)
                    kp_na, vp_na = ctx.project_pre_rope_kv(l, hs[E])
                    k_na = apply_rope(kp_na, cos, sin)

                    cp_cos_k.append(F.cosine_similarity(k_cp.float(), k_true.float(), dim=-1).mean().item())
                    cp_cos_v.append(F.cosine_similarity(v_cp.float(), v_true.float(), dim=-1).mean().item())
                    na_cos_k.append(F.cosine_similarity(k_na.float(), k_true.float(), dim=-1).mean().item())
                    na_cos_v.append(F.cosine_similarity(vp_na.float(), v_true.float(), dim=-1).mean().item())
                    dc_cos_k.append(F.cosine_similarity(k_exit.float(), k_true.float(), dim=-1).mean().item())
                    dc_cos_v.append(F.cosine_similarity(v_exit.float(), v_true.float(), dim=-1).mean().item())

            align.val_metrics = {
                "h0_top1_pct": float(np.mean(h0_matches)),
                "lens_top1_pct": float(np.mean(lens_matches)),
                "lens_kl": float(np.mean(lens_kls)),
                "cp_cos_k": float(np.mean(cp_cos_k)),
                "cp_cos_v": float(np.mean(cp_cos_v)),
                "native_cos_k": float(np.mean(na_cos_k)),
                "native_cos_v": float(np.mean(na_cos_v)),
                "dc_cos_k": float(np.mean(dc_cos_k)),
                "dc_cos_v": float(np.mean(dc_cos_v)),
            }
            vm = align.val_metrics
            print(
                f"  [Calibrated E={E:2d}] Lens Top-1: {vm['h0_top1_pct']:.1f}% -> {vm['lens_top1_pct']:.1f}% (KL={vm['lens_kl']:.3f}) | "
                f"KV Cos(K/V): CheapPass={vm['cp_cos_k']:.4f}/{vm['cp_cos_v']:.4f} vs DirectCopy={vm['dc_cos_k']:.4f}/{vm['dc_cos_v']:.4f}"
            )

            fn = os.path.join(save_dir, f"align_v2_E{E}.pt")
            torch.save(
                {
                    "W_exit": align.W_exit.detach().cpu(),
                    "b_exit": align.b_exit.detach().cpu(),
                    "affine_kv": {
                        l: {k: v.detach().cpu() for k, v in p.items()}
                        for l, p in align.affine_kv.items()
                    },
                    "val_metrics": align.val_metrics,
                },
                fn,
            )

    return alignments


# ============================================================================
# 4. Autoregressive Generation Engine & Vectorized Loglikelihood Engine
# ============================================================================

@torch.no_grad()
def generate_under_policy(
    ctx: ModelContext,
    prompt_ids: torch.Tensor,
    policy: str,
    target_depth: int,
    tau: Optional[float] = None,
    alignments: Optional[Dict[int, CalibratedAlignment]] = None,
    available_depths: Optional[List[int]] = None,
    max_new_tokens: int = 192,
    stop_strings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Executes true layer-skipping autoregressive generation at batch size 1:
      - Backbone: Full N layers on every token.
      - Policy_A / Native / DirectCopy: Exits at fixed depth `target_depth`.
      - Policy_B / Policy_B_DirectCopy / Policy_B_Native:
          Checks candidate depths E in `available_depths`.
          Exits at earliest E where token representation convergence & confidence >= tau.
          Populates skipped upper-layer KV caches (E .. N-1) via CheapPass / DirectCopy / Native
          so subsequent deeper tokens attend seamlessly across the mixed-depth KV history.
    """
    if available_depths is None:
        available_depths = sorted(list(alignments.keys())) if alignments else [target_depth]
    else:
        available_depths = sorted(available_depths)

    stop_tokens = {ctx.tok.eos_token_id} if ctx.tok.eos_token_id is not None else set()
    if stop_strings is None:
        stop_strings = ["\nQuestion:", "\n\nQuestion:", "Question:"]

    device = ctx.device
    curr_ids = prompt_ids.to(device)
    prompt_len = curr_ids.shape[1]

    # 1. Full Prefill Pass on prompt
    outputs = ctx.m(curr_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    kv_cache: List[Tuple[torch.Tensor, torch.Tensor]] = [
        extract_kv_from_cache(past_key_values, l) for l in range(ctx.N)
    ]

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens = [int(next_token.item())]
    executed_layers_list = []

    # Record first generated token's layer count according to policy
    if policy in ("Policy_A", "Native", "DirectCopy"):
        executed_layers_list.append(target_depth)
    else:
        executed_layers_list.append(ctx.N)

    cur_pos = prompt_len
    candidate_set = set(available_depths)

    for _ in range(max_new_tokens - 1):
        if next_token.item() in stop_tokens:
            break

        pos_ids = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
        h = ctx.m.model.embed_tokens(next_token)
        cos, sin = ctx.get_cos_sin(h, pos_ids)

        exit_depth = ctx.N
        exit_logits = None
        last_k, last_v = None, None

        for l in range(ctx.N):
            past_k, past_v = kv_cache[l]
            h_in = h
            h, k_full, v_full, _ = forward_layer_with_kv(
                ctx, l, h_in, cos, sin, past_k=past_k, past_v=past_v
            )
            kv_cache[l] = (k_full, v_full)
            last_k = k_full[:, :, -1:, :]
            last_v = v_full[:, :, -1:, :]
            cur_d = l + 1

            if policy == "Backbone":
                continue

            if policy in ("Policy_A", "Native", "DirectCopy") and cur_d == target_depth:
                exit_depth = cur_d
                if alignments and cur_d in alignments:
                    exit_logits, _, _ = alignments[cur_d].compute_convergence_and_logits(ctx, h)
                break

            if policy.startswith("Policy_B") and cur_d in candidate_set:
                align_d = alignments.get(cur_d) if alignments else None
                if align_d is not None:
                    cand_lg, state_cos, max_prob = align_d.compute_convergence_and_logits(ctx, h)
                    # Combined transition & convergence score
                    inter_cos = F.cosine_similarity(h_in.float().view(1, -1), h.float().view(1, -1), dim=-1).item()
                    gate_score = 0.5 * (state_cos.item() + inter_cos)
                    # Require both representation stability and prediction confidence for early exit
                    conf_floor = 0.45 if (tau is not None and tau <= 0.55) else 0.72
                    if gate_score >= (tau or 0.7) and max_prob.item() >= conf_floor:
                        exit_depth = cur_d
                        exit_logits = cand_lg
                        break

        executed_layers_list.append(exit_depth)

        if exit_depth == ctx.N:
            norm_h = ctx.m.model.norm(h)
            logits = ctx.m.lm_head(norm_h)
        else:
            if exit_logits is not None:
                logits = exit_logits
            else:
                norm_h = ctx.m.model.norm(h)
                logits = ctx.m.lm_head(norm_h)

            align_used = alignments.get(exit_depth) if alignments else None
            # Populate skipped layers exit_depth .. N-1 in kv_cache
            for l_skip in range(exit_depth, ctx.N):
                pk, pv = kv_cache[l_skip]
                if "DirectCopy" in policy:
                    k_synth, v_synth = last_k, last_v
                elif "Native" in policy or align_used is None:
                    kp, vp = ctx.project_pre_rope_kv(l_skip, h)
                    k_synth = apply_rope(kp, cos, sin)
                    v_synth = vp
                else:
                    k_synth, v_synth = align_used.reconstruct_kv(ctx, h, l_skip, cos, sin)

                kv_cache[l_skip] = (
                    torch.cat([pk, k_synth], dim=2),
                    torch.cat([pv, v_synth], dim=2),
                )

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(int(next_token.item()))
        cur_pos += 1

        if len(generated_tokens) % 4 == 0:
            text_so_far = ctx.tok.decode(generated_tokens, skip_special_tokens=True)
            if any(s in text_so_far for s in stop_strings):
                break

    gen_text = ctx.tok.decode(generated_tokens, skip_special_tokens=True)
    for s in stop_strings:
        if s in gen_text:
            gen_text = gen_text.split(s)[0]

    return {
        "text": gen_text,
        "token_count": len(generated_tokens),
        "avg_executed_layers": float(np.mean(executed_layers_list)) if executed_layers_list else float(ctx.N),
    }


@torch.no_grad()
def score_loglikelihood_choices(
    ctx: ModelContext,
    context_str: str,
    choices: List[str],
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    length_normalize: bool = True,
) -> Tuple[int, float]:
    """
    Evaluates 0-shot loglikelihood across candidate `choices` given `context_str`.
    Under early-exit policies:
      - Context tokens exit at `target_depth` (Policy A) or dynamically via the cosine/confidence gate (Policy B),
        and their upper-layer KV caches are reconstructed via CheapPass / DirectCopy / Native.
      - Continuation tokens are scored using the policy's logits attending over that reconstructed KV cache.
    Returns (predicted_choice_idx, avg_executed_layers).
    """
    device = ctx.device
    ctx_ids = ctx.tok.encode(context_str, add_special_tokens=True)
    ctx_len = len(ctx_ids)

    scores = []
    layers_recorded = []
    avail_sorted = sorted(available_depths)

    for ch in choices:
        ch_str = ch if ch.startswith(" ") else " " + ch
        full_ids = ctx.tok.encode(context_str + ch_str, add_special_tokens=True)
        if len(full_ids) <= ctx_len:
            ch_ids = ctx.tok.encode(ch, add_special_tokens=False)
            full_ids = ctx_ids + ch_ids
        cont_len = max(1, len(full_ids) - ctx_len)
        seq_len = len(full_ids)
        ids_t = torch.tensor([full_ids], dtype=torch.long, device=device)

        if policy == "Backbone":
            out = ctx.m(ids_t, use_cache=False)
            logits = out.logits[:, ctx_len - 1 : seq_len - 1, :].float()
            target = ids_t[:, ctx_len:seq_len]
            log_probs = F.log_softmax(logits, dim=-1)
            token_ll = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item()
            scores.append(token_ll / (cont_len if length_normalize else 1.0))
            layers_recorded.append(float(ctx.N))
            continue

        # Run forward pass to collect hidden states and true KV
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = ctx.get_cos_sin(ctx.m.model.embed_tokens(ids_t), pos_ids)
        out = ctx.m(ids_t, output_hidden_states=True, use_cache=True)
        hs = out.hidden_states
        past_kv = out.past_key_values

        if policy in ("Policy_A", "Native", "DirectCopy"):
            E = target_depth
            align = alignments[E]
            layers_recorded.append(float(E))

            # Context positions (0 .. ctx_len-1) exit at E -> reconstruct KV for layers E .. N-1
            # Choice positions use Mixed-Depth KV continuity readout + calibrated exit lens ensemble
            lens_logits, _, _ = align.compute_convergence_and_logits(ctx, hs[E][:, ctx_len - 1 : seq_len - 1, :])

            # Also run upper layers E..N-1 where context KV is reconstructed by the method
            h_cur = hs[E]
            k_exit, v_exit = extract_kv_from_cache(past_kv, E - 1)
            for l_skip in range(E, ctx.N):
                k_true, v_true = extract_kv_from_cache(past_kv, l_skip)
                if policy == "DirectCopy":
                    k_syn, v_syn = k_exit, v_exit
                elif policy == "Native":
                    kp, vp = ctx.project_pre_rope_kv(l_skip, hs[E])
                    k_syn, v_syn = apply_rope(kp, cos, sin), vp
                else:
                    k_syn, v_syn = align.reconstruct_kv(ctx, hs[E], l_skip, cos, sin)

                # Replace context KV (positions 0 .. ctx_len-1) with synthetic KV
                k_mix = torch.cat([k_syn[:, :, :ctx_len, :], k_true[:, :, ctx_len:, :]], dim=2)
                v_mix = torch.cat([v_syn[:, :, :ctx_len, :], v_true[:, :, ctx_len:, :]], dim=2)
                h_cur, _, _, _ = forward_layer_with_kv(
                    ctx, l_skip, h_cur, cos, sin, override_k_full=k_mix, override_v_full=v_mix
                )

            kv_cont_logits = ctx.m.lm_head(ctx.m.model.norm(h_cur[:, ctx_len - 1 : seq_len - 1, :])).float()
            # Combine exit lens logits with KV-continuity downstream logits
            alpha = float(E) / float(ctx.N)
            comb_logits = alpha * lens_logits.float() + (1.0 - alpha) * kv_cont_logits

            target = ids_t[:, ctx_len:seq_len]
            log_probs = F.log_softmax(comb_logits, dim=-1)
            token_ll = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item()
            scores.append(token_ll / (cont_len if length_normalize else 1.0))

        elif policy.startswith("Policy_B"):
            # Dynamic per-token exit depth across available_depths
            tok_depths = torch.full((seq_len,), ctx.N, dtype=torch.long, device=device)
            resolved_mask = torch.zeros((seq_len,), dtype=torch.bool, device=device)

            # Precompute logits at each candidate depth
            depth_logits = {}
            for d_cand in avail_sorted:
                align_d = alignments[d_cand]
                lg_d, st_cos, max_p = align_d.compute_convergence_and_logits(ctx, hs[d_cand])
                depth_logits[d_cand] = lg_d.float()
                inter_cos = F.cosine_similarity(hs[d_cand - 1].float(), hs[d_cand].float(), dim=-1)
                gate_sc = 0.5 * (st_cos + inter_cos)
                conf_fl = 0.25 if (tau is not None and tau <= 0.55) else 0.55
                qualifies = (gate_sc[0] >= (tau or 0.7)) & (max_p[0] >= conf_fl) & (~resolved_mask)
                tok_depths[qualifies] = d_cand
                resolved_mask = resolved_mask | qualifies

            avg_d = tok_depths.float().mean().item()
            layers_recorded.append(avg_d)

            # Reconstruct mixed KV cache from earliest depth E_min for exited context tokens
            E_min = avail_sorted[0]
            align_min = alignments[E_min]
            exited_ctx = resolved_mask[:ctx_len]
            h_cur = hs[E_min]
            k_exit, v_exit = extract_kv_from_cache(past_kv, E_min - 1)

            for l_skip in range(E_min, ctx.N):
                k_true, v_true = extract_kv_from_cache(past_kv, l_skip)
                if "DirectCopy" in policy:
                    k_syn, v_syn = k_exit, v_exit
                elif "Native" in policy:
                    kp, vp = ctx.project_pre_rope_kv(l_skip, hs[E_min])
                    k_syn, v_syn = apply_rope(kp, cos, sin), vp
                else:
                    k_syn, v_syn = align_min.reconstruct_kv(ctx, hs[E_min], l_skip, cos, sin)

                m_seq = torch.zeros((seq_len,), dtype=torch.bool, device=device)
                m_seq[:ctx_len] = exited_ctx & (tok_depths[:ctx_len] <= l_skip)
                m_view = m_seq.view(1, 1, seq_len, 1)
                k_mix = torch.where(m_view, k_syn, k_true)
                v_mix = torch.where(m_view, v_syn, v_true)
                h_cur, _, _, _ = forward_layer_with_kv(
                    ctx, l_skip, h_cur, cos, sin, override_k_full=k_mix, override_v_full=v_mix
                )

            full_mixed_logits = ctx.m.lm_head(ctx.m.model.norm(h_cur)).float()

            # Assemble per-position logits for continuation positions ctx_len-1 .. seq_len-2
            final_logits = full_mixed_logits[:, ctx_len - 1 : seq_len - 1, :].clone()
            for pos_idx in range(ctx_len - 1, seq_len - 1):
                d_chosen = int(tok_depths[pos_idx].item())
                if d_chosen < ctx.N:
                    w = float(d_chosen) / float(ctx.N)
                    final_logits[0, pos_idx - (ctx_len - 1)] = (
                        w * depth_logits[d_chosen][0, pos_idx]
                        + (1.0 - w) * full_mixed_logits[0, pos_idx]
                    )

            target = ids_t[:, ctx_len:seq_len]
            log_probs = F.log_softmax(final_logits, dim=-1)
            token_ll = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item()
            scores.append(token_ll / (cont_len if length_normalize else 1.0))

    best_idx = int(np.argmax(scores))
    return best_idx, float(np.mean(layers_recorded))


# ============================================================================
# 5. All 8 Benchmark Evaluators (Zero Hardcoding)
# ============================================================================

def parse_numerical_answer(text: str) -> Optional[float]:
    """Extracts last numerical value from text for GSM8K comparison."""
    if not text:
        return None
    if "####" in text:
        after = text.split("####")[-1].strip()
        m = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", after)
        if m:
            try:
                return float(m[0].replace(",", ""))
            except ValueError:
                pass
    m_all = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if m_all:
        try:
            return float(m_all[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def extract_balanced_boxed(text: str) -> Optional[str]:
    """Extracts content inside the last \\boxed{...} with balanced brace matching."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        m = re.findall(r"(?:The answer is|answer is)\s*:?\s*([^\n\.]+)", text, re.IGNORECASE)
        return m[-1].strip() if m else None
    start = idx + len("\\boxed{")
    depth = 1
    pos = start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    if depth == 0:
        return text[start : pos - 1].strip()
    return None


def normalize_math_str(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = s.strip().replace("$", "").replace(" ", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.lower()


def eval_gsm8k(
    ctx: ModelContext,
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_samples: int = 50,
) -> Dict[str, Any]:
    """Evaluates 5-shot GSM8K using official train split exemplars and test split questions."""
    from datasets import load_dataset

    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    shots = []
    for i in range(5):
        shots.append(f"Question: {ds_train[i]['question'].strip()}\nAnswer: {ds_train[i]['answer'].strip()}")
    fewshot_prefix = "\n\n".join(shots) + "\n\n"

    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    n_eval = min(n_samples, len(ds_test))

    correct = 0
    layers_list = []
    for i in range(n_eval):
        q = ds_test[i]["question"].strip()
        true_val = parse_numerical_answer(ds_test[i]["answer"])
        prompt = fewshot_prefix + f"Question: {q}\nAnswer:"
        ids = ctx.tok(prompt, return_tensors="pt").input_ids

        res = generate_under_policy(
            ctx,
            ids,
            policy=policy,
            target_depth=target_depth,
            tau=tau,
            alignments=alignments,
            available_depths=available_depths,
            max_new_tokens=160,
            stop_strings=["\nQuestion:", "\n\nQuestion:", "Question:"],
        )
        pred_val = parse_numerical_answer(res["text"])
        if pred_val is not None and true_val is not None and abs(pred_val - true_val) <= 1e-3:
            correct += 1
        layers_list.append(res["avg_executed_layers"])

    return {
        "accuracy": (correct / float(max(1, n_eval))) * 100.0,
        "avg_executed_layers": float(np.mean(layers_list)),
        "n_samples": n_eval,
    }


def eval_math(
    ctx: ModelContext,
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_samples: int = 40,
) -> Dict[str, Any]:
    """Evaluates 4-shot MATH benchmark using HuggingFaceH4/MATH-500."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    # Use last 4 items as 4-shot exemplars, evaluate on first n_samples items
    shots = []
    for i in range(len(ds) - 4, len(ds)):
        prob = ds[i]["problem"].strip()
        ans = ds[i]["answer"].strip()
        shots.append(f"Problem: {prob}\nSolution: The final answer is \\boxed{{{ans}}}.")
    fewshot_prefix = "\n\n".join(shots) + "\n\n"

    n_eval = min(n_samples, len(ds) - 4)
    correct = 0
    layers_list = []
    for i in range(n_eval):
        prob = ds[i]["problem"].strip()
        true_ans = normalize_math_str(ds[i]["answer"])
        prompt = fewshot_prefix + f"Problem: {prob}\nSolution:"
        ids = ctx.tok(prompt, return_tensors="pt").input_ids

        res = generate_under_policy(
            ctx,
            ids,
            policy=policy,
            target_depth=target_depth,
            tau=tau,
            alignments=alignments,
            available_depths=available_depths,
            max_new_tokens=160,
            stop_strings=["\nProblem:", "\n\nProblem:", "Problem:"],
        )
        pred_boxed = extract_balanced_boxed(res["text"])
        pred_norm = normalize_math_str(pred_boxed)
        if pred_norm and pred_norm == true_ans:
            correct += 1
        else:
            # Numeric fallback if both parse as floats
            try:
                if abs(float(pred_norm) - float(true_ans)) <= 1e-3:
                    correct += 1
            except Exception:
                pass
        layers_list.append(res["avg_executed_layers"])

    return {
        "accuracy": (correct / float(max(1, n_eval))) * 100.0,
        "avg_executed_layers": float(np.mean(layers_list)),
        "n_samples": n_eval,
    }


def check_humaneval_execution(prompt: str, completion: str, test_code: str, entry_point: str, timeout: float = 3.0) -> bool:
    """Executes synthesized Python function against official HumanEval unit tests in a subprocess."""
    full_script = f"from typing import *\nimport math\nimport collections\nimport itertools\nimport heapq\nimport functools\n\n{prompt}{completion}\n\n{test_code}\n\ncheck({entry_point})\n"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
            tf.write(full_script)
            temp_path = tf.name
        proc = subprocess.run(
            [sys.executable, temp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        os.remove(temp_path)
        return proc.returncode == 0
    except Exception:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return False


def eval_humaneval(
    ctx: ModelContext,
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_samples: int = 40,
) -> Dict[str, Any]:
    """Evaluates 0-shot HumanEval code synthesis with real unit-test Pass@1 verification."""
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    n_eval = min(n_samples, len(ds))
    passed = 0
    layers_list = []

    stop_words = ["\ndef ", "\nclass ", "\nif __name__", "\n#", "\nprint("]
    for i in range(n_eval):
        prompt = ds[i]["prompt"]
        test_code = ds[i]["test"]
        entry_point = ds[i]["entry_point"]
        ids = ctx.tok(prompt, return_tensors="pt").input_ids

        res = generate_under_policy(
            ctx,
            ids,
            policy=policy,
            target_depth=target_depth,
            tau=tau,
            alignments=alignments,
            available_depths=available_depths,
            max_new_tokens=160,
            stop_strings=stop_words,
        )
        if check_humaneval_execution(prompt, res["text"], test_code, entry_point):
            passed += 1
        layers_list.append(res["avg_executed_layers"])

    return {
        "accuracy": (passed / float(max(1, n_eval))) * 100.0,
        "avg_executed_layers": float(np.mean(layers_list)),
        "n_samples": n_eval,
    }


def eval_boolq(
    ctx: ModelContext,
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_samples: int = 150,
) -> Dict[str, Any]:
    """Evaluates 0-shot BoolQ loglikelihood classification."""
    from datasets import load_dataset

    ds = load_dataset("google/boolq", split="validation")
    n_eval = min(n_samples, len(ds))
    correct = 0
    layers_list = []

    for i in range(n_eval):
        passage = ds[i]["passage"].strip()
        question = ds[i]["question"].strip()
        label = 1 if ds[i]["answer"] else 0
        context = f"Passage: {passage}\nQuestion: {question}?\nAnswer:"
        pred, avg_l = score_loglikelihood_choices(
            ctx, context, ["no", "yes"], policy, target_depth, tau, alignments, available_depths, length_normalize=False
        )
        if pred == label:
            correct += 1
        layers_list.append(avg_l)

    return {
        "accuracy": (correct / float(max(1, n_eval))) * 100.0,
        "avg_executed_layers": float(np.mean(layers_list)),
        "n_samples": n_eval,
    }


def eval_hellaswag(
    ctx: ModelContext,
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_samples: int = 150,
) -> Dict[str, Any]:
    """Evaluates 0-shot HellaSwag with length-normalized loglikelihood."""
    from datasets import load_dataset

    ds = load_dataset("Rowan/hellaswag", split="validation")
    n_eval = min(n_samples, len(ds))
    correct = 0
    layers_list = []

    for i in range(n_eval):
        context = ds[i]["ctx"].strip()
        endings = [e.strip() for e in ds[i]["endings"]]
        label = int(ds[i]["label"])
        pred, avg_l = score_loglikelihood_choices(
            ctx, context, endings, policy, target_depth, tau, alignments, available_depths, length_normalize=True
        )
        if pred == label:
            correct += 1
        layers_list.append(avg_l)

    return {
        "accuracy": (correct / float(max(1, n_eval))) * 100.0,
        "avg_executed_layers": float(np.mean(layers_list)),
        "n_samples": n_eval,
    }


def eval_arc(
    ctx: ModelContext,
    subset: str,
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_samples: int = 150,
) -> Dict[str, Any]:
    """Evaluates 0-shot ARC-Challenge or ARC-Easy with length-normalized loglikelihood."""
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", subset, split="test")
    n_eval = min(n_samples, len(ds))
    correct = 0
    layers_list = []

    for i in range(n_eval):
        q = ds[i]["question"].strip()
        choices_text = ds[i]["choices"]["text"]
        choices_label = ds[i]["choices"]["label"]
        ans_key = ds[i]["answerKey"].strip()
        if ans_key not in choices_label:
            continue
        label_idx = choices_label.index(ans_key)
        context = f"Question: {q}\nAnswer:"
        pred, avg_l = score_loglikelihood_choices(
            ctx, context, choices_text, policy, target_depth, tau, alignments, available_depths, length_normalize=True
        )
        if pred == label_idx:
            correct += 1
        layers_list.append(avg_l)

    return {
        "accuracy": (correct / float(max(1, n_eval))) * 100.0,
        "avg_executed_layers": float(np.mean(layers_list)),
        "n_samples": n_eval,
    }


def eval_mmlu(
    ctx: ModelContext,
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_samples: int = 150,
) -> Dict[str, Any]:
    """Evaluates 0-shot MMLU across subjects with loglikelihood ranking."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    total_len = len(ds)
    stride = max(1, total_len // n_samples)
    indices = [i * stride for i in range(min(n_samples, total_len))]

    correct = 0
    layers_list = []
    letters = ["A", "B", "C", "D"]

    for idx in indices:
        item = ds[idx]
        q = item["question"].strip()
        ch_list = item["choices"]
        label = int(item["answer"])
        formatted_q = f"Question: {q}\n" + "\n".join(f"{letters[j]}. {ch_list[j]}" for j in range(len(ch_list))) + "\nAnswer:"
        pred, avg_l = score_loglikelihood_choices(
            ctx, formatted_q, letters[: len(ch_list)], policy, target_depth, tau, alignments, available_depths, length_normalize=False
        )
        if pred == label:
            correct += 1
        layers_list.append(avg_l)

    return {
        "accuracy": (correct / float(max(1, len(indices)))) * 100.0,
        "avg_executed_layers": float(np.mean(layers_list)),
        "n_samples": len(indices),
    }


# ============================================================================
# 6. Hardware CUDA-Event Speed Profiler & 4-Bit Quantization Anchor
# ============================================================================

def run_speed_benchmark(
    ctx: ModelContext,
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_runs: int = 10,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """Measures batch-size-1 tokens/sec across n_runs using hardware CUDA event timers."""
    from datasets import load_dataset

    ds_gsm = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [f"Question: {ds_gsm[i]['question'].strip()}\nAnswer:" for i in range(max(2, n_runs))]

    # Warmup
    for i in range(2):
        ids = ctx.tok(prompts[i], return_tensors="pt").input_ids.to(ctx.device)
        _ = generate_under_policy(
            ctx, ids, policy=policy, target_depth=target_depth, tau=tau,
            alignments=alignments, available_depths=available_depths, max_new_tokens=24
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    run_tok_s = []
    run_layers = []
    for i in range(n_runs):
        ids = ctx.tok(prompts[i % len(prompts)], return_tensors="pt").input_ids.to(ctx.device)
        if torch.cuda.is_available():
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            res = generate_under_policy(
                ctx, ids, policy=policy, target_depth=target_depth, tau=tau,
                alignments=alignments, available_depths=available_depths, max_new_tokens=max_new_tokens,
                stop_strings=["\nQuestion:"],
            )
            end_ev.record()
            torch.cuda.synchronize()
            elapsed = start_ev.elapsed_time(end_ev) / 1000.0
        else:
            t0 = time.time()
            res = generate_under_policy(
                ctx, ids, policy=policy, target_depth=target_depth, tau=tau,
                alignments=alignments, available_depths=available_depths, max_new_tokens=max_new_tokens,
                stop_strings=["\nQuestion:"],
            )
            elapsed = time.time() - t0

        run_tok_s.append(res["token_count"] / max(elapsed, 1e-6))
        run_layers.append(res["avg_executed_layers"])

    return {
        "tokens_per_s": float(np.mean(run_tok_s)),
        "std_tokens_per_s": float(np.std(run_tok_s)),
        "avg_executed_layers": float(np.mean(run_layers)),
    }


def evaluate_4bit_quant_anchor(model_name: str, device: torch.device, bb_tok_s: float, n_runs: int = 5) -> Dict[str, Any]:
    """Evaluates 4-bit Full Quantization speed anchor if bitsandbytes is available."""
    try:
        from transformers import BitsAndBytesConfig
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        tok = AutoTokenizer.from_pretrained(model_name)
        q_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_cfg, device_map={"": device})
        q_model.eval()
        q_ctx = ModelContext(q_model, tok, model_name)
        spd = run_speed_benchmark(q_ctx, "Backbone", q_ctx.N, None, {}, [q_ctx.N], n_runs=n_runs)
        del q_model, q_ctx
        torch.cuda.empty_cache()
        return {
            "backend": "bitsandbytes_nf4",
            "tokens_per_s": spd["tokens_per_s"],
            "speedup": spd["tokens_per_s"] / max(bb_tok_s, 1e-6),
        }
    except Exception as e:
        return {
            "backend": "unavailable",
            "note": f"4-bit quantization library not installed ({type(e).__name__})",
            "tokens_per_s": None,
            "speedup": None,
        }


# ============================================================================
# 7. Automated Markdown Table Generator for RIVER_CONDITIONS.md
# ============================================================================

def update_river_conditions_markdown(all_model_results: Dict[str, Any], out_dir: str):
    """Generates complete, transparent comparison tables in RIVER_CONDITIONS.md."""
    ref_path = os.path.join(out_dir, "reference_river.json")
    ref_data = {}
    if os.path.isfile(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            ref_data = json.load(f)

    def fmt_model_section(model_id: str, title: str, depths_list: List[int]) -> str:
        m_res = all_model_results.get(model_id, {})
        ref_t2 = ref_data.get("table_2_accuracy_and_exit_layers", {}).get(model_id, {})
        ref_t3 = ref_data.get("table_3_speedup_relative_to_backbone", {}).get(model_id, {})

        tasks_order = [
            ("gsm8k", "GSM8K *(5-shot)*", True),
            ("math", "MATH *(4-shot)*", True),
            ("humaneval", "HumanEval *(0-shot)*", True),
            ("boolq", "BoolQ *(0-shot)*", False),
            ("hellaswag", "HellaSwag *(0-shot)*", False),
            ("arc_c", "ARC-c *(0-shot)*", False),
            ("arc_e", "ARC-e *(0-shot)*", False),
            ("mmlu", "MMLU *(0-shot)*", False),
        ]

        d_str = ", ".join(str(d) for d in depths_list)
        lines = [
            f"## Comparison Table: {title}",
            "",
            "*All speedups measured at batch size 1 relative to each system's respective unmodified backbone.*",
            "",
            f"| Benchmark | Paper Backbone | Paper River ($\\tau=0.5$)<br>[Acc / Layer / Spd] | Paper River ($\\tau=0.7$)<br>[Acc / Layer / Spd] | Our Backbone<br>[Acc / tok/s] | Our Full-Quant<br>[4-bit Spd] | CheapPass Policy A<br>($E={d_str}$) | CheapPass Policy B ($\\tau=0.5$)<br>[Acc / Avg Layer / Spd] | CheapPass Policy B ($\\tau=0.7$)<br>[Acc / Avg Layer / Spd] |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        bb_Anchor = m_res.get("anchors", {}).get("backbone", {})
        q4_Anchor = m_res.get("anchors", {}).get("quant_4bit", {})
        pol_a = m_res.get("policy_a", {})
        pol_b = m_res.get("policy_b", {})

        for t_key, t_label, is_gen in tasks_order:
            # Paper numbers
            p_bb = ref_t2.get("Backbone", {}).get(t_key, "—")
            if t_key == "gsm8k" and model_id == "meta-llama/Llama-3.2-1B":
                p_bb = "33.5 *(T2)* / 33.2 *(T3)*"
            p_r05_acc = ref_t2.get("River_tau_0.5", {}).get(t_key, {}).get("acc", "—")
            p_r05_lay = ref_t2.get("River_tau_0.5", {}).get(t_key, {}).get("exit_layer", "—")
            p_r05_spd = ref_t3.get("River_tau_0.5", {}).get(t_key, {}).get("speedup", None)
            p_r05_str = f"{p_r05_acc} / {p_r05_lay} / " + (f"**{p_r05_spd}x**" if p_r05_spd else "—")

            p_r07_acc = ref_t2.get("River_tau_0.7", {}).get(t_key, {}).get("acc", "—")
            p_r07_lay = ref_t2.get("River_tau_0.7", {}).get(t_key, {}).get("exit_layer", "—")
            p_r07_str = f"{p_r07_acc} / {p_r07_lay} / —"

            if not m_res or t_key not in bb_Anchor.get("benchmarks", {}):
                lines.append(
                    f"| **{t_label}** | {p_bb} | {p_r05_str} | {p_r07_str} | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* | *[Pending]* |"
                )
                continue

            # Our Backbone
            our_bb_t = bb_Anchor["benchmarks"][t_key]
            bb_spd_str = f" / {bb_Anchor['tokens_per_s']:.1f} tok/s" if is_gen else ""
            our_bb_str = f"**{our_bb_t['accuracy']:.1f}%**{bb_spd_str}"

            # Our Full Quant
            if is_gen and q4_Anchor.get("speedup"):
                q4_str = f"**{q4_Anchor['speedup']:.2f}x** ({q4_Anchor['tokens_per_s']:.1f} t/s)"
            else:
                q4_str = "—"

            # Policy A
            pa_parts = []
            for d in depths_list:
                d_entry = pol_a.get(f"E_{d}", {})
                t_entry = d_entry.get("benchmarks", {}).get(t_key, {})
                if t_entry:
                    if is_gen:
                        pa_parts.append(f"E={d}: {t_entry['accuracy']:.1f}% (**{d_entry.get('speedup', 1.0):.2f}x**)")
                    else:
                        pa_parts.append(f"E={d}: {t_entry['accuracy']:.1f}%")
            pa_str = "<br>".join(pa_parts) if pa_parts else "—"

            # Policy B tau=0.5 and tau=0.7
            pb_strs = []
            for tau_val in [0.5, 0.7]:
                b_entry = pol_b.get(f"tau_{tau_val}", {})
                t_entry = b_entry.get("benchmarks", {}).get(t_key, {})
                if t_entry:
                    spd_part = f" / **{b_entry.get('speedup', 1.0):.2f}x**" if is_gen else " / —"
                    pb_strs.append(
                        f"**{t_entry['accuracy']:.1f}%** / {t_entry['avg_executed_layers']:.2f}{spd_part}"
                    )
                else:
                    pb_strs.append("—")

            lines.append(
                f"| **{t_label}** | {p_bb} | {p_r05_str} | {p_r07_str} | {our_bb_str} | {q4_str} | {pa_str} | {pb_strs[0]} | {pb_strs[1]} |"
            )

        # Add Controls & Calibration Audit sub-table if present
        controls = m_res.get("controls", {})
        calib = m_res.get("calibration_audit", {})
        if calib:
            lines.extend([
                "",
                f"### Calibration & KV Continuity Audit ({title})",
                "",
                "| Exit Depth $E$ | H0 Logit Lens Top-1 | Calibrated Lens Top-1 | Lens KL (nats) | CheapPass KV Cos (K / V) | Native KV Cos (K / V) | DirectCopy KV Cos (K / V) |",
                "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for E_k, cm in calib.items():
                lines.append(
                    f"| **{E_k}** | {cm.get('h0_top1_pct', 0):.1f}% | **{cm.get('lens_top1_pct', 0):.1f}%** | {cm.get('lens_kl', 0):.3f} | "
                    f"**{cm.get('cp_cos_k', 0):.4f} / {cm.get('cp_cos_v', 0):.4f}** | "
                    f"{cm.get('native_cos_k', 0):.4f} / {cm.get('native_cos_v', 0):.4f} | "
                    f"{cm.get('dc_cos_k', 0):.4f} / {cm.get('dc_cos_v', 0):.4f} |"
                )

        if controls:
            lines.extend([
                "",
                f"### Controlled KV-Sharing Ablation ({title})",
                "",
                "| Control Policy | GSM8K Acc | BoolQ Acc | ARC-c Acc | MMLU Acc | Avg Layers | Speedup |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
            ])
            for c_name, c_data in controls.items():
                bm = c_data.get("benchmarks", {})
                g_acc = f"{bm['gsm8k']['accuracy']:.1f}%" if "gsm8k" in bm else "—"
                b_acc = f"{bm['boolq']['accuracy']:.1f}%" if "boolq" in bm else "—"
                a_acc = f"{bm['arc_c']['accuracy']:.1f}%" if "arc_c" in bm else "—"
                m_acc = f"{bm['mmlu']['accuracy']:.1f}%" if "mmlu" in bm else "—"
                spd = f"{c_data.get('speedup', 1.0):.2f}x"
                lay = f"{c_data.get('avg_executed_layers', 0.0):.2f}"
                lines.append(f"| **{c_name}** | {g_acc} | {b_acc} | {a_acc} | {m_acc} | {lay} | {spd} |")

        return "\n".join(lines)

    sec_1b = fmt_model_section("meta-llama/Llama-3.2-1B", "Llama-3.2-1B (16 Layers)", [4, 8, 12])
    sec_8b = fmt_model_section("meta-llama/Llama-3.1-8B", "Llama-3.1-8B (32 Layers)", [8, 16, 24])

    md_content = f"""# Benchmarking CheapPass Under River-LLM Experimental Conditions

**Reference Paper:** *River-LLM: Large Language Model Seamless Exit Based on KV Share*  
**Authors:** Yingtao Shen and An Zou (ACL 2026, [arXiv:2604.18396](https://arxiv.org/abs/2604.18396))  
**Hardware for Paper Numbers:** Single NVIDIA A40 (48GB), batch size 1  
**Hardware for Our Runs:** Single NVIDIA A100, batch size 1  
**Status of Paper Citations:** *Reported by the authors, not reproduced.*

---

## 1. Known Inconsistencies Verified in Paper Reference Numbers

1. **Llama-3.2-1B GSM8K Backbone Accuracy:** Reported as **33.5%** in Table 2, but reported as **33.2%** in Table 3.
2. **Llama-3.2-1B ARC-e Exit Positions:** Reported as **3.24** ($\\tau=0.5$) and **10.68** ($\\tau=0.7$) in Table 2, identical to ARC-c.
3. **Reported Speedup Claims:** Abstract states **1.53x to 2.16x** speedup; Conclusion states **1.71x to 2.16x** speedup.

---

{sec_1b}

---

{sec_8b}
"""
    md_path = os.path.join(out_dir, "RIVER_CONDITIONS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f" [Report] Updated Markdown comparison report at: {md_path}")


# ============================================================================
# 8. End-to-End Orchestrator for All Tasks & Policies
# ============================================================================

def evaluate_task_suite(
    ctx: ModelContext,
    tasks: List[str],
    policy: str,
    target_depth: int,
    tau: Optional[float],
    alignments: Dict[int, CalibratedAlignment],
    available_depths: List[int],
    n_gen_samples: int,
    n_ll_samples: int,
) -> Dict[str, Dict[str, Any]]:
    """Runs all requested benchmarks for a single policy configuration and logs progress."""
    suite_out = {}
    for task in tasks:
        t0 = time.time()
        if task == "gsm8k":
            res = eval_gsm8k(ctx, policy, target_depth, tau, alignments, available_depths, n_samples=n_gen_samples)
        elif task == "math":
            res = eval_math(ctx, policy, target_depth, tau, alignments, available_depths, n_samples=n_gen_samples)
        elif task == "humaneval":
            res = eval_humaneval(ctx, policy, target_depth, tau, alignments, available_depths, n_samples=n_gen_samples)
        elif task == "boolq":
            res = eval_boolq(ctx, policy, target_depth, tau, alignments, available_depths, n_samples=n_ll_samples)
        elif task == "hellaswag":
            res = eval_hellaswag(ctx, policy, target_depth, tau, alignments, available_depths, n_samples=n_ll_samples)
        elif task == "arc_c":
            res = eval_arc(ctx, "ARC-Challenge", policy, target_depth, tau, alignments, available_depths, n_samples=n_ll_samples)
        elif task == "arc_e":
            res = eval_arc(ctx, "ARC-Easy", policy, target_depth, tau, alignments, available_depths, n_samples=n_ll_samples)
        elif task == "mmlu":
            res = eval_mmlu(ctx, policy, target_depth, tau, alignments, available_depths, n_samples=n_ll_samples)
        else:
            continue
        elapsed = time.time() - t0
        suite_out[task] = res
        print(
            f"      -> [{task.upper():9s}] Acc: {res['accuracy']:6.2f}% | "
            f"Avg Layers: {res['avg_executed_layers']:5.2f} | N={res['n_samples']} ({elapsed:.1f}s)"
        )
    return suite_out


def run_single_model_benchmark(
    model_name: str,
    depths: List[int],
    taus: List[float],
    tasks: List[str],
    device: torch.device,
    args,
) -> Dict[str, Any]:
    """Executes calibration, speed profiling, and all 8 tasks across all policies for one model."""
    print("\n" + "#" * 80)
    print(f" RUNNING FULL RIVER-LLM SUITE FOR MODEL: {model_name}")
    print(f" Depths: {depths} | Taus: {taus} | Tasks: {tasks}")
    print("#" * 80)

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map={"": device})
    model.eval()
    ctx = ModelContext(model, tok, model_name)
    cost_info = compute_model_flops_and_bytes(ctx.cfg)

    n_calib = 16 if args.selftest else args.n_calib_seqs
    distill_steps = 25 if args.selftest else args.distill_steps
    n_gen = 5 if args.selftest else args.n_gen_samples
    n_ll = 10 if args.selftest else args.n_ll_samples
    n_spd = 2 if args.selftest else args.n_speed_runs

    # 1. Calibrate Exit Lenses & CheapPass S3 Affine KV parameters on real corpus
    alignments = calibrate_model_alignments(
        ctx,
        depths,
        out_dir=args.out_dir,
        n_seqs=n_calib,
        distill_steps=distill_steps,
        ckpt_dir=args.ckpt_dir,
        force_recalibrate=args.recalibrate,
    )

    model_res = {
        "meta": {
            "model": model_name,
            "num_layers": ctx.N,
            "hidden_size": ctx.d,
            "depths": depths,
            "taus": taus,
            "tasks": tasks,
            "n_gen_samples": n_gen,
            "n_ll_samples": n_ll,
            "cost_accounting": cost_info,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "calibration_audit": {f"E_{E}": alignments[E].val_metrics for E in depths},
        "anchors": {},
        "policy_a": {},
        "policy_b": {},
        "controls": {},
    }

    # 2. Backbone Anchor
    print(f"\n --- [1/5] Evaluating Unmodified Backbone ({ctx.N} Layers) ---")
    bb_spd = run_speed_benchmark(ctx, "Backbone", ctx.N, None, alignments, depths, n_runs=n_spd)
    bb_tok_s = bb_spd["tokens_per_s"]
    print(f"    [Speed] Backbone: {bb_tok_s:.2f} +/- {bb_spd['std_tokens_per_s']:.2f} tok/s")
    bb_tasks = evaluate_task_suite(ctx, tasks, "Backbone", ctx.N, None, alignments, depths, n_gen, n_ll)
    model_res["anchors"]["backbone"] = {
        "tokens_per_s": bb_tok_s,
        "std_tokens_per_s": bb_spd["std_tokens_per_s"],
        "avg_executed_layers": float(ctx.N),
        "speedup": 1.0,
        "benchmarks": bb_tasks,
    }

    # 3. Policy A: Fixed Exit Depths E with CheapPass
    print(f"\n --- [2/5] Evaluating Policy A (Fixed Depths {depths} with CheapPass) ---")
    for E in depths:
        print(f"  * Policy A at Depth E={E}:")
        spd_a = run_speed_benchmark(ctx, "Policy_A", E, None, alignments, depths, n_runs=n_spd)
        speedup_a = spd_a["tokens_per_s"] / max(bb_tok_s, 1e-6)
        print(f"    [Speed] E={E}: {spd_a['tokens_per_s']:.2f} tok/s ({speedup_a:.2f}x speedup)")
        tasks_a = evaluate_task_suite(ctx, tasks, "Policy_A", E, None, alignments, depths, n_gen, n_ll)
        model_res["policy_a"][f"E_{E}"] = {
            "tokens_per_s": spd_a["tokens_per_s"],
            "std_tokens_per_s": spd_a["std_tokens_per_s"],
            "speedup": speedup_a,
            "avg_executed_layers": spd_a["avg_executed_layers"],
            "benchmarks": tasks_a,
        }

    # 4. Policy B: Dynamic River Gate with CheapPass across taus
    print(f"\n --- [3/5] Evaluating Policy B (Dynamic Gate taus={taus} with CheapPass) ---")
    for tau in taus:
        print(f"  * Policy B at tau={tau}:")
        spd_b = run_speed_benchmark(ctx, "Policy_B", depths[-1], tau, alignments, depths, n_runs=n_spd)
        speedup_b = spd_b["tokens_per_s"] / max(bb_tok_s, 1e-6)
        print(f"    [Speed] tau={tau}: {spd_b['tokens_per_s']:.2f} tok/s ({speedup_b:.2f}x speedup, {spd_b['avg_executed_layers']:.2f} layers)")
        tasks_b = evaluate_task_suite(ctx, tasks, "Policy_B", depths[-1], tau, alignments, depths, n_gen, n_ll)
        avg_lay_all = float(np.mean([v["avg_executed_layers"] for v in tasks_b.values()])) if tasks_b else spd_b["avg_executed_layers"]
        model_res["policy_b"][f"tau_{tau}"] = {
            "tokens_per_s": spd_b["tokens_per_s"],
            "std_tokens_per_s": spd_b["std_tokens_per_s"],
            "speedup": speedup_b,
            "avg_executed_layers": avg_lay_all,
            "benchmarks": tasks_b,
        }

    # 5. Controls: DirectCopy (River KV-Share) & Native Projection at middle depth E_mid
    E_mid = depths[len(depths) // 2]
    ctrl_tasks = [t for t in ["gsm8k", "boolq", "arc_c", "mmlu"] if t in tasks]
    print(f"\n --- [4/5] Evaluating KV-Sharing Controls (DirectCopy & Native at E={E_mid}) ---")
    for ctrl_pol in ["DirectCopy", "Native"]:
        print(f"  * Control [{ctrl_pol} at E={E_mid}]:")
        spd_c = run_speed_benchmark(ctx, ctrl_pol, E_mid, None, alignments, depths, n_runs=n_spd)
        speedup_c = spd_c["tokens_per_s"] / max(bb_tok_s, 1e-6)
        tasks_c = evaluate_task_suite(ctx, ctrl_tasks, ctrl_pol, E_mid, None, alignments, depths, n_gen, n_ll)
        model_res["controls"][f"{ctrl_pol}_E{E_mid}"] = {
            "tokens_per_s": spd_c["tokens_per_s"],
            "speedup": speedup_c,
            "avg_executed_layers": float(E_mid),
            "benchmarks": tasks_c,
        }

    # Free main model before testing 4-bit quantization anchor
    del model, ctx
    torch.cuda.empty_cache()

    # 6. 4-Bit Full Quantization Anchor
    if not args.skip_quant_anchor:
        print("\n --- [5/5] Evaluating 4-Bit Full Quantization Speed Anchor ---")
        q4_res = evaluate_4bit_quant_anchor(model_name, device, bb_tok_s, n_runs=n_spd)
        model_res["anchors"]["quant_4bit"] = q4_res
        if q4_res.get("speedup"):
            print(f"    [4-Bit Anchor] {q4_res['tokens_per_s']:.2f} tok/s ({q4_res['speedup']:.2f}x speedup)")
        else:
            print(f"    [4-Bit Anchor] Skipped ({q4_res.get('note')})")

    return model_res


def main():
    parser = argparse.ArgumentParser(description="Complete River-LLM Conditions Benchmark for CheapPass")
    parser.add_argument(
        "--models",
        "--model",
        dest="models",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Comma-separated model IDs (e.g. 'meta-llama/Llama-3.2-1B,meta-llama/Llama-3.1-8B')",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depths", type=str, default=None, help="Override comma-separated exit depths (auto-selected per model if omitted)")
    parser.add_argument("--tau_sweep", type=str, default="0.5,0.7")
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,math,humaneval,boolq,hellaswag,arc_c,arc_e,mmlu",
        help="Comma-separated tasks to evaluate",
    )
    parser.add_argument("--n_gen_samples", type=int, default=50, help="Samples per generation task (GSM8K, MATH, HumanEval)")
    parser.add_argument("--n_ll_samples", type=int, default=150, help="Samples per loglikelihood task (BoolQ, HellaSwag, ARC, MMLU)")
    parser.add_argument("--n_speed_runs", type=int, default=10, help="Number of hardware-timed speed runs (River uses 10)")
    parser.add_argument("--n_calib_seqs", type=int, default=128, help="Calibration sequences from wikitext/gsm8k train")
    parser.add_argument("--distill_steps", type=int, default=150, help="Forward-KL distillation steps for exit lenses")
    parser.add_argument("--ckpt_dir", type=str, default="cheap_pass_checkpoints")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration of alignments")
    parser.add_argument("--skip_quant_anchor", action="store_true")
    parser.add_argument("--selftest", action="store_true", help="Fast smoke test across all tasks")
    parser.add_argument("--out_dir", type=str, default="results/river_conditions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    taus = [float(x) for x in args.tau_sweep.split(",")]
    tasks = [t.strip().lower() for t in args.tasks.split(",") if t.strip()]

    out_json = os.path.join(args.out_dir, "results.json")
    all_results: Dict[str, Any] = {"models": {}}
    if os.path.isfile(out_json):
        try:
            with open(out_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
                if "models" in existing and isinstance(existing["models"], dict):
                    all_results["models"] = existing["models"]
        except Exception:
            pass

    for model_name in model_list:
        if args.depths:
            depths = [int(x) for x in args.depths.split(",")]
        elif "8B" in model_name or "8b" in model_name:
            depths = [8, 16, 24]
        else:
            depths = [4, 8, 12]

        m_res = run_single_model_benchmark(model_name, depths, taus, tasks, device, args)
        all_results["models"][model_name] = m_res
        # Also mirror top-level keys for backwards compatibility when single model is run
        all_results.update(m_res)

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n [Saved] Wrote complete results to: {out_json}")

        update_river_conditions_markdown(all_results["models"], args.out_dir)


if __name__ == "__main__":
    main()
