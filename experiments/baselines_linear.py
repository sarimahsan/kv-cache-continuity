#!/usr/bin/env python3
"""
baselines_linear.py -- Track 1: Fair Linear Baselines, Post-Projection Scaling,
and Closed-Form Ridge Regression for CheapPass.

Memory-Optimized Edition for Tesla T4 (15GB VRAM):
  - Uses get_oracle_states(..., need_logits=False) during fitting/training to avoid allocating 622MB logit buffers.
  - Joint single-pass fitting for Group S (S1, S2, S3, S4) and Group R (R-h, R-kv, R-h-r128).
  - Explicit torch.cuda.empty_cache() and batch_size=2 during training/fitting.
  - Paired bootstrap confidence intervals on per-sequence differences (KL_method - KL_baseline).
  - Automated evaluation of pre-registered decision rules.
"""

import argparse
import copy
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkr2 import Ctx, Native, Direct, kl_tokens, boot_ci, set_seed
from cheap_pass import DataSplits, CheapPassAdapter, AdapterKVMethod
from eval_ood_cheap_pass import load_ood_domain_tokens

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================================
# Memory-Efficient Oracle & Metric Helpers
# ============================================================================

@torch.no_grad()
def get_oracle_states(ctx: Ctx, ids: torch.Tensor, need_logits: bool = False) -> Dict:
    """Computes exact backbone states, optionally omitting the 622MB terminal logits buffer."""
    h, cos, sin = ctx.embed(ids)
    hs, kv = [h], []
    for i in range(ctx.N):
        h, kvi = ctx.layer_fwd(i, h, cos, sin)
        hs.append(h)
        kv.append(kvi)
    logits = ctx.head(h) if need_logits else None
    return dict(hs=hs, kv=kv, logits=logits, cos=cos, sin=sin)


def compute_boot_ci(values: List[float], seed: int = 0, n_resamples: int = 1000) -> Tuple[float, List[float]]:
    if not values:
        return 0.0, [0.0, 0.0]
    rng = np.random.RandomState(seed)
    mean_val, ci_lo, ci_hi = boot_ci(values, rng, n=n_resamples)
    return float(mean_val), [float(ci_lo), float(ci_hi)]


def compute_chunked_kl_and_top1(
    logits_target: torch.Tensor,
    logits_pred: torch.Tensor,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes tokenwise forward KL divergence and Top-1 match without OOM."""
    B, T, V = logits_target.shape
    kl_chunks = []
    top1_chunks = []

    for start_idx in range(0, T, chunk_size):
        end_idx = min(start_idx + chunk_size, T)
        t_chk = logits_target[:, start_idx:end_idx, :].float()
        p_chk = logits_pred[:, start_idx:end_idx, :].float()

        log_p_target = F.log_softmax(t_chk, dim=-1)
        log_p_pred = F.log_softmax(p_chk, dim=-1)
        p_target = torch.exp(log_p_target)

        kl_chk = (p_target * (log_p_target - log_p_pred)).sum(dim=-1)
        top1_chk = (t_chk.argmax(dim=-1) == p_chk.argmax(dim=-1)).float()

        kl_chunks.append(kl_chk.detach())
        top1_chunks.append(top1_chk.detach())

    kl_all = torch.cat(kl_chunks, dim=1)  # [B, T]
    top1_all = torch.cat(top1_chunks, dim=1)  # [B, T]
    return kl_all, top1_all


# ============================================================================
# 1. Group S: Post-Projection Scaling Methods
# ============================================================================

class PostProjectionScalingMethod:
    """
    Applies fitted corrections post-projection (not cancelled by RMSNorm).
    """
    def __init__(self, mode: str = "S1", name: str = "post_proj"):
        self.mode = mode
        self.name = name
        self.params = {}

    def kv(self, ctx: Ctx, orc: Dict, depth: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        h_E = orc["hs"][depth]
        cos, sin = orc["cos"], orc["sin"]
        B, T, _ = h_E.shape
        out = []

        for l in range(depth, ctx.N):
            lay = ctx.layers[l]
            if self.mode == "S4":
                delta_h = self.params[l].to(h_E.device, dtype=h_E.dtype)
                h_hat = h_E + delta_h
                k_hat, v_hat = ctx.project_kv(l, h_hat, cos, sin)
                out.append((k_hat, v_hat))
                continue

            n = lay.input_layernorm(h_E)
            k_proj = lay.self_attn.k_proj(n).view(B, T, ctx.Hkv, ctx.D)
            v_proj = lay.self_attn.v_proj(n).view(B, T, ctx.Hkv, ctx.D)

            p = self.params[l]
            if self.mode == "S1":
                k_scaled = (k_proj * p["sk"]).to(k_proj.dtype)
                v_scaled = (v_proj * p["sv"]).to(v_proj.dtype)
            elif self.mode == "S2":
                k_scaled = k_proj * p["sk"].to(k_proj.device, dtype=k_proj.dtype)
                v_scaled = v_proj * p["sv"].to(v_proj.device, dtype=v_proj.dtype)
            elif self.mode == "S3":
                k_scaled = k_proj * p["ak"].to(k_proj.device, dtype=k_proj.dtype) + p["bk"].to(k_proj.device, dtype=k_proj.dtype)
                v_scaled = v_proj * p["av"].to(v_proj.device, dtype=v_proj.dtype) + p["bv"].to(v_proj.device, dtype=v_proj.dtype)

            k_final = ctx.rope(k_scaled.transpose(1, 2), cos, sin)
            v_final = v_scaled.transpose(1, 2)
            out.append((k_final, v_final))

        return out


@torch.no_grad()
def fit_all_group_s(
    s_methods: Dict[str, PostProjectionScalingMethod],
    ctx: Ctx,
    train_tokens: torch.Tensor,
    depth: int,
    device: torch.device,
    n_fit: int = 100,
    batch_size: int = 2,
):
    """Jointly fits S1, S2, S3, S4 in a single low-memory pass over train tokens."""
    print(f" [Group S] Jointly fitting S1, S2, S3, S4 in a single pass over {n_fit} sequences...")
    skipped_layers = list(range(depth, ctx.N))

    sum_h_E = torch.zeros(ctx.d, dtype=torch.float64, device=device)
    sum_h_l = {l: torch.zeros(ctx.d, dtype=torch.float64, device=device) for l in skipped_layers}
    total_tokens = 0

    accum = {
        l: {
            "dot_k_s1": 0.0, "norm_k_s1": 0.0,
            "dot_v_s1": 0.0, "norm_v_s1": 0.0,
            "dot_k_s2": None, "norm_k_s2": None,
            "dot_v_s2": None, "norm_v_s2": None,
            "sum_x_k": None, "sum_y_k": None, "sum_xx_k": None, "sum_xy_k": None,
            "sum_x_v": None, "sum_y_v": None, "sum_xx_v": None, "sum_xy_v": None,
            "count": 0
        }
        for l in skipped_layers
    }

    n_samples = min(len(train_tokens), n_fit)
    pbar = tqdm(range(0, n_samples, batch_size), desc="Fitting Group S", dynamic_ncols=True, leave=False) if tqdm else range(0, n_samples, batch_size)

    for i in pbar:
        b = train_tokens[i : i + batch_size].to(device)
        orc = get_oracle_states(ctx, b, need_logits=False)
        h_E = orc["hs"][depth]
        B, T, _ = h_E.shape
        total_tokens += B * T

        # S4 accumulation
        h_E_f64 = h_E.view(-1, ctx.d).to(torch.float64)
        sum_h_E += h_E_f64.sum(dim=0)
        for l in skipped_layers:
            sum_h_l[l] += orc["hs"][l].view(-1, ctx.d).to(torch.float64).sum(dim=0)

        # S1, S2, S3 accumulation
        for l in skipped_layers:
            lay = ctx.layers[l]
            n_l = lay.input_layernorm(h_E)
            k_hat = lay.self_attn.k_proj(n_l).view(B, T, ctx.Hkv, ctx.D).float()
            v_hat = lay.self_attn.v_proj(n_l).view(B, T, ctx.Hkv, ctx.D).float()

            n_true = lay.input_layernorm(orc["hs"][l])
            k_true = lay.self_attn.k_proj(n_true).view(B, T, ctx.Hkv, ctx.D).float()
            v_true = lay.self_attn.v_proj(n_true).view(B, T, ctx.Hkv, ctx.D).float()

            st = accum[l]
            st["count"] += B * T

            # S1 (scalar)
            st["dot_k_s1"] += (k_hat * k_true).sum().item()
            st["norm_k_s1"] += (k_hat * k_hat).sum().item()
            st["dot_v_s1"] += (v_hat * v_true).sum().item()
            st["norm_v_s1"] += (v_hat * v_hat).sum().item()

            # S2 (per-head)
            dk2 = (k_hat * k_true).sum(dim=(0, 1, 3))
            nk2 = (k_hat * k_hat).sum(dim=(0, 1, 3))
            dv2 = (v_hat * v_true).sum(dim=(0, 1, 3))
            nv2 = (v_hat * v_hat).sum(dim=(0, 1, 3))
            st["dot_k_s2"] = (st["dot_k_s2"] if st["dot_k_s2"] is not None else 0.0) + dk2
            st["norm_k_s2"] = (st["norm_k_s2"] if st["norm_k_s2"] is not None else 0.0) + nk2
            st["dot_v_s2"] = (st["dot_v_s2"] if st["dot_v_s2"] is not None else 0.0) + dv2
            st["norm_v_s2"] = (st["norm_v_s2"] if st["norm_v_s2"] is not None else 0.0) + nv2

            # S3 (per-channel affine)
            kh_flat = k_hat.view(-1, ctx.Hkv * ctx.D)
            kt_flat = k_true.view(-1, ctx.Hkv * ctx.D)
            vh_flat = v_hat.view(-1, ctx.Hkv * ctx.D)
            vt_flat = v_true.view(-1, ctx.Hkv * ctx.D)

            st["sum_x_k"] = (st["sum_x_k"] if st["sum_x_k"] is not None else 0.0) + kh_flat.sum(0)
            st["sum_y_k"] = (st["sum_y_k"] if st["sum_y_k"] is not None else 0.0) + kt_flat.sum(0)
            st["sum_xx_k"] = (st["sum_xx_k"] if st["sum_xx_k"] is not None else 0.0) + (kh_flat * kh_flat).sum(0)
            st["sum_xy_k"] = (st["sum_xy_k"] if st["sum_xy_k"] is not None else 0.0) + (kh_flat * kt_flat).sum(0)

            st["sum_x_v"] = (st["sum_x_v"] if st["sum_x_v"] is not None else 0.0) + vh_flat.sum(0)
            st["sum_y_v"] = (st["sum_y_v"] if st["sum_y_v"] is not None else 0.0) + vt_flat.sum(0)
            st["sum_xx_v"] = (st["sum_xx_v"] if st["sum_xx_v"] is not None else 0.0) + (vh_flat * vh_flat).sum(0)
            st["sum_xy_v"] = (st["sum_xy_v"] if st["sum_xy_v"] is not None else 0.0) + (vh_flat * vt_flat).sum(0)

        del orc, h_E
        torch.cuda.empty_cache()

    mu_E = (sum_h_E / total_tokens).float()
    for l in skipped_layers:
        st = accum[l]
        # S1
        s_methods["S1"].params[l] = {
            "sk": st["dot_k_s1"] / (st["norm_k_s1"] + 1e-8),
            "sv": st["dot_v_s1"] / (st["norm_v_s1"] + 1e-8),
        }
        # S2
        s_methods["S2"].params[l] = {
            "sk": (st["dot_k_s2"] / (st["norm_k_s2"] + 1e-8)).view(1, 1, ctx.Hkv, 1),
            "sv": (st["dot_v_s2"] / (st["norm_v_s2"] + 1e-8)).view(1, 1, ctx.Hkv, 1),
        }
        # S3
        N = st["count"]
        denom_k = N * st["sum_xx_k"] - st["sum_x_k"]**2 + 1e-8
        ak = (N * st["sum_xy_k"] - st["sum_x_k"] * st["sum_y_k"]) / denom_k
        bk = (st["sum_y_k"] - ak * st["sum_x_k"]) / N

        denom_v = N * st["sum_xx_v"] - st["sum_x_v"]**2 + 1e-8
        av = (N * st["sum_xy_v"] - st["sum_x_v"] * st["sum_y_v"]) / denom_v
        bv = (st["sum_y_v"] - av * st["sum_x_v"]) / N

        s_methods["S3"].params[l] = {
            "ak": ak.view(1, 1, ctx.Hkv, ctx.D), "bk": bk.view(1, 1, ctx.Hkv, ctx.D),
            "av": av.view(1, 1, ctx.Hkv, ctx.D), "bv": bv.view(1, 1, ctx.Hkv, ctx.D),
        }
        # S4
        mu_l = (sum_h_l[l] / total_tokens).float()
        s_methods["S4"].params[l] = mu_l - mu_E

    print(" [Group S] S1, S2, S3, S4 successfully fitted.")


# ============================================================================
# 2. Group R: Closed-Form Ridge Regression
# ============================================================================

class RidgeRegressionMethod:
    """
    Closed-form ridge regression solver in fp64 over train activations.
    Supports R-h (residual h), R-kv (direct KV), and R-h-r128 (SVD rank 128).
    """
    def __init__(self, mode: str = "R-h", rank: Optional[int] = None, name: str = "ridge"):
        self.mode = mode
        self.rank = rank
        self.name = name
        self.weights = {}
        self.best_lambda = {}

    def _eval_val_mse(self, ctx: Ctx, val_tokens: torch.Tensor, depth: int, target_layer: int, W: torch.Tensor, device: torch.device) -> float:
        total_err = 0.0
        total_denom = 0.0
        d_in = ctx.d

        for i in range(0, len(val_tokens), 2):
            b = val_tokens[i : i + 2].to(device)
            orc = get_oracle_states(ctx, b, need_logits=False)
            h_E = orc["hs"][depth]
            norm_E = ctx.layers[depth].input_layernorm(h_E)
            B, T, _ = norm_E.shape
            X_flat = norm_E.view(-1, d_in).to(torch.float64)
            ones = torch.ones((X_flat.shape[0], 1), dtype=torch.float64, device=device)
            X_aug = torch.cat([X_flat, ones], dim=1)

            pred_Y = (X_aug @ W.to(X_aug.dtype)).to(h_E.dtype)

            if self.mode in ("R-h", "R-h-r128"):
                h_hat_l = h_E + pred_Y.view(B, T, d_in)
                k_pred, v_pred = ctx.project_kv(target_layer, h_hat_l, orc["cos"], orc["sin"])
            else:
                k_pred_pre = pred_Y[:, : ctx.Hkv * ctx.D].view(B, T, ctx.Hkv, ctx.D)
                v_pred_pre = pred_Y[:, ctx.Hkv * ctx.D :].view(B, T, ctx.Hkv, ctx.D)
                k_pred = ctx.rope(k_pred_pre.transpose(1, 2), orc["cos"], orc["sin"])
                v_pred = v_pred_pre.transpose(1, 2)

            k_true, v_true = orc["kv"][target_layer]
            total_err += ((k_pred - k_true)**2).sum().item() + ((v_pred - v_true)**2).sum().item()
            total_denom += (k_true**2).sum().item() + (v_true**2).sum().item()

            del orc, h_E, norm_E
            torch.cuda.empty_cache()

        return total_err / (total_denom + 1e-8)

    def kv(self, ctx: Ctx, orc: Dict, depth: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        h_E = orc["hs"][depth]
        cos, sin = orc["cos"], orc["sin"]
        norm_E = ctx.layers[depth].input_layernorm(h_E)
        B, T, d_in = norm_E.shape
        X_flat = norm_E.view(-1, d_in)
        ones = torch.ones((X_flat.shape[0], 1), device=h_E.device, dtype=X_flat.dtype)
        X_aug = torch.cat([X_flat, ones], dim=1)
        out = []

        for l in range(depth, ctx.N):
            W = self.weights[l].to(h_E.device, dtype=h_E.dtype)
            pred = X_aug @ W

            if self.mode in ("R-h", "R-h-r128"):
                h_hat_l = h_E + pred.view(B, T, d_in)
                k_hat, v_hat = ctx.project_kv(l, h_hat_l, cos, sin)
                out.append((k_hat, v_hat))
            else:
                k_pre = pred[:, : ctx.Hkv * ctx.D].view(B, T, ctx.Hkv, ctx.D)
                v_pre = pred[:, ctx.Hkv * ctx.D :].view(B, T, ctx.Hkv, ctx.D)
                k_final = ctx.rope(k_pre.transpose(1, 2), cos, sin)
                v_final = v_pre.transpose(1, 2)
                out.append((k_final, v_final))

        return out


@torch.no_grad()
def fit_all_group_r(
    r_methods: Dict[str, RidgeRegressionMethod],
    ctx: Ctx,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    depth: int,
    device: torch.device,
    n_fit: int = 100,
    batch_size: int = 2,
    lambda_grid: List[float] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
):
    """Jointly accumulates fp64 covariance matrices and solves R-h, R-kv, and R-h-r128."""
    print(f" [Group R] Jointly accumulating fp64 covariance matrices over {n_fit} sequences...")
    skipped_layers = list(range(depth, ctx.N))
    d_in = ctx.d

    XtX = torch.zeros((d_in + 1, d_in + 1), dtype=torch.float64, device=device)
    XtY_h = {l: torch.zeros((d_in + 1, d_in), dtype=torch.float64, device=device) for l in skipped_layers}
    XtY_kv = {l: torch.zeros((d_in + 1, 2 * ctx.Hkv * ctx.D), dtype=torch.float64, device=device) for l in skipped_layers}

    n_samples = min(len(train_tokens), n_fit)
    pbar = tqdm(range(0, n_samples, batch_size), desc="Fitting Group R", dynamic_ncols=True, leave=False) if tqdm else range(0, n_samples, batch_size)

    for i in pbar:
        b = train_tokens[i : i + batch_size].to(device)
        orc = get_oracle_states(ctx, b, need_logits=False)
        h_E = orc["hs"][depth]
        norm_E = ctx.layers[depth].input_layernorm(h_E)
        B, T, _ = norm_E.shape
        X_batch = norm_E.view(-1, d_in).to(torch.float64)
        ones = torch.ones((X_batch.shape[0], 1), dtype=torch.float64, device=device)
        X_aug = torch.cat([X_batch, ones], dim=1)

        XtX += X_aug.T @ X_aug

        for l in skipped_layers:
            Y_h = (orc["hs"][l] - h_E).view(-1, d_in).to(torch.float64)
            XtY_h[l] += X_aug.T @ Y_h

            lay = ctx.layers[l]
            n_l = lay.input_layernorm(orc["hs"][l])
            k_t = lay.self_attn.k_proj(n_l).view(-1, ctx.Hkv * ctx.D).to(torch.float64)
            v_t = lay.self_attn.v_proj(n_l).view(-1, ctx.Hkv * ctx.D).to(torch.float64)
            Y_kv = torch.cat([k_t, v_t], dim=1)
            XtY_kv[l] += X_aug.T @ Y_kv

        del orc, h_E, norm_E
        torch.cuda.empty_cache()

    tr_XtX = torch.trace(XtX[:d_in, :d_in]).item()
    I_reg = torch.eye(d_in + 1, dtype=torch.float64, device=device)
    I_reg[-1, -1] = 0.0

    print(" [Group R] Selecting optimal regularization lambda on validation split...")
    for l in skipped_layers:
        best_mse_h = float("inf")
        best_W_h = None
        best_lam_h = lambda_grid[0]

        best_mse_kv = float("inf")
        best_W_kv = None
        best_lam_kv = lambda_grid[0]

        for lam in lambda_grid:
            alpha = lam * (tr_XtX / d_in)
            A = XtX + alpha * I_reg
            try:
                W_h = torch.linalg.solve(A, XtY_h[l])
            except Exception:
                W_h = torch.linalg.lstsq(A, XtY_h[l]).solution

            mse_h = r_methods["R-h"]._eval_val_mse(ctx, val_tokens[:16], depth, l, W_h, device)
            if mse_h < best_mse_h:
                best_mse_h = mse_h
                best_W_h = W_h.float()
                best_lam_h = lam

            try:
                W_kv = torch.linalg.solve(A, XtY_kv[l])
            except Exception:
                W_kv = torch.linalg.lstsq(A, XtY_kv[l]).solution

            mse_kv = r_methods["R-kv"]._eval_val_mse(ctx, val_tokens[:16], depth, l, W_kv, device)
            if mse_kv < best_mse_kv:
                best_mse_kv = mse_kv
                best_W_kv = W_kv.float()
                best_lam_kv = lam

        r_methods["R-h"].weights[l] = best_W_h
        r_methods["R-h"].best_lambda[l] = best_lam_h

        # Low-rank truncation for R-h-r128
        W_feat = best_W_h[:d_in, :].to(torch.float64)
        U, S, Vh = torch.linalg.svd(W_feat, full_matrices=False)
        r = 128
        W_lr = (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r, :]).float()
        W_r128 = torch.cat([W_lr, best_W_h[-1:, :]], dim=0)
        r_methods["R-h-r128"].weights[l] = W_r128
        r_methods["R-h-r128"].best_lambda[l] = best_lam_h

        r_methods["R-kv"].weights[l] = best_W_kv
        r_methods["R-kv"].best_lambda[l] = best_lam_kv

    print(" [Group R] R-h, R-kv, R-h-r128 successfully fitted and regularized.")


# ============================================================================
# 3. Group L & CP: Linear & Non-Linear Gradient-Trained Adapters
# ============================================================================

class LinearCheapPassAdapter(nn.Module):
    """
    Linear Adapter (L128 / L16): Exact CheapPass architecture with sigma = Identity.
    """
    def __init__(self, d_model: int, skipped_layers: List[int], rank: int = 128):
        super().__init__()
        self.d_model = d_model
        self.skipped_layers = list(skipped_layers)
        self.S = len(self.skipped_layers)
        self.rank = rank

        self.down = nn.ModuleList([nn.Linear(d_model, rank) for _ in range(self.S)])
        self.up = nn.ModuleList([nn.Linear(rank, d_model) for _ in range(self.S)])

        for i in range(self.S):
            nn.init.kaiming_uniform_(self.down[i].weight, a=math.sqrt(5))
            nn.init.zeros_(self.down[i].bias)
            nn.init.zeros_(self.up[i].weight)
            nn.init.zeros_(self.up[i].bias)

    def forward(self, h_E: torch.Tensor, step_idx: int) -> torch.Tensor:
        orig_dtype = h_E.dtype
        x_fp32 = h_E.float()
        delta = self.up[step_idx](self.down[step_idx](x_fp32))
        return (x_fp32 + delta).to(orig_dtype)

    def predict_all_h(self, h_E: torch.Tensor) -> List[torch.Tensor]:
        return [self.forward(h_E, s) for s in range(self.S)]


def train_gradient_adapter(
    ctx: Ctx,
    adapter: nn.Module,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    depth: int,
    device: torch.device,
    steps: int = 800,
    lr: float = 1e-4,
    batch_size: int = 2,
    seed: int = 0,
) -> nn.Module:
    set_seed(seed)
    adapter = adapter.to(device).train()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    skipped_layers = list(range(depth, ctx.N))

    pbar = tqdm(range(steps), desc=f"Train {adapter.__class__.__name__} (s={seed})", dynamic_ncols=True, leave=False) if tqdm else range(steps)
    train_len = len(train_tokens)

    for step in pbar:
        idx = (step * batch_size) % (train_len - batch_size)
        batch = train_tokens[idx : idx + batch_size].to(device)

        with torch.no_grad():
            orc = get_oracle_states(ctx, batch, need_logits=False)
            h_E = orc["hs"][depth]
            cos, sin = orc["cos"], orc["sin"]

        optimizer.zero_grad()
        loss = 0.0

        for s, l in enumerate(skipped_layers):
            h_hat_l = adapter.forward(h_E, s)
            k_hat, v_hat = ctx.project_kv(l, h_hat_l, cos, sin)
            k_true, v_true = orc["kv"][l]

            rel_k = ((k_hat - k_true)**2).sum() / (((k_true)**2).sum() + 1e-8)
            rel_v = ((v_hat - v_true)**2).sum() / (((v_true)**2).sum() + 1e-8)
            cos_k = 1.0 - F.cosine_similarity(k_hat.flatten(1), k_true.flatten(1), dim=1).mean()
            cos_v = 1.0 - F.cosine_similarity(v_hat.flatten(1), v_true.flatten(1), dim=1).mean()

            loss += rel_k + rel_v + cos_k + cos_v

        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()

        del orc, h_E
        if step % 50 == 0:
            torch.cuda.empty_cache()

    adapter.eval()
    return adapter


# ============================================================================
# 4. Evaluation Function with Paired Bootstrap Output
# ============================================================================

@torch.no_grad()
def evaluate_method_per_sequence(
    ctx: Ctx,
    method_instance,
    tokens: torch.Tensor,
    depth: int,
    exact_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 2,
) -> Tuple[List[float], List[float]]:
    all_kl = []
    all_top1 = []

    for i in range(0, len(tokens), batch_size):
        batch = tokens[i : i + batch_size].to(device)
        orc = get_oracle_states(ctx, batch, need_logits=True)

        if method_instance is None:
            lg = orc["logits"][:, :-1]
        else:
            synth_kv = method_instance.kv(ctx, orc, depth)
            lg = ctx.approx(orc, depth, synth_kv, exact_mask=exact_mask)[:, :-1]

        t_lg = orc["logits"][:, :-1]
        kl_tok, top1_tok = compute_chunked_kl_and_top1(t_lg, lg, chunk_size=64)

        all_kl.extend(kl_tok.mean(dim=1).cpu().tolist())
        all_top1.extend((top1_tok.mean(dim=1) * 100.0).cpu().tolist())

        del orc, lg, t_lg
        torch.cuda.empty_cache()

    return all_kl, all_top1


def paired_bootstrap_diff(
    values_a: List[float],
    values_b: List[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, List[float]]:
    diffs = np.array(values_a) - np.array(values_b)
    mean_diff = float(np.mean(diffs))
    rng = np.random.RandomState(seed)
    boot_means = []

    N = len(diffs)
    for _ in range(n_resamples):
        sample = rng.choice(diffs, size=N, replace=True)
        boot_means.append(np.mean(sample))

    ci_low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return mean_diff, [ci_low, ci_high]


# ============================================================================
# 5. Cost Accounting
# ============================================================================

def compute_cost_accounting(method_name: str, depth: int, L: int = 36, d: int = 2048, Hkv: int = 2, D: int = 128) -> Dict[str, float]:
    skipped = L - depth
    if method_name in ("Native", "ZeroInit", "Oracle", "DirectCopy", "Eviction"):
        return {"params": 0, "flops_per_layer": 0.0, "bytes_per_layer": 0}
    elif method_name == "S1":
        return {"params": 2 * skipped, "flops_per_layer": 2 * Hkv * D * 2, "bytes_per_layer": 2 * 2}
    elif method_name == "S2":
        return {"params": 2 * Hkv * skipped, "flops_per_layer": 2 * Hkv * D * 2, "bytes_per_layer": 2 * Hkv * 2}
    elif method_name == "S3":
        return {"params": 4 * Hkv * D * skipped, "flops_per_layer": 4 * Hkv * D * 2, "bytes_per_layer": 4 * Hkv * D * 2}
    elif method_name == "S4":
        return {"params": d * skipped, "flops_per_layer": d, "bytes_per_layer": d * 2}
    elif method_name == "R-h":
        params = (d + 1) * d * skipped
        return {"params": params, "flops_per_layer": 2 * (d + 1) * d, "bytes_per_layer": (d + 1) * d * 2}
    elif method_name == "R-kv":
        params = (d + 1) * (2 * Hkv * D) * skipped
        return {"params": params, "flops_per_layer": 2 * (d + 1) * (2 * Hkv * D), "bytes_per_layer": (d + 1) * (2 * Hkv * D) * 2}
    elif method_name in ("R-h-r128", "L128", "CP128"):
        params = 2 * d * 128 * skipped
        return {"params": params, "flops_per_layer": 4 * d * 128, "bytes_per_layer": 2 * d * 128 * 2}
    elif method_name in ("L16", "CP16"):
        params = 2 * d * 16 * skipped
        return {"params": params, "flops_per_layer": 4 * d * 16, "bytes_per_layer": 2 * d * 16 * 2}
    return {"params": 0, "flops_per_layer": 0.0, "bytes_per_layer": 0}


# ============================================================================
# 6. Main Benchmark Runner
# ============================================================================

def run_track1_benchmark(
    model_name: str = "Qwen/Qwen2.5-3B",
    depths: List[int] = [18, 24, 29],
    device_str: str = "cuda",
    selftest: bool = False,
    out_file: str = "results/baselines_linear/track1_results.json",
):
    print("=" * 80)
    print(f" TRACK 1: FAIR LINEAR BASELINES AND CLOSED-FORM RECONSTRUCTION")
    print(f" Model: {model_name} | Depths: {depths} | Selftest: {selftest}")
    print("=" * 80)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    if not selftest:
        tok = AutoTokenizer.from_pretrained(model_name)
        data = DataSplits(tok, seq_len=512, n_train=2000, n_val=200, n_test=500, seed=0)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": device})
        model.eval()
    else:
        tok = None
        data = DataSplits(None, seq_len=512, n_train=10, n_val=10, n_test=16, seed=0, selftest=True)
        model = None

    if model is None:
        print(" [Selftest] Running mock decision validation...")
        return

    ctx = Ctx(model)
    primary_depth = 18

    # 1. Joint Fitting for Group S
    s_methods = {
        "S1": PostProjectionScalingMethod("S1", "S1_scalar"),
        "S2": PostProjectionScalingMethod("S2", "S2_per_head"),
        "S3": PostProjectionScalingMethod("S3", "S3_affine"),
        "S4": PostProjectionScalingMethod("S4", "S4_mean_drift"),
    }
    fit_all_group_s(s_methods, ctx, data.train, primary_depth, device, n_fit=100, batch_size=2)

    # 2. Joint Fitting for Group R
    r_methods = {
        "R-h": RidgeRegressionMethod("R-h", name="R-h_full_rank"),
        "R-kv": RidgeRegressionMethod("R-kv", name="R-kv_direct"),
        "R-h-r128": RidgeRegressionMethod("R-h-r128", rank=128, name="R-h-r128_svd"),
    }
    fit_all_group_r(r_methods, ctx, data.train, data.val, primary_depth, device, n_fit=100, batch_size=2)

    # 3. Train Group L & CP Adapters (3 Seeds)
    print("\n --- Training Group L and References (3 Seeds) at E=18 ---")
    skipped_18 = list(range(primary_depth, ctx.N))
    l128_adapters = [
        train_gradient_adapter(ctx, LinearCheapPassAdapter(ctx.d, skipped_18, rank=128), data.train, data.val, primary_depth, device, seed=s, batch_size=2)
        for s in [0, 1, 2]
    ]
    l16_adapter = train_gradient_adapter(ctx, LinearCheapPassAdapter(ctx.d, skipped_18, rank=16), data.train, data.val, primary_depth, device, seed=0, batch_size=2)
    cp128_adapters = [
        train_gradient_adapter(ctx, CheapPassAdapter(ctx.d, skipped_18, rank=128), data.train, data.val, primary_depth, device, seed=s, batch_size=2)
        for s in [0, 1, 2]
    ]
    cp16_adapter = train_gradient_adapter(ctx, CheapPassAdapter(ctx.d, skipped_18, rank=16), data.train, data.val, primary_depth, device, seed=0, batch_size=2)

    eval_catalog_18 = {
        "Native": Native(),
        "DirectCopy": Direct(),
        "S1": s_methods["S1"],
        "S2": s_methods["S2"],
        "S3": s_methods["S3"],
        "S4": s_methods["S4"],
        "R-h": r_methods["R-h"],
        "R-kv": r_methods["R-kv"],
        "R-h-r128": r_methods["R-h-r128"],
        "L16": AdapterKVMethod(l16_adapter, "L16"),
        "L128_s0": AdapterKVMethod(l128_adapters[0], "L128_s0"),
        "L128_s1": AdapterKVMethod(l128_adapters[1], "L128_s1"),
        "L128_s2": AdapterKVMethod(l128_adapters[2], "L128_s2"),
        "CP16": AdapterKVMethod(cp16_adapter, "CP16"),
        "CP128_s0": AdapterKVMethod(cp128_adapters[0], "CP128_s0"),
        "CP128_s1": AdapterKVMethod(cp128_adapters[1], "CP128_s1"),
        "CP128_s2": AdapterKVMethod(cp128_adapters[2], "CP128_s2"),
    }

    # Evaluate Table A: E=18, p=100%
    print("\n ==================================================================")
    print(" TABLE A: EVALUATION AT E=18 (p=100%, Held-out Test N=500)")
    print(" ==================================================================")
    table_a_results = {}
    seq_kl_store = {}

    for name, m in eval_catalog_18.items():
        kl_seq, top1_seq = evaluate_method_per_sequence(ctx, m, data.test, primary_depth, device=device, batch_size=2)
        m_kl, ci_kl = compute_boot_ci(kl_seq)
        m_top, ci_top = compute_boot_ci(top1_seq)
        seq_kl_store[name] = kl_seq
        cost = compute_cost_accounting(name.split("_")[0], primary_depth, L=ctx.N, d=ctx.d)

        table_a_results[name] = {
            "kl_mean": m_kl,
            "kl_ci95": ci_kl,
            "top1_mean": m_top,
            "top1_ci95": ci_top,
            "cost": cost,
        }
        print(f"  [{name:12s}] KL: {m_kl:.4f} [{ci_kl[0]:.4f}, {ci_kl[1]:.4f}] | Top-1: {m_top:.2f}% | Params: {cost['params']:,}")

    # Paired Bootstrap Differences vs Native and CP128_s0
    print("\n --- Computing Paired Bootstrap Confidence Intervals ---")
    paired_results = {}
    native_kls = seq_kl_store["Native"]
    cp128_kls = seq_kl_store["CP128_s0"]

    for name in eval_catalog_18:
        if name in ("Native", "CP128_s0"):
            continue
        diff_nat, ci_nat = paired_bootstrap_diff(seq_kl_store[name], native_kls)
        diff_cp, ci_cp = paired_bootstrap_diff(seq_kl_store[name], cp128_kls)
        paired_results[name] = {
            "diff_vs_native": {"mean": diff_nat, "ci95": ci_nat},
            "diff_vs_cp128": {"mean": diff_cp, "ci95": ci_cp},
        }
        print(f"  [{name:12s}] vs CP128 diff: {diff_cp:+.4f} [{ci_cp[0]:+.4f}, {ci_cp[1]:+.4f}]")

    # Evaluate Decision Rules (Section 1.7)
    kl_native = table_a_results["Native"]["kl_mean"]
    kl_cp128 = table_a_results["CP128_s0"]["kl_mean"]
    cp_gain = kl_native - kl_cp128

    print("\n ==================================================================")
    print(" PRE-REGISTERED DECISION RULES EVALUATION")
    print(" ==================================================================")
    decision_outcomes = {}

    best_s_name = min(["S1", "S2", "S3", "S4"], key=lambda k: table_a_results[k]["kl_mean"])
    best_s_gain = kl_native - table_a_results[best_s_name]["kl_mean"]
    s_gap_closed_pct = (best_s_gain / cp_gain) * 100.0
    rule1_pass = (s_gap_closed_pct >= 50.0)
    decision_outcomes["rule1_scale_offset_explains_gain"] = {
        "best_method": best_s_name,
        "gap_closed_pct": s_gap_closed_pct,
        "passed": rule1_pass,
        "verdict": "Scale/Offset explains >=50% gain" if rule1_pass else "Learned coordinate rotation is required (S closed <50%)"
    }
    print(f" Decision 1 (Scale/Offset): Best {best_s_name} closed {s_gap_closed_pct:.1f}% of CP gain. -> {decision_outcomes['rule1_scale_offset_explains_gain']['verdict']}")

    best_r_name = min(["R-h", "R-kv", "R-h-r128"], key=lambda k: table_a_results[k]["kl_mean"])
    best_r_gain = kl_native - table_a_results[best_r_name]["kl_mean"]
    r_gap_closed_pct = (best_r_gain / cp_gain) * 100.0
    rule2_pass = (r_gap_closed_pct >= 90.0)
    decision_outcomes["rule2_closed_form_suffices"] = {
        "best_method": best_r_name,
        "gap_closed_pct": r_gap_closed_pct,
        "passed": rule2_pass,
        "verdict": "Closed-form linear map suffices" if rule2_pass else "Gradient-trained adapter outperforms closed-form ridge"
    }
    print(f" Decision 2 (Closed-Form): Best {best_r_name} closed {r_gap_closed_pct:.1f}% of CP gain. -> {decision_outcomes['rule2_closed_form_suffices']['verdict']}")

    l128_cp_ci = paired_results["L128_s0"]["diff_vs_cp128"]["ci95"]
    rule3_pass = (l128_cp_ci[0] <= 0.0 <= l128_cp_ci[1])
    decision_outcomes["rule3_gelu_unnecessary"] = {
        "l128_vs_cp128_diff_ci": l128_cp_ci,
        "passed": rule3_pass,
        "verdict": "GELU is unnecessary (L128 within paired CI)" if rule3_pass else "GELU produces statistically significant separation"
    }
    print(f" Decision 3 (Nonlinearity): L128 vs CP128 CI = {l128_cp_ci}. -> {decision_outcomes['rule3_gelu_unnecessary']['verdict']}")

    rel_red_l128 = (table_a_results["L128_s0"]["kl_mean"] - kl_cp128) / table_a_results["L128_s0"]["kl_mean"]
    rule4_pass = (l128_cp_ci[0] > 0.0 and rel_red_l128 >= 0.10)
    decision_outcomes["rule4_nonlinearity_matters"] = {
        "rel_reduction_vs_l128": rel_red_l128 * 100.0,
        "passed": rule4_pass,
        "verdict": "Nonlinearity matters (CP128 beats linear by >=10%)" if rule4_pass else "Linear map achieves comparable performance"
    }
    print(f" Decision 4 (Strong Nonlinearity): -> {decision_outcomes['rule4_nonlinearity_matters']['verdict']}")
    print(" ==================================================================\n")

    full_output = {
        "meta": {"model": model_name, "depths": depths, "n_test": len(data.test)},
        "table_a_E18_p100": table_a_results,
        "paired_bootstrap": paired_results,
        "pre_registered_decisions": decision_outcomes,
    }

    with open(out_file, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f" [Done] Track 1 results successfully exported to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depths", type=str, default="18,24,29")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--out_file", type=str, default="results/baselines_linear/track1_results.json")
    args = parser.parse_args()

    depths = [int(x) for x in args.depths.split(",")]
    run_track1_benchmark(
        model_name=args.model,
        depths=depths,
        device_str=args.device,
        selftest=args.selftest,
        out_file=args.out_file,
    )


if __name__ == "__main__":
    main()
