#!/usr/bin/env python3
"""
baselines_linear.py -- Track 1: Fair Linear Baselines, Post-Projection Scaling,
and Closed-Form Ridge Regression for CheapPass.

Implements all methods specified in Track 1:
  1. Group S (Post-Projection Scaling, closed-form least squares on train):
     - S1: Per-layer scalar on K and V (2 params/layer).
     - S2: Per-KV-head scalar on K and V (4 params/layer for Qwen).
     - S3: Per-channel affine (scale + shift) on K and V (4 * 256 params/layer).
     - S4: Mean-drift correction on h: h_hat_l = h_E + (mu(h_l) - mu(h_E)).
  2. Group R (Closed-Form Ridge Regression in fp64):
     - R-h: Full-rank map from x = RMSNorm(h_E) to residual target (h_l - h_E) + bias.
     - R-kv: Direct map from x to [K_l, V_l] pre-RoPE (2048 -> 512).
     - R-h-r128: Low-rank SVD truncation of R-h to rank 128 (parameter-matched to CheapPass).
     - lambda grid: {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} tuned on validation split.
  3. Group L (Linear Gradient-Trained Adapters):
     - L128: CheapPass with sigma = Identity, rank 128, W_up zero-init (3 seeds).
     - L16: Same, rank 16.
  4. References:
     - CP128 (GELU, r=128, 3 seeds), CP16, Native, ZeroInit, Eviction, Oracle, DirectCopy.
  5. Paired Bootstrap Confidence Intervals on differences (KL_method - KL_baseline).
  6. Automated Pre-Registered Decision Rules evaluation.
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
    Base class for Group S: Applies learned/fitted transformations post-projection.
    """
    def __init__(self, mode: str = "S1", name: str = "post_proj"):
        self.mode = mode
        self.name = name
        self.params = {}

    def fit(self, ctx: Ctx, train_tokens: torch.Tensor, depth: int, device: torch.device):
        skipped_layers = list(range(depth, ctx.N))

        if self.mode == "S4":
            sum_h_E = torch.zeros(ctx.d, dtype=torch.float64, device=device)
            sum_h_l = {l: torch.zeros(ctx.d, dtype=torch.float64, device=device) for l in skipped_layers}
            total_tokens = 0

            for i in range(0, min(len(train_tokens), 200), 4):
                b = train_tokens[i : i + 4].to(device)
                orc = ctx.oracle(b)
                h_E = orc["hs"][depth].view(-1, ctx.d).to(torch.float64)
                sum_h_E += h_E.sum(dim=0)
                total_tokens += h_E.shape[0]
                for l in skipped_layers:
                    sum_h_l[l] += orc["hs"][l].view(-1, ctx.d).to(torch.float64).sum(dim=0)

            mu_E = (sum_h_E / total_tokens).float()
            mu_l = {l: (sum_h_l[l] / total_tokens).float() for l in skipped_layers}
            self.params = {l: (mu_l[l] - mu_E) for l in skipped_layers}
            return

        accum = {
            l: {
                "dot_k": None, "norm_k": None,
                "dot_v": None, "norm_v": None,
                "sum_x_k": None, "sum_y_k": None, "sum_xx_k": None, "sum_xy_k": None,
                "sum_x_v": None, "sum_y_v": None, "sum_xx_v": None, "sum_xy_v": None,
                "count": 0
            }
            for l in skipped_layers
        }

        n_fit = min(len(train_tokens), 250)
        for i in range(0, n_fit, 4):
            b = train_tokens[i : i + 4].to(device)
            orc = ctx.oracle(b)
            h_E = orc["hs"][depth]
            B, T, _ = h_E.shape

            for l in skipped_layers:
                lay = ctx.layers[l]
                n = lay.input_layernorm(h_E)
                k_hat = lay.self_attn.k_proj(n).view(B, T, ctx.Hkv, ctx.D).float()
                v_hat = lay.self_attn.v_proj(n).view(B, T, ctx.Hkv, ctx.D).float()

                n_true = lay.input_layernorm(orc["hs"][l])
                k_true = lay.self_attn.k_proj(n_true).view(B, T, ctx.Hkv, ctx.D).float()
                v_true = lay.self_attn.v_proj(n_true).view(B, T, ctx.Hkv, ctx.D).float()

                st = accum[l]
                st["count"] += B * T

                if self.mode == "S1":
                    dk = (k_hat * k_true).sum().item()
                    nk = (k_hat * k_hat).sum().item()
                    dv = (v_hat * v_true).sum().item()
                    nv = (v_hat * v_hat).sum().item()
                    st["dot_k"] = (st["dot_k"] or 0.0) + dk
                    st["norm_k"] = (st["norm_k"] or 0.0) + nk
                    st["dot_v"] = (st["dot_v"] or 0.0) + dv
                    st["norm_v"] = (st["norm_v"] or 0.0) + nv

                elif self.mode == "S2":
                    dk = (k_hat * k_true).sum(dim=(0, 1, 3))
                    nk = (k_hat * k_hat).sum(dim=(0, 1, 3))
                    dv = (v_hat * v_true).sum(dim=(0, 1, 3))
                    nv = (v_hat * v_hat).sum(dim=(0, 1, 3))
                    st["dot_k"] = (st["dot_k"] if st["dot_k"] is not None else 0.0) + dk
                    st["norm_k"] = (st["norm_k"] if st["norm_k"] is not None else 0.0) + nk
                    st["dot_v"] = (st["dot_v"] if st["dot_v"] is not None else 0.0) + dv
                    st["norm_v"] = (st["norm_v"] if st["norm_v"] is not None else 0.0) + nv

                elif self.mode == "S3":
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

        for l in skipped_layers:
            st = accum[l]
            if self.mode == "S1":
                sk = st["dot_k"] / (st["norm_k"] + 1e-8)
                sv = st["dot_v"] / (st["norm_v"] + 1e-8)
                self.params[l] = {"sk": sk, "sv": sv}
            elif self.mode == "S2":
                sk = (st["dot_k"] / (st["norm_k"] + 1e-8)).view(1, 1, ctx.Hkv, 1)
                sv = (st["dot_v"] / (st["norm_v"] + 1e-8)).view(1, 1, ctx.Hkv, 1)
                self.params[l] = {"sk": sk, "sv": sv}
            elif self.mode == "S3":
                N = st["count"]
                denom_k = N * st["sum_xx_k"] - st["sum_x_k"]**2 + 1e-8
                ak = (N * st["sum_xy_k"] - st["sum_x_k"] * st["sum_y_k"]) / denom_k
                bk = (st["sum_y_k"] - ak * st["sum_x_k"]) / N

                denom_v = N * st["sum_xx_v"] - st["sum_x_v"]**2 + 1e-8
                av = (N * st["sum_xy_v"] - st["sum_x_v"] * st["sum_y_v"]) / denom_v
                bv = (st["sum_y_v"] - av * st["sum_x_v"]) / N

                self.params[l] = {
                    "ak": ak.view(1, 1, ctx.Hkv, ctx.D), "bk": bk.view(1, 1, ctx.Hkv, ctx.D),
                    "av": av.view(1, 1, ctx.Hkv, ctx.D), "bv": bv.view(1, 1, ctx.Hkv, ctx.D),
                }

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
                k_scaled = k_proj * p["sk"]
                v_scaled = v_proj * p["sv"]
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

    def fit(
        self,
        ctx: Ctx,
        train_tokens: torch.Tensor,
        val_tokens: torch.Tensor,
        depth: int,
        device: torch.device,
        lambda_grid: List[float] = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    ):
        skipped_layers = list(range(depth, ctx.N))
        d_in = ctx.d

        print(f" [{self.name}] Accumulating fp64 covariance matrices over train tokens...")
        XtX = torch.zeros((d_in + 1, d_in + 1), dtype=torch.float64, device=device)
        XtY = {l: None for l in skipped_layers}

        n_fit = min(len(train_tokens), 250)
        for i in range(0, n_fit, 4):
            b = train_tokens[i : i + 4].to(device)
            orc = ctx.oracle(b)
            h_E = orc["hs"][depth]
            norm_E = ctx.layers[depth].input_layernorm(h_E)
            B, T, _ = norm_E.shape
            X_batch = norm_E.view(-1, d_in).to(torch.float64)
            ones = torch.ones((X_batch.shape[0], 1), dtype=torch.float64, device=device)
            X_aug = torch.cat([X_batch, ones], dim=1)

            XtX += X_aug.T @ X_aug

            for l in skipped_layers:
                if self.mode in ("R-h", "R-h-r128"):
                    h_l = orc["hs"][l]
                    Y_batch = (h_l - h_E).view(-1, d_in).to(torch.float64)
                elif self.mode == "R-kv":
                    lay = ctx.layers[l]
                    n_l = lay.input_layernorm(orc["hs"][l])
                    k_t = lay.self_attn.k_proj(n_l).view(-1, ctx.Hkv * ctx.D).to(torch.float64)
                    v_t = lay.self_attn.v_proj(n_l).view(-1, ctx.Hkv * ctx.D).to(torch.float64)
                    Y_batch = torch.cat([k_t, v_t], dim=1)

                if XtY[l] is None:
                    XtY[l] = torch.zeros((d_in + 1, Y_batch.shape[1]), dtype=torch.float64, device=device)
                XtY[l] += X_aug.T @ Y_batch

        print(f" [{self.name}] Selecting optimal regularization lambda on validation split...")
        tr_XtX = torch.trace(XtX[:d_in, :d_in]).item()
        I_reg = torch.eye(d_in + 1, dtype=torch.float64, device=device)
        I_reg[-1, -1] = 0.0

        for l in skipped_layers:
            best_rel_mse = float("inf")
            best_W = None
            best_lam = lambda_grid[0]

            for lam in lambda_grid:
                alpha = lam * (tr_XtX / d_in)
                A = XtX + alpha * I_reg
                try:
                    W_sol = torch.linalg.solve(A, XtY[l])
                except Exception:
                    W_sol = torch.linalg.lstsq(A, XtY[l]).solution

                if self.mode == "R-h-r128" and self.rank is not None:
                    W_feat = W_sol[:d_in, :]
                    U, S, Vh = torch.linalg.svd(W_feat, full_matrices=False)
                    W_lr = U[:, : self.rank] @ torch.diag(S[: self.rank]) @ Vh[: self.rank, :]
                    W_sol = torch.cat([W_lr, W_sol[-1:, :]], dim=0)

                rel_mse = self._eval_val_mse(ctx, val_tokens[:32], depth, l, W_sol, device)
                if rel_mse < best_rel_mse:
                    best_rel_mse = rel_mse
                    best_W = W_sol.float()
                    best_lam = lam

            self.weights[l] = best_W
            self.best_lambda[l] = best_lam

        print(f" [{self.name}] Fitting complete. Best lambdas: {list(self.best_lambda.values())[:3]}...")

    def _eval_val_mse(self, ctx: Ctx, val_tokens: torch.Tensor, depth: int, target_layer: int, W: torch.Tensor, device: torch.device) -> float:
        total_err = 0.0
        total_denom = 0.0
        d_in = ctx.d

        for i in range(0, len(val_tokens), 4):
            b = val_tokens[i : i + 4].to(device)
            orc = ctx.oracle(b)
            h_E = orc["hs"][depth]
            norm_E = ctx.layers[depth].input_layernorm(h_E)
            B, T, _ = norm_E.shape
            X_flat = norm_E.view(-1, d_in).to(torch.float64)
            ones = torch.ones((X_flat.shape[0], 1), dtype=torch.float64, device=device)
            X_aug = torch.cat([X_flat, ones], dim=1)

            pred_Y = (X_aug @ W).float()

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
    steps: int = 1000,
    lr: float = 1e-4,
    batch_size: int = 4,
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
            orc = ctx.oracle(batch)
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
    batch_size: int = 4,
) -> Tuple[List[float], List[float]]:
    """Evaluates method returning per-sequence KL and Top-1 match lists."""
    all_kl = []
    all_top1 = []

    for i in range(0, len(tokens), batch_size):
        batch = tokens[i : i + batch_size].to(device)
        orc = ctx.oracle(batch)

        if method_instance is None:
            # Oracle baseline
            lg = orc["logits"][:, :-1]
        else:
            synth_kv = method_instance.kv(ctx, orc, depth)
            lg = ctx.approx(orc, depth, synth_kv, exact_mask=exact_mask)[:, :-1]

        t_lg = orc["logits"][:, :-1]
        kl_tok, top1_tok = compute_chunked_kl_and_top1(t_lg, lg, chunk_size=64)

        all_kl.extend(kl_tok.mean(dim=1).cpu().tolist())
        all_top1.extend((top1_tok.mean(dim=1) * 100.0).cpu().tolist())

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

    # 1. Fit Group S Methods
    print("\n --- Fitting Group S Methods (S1, S2, S3, S4) at E=18 ---")
    s_methods = {
        "S1": PostProjectionScalingMethod("S1", "S1_scalar"),
        "S2": PostProjectionScalingMethod("S2", "S2_per_head"),
        "S3": PostProjectionScalingMethod("S3", "S3_affine"),
        "S4": PostProjectionScalingMethod("S4", "S4_mean_drift"),
    }
    for name, m in s_methods.items():
        print(f" Fitting {name}...")
        m.fit(ctx, data.train, primary_depth, device)

    # 2. Fit Group R Methods
    print("\n --- Fitting Group R Methods (R-h, R-kv, R-h-r128) at E=18 ---")
    r_methods = {
        "R-h": RidgeRegressionMethod("R-h", name="R-h_full_rank"),
        "R-kv": RidgeRegressionMethod("R-kv", name="R-kv_direct"),
        "R-h-r128": RidgeRegressionMethod("R-h-r128", rank=128, name="R-h-r128_svd"),
    }
    for name, m in r_methods.items():
        print(f" Fitting {name}...")
        m.fit(ctx, data.train, data.val, primary_depth, device)

    # 3. Train Group L & CP Adapters (3 Seeds)
    print("\n --- Training Group L and References (3 Seeds) at E=18 ---")
    skipped_18 = list(range(primary_depth, ctx.N))
    l128_adapters = [
        train_gradient_adapter(ctx, LinearCheapPassAdapter(ctx.d, skipped_18, rank=128), data.train, data.val, primary_depth, device, seed=s)
        for s in [0, 1, 2]
    ]
    l16_adapter = train_gradient_adapter(ctx, LinearCheapPassAdapter(ctx.d, skipped_18, rank=16), data.train, data.val, primary_depth, device, seed=0)
    cp128_adapters = [
        train_gradient_adapter(ctx, CheapPassAdapter(ctx.d, skipped_18, rank=128), data.train, data.val, primary_depth, device, seed=s)
        for s in [0, 1, 2]
    ]
    cp16_adapter = train_gradient_adapter(ctx, CheapPassAdapter(ctx.d, skipped_18, rank=16), data.train, data.val, primary_depth, device, seed=0)

    # Construct evaluation catalog for E=18
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
        kl_seq, top1_seq = evaluate_method_per_sequence(ctx, m, data.test, primary_depth, device=device)
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

    # Rule 1: S-methods close >= 50% of CP128 gain?
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

    # Rule 2: Closed-form ridge closes >= 90% of CP128 gain?
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

    # Rule 3: L128 within paired CI of CP128?
    l128_cp_ci = paired_results["L128_s0"]["diff_vs_cp128"]["ci95"]
    rule3_pass = (l128_cp_ci[0] <= 0.0 <= l128_cp_ci[1])
    decision_outcomes["rule3_gelu_unnecessary"] = {
        "l128_vs_cp128_diff_ci": l128_cp_ci,
        "passed": rule3_pass,
        "verdict": "GELU is unnecessary (L128 within paired CI)" if rule3_pass else "GELU produces statistically significant separation"
    }
    print(f" Decision 3 (Nonlinearity): L128 vs CP128 CI = {l128_cp_ci}. -> {decision_outcomes['rule3_gelu_unnecessary']['verdict']}")

    # Rule 4: CP128 beats L128 and all R with paired CI excluding 0 and >= 10% relative reduction?
    rel_red_l128 = (table_a_results["L128_s0"]["kl_mean"] - kl_cp128) / table_a_results["L128_s0"]["kl_mean"]
    rule4_pass = (l128_cp_ci[0] > 0.0 and rel_red_l128 >= 0.10)
    decision_outcomes["rule4_nonlinearity_matters"] = {
        "rel_reduction_vs_l128": rel_red_l128 * 100.0,
        "passed": rule4_pass,
        "verdict": "Nonlinearity matters (CP128 beats linear by >=10%)" if rule4_pass else "Linear map achieves comparable performance"
    }
    print(f" Decision 4 (Strong Nonlinearity): -> {decision_outcomes['rule4_nonlinearity_matters']['verdict']}")
    print(" ==================================================================\n")

    # Save Output JSON
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
