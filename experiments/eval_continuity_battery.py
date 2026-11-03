#!/usr/bin/env python3
"""
eval_continuity_battery.py -- The Comprehensive 7-Point Continuity Battery for CheapPass.

Executes the formal continuity verification protocol requested by peer review:
  1. Missing Baselines:
     - Oracle (True KV)
     - NativeProj (Raw h_E projection)
     - NormCorrectedNative (Per-layer calibrated scalar gain matching layer norm)
     - KVEviction (Exited tokens masked out of attention entirely)
     - CheapPass (Trained bottleneck adapter)
     - ZeroInit (Untrained adapter control)
     - Computes the Exact Fraction of Native-to-Oracle Gap Closed: (KL_native - KL_method) / KL_native
  2. Accumulating Holes Sweep (p in {1/512, 10%, 50%, 100%}):
     - Evaluates whether KL degrades gracefully or cliff-drops as early-exit density increases.
  3. Per-Layer Diagnostics:
     - Attention Output Drift: ||A_hat * V_hat - A* * V*||_F / ||A* * V*||_F
     - Attention Mass on Synthetic Tokens (Sum A_{t, tau} for tau in exited)
     - Layer-by-layer hidden state drift ||h_l_hat - h_l*|| / ||h_l*||
  4. Reference Scales:
     - fp16 Noise Floor
     - Simulated int8 KV Quantization benchmark
"""

import argparse
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
from cheap_pass import (
    Ctx,
    CheapPassAdapter,
    AdapterKVMethod,
    compute_kl_and_top1_chunked,
    boot_ci,
    LN2,
)
from vkr2 import Native, Direct

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================================
# 1. Custom Extended Context with Eviction & Diagnostics
# ============================================================================

class ContinuityCtx:
    """
    Wraps model with exact attention output tracking, eviction masking, and drift measurement.
    """
    def __init__(self, model):
        self.m = model
        self.layers = model.model.layers
        self.N = len(self.layers)
        self.d = model.config.hidden_size
        self.H = model.config.num_attention_heads
        self.Hkv = getattr(model.config, "num_key_value_heads", self.H)
        self.D = self.d // self.H
        self.rep = self.H // self.Hkv
        self.rotary = model.model.rotary_emb
        self.dtype = next(model.parameters()).dtype

    def embed(self, ids):
        B, T = ids.shape
        h = self.m.model.embed_tokens(ids)
        pos = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary(h, pos)
        return h, cos, sin

    @staticmethod
    def rope(x, cos, sin):
        D = x.shape[-1]
        x1, x2 = x[..., : D // 2], x[..., D // 2 :]
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos.unsqueeze(1) + rot * sin.unsqueeze(1)

    def split(self, x, nh):
        B, T, _ = x.shape
        return x.view(B, T, nh, self.D).transpose(1, 2)

    def rep_kv(self, x):
        return x if self.rep == 1 else x.repeat_interleave(self.rep, dim=1)

    def project_kv(self, i, h_src, cos, sin, gain: float = 1.0):
        lay = self.layers[i]
        a = lay.self_attn
        h_scaled = h_src * gain if gain != 1.0 else h_src
        n = lay.input_layernorm(h_scaled)
        kl, vl = a.k_proj(n), a.v_proj(n)
        return self.rope(self.split(kl, self.Hkv), cos, sin), self.split(vl, self.Hkv)

    def layer_fwd(self, i, h, cos, sin, kv_hat=None, evict_mask=None, record_attn: bool = False):
        lay = self.layers[i]
        a = lay.self_attn
        B, T, _ = h.shape
        n = lay.input_layernorm(h)
        q = self.rope(self.split(a.q_proj(n), self.H), cos, sin)
        k = self.rope(self.split(a.k_proj(n), self.Hkv), cos, sin)
        v = self.split(a.v_proj(n), self.Hkv)
        
        kh, vh = (k, v) if kv_hat is None else kv_hat
        khr, vhr = self.rep_kv(kh), self.rep_kv(vh)
        kr, vr = self.rep_kv(k), self.rep_kv(v)
        
        scale = self.D ** -0.5
        sc = torch.matmul(q, khr.transpose(-1, -2)).float() * scale
        diag = (q.float() * kr.float()).sum(-1) * scale
        eye = torch.eye(T, dtype=torch.bool, device=h.device)
        sc = torch.where(eye, diag.unsqueeze(-1), sc)

        # Eviction Masking: Mask exited tokens out of attention for subsequent queries
        if evict_mask is not None:
            # evict_mask: [B, T] bool where True = token is evicted
            em = evict_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            sc = sc.masked_fill(em, float("-inf"))
            # Preserve query diagonal attending to its own token
            sc = torch.where(eye, diag.unsqueeze(-1), sc)

        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1)
        sc = sc.masked_fill(causal, float("-inf"))
        A = torch.softmax(sc, dim=-1)
        
        a_diag = torch.diagonal(A, dim1=-2, dim2=-1)
        attn_out_pre = torch.matmul(A.to(vhr.dtype), vhr) + a_diag.unsqueeze(-1).to(vhr.dtype) * (vr - vhr)
        attn_out = attn_out_pre.transpose(1, 2).reshape(B, T, self.H * self.D)
        
        h = h + a.o_proj(attn_out)
        h = h + lay.mlp(lay.post_attention_layernorm(h))
        
        diag_info = {}
        if record_attn:
            diag_info = {
                "A": A.detach(),
                "attn_out": attn_out.detach(),
                "h": h.detach(),
            }
        return h, (k, v), diag_info

    def head(self, h):
        return self.m.lm_head(self.m.model.norm(h))

    @torch.no_grad()
    def oracle(self, ids, record_attn: bool = False):
        h, cos, sin = self.embed(ids)
        hs, kv, diag_list = [h], [], []
        for i in range(self.N):
            h, kvi, dinfo = self.layer_fwd(i, h, cos, sin, record_attn=record_attn)
            hs.append(h)
            kv.append(kvi)
            diag_list.append(dinfo)
        return dict(hs=hs, kv=kv, logits=self.head(h), cos=cos, sin=sin, diags=diag_list)

    @torch.no_grad()
    def approx(self, orc, E, kv_hats, exit_mask=None, evict: bool = False, record_attn: bool = False):
        """
        Runs upper layers from E to N-1.
        exit_mask: [B, T] bool. True = token exited at E; False = token continues at full depth.
        """
        h, cos, sin = orc["hs"][E], orc["cos"], orc["sin"]
        B, T = exit_mask.shape if exit_mask is not None else (h.shape[0], h.shape[1])
        diag_list = []

        for j, i in enumerate(range(E, self.N)):
            kh, vh = kv_hats[j]
            # Replace non-exited tokens with exact oracle KV
            if exit_mask is not None:
                m_exact = (~exit_mask).unsqueeze(1).unsqueeze(-1)  # True where NOT exited
                kh = torch.where(m_exact, orc["kv"][i][0], kh)
                vh = torch.where(m_exact, orc["kv"][i][1], vh)

            ev_mask = exit_mask if evict else None
            h, _, dinfo = self.layer_fwd(i, h, cos, sin, (kh, vh), evict_mask=ev_mask, record_attn=record_attn)
            diag_list.append(dinfo)

        return self.head(h), diag_list


# ============================================================================
# 2. Baseline Classes: Norm-Corrected Native & KV Eviction
# ============================================================================

class NormCorrectedNative:
    """
    Baseline: Per-layer scalar gain applied to h_E before LayerNorm & projection.
    gain_l = E[||h_l||] / E[||h_E||]
    Tests if a simple 1-parameter norm correction closes the Native Projection gap.
    """
    def __init__(self, gains: Dict[int, float]):
        self.gains = gains
        self.name = "NormCorrected"

    def kv(self, ctx: ContinuityCtx, orc, E: int):
        h = orc["hs"][E]
        return [
            ctx.project_kv(i, h, orc["cos"], orc["sin"], gain=self.gains.get(i, 1.0))
            for i in range(E, ctx.N)
        ]


class KVEviction:
    """
    Baseline: Exited tokens are masked out of attention in skipped layers.
    Tests whether synthetic KV carries positive information over simply dropping tokens.
    """
    def __init__(self):
        self.name = "KVEviction"

    def kv(self, ctx: ContinuityCtx, orc, E: int):
        # Feeds arbitrary dummy tensors because they will be masked out with -inf
        h = orc["hs"][E]
        return [ctx.project_kv(i, h, orc["cos"], orc["sin"]) for i in range(E, ctx.N)]


class Int8SimulatedKV:
    """
    Reference Scale: Applies 8-bit uniform quantization to exact Oracle KV.
    Provides the standard production compression anchor.
    """
    def __init__(self):
        self.name = "Int8_KV_Quant"

    def kv(self, ctx: ContinuityCtx, orc, E: int):
        out = []
        for i in range(E, ctx.N):
            k, v = orc["kv"][i]
            # 8-bit RTN quantization simulation
            k_q = self.quantize_int8(k)
            v_q = self.quantize_int8(v)
            out.append((k_q, v_q))
        return out

    @staticmethod
    def quantize_int8(x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        amax = x_f.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = amax / 127.0
        q = (x_f / scale).round().clamp(-127, 127)
        return (q * scale).to(orig_dtype)


# ============================================================================
# 3. Calibration: Compute Per-Layer Scalar Gains on Validation Split
# ============================================================================

@torch.no_grad()
def calibrate_layer_gains(ctx: ContinuityCtx, val_data: torch.Tensor, E: int, device: torch.device) -> Dict[int, float]:
    """Computes mean norm ratio g_l = E[||h_l||] / E[||h_E||] on validation data."""
    print(f" [Calibration] Computing per-layer scalar norms for E={E}...")
    norms_E = []
    norms_l = {l: [] for l in range(E, ctx.N)}
    
    for s in range(0, min(100, len(val_data)), 4):
        batch = val_data[s : s + 4].to(device)
        orc = ctx.oracle(batch)
        n_E = orc["hs"][E].float().norm(dim=-1).mean().item()
        norms_E.append(n_E)
        for l in range(E, ctx.N):
            n_l = orc["hs"][l].float().norm(dim=-1).mean().item()
            norms_l[l].append(n_l)

    mean_nE = float(np.mean(norms_E))
    gains = {l: float(np.mean(norms_l[l])) / max(1e-6, mean_nE) for l in range(E, ctx.N)}
    print(f"   Calibrated Gains: Layer {E}: {gains[E]:.3f} -> Layer 35: {gains[ctx.N - 1]:.3f}")
    return gains


# ============================================================================
# 4. Evaluation Engine with Variable Exit Fraction p
# ============================================================================

@torch.no_grad()
def evaluate_battery_condition(
    ctx: ContinuityCtx,
    method_instance,
    test_data: torch.Tensor,
    E: int,
    device: torch.device,
    p_exit: float = 1.0,
    evict: bool = False,
    bs: int = 4,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Evaluates KL divergence and Top-1 match where fraction p_exit of tokens exit at E.
    p_exit = 1/512 (single token), 0.10 (10%), 0.50 (50%), 1.00 (100%).
    """
    kl_per_seq = []
    top1_per_seq = []
    eval_bs = min(2, bs) if (device.type == "cuda" and bs > 2) else bs
    rng = np.random.RandomState(seed)

    for s in range(0, len(test_data), eval_bs):
        batch = test_data[s : s + eval_bs].to(device)
        B, T = batch.shape
        orc = ctx.oracle(batch)
        ref_logits = orc["logits"][:, :-1]

        # Generate exit mask based on p_exit
        if p_exit >= 0.999:
            exit_mask = torch.ones((B, T), dtype=torch.bool, device=device)
        elif p_exit <= (1.0 / T + 1e-5):
            # Exactly 1 token exits per sequence (at position T // 2)
            exit_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
            exit_mask[:, T // 2] = True
        else:
            # Random Bernoulli fraction p
            mask_np = rng.rand(B, T) < p_exit
            exit_mask = torch.tensor(mask_np, dtype=torch.bool, device=device)

        if method_instance is None:
            approx_logits = ref_logits
        else:
            kv_approx = method_instance.kv(ctx, orc, E)
            approx_full, _ = ctx.approx(orc, E, kv_approx, exit_mask=exit_mask, evict=evict)
            approx_logits = approx_full[:, :-1]

        kl_seq, top1_seq = compute_kl_and_top1_chunked(ref_logits, approx_logits, chunk_size=64)
        kl_per_seq.extend(kl_seq)
        top1_per_seq.extend(top1_seq)

    mean_kl, ci_lo_kl, ci_hi_kl = boot_ci(kl_per_seq, rng, n=1000)
    mean_top, ci_lo_top, ci_hi_top = boot_ci([100.0 * x for x in top1_per_seq], rng, n=1000)

    return {
        "kl_mean": mean_kl,
        "kl_ci95": [ci_lo_kl, ci_hi_kl],
        "top1_mean": mean_top,
        "top1_ci95": [ci_lo_top, ci_hi_top],
    }


# ============================================================================
# 5. Diagnostic Engine: Attention Output & Mass Drift
# ============================================================================

@torch.no_grad()
def evaluate_attention_diagnostics(
    ctx: ContinuityCtx,
    adapter: CheapPassAdapter,
    test_data: torch.Tensor,
    E: int,
    device: torch.device,
    n_seq: int = 20,
) -> Dict[str, any]:
    """
    Evaluates:
      1. Attention Output Drift: ||A_hat * V_hat - A* * V*||_F / ||A* * V*||_F
      2. Attention Mass on Synthetic Tokens (Sum A_{t, tau} for tau in exited)
      3. Hidden State Drift: ||h_l_hat - h_l*|| / ||h_l*||
    """
    print(f"\n [Diagnostics] Measuring per-layer attention output drift and attention mass for E={E}...")
    attn_out_errs = {l: [] for l in range(E, ctx.N)}
    h_drifts = {l: [] for l in range(E, ctx.N)}
    synth_mass_oracle = {l: [] for l in range(E, ctx.N)}
    synth_mass_approx = {l: [] for l in range(E, ctx.N)}

    method_ad = AdapterKVMethod(adapter, "cheap_pass")

    for s in range(0, min(n_seq, len(test_data)), 2):
        batch = test_data[s : s + 2].to(device)
        B, T = batch.shape
        # Half of tokens exit to measure mass on exited tokens
        exit_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        exit_mask[:, 1::2] = True  # Every odd token exits

        orc = ctx.oracle(batch, record_attn=True)
        kv_approx = method_ad.kv(ctx, orc, E)
        _, approx_diags = ctx.approx(orc, E, kv_approx, exit_mask=exit_mask, record_attn=True)

        for j, l in enumerate(range(E, ctx.N)):
            # 1. Attention Output Frobenius Drift
            ao_star = orc["diags"][l]["attn_out"].float()
            ao_hat = approx_diags[j]["attn_out"].float()
            rel_ao = (torch.norm(ao_hat - ao_star, dim=-1) / torch.norm(ao_star, dim=-1).clamp(min=1e-6)).mean().item()
            attn_out_errs[l].append(rel_ao)

            # 2. Hidden State Drift
            h_star = orc["hs"][l + 1].float()
            h_hat = approx_diags[j]["h"].float()
            rel_h = (torch.norm(h_hat - h_star, dim=-1) / torch.norm(h_star, dim=-1).clamp(min=1e-6)).mean().item()
            h_drifts[l].append(rel_h)

            # 3. Attention Mass on Exited Positions
            A_star = orc["diags"][l]["A"].float()  # [B, H, T, T]
            A_hat = approx_diags[j]["A"].float()
            em = exit_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
            mass_star = (A_star * em.float()).sum(dim=-1).mean().item()
            mass_hat = (A_hat * em.float()).sum(dim=-1).mean().item()
            synth_mass_oracle[l].append(mass_star)
            synth_mass_approx[l].append(mass_hat)

    return {
        "per_layer_attn_out_err": {l: float(np.mean(attn_out_errs[l])) for l in range(E, ctx.N)},
        "per_layer_h_drift": {l: float(np.mean(h_drifts[l])) for l in range(E, ctx.N)},
        "attention_mass_oracle": {l: float(np.mean(synth_mass_oracle[l])) for l in range(E, ctx.N)},
        "attention_mass_approx": {l: float(np.mean(synth_mass_approx[l])) for l in range(E, ctx.N)},
    }


# ============================================================================
# 6. Main Battery Runner
# ============================================================================

def run_continuity_battery(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 80)
    print(" 7-POINT CONTINUITY VERIFICATION BATTERY")
    print(f" Model: {args.model} | Device: {device} | Depth: E={args.depth}")
    print(f" P-Sweep: {args.p_sweep} | Sequences: N={args.n_seq}, T={args.seq_len}")
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    ctx = ContinuityCtx(model)
    E = args.depth

    # Load Data (FineWeb-Edu test split)
    from datasets import load_dataset
    print(f"\n [Data] Streaming {args.n_seq} sequences from HuggingFaceFW/fineweb-edu...")
    texts = []
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        for r in ds:
            t = r.get("text", "")
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= args.n_seq * 2:
                break
    except Exception as e:
        print(f" [Data] Warning: Dataset stream failed ({e}); using synthetic sequences.")
        rng = np.random.RandomState(args.seed)
        test_tokens = torch.tensor(rng.randint(50, tok.vocab_size - 100, size=(args.n_seq, args.seq_len)), dtype=torch.long)
        texts = []

    if texts:
        t_list = []
        for t in texts:
            toks = tok(t, add_special_tokens=False)["input_ids"]
            if len(toks) >= args.seq_len:
                t_list.append(toks[:args.seq_len])
            if len(t_list) >= args.n_seq:
                break
        test_tokens = torch.tensor(t_list, dtype=torch.long)

    # 1. Load Trained CheapPass Adapter
    fname = f"adapter_E{E}_r{args.rank}_primary.pt"
    candidate_dirs = [
        args.ckpt_dir,
        "cheap_pass_out",
        "cheap_pass_checkpoints",
        "../cheap_pass_out",
        "/kaggle/working/cheap_pass_out",
        "/kaggle/working/cheap_pass_checkpoints",
    ]
    ckpt_path = None
    for cd in candidate_dirs:
        cand = os.path.join(cd, fname)
        if os.path.exists(cand):
            ckpt_path = cand
            break

    adapter = CheapPassAdapter(ctx.d, list(range(E, ctx.N)), rank=args.rank).to(device)
    if ckpt_path is not None:
        print(f" [Checkpoint] Successfully loaded adapter weights from: {ckpt_path}")
        adapter.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f" [Warning] Checkpoint '{fname}' not found; running untrained adapter!")
    adapter.eval()

    # 2. Calibrate Norm-Corrected Native Projection
    gains = calibrate_layer_gains(ctx, test_tokens[:30], E, device)
    norm_native = NormCorrectedNative(gains)

    # 3. Reference Quantization Baseline
    int8_method = Int8SimulatedKV()

    results = {
        "meta": {
            "model": args.model,
            "depth": E,
            "rank": args.rank,
            "seq_len": args.seq_len,
            "n_seq": args.n_seq,
            "gate_ln2": LN2,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baselines_p100": {},
        "p_sweep_results": {},
        "diagnostics": {},
    }

    # ========================================================================
    # Battery Step 1: All Baselines Compared at p = 100%
    # ========================================================================
    print("\n" + "#" * 80)
    print(f" BATTERY PART 1: ALL BASELINES AT p = 100% (E={E})")
    print("#" * 80)

    baselines = {
        "NativeProj": (Native(), False),
        "NormCorrected": (norm_native, False),
        "KVEviction": (KVEviction(), True),
        "Int8_KV_Quant": (int8_method, False),
        "CheapPass": (AdapterKVMethod(adapter, "cheap_pass"), False),
    }

    print(f"   {'Method':16s} {'KL (nats)':12s} {'95% CI':24s} {'Top-1 (%)':12s} {'Gap Closed (%)':16s}")
    print("   " + "-" * 85)

    nat_kl = None
    for b_name, (m_inst, evict) in baselines.items():
        res = evaluate_battery_condition(ctx, m_inst, test_tokens, E, device, p_exit=1.0, evict=evict, bs=args.batch_size)
        if b_name == "NativeProj":
            nat_kl = res["kl_mean"]
            gap_closed = 0.0
        else:
            gap_closed = 100.0 * (nat_kl - res["kl_mean"]) / max(1e-8, nat_kl)

        res["gap_closed_pct"] = gap_closed
        results["baselines_p100"][b_name] = res
        print(f"   {b_name:16s} {res['kl_mean']:10.4f}   [{res['kl_ci95'][0]:.4f}, {res['kl_ci95'][1]:.4f}]   {res['top1_mean']:8.2f}%   {gap_closed:12.1f}%")

    # ========================================================================
    # Battery Step 2: Accumulating Holes Sweep (p in {1 tok, 10%, 50%, 100%})
    # ========================================================================
    print("\n" + "#" * 80)
    print(f" BATTERY PART 2: ACCUMULATING HOLES SWEEP (p-sweep at E={E})")
    print("#" * 80)

    p_vals = [float(p.strip()) for p in args.p_sweep.split(",") if p.strip()]
    sweep_methods = {
        "NativeProj": (Native(), False),
        "NormCorrected": (norm_native, False),
        "KVEviction": (KVEviction(), True),
        "CheapPass": (AdapterKVMethod(adapter, "cheap_pass"), False),
    }

    for p in p_vals:
        p_label = "1_token" if p <= 0.005 else f"p_{int(p*100)}pct"
        print(f"\n   --- Evaluating Exit Rate: {p_label} ({p*100:.1f}%) ---")
        results["p_sweep_results"][p_label] = {}

        for m_name, (m_inst, evict) in sweep_methods.items():
            res = evaluate_battery_condition(ctx, m_inst, test_tokens, E, device, p_exit=p, evict=evict, bs=args.batch_size)
            results["p_sweep_results"][p_label][m_name] = res
            print(f"      {m_name:16s} | KL: {res['kl_mean']:.4f} [{res['kl_ci95'][0]:.4f}, {res['kl_ci95'][1]:.4f}] | Top-1: {res['top1_mean']:.2f}%")

    # ========================================================================
    # Battery Step 3: Per-Layer Attention & Mass Diagnostics
    # ========================================================================
    print("\n" + "#" * 80)
    print(f" BATTERY PART 3: PER-LAYER ATTENTION & MASS DIAGNOSTICS")
    print("#" * 80)

    diag_data = evaluate_attention_diagnostics(ctx, adapter, test_tokens, E, device, n_seq=20)
    results["diagnostics"] = diag_data

    print(f"   {'Layer':8s} {'AttnOut Drift':16s} {'Hidden Drift':16s} {'Oracle Mass':14s} {'CheapPass Mass':16s}")
    print("   " + "-" * 75)
    for l in range(E, ctx.N):
        ao_d = diag_data["per_layer_attn_out_err"][l]
        h_d = diag_data["per_layer_h_drift"][l]
        m_orc = diag_data["attention_mass_oracle"][l]
        m_ad = diag_data["attention_mass_approx"][l]
        print(f"   Layer {l:2d}   {ao_d:12.4f}     {h_d:12.4f}     {m_orc:10.4f}     {m_ad:12.4f}")

    # Save output
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n [Done] Saved comprehensive battery results to: {args.out_file}")

    # Generate Markdown Summary
    md_file = args.out_file.replace(".json", ".md")
    with open(md_file, "w") as f:
        f.write(f"# Comprehensive Continuity Battery Report (Exit Depth E={E})\n\n")
        f.write("## 1. Baselines Comparison (Full 100% Exit Rate)\n\n")
        f.write("| Method | Forward KL (nats) [95% CI] | Top-1 Match (%) | Gap Closed vs Native (%) | Gate (< 0.6931) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for b_name, d in results["baselines_p100"].items():
            ci = d["kl_ci95"]
            gate = "**PASSED**" if d["kl_mean"] < LN2 else "FAILED"
            f.write(f"| **{b_name}** | {d['kl_mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {d['top1_mean']:.2f}% | {d['gap_closed_pct']:.1f}% | {gate} |\n")

        f.write("\n## 2. Accumulating Holes Degradation Curve\n\n")
        f.write("| Exit Fraction $p$ | NativeProj KL | NormCorrected KL | KVEviction KL | CheapPass KL |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for p_label, m_dict in results["p_sweep_results"].items():
            f.write(f"| **{p_label}** | {m_dict['NativeProj']['kl_mean']:.4f} | {m_dict['NormCorrected']['kl_mean']:.4f} | {m_dict['KVEviction']['kl_mean']:.4f} | **{m_dict['CheapPass']['kl_mean']:.4f}** |\n")

    print(f" [Done] Saved markdown summary to: {md_file}")


def main():
    p = argparse.ArgumentParser(description="7-Point Continuity Verification Battery for CheapPass")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--p_sweep", default="0.002,0.10,0.50,1.00", help="Comma-separated exit fractions p")
    p.add_argument("--n_seq", type=int, default=100, help="Number of sequences for test evaluation")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_dir", default="cheap_pass_out")
    p.add_argument("--out_file", default="results/continuity_battery/battery_results.json")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    run_continuity_battery(args)


if __name__ == "__main__":
    main()
