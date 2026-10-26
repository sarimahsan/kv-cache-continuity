#!/usr/bin/env python3
"""
vkr3.py -- attacking the missing-KV problem of early-exit / depth-adaptive Transformers from SEVEN angles.
Needs vkr2.py in the same folder (reuses its harness, data, metrics).  Everything is teacher-forced on a
frozen backbone unless stated; the reference is always the untouched full model.

WHY v3.  In v2, a ~0.3M-param repair (KVC) removes 70-90% of native-projection KL, but (a) the remaining KL is
not zero, (b) KL gains did not turn into generation gains, (c) nothing addresses the exiting token's own readout.
v3 asks WHY the residual remains and tests ways around it:

  A. CAPACITY vs INFORMATION      --exp rank      KVC rank sweep. KL keeps falling with params -> capacity-limited;
                                                  flat -> a token-local map from h_L cannot know the missing context.
  B. TRAJECTORY (shadow trunk)    method shadow*  tiny low-rank residual network that walks the residual stream from
                                                  L to N and emits KV at every layer (MimicNet / EAGLE-style feature
                                                  prediction).  Suffixes: S=shared weights (recurrent), X=causal
                                                  depthwise conv over previous tokens (ADEPT-style "inertia"), C=also
                                                  per-layer K/V corrections.   e.g. shadow, shadowS, shadowX, shadowSXC
  C. CONTEXT in the latent        method kvcx     KVC + causal depthwise conv over time in the latent code.
  D. ONE module for ALL depths    method kvcm     one generator conditioned on exit depth (deployment-realistic).
  E. CO-ADAPTATION (SwiftKV-like) method coadapt  KVC + LoRA on the UPPER layers (q,o,mlp[,kv]) trained jointly, so the
                                                  queries/MLPs learn to live with generated KV.  Reports the damage to
                                                  the model with EXACT history (selfcheck) - must stay ~0.
                                  method joint    coadapt + LoRA on LOWER layers + exit-readout adapter + exit loss:
                                                  the whole early-exit pipeline (readout AND KV) in one conversion.
                                  method exithead frozen backbone + readout adapter only (baseline for `joint`).
  F. STALENESS / BACKFILL curve   --exp refresh   history older than R tokens is EXACT (lazy backfill), the last R are
                                                  approximate.  Also protects the first k tokens (attention sinks;
                                                  prompts are always prefilled at full depth in practice).
  G. MIXED exits + on-policy      --exp mix       random fraction of history exact (some tokens didn't exit);
                                  --p_exact p     train with that mixture;  --onpolicy_n N  train on the model's OWN
                                                  samples (DAgger-style) instead of only WikiText.
  + --exp diag   per-layer K/V relative error + KL by token position (where does the damage live?).

Typical runs (Colab/Kaggle):
  # 0) sanity, no downloads, CPU, ~1 min
  python vkr3.py --selftest --exp main,refresh,mix,diag --methods adept_hsm,kvc,kvcx,shadow,shadowSX,kvcm,coadapt,joint,exithead
  # 1) cheap diagnostics: is it capacity or information?
  python vkr3.py --model Qwen/Qwen2.5-3B --depths 18,24 --exp rank --ranks 16,64,256,1024
  # 2) new repair modules vs the old best (reuses kvc/adept_hsm checkpoints from vkr2 via --ckpt_dir)
  python vkr3.py --model Qwen/Qwen2.5-3B --depths 18,24 --methods adept_hsm,kvc,kvcx,shadow,shadowS,shadowX --ckpt_dir vkr_out
  # 3) the big bet: co-adaptation (+ joint conversion with exit loss)
  python vkr3.py --model Qwen/Qwen2.5-3B --depths 18,24 --methods kvc,coadapt,joint,exithead --train_steps 2000 --batch 2
  # 4) staleness / backfill curve and mixed exits for whatever is trained
  python vkr3.py --model Qwen/Qwen2.5-3B --depths 18,24 --methods kvc --exp refresh,mix,diag
  # 5) on-policy training (generates samples from the full model first)
  python vkr3.py --model Qwen/Qwen2.5-3B --depths 18 --methods kvc --onpolicy_n 512 --p_exact 0.3
"""
import argparse
import contextlib
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from vkr2 import (Ctx, Native, Direct, OracleK, OracleV, HSM, VKR, KVC, n_params, kvc_rank, kl_tokens, vmse,
                  make_scaler, boot_ci, Data, set_seed, rollout_metrics, gen_nll)


# ============================================================================ args
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depths", default="18,24")
    p.add_argument("--exp", default="main", help="comma list of: main,rank,refresh,mix,diag  (or 'all' = main,refresh,mix,diag)")
    p.add_argument("--methods", default="adept_hsm,kvc,kvcx,shadow",
                   help="comma list: adept_hsm,kvc,kvcx,shadow[S][X][C],kvcm,coadapt,coshadow,joint,exithead")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--train_steps", type=int, default=1500)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--rank", type=int, default=64, help="VKR rank defining the parameter budget (hsm/kvc matched to it)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.5, help="weight of normalised V-MSE")
    p.add_argument("--beta", type=float, default=0.2, help="weight of hidden-trajectory loss (shadow*)")
    p.add_argument("--srank", type=int, default=16, help="latent rank of each shadow block")
    p.add_argument("--ctxk", type=int, default=4, help="causal conv kernel for kvcx / shadow*X")
    p.add_argument("--mrank", type=int, default=64, help="rank of the multi-depth kvcm")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_lr", type=float, default=2e-4)
    p.add_argument("--lora_targets", default="q,o,mlp", help="subset of q,o,mlp,kv")
    p.add_argument("--exit_loss", type=float, default=1.0, help="weight of the exit-readout KL (joint / exithead)")
    p.add_argument("--keep_w", type=float, default=0.0, help="weight of a KL(exact-history adapted model || teacher) term (coadapt/joint)")
    p.add_argument("--p_exact", type=float, default=0.0, help="train with this fraction of history positions EXACT (mixed exits)")
    p.add_argument("--train_first", type=int, default=0, help="train with a random 0..k of the FIRST tokens exact (prompt/attention-sink realism)")
    p.add_argument("--onpolicy_n", type=int, default=0, help="# self-generated training sequences (0 = off)")
    p.add_argument("--op_frac", type=float, default=0.5, help="fraction of train batches drawn from the on-policy pool")
    p.add_argument("--ranks", default="16,64,256,1024")
    p.add_argument("--refresh_list", default="1,2,4,8,16,32,64,0", help="R values (0 = never backfill)")
    p.add_argument("--first_list", default="0,1,4")
    p.add_argument("--pexact_list", default="0,0.25,0.5,0.75")
    p.add_argument("--eval_n", type=int, default=120)
    p.add_argument("--eval_bs", type=int, default=6)
    p.add_argument("--rollout_n", type=int, default=0)
    p.add_argument("--rollout_new", type=int, default=64)
    p.add_argument("--rollout_prompt", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="vkr3_out")
    p.add_argument("--ckpt_dir", default=None)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no_c4", action="store_true")
    p.add_argument("--c4_docs", type=int, default=800)
    p.add_argument("--selftest", action="store_true")
    return p.parse_args()


# ============================================================================ LoRA + context
class Holder:
    on = False


class LoRALinear(nn.Module):
    """y = base(x) + (alpha/r) * B A x, active only while holder.on (so the teacher pass is untouched)."""

    def __init__(self, base, r, alpha, holder):
        super().__init__()
        self.base, self.holder, self.scale = base, holder, alpha / r
        dev = base.weight.device
        self.A = nn.Parameter(torch.zeros(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        self.reset()

    def reset(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B.zero_()

    def forward(self, x):
        y = self.base(x)
        if not self.holder.on:
            return y
        return y + (F.linear(F.linear(x.float(), self.A), self.B) * self.scale).to(y.dtype)


class Ctx3(Ctx):
    def __init__(self, model):
        super().__init__(model)
        self.holder = Holder()
        self.lora = []          # list of (layer_idx, LoRALinear)

    # ---- LoRA management
    def install_lora(self, lmin, r, alpha, targets):
        targets = set(targets.split(","))
        for i in range(lmin, self.N):
            lay, a = self.layers[i], self.layers[i].self_attn
            def wrap(parent, name):
                m = LoRALinear(getattr(parent, name), r, alpha, self.holder)
                setattr(parent, name, m)
                self.lora.append((i, m))
            if "q" in targets:
                wrap(a, "q_proj")
            if "o" in targets:
                wrap(a, "o_proj")
            if "kv" in targets:
                wrap(a, "k_proj")
                wrap(a, "v_proj")
            if "mlp" in targets:
                for nm in ("gate_proj", "up_proj", "down_proj"):
                    wrap(lay.mlp, nm)
        print(f"  LoRA installed on layers {lmin}..{self.N - 1}, targets={sorted(targets)}, r={r}, "
              f"{sum(m.A.numel() + m.B.numel() for _, m in self.lora):,} params")

    def lora_params(self, lmin=0):
        return [p for i, m in self.lora if i >= lmin for p in (m.A, m.B)]

    def lora_reset(self):
        for _, m in self.lora:
            m.reset()

    def lora_state(self):
        return [(m.A.detach().cpu().clone(), m.B.detach().cpu().clone()) for _, m in self.lora]

    def lora_load(self, st):
        with torch.no_grad():
            for (_, m), (a, b) in zip(self.lora, st):
                m.A.copy_(a)
                m.B.copy_(b)

    @contextlib.contextmanager
    def lora_on(self, flag=True):
        prev = self.holder.on
        self.holder.on = flag
        try:
            yield
        finally:
            self.holder.on = prev

    # ---- teacher is ALWAYS the untouched model
    def oracle(self, ids):
        with self.lora_on(False):
            return super().oracle(ids)

    def run_lower(self, ids, L):
        h, cos, sin = self.embed(ids)
        for i in range(L):
            h, _ = self.layer_fwd(i, h, cos, sin)
        return h

    # ---- attention with a per-(query,key) "this history entry is exact" mask
    def layer_fwd(self, i, h, cos, sin, kv_hat=None, kv_ex=None, ehist=None):
        if ehist is None:
            return super().layer_fwd(i, h, cos, sin, kv_hat)
        lay, a = self.layers[i], self.layers[i].self_attn
        B, T, _ = h.shape
        n = lay.input_layernorm(h)
        q = self.rope(self.split(a.q_proj(n), self.H), cos, sin)
        k = self.rope(self.split(a.k_proj(n), self.Hkv), cos, sin)
        v = self.split(a.v_proj(n), self.Hkv)
        kh, vh = (k, v) if kv_hat is None else kv_hat
        khr, vhr, kr, vr = self.rep_kv(kh), self.rep_kv(vh), self.rep_kv(k), self.rep_kv(v)
        kxr, vxr = self.rep_kv(kv_ex[0]), self.rep_kv(kv_ex[1])
        scale = self.D ** -0.5
        E = ehist.view(1, 1, T, T)
        sc = torch.where(E, torch.matmul(q, kxr.transpose(-1, -2)).float(),
                         torch.matmul(q, khr.transpose(-1, -2)).float()) * scale
        diag = (q.float() * kr.float()).sum(-1) * scale
        eye = torch.eye(T, dtype=torch.bool, device=h.device)
        sc = torch.where(eye, diag.unsqueeze(-1), sc)
        sc = sc.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=h.device), 1), float("-inf"))
        A = torch.softmax(sc, dim=-1)
        a_diag = torch.diagonal(A, dim1=-2, dim2=-1)
        A_ex = A * E
        A_hat = A * (~E & ~eye)
        dt = vhr.dtype
        out = torch.matmul(A_ex.to(dt), vxr) + torch.matmul(A_hat.to(dt), vhr) + a_diag.unsqueeze(-1).to(dt) * vr
        out = out.transpose(1, 2).reshape(B, T, self.H * self.D)
        h = h + a.o_proj(out)
        h = h + lay.mlp(lay.post_attention_layernorm(h))
        return h, (k, v)

    def approx(self, orc, L, kv_hats, exact_mask=None, last_only=False, ehist=None):
        h, cos, sin = orc["hs"][L], orc["cos"], orc["sin"]
        T = h.shape[1]
        if ehist is None and exact_mask is not None:
            eye = torch.eye(T, dtype=torch.bool, device=h.device)
            ehist = exact_mask.view(1, -1).expand(T, -1) & ~eye
        for j, i in enumerate(range(L, self.N)):
            kv_ex = orc["kv"][i] if ehist is not None else None
            h, _ = self.layer_fwd(i, h, cos, sin, kv_hats[j], kv_ex, ehist)
        return self.head(h, last_only)


def hist_mask(T, device, refresh=0, first=0, exact_cols=None, all_exact=False):
    """[T,T] bool, True where query t reads the EXACT (oracle) K/V of key j (j<t)."""
    t = torch.arange(T, device=device)
    below = t.view(1, -1) < t.view(-1, 1)                       # j < t
    if all_exact:
        return below
    E = torch.zeros(T, T, dtype=torch.bool, device=device)
    if refresh and refresh > 0:
        E |= t.view(1, -1) < ((t // refresh) * refresh).view(-1, 1)
    if first > 0:
        E[:, :first] = True
    if exact_cols is not None:
        E |= exact_cols.view(1, -1)
    return E & below


# ============================================================================ new modules
class Shadow(nn.Module):
    """Tiny residual network that walks h_L -> h_N, emitting native (+optional corrected) KV at every skipped layer.
    name suffix letters: S shared weights, X causal depthwise conv over time, C per-layer K/V corrections."""

    def __init__(self, d, S, rank, kvdim, flags="", ctxk=4):
        super().__init__()
        self.flags, self.S, self.rank = flags, S, rank
        self.share, self.use_conv, self.corr = "S" in flags, "X" in flags, "C" in flags
        self.ctxk = ctxk
        nb = 1 if self.share else S
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(nb)])
        self.down = nn.ModuleList([nn.Linear(d, rank) for _ in range(nb)])
        self.up = nn.ModuleList([nn.Linear(rank, d) for _ in range(nb)])
        for u in self.up:
            nn.init.zeros_(u.weight)
            nn.init.zeros_(u.bias)
        self.emb = nn.Embedding(S, rank)
        nn.init.normal_(self.emb.weight, std=0.02)
        if self.use_conv:
            self.conv = nn.ModuleList([nn.Conv1d(rank, rank, ctxk, groups=rank) for _ in range(nb)])
            for c in self.conv:
                nn.init.normal_(c.weight, std=0.02)
                nn.init.zeros_(c.bias)
        if self.corr:
            self.up_k = nn.Parameter(torch.zeros(S, rank, kvdim))
            self.up_v = nn.Parameter(torch.zeros(S, rank, kvdim))
        self.name = "shadow" + flags
        self.hh = []

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        x, out, self.hh = h.float(), [], []
        for s, i in enumerate(range(L, ctx.N)):
            b = 0 if self.share else s
            z = self.down[b](self.ln[b](x)) + self.emb.weight[s]
            if self.use_conv:
                zc = F.pad(z.transpose(1, 2), (self.ctxk - 1, 0))
                z = z + self.conv[b](zc).transpose(1, 2)
            z = F.gelu(z)
            dk = dv = None
            if self.corr:
                dk, dv = z @ self.up_k[s], z @ self.up_v[s]
            out.append(ctx.project_kv(i, x.to(h.dtype), orc["cos"], orc["sin"], dk, dv))
            self.hh.append(x)
            x = x + self.up[b](z)
        return out

    @staticmethod
    def _rms(x):
        x = x.float()
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def aux(self, orc, L):
        """RMS-normalised hidden-state trajectory loss (what the next layer's input norm actually sees)."""
        loss = 0.0
        for s, i in enumerate(range(L, len(orc["kv"]))):
            if s == 0:
                continue
            loss = loss + (self._rms(self.hh[s]) - self._rms(orc["hs"][i])).pow(2).mean()
        return loss / max(1, len(self.hh) - 1)


class KVCx(KVC):
    """KVC whose latent code also sees the previous ctxk-1 tokens (causal depthwise conv)."""
    name = "kvcx"

    def __init__(self, d, S, rank, kvdim, ctxk=4):
        super().__init__(d, S, rank, kvdim)
        self.ctxk = ctxk
        self.conv = nn.Conv1d(rank, rank, ctxk, groups=rank)
        nn.init.normal_(self.conv.weight, std=0.02)
        nn.init.zeros_(self.conv.bias)

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        z0 = self.down(self.ln(h.float()))
        z0 = z0 + self.conv(F.pad(z0.transpose(1, 2), (self.ctxk - 1, 0))).transpose(1, 2)
        z = F.gelu(z0.unsqueeze(2) + self.emb.weight.unsqueeze(0).unsqueeze(0))
        dK = torch.einsum("btsr,srk->btsk", z, self.up_k)
        dV = torch.einsum("btsr,srk->btsk", z, self.up_v)
        return [ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=dK[:, :, s], dv=dV[:, :, s])
                for s, i in enumerate(range(L, ctx.N))]


class KVCMulti(nn.Module):
    """ONE generator for every exit depth: code = down(LN(h_L)) + layer-embedding + exit-depth-embedding."""
    name = "kvcm"

    def __init__(self, d, N, depths, rank, kvdim):
        super().__init__()
        self.depths, self.N, self.Lmin = list(depths), N, min(depths)
        self.ln = nn.LayerNorm(d)
        self.down = nn.Linear(d, rank)
        self.emb_l = nn.Embedding(N, rank)
        self.emb_e = nn.Embedding(len(self.depths), rank)
        nn.init.normal_(self.emb_l.weight, std=0.02)
        nn.init.normal_(self.emb_e.weight, std=0.02)
        self.up_k = nn.Parameter(torch.zeros(N - self.Lmin, rank, kvdim))
        self.up_v = nn.Parameter(torch.zeros(N - self.Lmin, rank, kvdim))

    def kv(self, ctx, orc, L):
        h = orc["hs"][L]
        e = self.depths.index(L)
        z0 = self.down(self.ln(h.float()))
        ls = torch.arange(L, ctx.N, device=h.device)
        z = F.gelu(z0.unsqueeze(2) + self.emb_l(ls).view(1, 1, -1, z0.shape[-1])
                   + self.emb_e.weight[e].view(1, 1, 1, -1))
        uk, uv = self.up_k[L - self.Lmin:], self.up_v[L - self.Lmin:]
        dK = torch.einsum("btsr,srk->btsk", z, uk)
        dV = torch.einsum("btsr,srk->btsk", z, uv)
        return [ctx.project_kv(i, h, orc["cos"], orc["sin"], dk=dK[:, :, s], dv=dV[:, :, s])
                for s, i in enumerate(range(L, ctx.N))]


class ExitHead(nn.Module):
    """Readout adapter: h_L -> logits (residual bottleneck, then the frozen final norm + lm_head)."""

    def __init__(self, d, width):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, width)
        self.w2 = nn.Linear(width, d)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def logits(self, ctx, h):
        h32 = h.float()
        return ctx.head((h32 + self.w2(F.gelu(self.w1(self.ln(h32))))).to(h.dtype))


class KVCLora(KVC):
    """KVC + LoRA on the upper layers (co-adaptation).  The LoRA weights live in ctx, state kept in .lora_st."""
    name = "coadapt"
    uses_lora, lora_lower, has_exit = True, False, False
    lora_st = None


class CoShadow(Shadow):
    """Shadow trunk + LoRA on the upper layers (co-adaptation on top of the best frozen-backbone repair)."""
    uses_lora, lora_lower, has_exit = True, False, False
    lora_st = None

    def __init__(self, d, S, rank, kvdim, flags="C", ctxk=4):
        super().__init__(d, S, rank, kvdim, flags, ctxk)
        self.name = "coshadow"


class KVCJoint(KVC):
    """Whole early-exit pipeline: LoRA on ALL layers + KVC + exit-readout adapter."""
    name = "joint"
    uses_lora, lora_lower, has_exit = True, True, True
    lora_st = None

    def __init__(self, d, S, rank, kvdim, exit_width):
        super().__init__(d, S, rank, kvdim)
        self.exit = ExitHead(d, exit_width)

    def exit_logits(self, ctx, h):
        return self.exit.logits(ctx, h)


class ExitOnly(nn.Module):
    """Frozen backbone + readout adapter only; KV is native.  Baseline for `joint`."""
    name = "exithead"
    uses_lora, lora_lower, has_exit, kv_trainable = False, False, True, False

    def __init__(self, d, exit_width):
        super().__init__()
        self.exit = ExitHead(d, exit_width)

    def kv(self, ctx, orc, L):
        return Native().kv(ctx, orc, L)

    def exit_logits(self, ctx, h):
        return self.exit.logits(ctx, h)


def flags_of(m):
    return (getattr(m, "uses_lora", False), getattr(m, "lora_lower", False), getattr(m, "has_exit", False),
            getattr(m, "kv_trainable", True))


# ============================================================================ builders
def budget_of(ctx, L, args):
    S = ctx.N - L
    return n_params(VKR(ctx.d, S, args.rank, ctx.kvdim)), S


def build(name, ctx, L, args, device):
    bud, S = budget_of(ctx, L, args)
    if name == "adept_hsm":
        m = HSM(ctx.d, max(8, int(round(bud / (2 * ctx.d + 1)))))
    elif name == "kvc":
        m = KVC(ctx.d, S, kvc_rank(bud, ctx.d, S, ctx.kvdim), ctx.kvdim)
    elif name == "kvcx":
        m = KVCx(ctx.d, S, kvc_rank(bud, ctx.d, S, ctx.kvdim), ctx.kvdim, args.ctxk)
    elif name.startswith("shadow"):
        m = Shadow(ctx.d, S, args.srank, ctx.kvdim, name[len("shadow"):], args.ctxk)
    elif name == "coshadow":
        m = CoShadow(ctx.d, S, args.srank, ctx.kvdim, "C", args.ctxk)
    elif name == "coadapt":
        m = KVCLora(ctx.d, S, kvc_rank(bud, ctx.d, S, ctx.kvdim), ctx.kvdim)
    elif name == "joint":
        m = KVCJoint(ctx.d, S, kvc_rank(bud, ctx.d, S, ctx.kvdim), ctx.kvdim, max(8, int(round(bud / (2 * ctx.d + 1)))))
    elif name == "exithead":
        m = ExitOnly(ctx.d, max(8, int(round(bud / (2 * ctx.d + 1)))))
    else:
        raise ValueError(name)
    m.name = name if not name.startswith("shadow") else m.name
    return m.to(device)


# ============================================================================ data helpers
class MixData:
    def __init__(self, base, op, frac):
        self.base, self.op, self.frac = base, op, frac
        self.eval = base.eval

    def sample(self, B):
        if self.op is not None and random.random() < self.frac:
            return self.op[torch.randint(0, len(self.op), (B,))]
        return self.base.sample(B)


@torch.no_grad()
def make_onpolicy(model, data, n, T, device, args, prompt_len=16, bs=16):
    outs, eos = [], (model.config.eos_token_id if isinstance(model.config.eos_token_id, int) else 0)
    pad = model.config.pad_token_id if model.config.pad_token_id is not None else eos
    while sum(len(o) for o in outs) < n:
        st = torch.randint(0, len(data.train) - prompt_len - 1, (bs,))
        prompts = torch.stack([data.train[s:s + prompt_len] for s in st]).to(device)
        g = model.generate(input_ids=prompts, attention_mask=torch.ones_like(prompts), max_new_tokens=T - prompt_len,
                           min_new_tokens=T - prompt_len, do_sample=True, temperature=1.0, top_p=0.95, pad_token_id=pad)
        outs.append(g[:, :T].cpu())
    return torch.cat(outs)[:n]


# ============================================================================ training
def make_student(ctx, orc, ids, L, lower):
    """orc-like dict whose hs[L] is the STUDENT's exit state (differs from teacher only if lower LoRA is on)."""
    if not lower:
        return orc
    hL = ctx.run_lower(ids, L)
    st = dict(orc)
    st["hs"] = list(orc["hs"])
    st["hs"][L] = hL
    return st


def train3(ctx, method, L, data, args, device):
    uses_lora, lower, has_exit, kv_tr = flags_of(method)
    multi = isinstance(method, KVCMulti)
    kv_params = [p for p in method.parameters() if p.requires_grad]
    groups = [dict(params=kv_params, lr=args.lr)]
    if uses_lora:
        ctx.lora_reset()
        lp = ctx.lora_params(0 if lower else L)
        groups.append(dict(params=lp, lr=args.lora_lr))
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    total = args.train_steps
    warm = max(1, min(50, total // 10))
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = make_scaler(ctx.dtype == torch.float16 and device.type == "cuda", device)
    every = max(1, total // 10)
    acc = dict(kl=0.0, v=0.0, aux=0.0, ex=0.0)
    t0 = time.time()
    all_params = [p for g in groups for p in g["params"]]
    for step in range(1, total + 1):
        f = min(1.0, step / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * f
        Ls = random.choice(method.depths) if multi else L
        ids = data.sample(args.batch).to(device)
        T = ids.shape[1]
        orc = ctx.oracle(ids)
        loss = 0.0
        with ctx.lora_on(uses_lora):
            st = make_student(ctx, orc, ids, Ls, lower)
            if kv_tr:
                kvh = method.kv(ctx, st, Ls)
                ehist = None
                if args.p_exact > 0 or args.train_first > 0:
                    fk = random.randint(0, args.train_first) if args.train_first > 0 else 0
                    cols = (torch.rand(T, device=device) < args.p_exact) if args.p_exact > 0 else None
                    ehist = hist_mask(T, device, first=fk, exact_cols=cols)
                logits = ctx.approx(st, Ls, kvh, ehist=ehist)
                kl = kl_tokens(orc["logits"][:, :-1], logits[:, :-1]).mean()
                vm = vmse(orc, kvh, Ls)
                loss = loss + kl + args.alpha * vm
                acc["kl"] += kl.item()
                acc["v"] += vm.item()
                if hasattr(method, "aux"):
                    au = method.aux(st, Ls)
                    loss = loss + args.beta * au
                    acc["aux"] += float(au.detach()) if torch.is_tensor(au) else float(au)
                if uses_lora and args.keep_w > 0:
                    lg_ex = ctx.approx(st, Ls, kvh, ehist=hist_mask(T, device, all_exact=True))
                    loss = loss + args.keep_w * kl_tokens(orc["logits"][:, :-1], lg_ex[:, :-1]).mean()
            if has_exit:
                ex = kl_tokens(orc["logits"][:, :-1], method.exit_logits(ctx, st["hs"][Ls])[:, :-1]).mean()
                loss = loss + args.exit_loss * ex
                acc["ex"] += ex.item()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        scaler.step(opt)
        scaler.update()
        if step % every == 0 or step == total:
            n = every if step % every == 0 else step % every
            print(f"    [{method.name} L={L}] step {step}/{total}  " + "  ".join(
                f"{k} {v / n:.4f}" for k, v in acc.items() if v) + f"  ({time.time() - t0:.0f}s)", flush=True)
            acc = {k: 0.0 for k in acc}
    if uses_lora:
        method.lora_st = ctx.lora_state()


def ckpt_path(args, ckpt_dir, name, L):
    tag = f"_s{args.seed}" if args.seed else ""
    vt = ("_op" if args.onpolicy_n else "") + (f"_pe{args.p_exact}" if args.p_exact else "")\
        + (f"_tf{args.train_first}" if args.train_first else "") \
        + (f"_kw{args.keep_w}" if args.keep_w else "")
    sfx = f"_r{args.srank}" if name.startswith(("shadow", "coshadow")) else ""
    if name in ("adept_hsm", "kvc"):                 # share vkr2 checkpoints when the recipe is the default one
        vt = vt if vt else ""
    return os.path.join(ckpt_dir, f"{name}{sfx}_L{L}{vt}{tag}.pt")


def get_method(name, ctx, L, data, args, device, ckpt_dir, cache, depths=None):
    key = (name, L if name != "kvcm" else -1)
    if key in cache:
        return cache[key]
    if name == "kvcm":
        m = KVCMulti(ctx.d, ctx.N, depths, args.mrank, ctx.kvdim).to(device)
        path = os.path.join(ckpt_dir, f"kvcm_{'-'.join(map(str, depths))}_r{args.mrank}.pt")
    else:
        m = build(name, ctx, L, args, device)
        path = ckpt_path(args, ckpt_dir, name, L)
    print(f"  {name}: {n_params(m):,} params" + (f" (+ LoRA {sum(p.numel() for p in ctx.lora_params(0 if flags_of(m)[1] else L)):,})"
                                               if flags_of(m)[0] else ""))
    if os.path.exists(path) and not args.retrain:
        sd = torch.load(path, map_location=device)
        if isinstance(sd, dict) and "m" in sd:
            m.load_state_dict(sd["m"])
            m.lora_st = sd["lora"]
        else:
            m.load_state_dict(sd)
        print(f"  loaded {path}")
    elif args.eval_only:
        raise SystemExit(f"--eval_only but checkpoint missing: {path}")
    else:
        print(f"  training {name} (no checkpoint at {path})")
        set_seed(args.seed * 1000 + L)
        train3(ctx, m, L if name != "kvcm" else depths[0], data, args, device)
        torch.save({"m": m.state_dict(), "lora": m.lora_st} if flags_of(m)[0] else m.state_dict(), path)
    m.eval()
    cache[key] = m
    return m


# ============================================================================ evaluation
def activate(ctx, m):
    if m is not None and getattr(m, "uses_lora", False):
        ctx.lora_load(m.lora_st)


@torch.no_grad()
def evaluate3(ctx, method, L, chunks, bs, device, refresh=0, first=0, p_exact=0.0, all_exact=False, seed=0):
    """KL(full || approx) per sequence (+ per-position KL, top-1).  method=None means the full model with exact history."""
    uses_lora, lower, _, _ = flags_of(method) if method is not None else (False, False, False, True)
    activate(ctx, method)
    g = torch.Generator(device="cpu").manual_seed(seed)
    kl_seq, top1, kl_pos, nb = [], [], 0.0, 0
    for s in range(0, len(chunks), bs):
        ids = chunks[s:s + bs].to(device)
        T = ids.shape[1]
        orc = ctx.oracle(ids)
        with ctx.lora_on(uses_lora):
            st = make_student(ctx, orc, ids, L, lower)
            kvh = method.kv(ctx, st, L)
            ehist = None
            if all_exact:
                ehist = hist_mask(T, device, all_exact=True)
            elif refresh or first or p_exact > 0:
                cols = (torch.rand(T, generator=g) < p_exact).to(device) if p_exact > 0 else None
                ehist = hist_mask(T, device, refresh=refresh, first=first, exact_cols=cols)
            lg = ctx.approx(st, L, kvh, ehist=ehist)
        lg, rl = lg[:, :-1], orc["logits"][:, :-1]
        k = kl_tokens(rl, lg)
        kl_seq += k.mean(1).cpu().tolist()
        kl_pos = kl_pos + k.sum(0).cpu().numpy()
        top1 += (lg.argmax(-1) == rl.argmax(-1)).float().mean(1).cpu().tolist()
        nb += k.shape[0]
    return dict(kl=float(np.mean(kl_seq)), top1=100 * float(np.mean(top1)), kl_seq=np.array(kl_seq), kl_pos=kl_pos / nb)


@torch.no_grad()
def eval_exit(ctx, method, L, chunks, bs, device):
    activate(ctx, method)
    uses_lora, lower, _, _ = flags_of(method)
    kls, t1 = [], []
    for s in range(0, len(chunks), bs):
        ids = chunks[s:s + bs].to(device)
        orc = ctx.oracle(ids)
        with ctx.lora_on(uses_lora):
            st = make_student(ctx, orc, ids, L, lower)
            lg = method.exit_logits(ctx, st["hs"][L])[:, :-1]
        rl = orc["logits"][:, :-1]
        kls += kl_tokens(rl, lg).mean(1).cpu().tolist()
        t1 += (lg.argmax(-1) == rl.argmax(-1)).float().mean(1).cpu().tolist()
    return dict(kl=float(np.mean(kls)), top1=100 * float(np.mean(t1)))


@torch.no_grad()
def layer_errors(ctx, method, L, chunks, device, n=16):
    activate(ctx, method)
    uses_lora, lower, _, _ = flags_of(method)
    ids = chunks[:n].to(device)
    orc = ctx.oracle(ids)
    with ctx.lora_on(uses_lora):
        st = make_student(ctx, orc, ids, L, lower)
        kvh = method.kv(ctx, st, L)
    rows = []
    for j, i in enumerate(range(L, ctx.N)):
        row = []
        for a, b in zip(kvh[j], orc["kv"][i]):
            a, b = a.float(), b.float()
            rel = ((a - b).pow(2).sum() / b.pow(2).sum()).item()
            cos = F.cosine_similarity(a.flatten(0, 1), b.flatten(0, 1), dim=-1).mean().item()
            row += [rel, cos]
        rows.append([i] + row)          # layer, relK, cosK, relV, cosV
    return rows


@torch.no_grad()
def rollout3(ctx, method, L, prompts, n_new, device):
    outs = []
    uses_lora, lower, _, _ = flags_of(method) if method is not None else (False, False, False, True)
    activate(ctx, method)
    for p in prompts:
        cur = p.unsqueeze(0).to(device)
        plen = cur.shape[1]
        for _ in range(n_new):
            orc = ctx.oracle(cur)
            if method is None:
                lg = orc["logits"][:, -1]
            else:
                mask = torch.arange(cur.shape[1], device=device) < plen
                with ctx.lora_on(uses_lora):
                    st = make_student(ctx, orc, cur, L, lower)
                    lg = ctx.approx(st, L, method.kv(ctx, st, L), exact_mask=mask, last_only=True)[:, -1]
            cur = torch.cat([cur, lg.argmax(-1, keepdim=True)], dim=1)
        outs.append(cur[0, plen:].cpu())
    return outs


def fmt_ci(x):
    return f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"


# ============================================================================ main
def main():
    args = parse_args()
    set_seed(args.seed)
    ckpt_dir = args.ckpt_dir or args.out
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    exps = ["main", "refresh", "mix", "diag"] if args.exp == "all" else [e.strip() for e in args.exp.split(",")]
    want = [m.strip() for m in args.methods.split(",") if m.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() and not args.selftest else "cpu")

    if args.selftest:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                          rope_theta=10000.0, tie_word_embeddings=False)
        model, tok = Qwen2ForCausalLM(cfg).float().eval(), None
        if args.depths == "18,24":
            args.depths = "0.5,0.75"
        args.train_steps = min(args.train_steps, 40)
        args.eval_n, args.eval_bs, args.seq_len = 16, 4, 48
        args.rollout_prompt, args.rollout_new = 8, 8
        args.rollout_n = args.rollout_n or 2
        args.ranks, args.refresh_list = "8,32", "1,4,0"
        args.srank = min(args.srank, 8)
        args.onpolicy_n = args.onpolicy_n
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        dt = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if device.type == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt)
        model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    ctx = Ctx3(model)
    print(f"model={args.model if not args.selftest else 'tiny-random-qwen2'} layers={ctx.N} d={ctx.d} H={ctx.H} "
          f"Hkv={ctx.Hkv} D={ctx.D} dtype={ctx.dtype} device={device}")
    depths = []
    for x in args.depths.split(","):
        x = x.strip()
        L = int(round(float(x) * ctx.N)) if "." in x else int(x)
        assert 1 <= L < ctx.N
        depths.append(L)

    data = Data(tok, args, model.config.vocab_size, need_train=not args.eval_only)
    train_data = data
    if args.onpolicy_n > 0 and not args.eval_only:
        opath = os.path.join(ckpt_dir, f"onpolicy_{args.model.replace('/', '_')}_{args.onpolicy_n}_T{args.seq_len}.pt")
        if os.path.exists(opath):
            op = torch.load(opath)
        else:
            print(f"  generating {args.onpolicy_n} on-policy sequences from the full model ...")
            op = make_onpolicy(model, data, args.onpolicy_n, args.seq_len, device, args)
            torch.save(op, opath)
        train_data = MixData(data, op.cpu(), args.op_frac)

    lora_methods = [m for m in want if m in ("coadapt", "joint", "coshadow")]
    if lora_methods:
        ctx.install_lora(0 if "joint" in lora_methods else min(depths), args.lora_r, args.lora_alpha, args.lora_targets)

    # ---- sanity: new attention path must reproduce the old one
    dom0 = list(data.eval)[0]
    ids = data.eval[dom0][:2].to(device)
    orc = ctx.oracle(ids)
    Lq = depths[0]
    lg = ctx.approx(orc, Lq, Native().kv(ctx, orc, Lq), ehist=hist_mask(ids.shape[1], device, all_exact=True))
    k1 = kl_tokens(orc["logits"][:, :-1], lg[:, :-1]).mean().item()
    lg0 = ctx.approx(orc, Lq, Native().kv(ctx, orc, Lq))
    lg1 = ctx.approx(orc, Lq, Native().kv(ctx, orc, Lq), ehist=hist_mask(ids.shape[1], device))   # no exact entries
    k2 = kl_tokens(lg0[:, :-1], lg1[:, :-1]).mean().item()
    sh = Shadow(ctx.d, ctx.N - Lq, 4, ctx.kvdim, "SXC", 4).to(device)
    ks, kn = sh.kv(ctx, orc, Lq), Native().kv(ctx, orc, Lq)
    k3 = max((a - b).abs().max().item() for (a, _), (b, _) in zip(ks, kn))
    print(f"sanity: KL(all-exact history vs oracle)={k1:.2e}  KL(mask path vs plain path)={k2:.2e}  "
          f"max|shadow@init - native|={k3:.2e}")
    tol = 0.05 if ctx.dtype != torch.float32 else 1e-4
    if k1 > tol or k2 > tol or k3 > (1e-2 if ctx.dtype != torch.float32 else 1e-4):
        raise SystemExit("SANITY CHECK FAILED")

    rng = np.random.RandomState(0)
    cache, report = {}, {}
    fixed = {"oracleK": OracleK(), "oracleV": OracleV(), "native": Native(), "direct": Direct()}

    def paired(a, b):
        return boot_ci(a["kl_seq"] - b["kl_seq"], rng)

    # ------------------------------------------------------------------ rank sweep (capacity vs information)
    if "rank" in exps:
        L = depths[0]
        report["rank"] = {}
        print(f"\n##### RANK SWEEP at L={L}: does KL keep falling with capacity?")
        bud, S = budget_of(ctx, L, args)
        for r in [int(x) for x in args.ranks.split(",")]:
            m = KVC(ctx.d, S, r, ctx.kvdim).to(device)
            m.name = f"kvc_r{r}"
            path = os.path.join(ckpt_dir, f"kvc_rank{r}_L{L}.pt")
            if os.path.exists(path) and not args.retrain:
                m.load_state_dict(torch.load(path, map_location=device))
            elif args.eval_only:
                raise SystemExit(f"missing {path}")
            else:
                set_seed(args.seed * 1000 + L)
                train3(ctx, m, L, train_data, args, device)
                torch.save(m.state_dict(), path)
            m.eval()
            report["rank"][r] = {}
            for dom, ch in data.eval.items():
                e = evaluate3(ctx, m, L, ch, args.eval_bs, device)
                report["rank"][r][dom] = dict(kl=e["kl"], top1=e["top1"], params=n_params(m))
                print(f"  rank {r:5d} params {n_params(m):>11,}  [{dom}] KL {e['kl']:.4f}  top1 {e['top1']:.2f}")
        print("  read: KL still dropping at rank 1024 -> capacity-limited (bigger/smarter generators help);\n"
              "        flat -> information-limited (needs context / co-adaptation / exact backfill).")

    # ------------------------------------------------------------------ per-depth experiments
    need_methods = any(e in exps for e in ("main", "refresh", "mix", "diag"))
    for L in (depths if need_methods else []):
        print(f"\n=== exit depth L={L}/{ctx.N} ({100 * L / ctx.N:.0f}%) ===")
        built = {}
        for n in want:
            built[n] = get_method(n, ctx, L, train_data, args, device, ckpt_dir, cache, depths=depths)
        allm = {**fixed, **built}
        rep = report.setdefault(L, {})

        if "main" in exps:
            for dom, ch in data.eval.items():
                res = {n: evaluate3(ctx, m, L, ch, args.eval_bs, device) for n, m in allm.items()}
                nat = res["native"]["kl"]
                out = {}
                print(f"  [{dom}] N={len(ch)}  KL vs full model (teacher-forced, attention-only)")
                for n, r in res.items():
                    mean, lo, hi = boot_ci(r["kl_seq"], rng)
                    o = dict(kl=r["kl"], ci=(lo, hi), top1=r["top1"], red=100 * (nat - r["kl"]) / nat,
                             params=n_params(allm[n]) if not isinstance(allm[n], (Native, Direct, OracleK, OracleV)) else 0)
                    ex = ""
                    if "kvc" in res and n in built and n != "kvc":
                        o["dkl_vs_kvc"] = paired(r, res["kvc"])
                        ex += f"  vs kvc {fmt_ci(o['dkl_vs_kvc'])}"
                    if "adept_hsm" in res and n in built and n != "adept_hsm":
                        o["dkl_vs_hsm"] = paired(r, res["adept_hsm"])
                        ex += f"  vs hsm {fmt_ci(o['dkl_vs_hsm'])}"
                    out[n] = o
                    print(f"    {n:12s} KL {r['kl']:.4f} [{lo:.4f},{hi:.4f}] top1 {r['top1']:6.2f} red {o['red']:5.1f}% "
                          f"params {o['params']:>10,d}{ex}")
                ek, ev = res["oracleK"]["kl"], res["oracleV"]["kl"]
                out["_value_share"] = 100 * ek / (ek + ev) if ek + ev > 0 else float("nan")
                # damage check for converted models: adapted model, EXACT history, vs untouched model
                for n, m in built.items():
                    if getattr(m, "uses_lora", False):
                        e = evaluate3(ctx, m, L, ch, args.eval_bs, device, all_exact=True)
                        out[n]["selfcheck_exact_kl"] = e["kl"]
                        print(f"    {n:12s} selfcheck: adapted model with EXACT history vs original: KL {e['kl']:.4f} "
                              f"(must be ~0, else the conversion damaged the model)")
                    if getattr(m, "has_exit", False):
                        ex_m = eval_exit(ctx, m, L, ch, args.eval_bs, device)
                        out[n]["exit_kl"], out[n]["exit_top1"] = ex_m["kl"], ex_m["top1"]
                        print(f"    {n:12s} EXIT readout from h_L: KL {ex_m['kl']:.4f}  top1 {ex_m['top1']:.2f}%")
                rep.setdefault("main", {})[dom] = out

            if args.rollout_n > 0:
                prompts = [c[: args.rollout_prompt] for c in data.eval[dom0][: args.rollout_n]]
                ref = rollout3(ctx, None, L, prompts, args.rollout_new, device)
                ref_nll = float(np.mean(gen_nll(ctx, prompts, ref, device)))
                print(f"  rollouts ({len(prompts)} prompts x {args.rollout_new}): full-model text NLL {ref_nll:.3f}")
                gens = {}
                for n, m in {"direct": Direct(), "native": Native(), **built}.items():
                    if getattr(m, "kv_trainable", True) is False and n != "exithead":
                        continue
                    g = rollout3(ctx, m, L, prompts, args.rollout_new, device)
                    gens[n] = g
                    met = rollout_metrics(ref, g, args.rollout_new)
                    nl = gen_nll(ctx, prompts, g, device)
                    mm, lo, hi = boot_ci(met["match_each"], rng)
                    nm, nlo, nhi = boot_ci(nl, rng)
                    print(f"    {n:12s} match {mm:6.2f}% [{lo:.1f},{hi:.1f}]  first-div {met['first_div']:.1f}  "
                          f"genNLL {nm:.3f} [{nlo:.3f},{nhi:.3f}]")
                    rep.setdefault("rollout", {})[n] = dict(match=mm, match_ci=(lo, hi), first_div=met["first_div"],
                                                            gen_nll=nm, gen_nll_ci=(nlo, nhi), ref_nll=ref_nll,
                                                            tokens=[x.tolist() for x in g])

        if "refresh" in exps:
            print(f"  --- STALENESS / BACKFILL: history older than R tokens is EXACT (R=0: never), first-k tokens exact")
            rep["refresh"] = {}
            for n, m in {"native": Native(), **built}.items():
                if getattr(m, "kv_trainable", True) is False and n != "exithead":
                    continue
                for dom, ch in data.eval.items():
                    row = []
                    for R in [int(x) for x in args.refresh_list.split(",")]:
                        e = evaluate3(ctx, m, L, ch, args.eval_bs, device, refresh=R)
                        row.append((R, e["kl"]))
                        rep["refresh"].setdefault(n, {}).setdefault(dom, {})[R] = e["kl"]
                    print(f"    {n:12s} [{dom}] " + "  ".join(f"R={R}:{k:.4f}" for R, k in row))
                    row = []
                    for k0 in [int(x) for x in args.first_list.split(",")]:
                        e = evaluate3(ctx, m, L, ch, args.eval_bs, device, first=k0)
                        row.append((k0, e["kl"]))
                        rep["refresh"].setdefault(n + "_first", {}).setdefault(dom, {})[k0] = e["kl"]
                    print(f"    {n:12s} [{dom}] exact-first-k: " + "  ".join(f"k={k}:{v:.4f}" for k, v in row))

        if "mix" in exps:
            print(f"  --- MIXED EXITS: random fraction p of history positions are EXACT (those tokens did not exit)")
            rep["mix"] = {}
            for n, m in {"native": Native(), **built}.items():
                if getattr(m, "kv_trainable", True) is False and n != "exithead":
                    continue
                for dom, ch in data.eval.items():
                    row = []
                    for pe in [float(x) for x in args.pexact_list.split(",")]:
                        e = evaluate3(ctx, m, L, ch, args.eval_bs, device, p_exact=pe)
                        row.append((pe, e["kl"]))
                        rep["mix"].setdefault(n, {}).setdefault(dom, {})[pe] = e["kl"]
                    print(f"    {n:12s} [{dom}] " + "  ".join(f"p={p}:{k:.4f}" for p, k in row))

        if "diag" in exps:
            print(f"  --- DIAGNOSTICS: per-layer relative K/V error (||a-b||^2/||b||^2) and KL by token position")
            rep["diag"] = {}
            for n, m in {"native": Native(), **built}.items():
                if getattr(m, "kv_trainable", True) is False and n != "exithead":
                    continue
                rows = layer_errors(ctx, m, L, data.eval[dom0], device)
                rep["diag"][n] = dict(layers=rows)
                mid = rows[len(rows) // 2]
                print(f"    {n:12s} mean relK {np.mean([r[1] for r in rows]):.3f} cosK {np.mean([r[2] for r in rows]):.3f} | "
                      f"relV {np.mean([r[3] for r in rows]):.3f} cosV {np.mean([r[4] for r in rows]):.3f}   (layer {mid[0]}: "
                      f"relK {mid[1]:.3f} relV {mid[3]:.3f})")
                e = evaluate3(ctx, m, L, data.eval[dom0], args.eval_bs, device)
                kp = e["kl_pos"]
                T = len(kp)
                bins = [(0, 1), (1, 4), (4, 16), (16, 64), (64, T)]
                rep["diag"][n]["kl_by_pos"] = {f"{a}-{b}": float(kp[a:b].mean()) for a, b in bins if a < T}
                print(f"    {n:12s} KL by position: " + "  ".join(f"[{a},{min(b, T)}):{kp[a:b].mean():.3f}" for a, b in bins if a < T))

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open(os.path.join(args.out, "results_v3.json"), "w") as f:
        json.dump(dict(args=vars(args), report=clean(report)), f, indent=1)
    print(f"\nDone. Wrote {args.out}/results_v3.json  (checkpoints in {ckpt_dir})")
    print("How to read: 'dKL vs kvc' < 0 with a CI excluding 0 => beats the v2 champion.  'selfcheck' must be ~0.\n"
          "A repair only counts as a SOLUTION if genNLL (rollouts) approaches the full model's, not just KL.")


if __name__ == "__main__":
    main()