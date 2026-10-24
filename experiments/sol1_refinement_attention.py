#!/usr/bin/env python3
"""
experiments/sol1_refinement_attention.py

Solution 1: Refinement Attention Bridge (Attacks E_readout).

Problem:
  Intermediate hidden state h_L lives on an uncalibrated manifold (41-55% Top-1),
  lacking the higher-order contextual aggregation that layers L..N-1 would have provided.

Solution:
  Use h_L as a Query into the already-projected top-layer KV cache (K_hat_top, V_hat_top)
  to perform a single, fast cross-attention pass that refines h_L before the terminal
  unembedding head.

Usage:
  # Quick CPU self-test:
  python experiments/sol1_refinement_attention.py --selftest

  # GPU training & evaluation on Qwen2.5-3B:
  python experiments/sol1_refinement_attention.py --model Qwen/Qwen2.5-3B --depth 18 --train_steps 1000
"""

import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local imports work whether run from repo root or experiments/
curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curr_dir)
sys.path.insert(0, os.path.dirname(curr_dir))

from vkr2 import Ctx, Data, boot_ci, kl_tokens, make_scaler, n_params, set_seed


# -----------------------------------------------------------------------------
# Module: Refinement Attention Bridge
# -----------------------------------------------------------------------------
class RefinementAttentionBridge(nn.Module):
    """
    Refines intermediate hidden state h_L by cross-attending over the
    projected KV cache of the upper layers.
    """
    def __init__(self, d_model, n_heads, head_dim, n_kv_heads, d_ffn=None):
        super().__init__()
        self.d_model = d_model
        self.H = n_heads
        self.Hkv = n_kv_heads
        self.D = head_dim
        self.rep = self.H // self.Hkv
        d_ffn = d_ffn or (2 * d_model)

        # Cross-Attention over projected KV
        self.ln_q = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, self.H * self.D, bias=False)
        self.out_proj = nn.Linear(self.H * self.D, d_model, bias=False)

        # Light refinement MLP (SwiGLU)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

        # Zero-initialize the residual projections so initialization is an exact identity
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.down_proj.weight)

    def forward(self, h_L, kv_top, cos, sin):
        """
        h_L: [B, T, d_model] - intermediate hidden state at exit depth L
        kv_top: tuple (k_hat, v_hat) from top-layer native projection:
                k_hat: [B, Hkv, T, D] (already has RoPE applied)
                v_hat: [B, Hkv, T, D]
        cos, sin: rotary position embeddings for query alignment
        """
        B, T, _ = h_L.shape
        q_raw = self.q_proj(self.ln_q(h_L)).view(B, T, self.H, self.D).transpose(1, 2)

        # Apply RoPE to Query
        D = self.D
        q1, q2 = q_raw[..., : D // 2], q_raw[..., D // 2 :]
        rot = torch.cat((-q2, q1), dim=-1)
        q = q_raw * cos.unsqueeze(1) + rot * sin.unsqueeze(1)

        k_hat, v_hat = kv_top
        if self.rep > 1:
            k_hat = k_hat.repeat_interleave(self.rep, dim=1)
            v_hat = v_hat.repeat_interleave(self.rep, dim=1)

        # Causal Attention over top projected cache
        scale = self.D ** -0.5
        scores = torch.matmul(q, k_hat.transpose(-1, -2)).float() * scale

        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=h_L.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1).to(v_hat.dtype)

        attn_out = torch.matmul(attn_weights, v_hat)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.H * self.D)
        h_refined = h_L + self.out_proj(attn_out)

        # SwiGLU MLP stage
        h_norm = self.ln_mlp(h_refined)
        mlp_out = self.down_proj(F.silu(self.gate_proj(h_norm)) * self.up_proj(h_norm))
        return h_refined + mlp_out


class StandardMLPAdapter(nn.Module):
    """Standard baseline adapter (Tuned Lens paradigm) that operates purely token-by-token."""
    def __init__(self, d_model, width=None):
        super().__init__()
        width = width or (2 * d_model)
        self.ln = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, width)
        self.w2 = nn.Linear(width, d_model)
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)

    def forward(self, h_L):
        h32 = h_L.float()
        return h_L + self.w2(F.gelu(self.w1(self.ln(h32)))).to(h_L.dtype)


# -----------------------------------------------------------------------------
# Training & Evaluation
# -----------------------------------------------------------------------------
def train_bridge(ctx, bridge, L, data, args, device):
    bridge.train()
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = make_scaler(ctx.dtype == torch.float16 and device.type == "cuda", device)
    warmup_steps = min(50, args.train_steps // 10)

    t0 = time.time()
    for step in range(1, args.train_steps + 1):
        # Cosine learning rate schedule
        if step <= warmup_steps:
            lr_scale = float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(max(1, args.train_steps - warmup_steps))
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr * lr_scale

        ids = data.sample(args.batch).to(device)
        orc = ctx.oracle(ids)
        h_L = orc["hs"][L]

        # Top-layer native projection (target layer N - 1)
        top_idx = ctx.N - 1
        kv_top = ctx.project_kv(top_idx, h_L, orc["cos"], orc["sin"])

        # Forward pass through refinement bridge
        h_ref = bridge(h_L, kv_top, orc["cos"], orc["sin"])
        logits_ref = ctx.head(h_ref)[:, :-1]
        logits_orc = orc["logits"][:, :-1]

        loss = kl_tokens(logits_orc, logits_ref).mean()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % max(1, args.train_steps // 10) == 0 or step == args.train_steps:
            print(f"  [Bridge L={L}] Step {step:4d}/{args.train_steps} | KL Loss: {loss.item():.4f} | Elapsed: {time.time()-t0:.1f}s")


@torch.no_grad()
def evaluate_readouts(ctx, bridge, mlp_adapter, L, chunks, bs, device):
    """
    Evaluates 3 readout modes from intermediate hidden state h_L:
      1. Zero-shot direct readout: head(h_L)
      2. Standard token-local MLP Adapter: head(mlp(h_L))
      3. Refinement Attention Bridge: head(bridge(h_L, top_kv))
    """
    if bridge:
        bridge.eval()
    if mlp_adapter:
        mlp_adapter.eval()

    results = {"zero_shot": [], "mlp": [], "bridge": []}
    top_idx = ctx.N - 1

    for s in range(0, len(chunks), bs):
        ids = chunks[s : s + bs].to(device)
        orc = ctx.oracle(ids)
        h_L = orc["hs"][L]
        logits_orc = orc["logits"][:, :-1]
        orc_preds = logits_orc.argmax(-1)

        # 1. Zero-shot Readout
        lg_zero = ctx.head(h_L)[:, :-1]
        kl_zero = kl_tokens(logits_orc, lg_zero).mean(1).cpu().tolist()
        top1_zero = (lg_zero.argmax(-1) == orc_preds).float().mean(1).cpu().tolist()
        results["zero_shot"].append((kl_zero, top1_zero))

        # 2. Standard MLP Adapter
        if mlp_adapter is not None:
            h_mlp = mlp_adapter(h_L)
            lg_mlp = ctx.head(h_mlp)[:, :-1]
            kl_mlp = kl_tokens(logits_orc, lg_mlp).mean(1).cpu().tolist()
            top1_mlp = (lg_mlp.argmax(-1) == orc_preds).float().mean(1).cpu().tolist()
            results["mlp"].append((kl_mlp, top1_mlp))

        # 3. Refinement Attention Bridge
        if bridge is not None:
            kv_top = ctx.project_kv(top_idx, h_L, orc["cos"], orc["sin"])
            h_ref = bridge(h_L, kv_top, orc["cos"], orc["sin"])
            lg_ref = ctx.head(h_ref)[:, :-1]
            kl_ref = kl_tokens(logits_orc, lg_ref).mean(1).cpu().tolist()
            top1_ref = (lg_ref.argmax(-1) == orc_preds).float().mean(1).cpu().tolist()
            results["bridge"].append((kl_ref, top1_ref))

    out = {}
    for key in ["zero_shot", "mlp", "bridge"]:
        if not results[key]:
            continue
        all_kl = [x for batch in results[key] for x in batch[0]]
        all_t1 = [x for batch in results[key] for x in batch[1]]
        out[key] = {
            "kl": float(np.mean(all_kl)),
            "top1": float(np.mean(all_t1)) * 100.0,
            "kl_ci": boot_ci(all_kl, np.random.RandomState(0)),
            "top1_ci": boot_ci(all_t1, np.random.RandomState(0)),
        }
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Solution 1: Refinement Attention Bridge")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depth", type=str, default="18", help="Exit layer depth (e.g. 18 or 0.5)")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--train_steps", type=int, default=1000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--eval_n", type=int, default=60)
    p.add_argument("--eval_bs", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="vkr_sol1_out")
    p.add_argument("--no_c4", action="store_true")
    p.add_argument("--c4_docs", type=int, default=400)
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.selftest else "cpu")
    os.makedirs(args.out, exist_ok=True)

    if args.selftest:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(
            vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
            rope_theta=10000.0, tie_word_embeddings=False
        )
        model = Qwen2ForCausalLM(cfg).float().eval()
        tok = None
        args.depth = "4"
        args.train_steps = 30
        args.eval_n, args.eval_bs, args.seq_len = 12, 4, 32
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        dt = torch.bfloat16 if (device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(device).eval()

    for param in model.parameters():
        param.requires_grad_(False)

    ctx = Ctx(model)
    L = int(round(float(args.depth) * ctx.N)) if "." in args.depth else int(args.depth)
    print(f"\n=======================================================")
    print(f" Solution 1: Refinement Attention Bridge at Layer L={L}/{ctx.N}")
    print(f" Model: {args.model if not args.selftest else 'tiny-random'} | Device: {device}")
    print(f"=======================================================")

    data = Data(tok, args, model.config.vocab_size, need_train=True)

    # 1. Instantiate Refinement Bridge
    bridge = RefinementAttentionBridge(
        d_model=ctx.d,
        n_heads=ctx.H,
        head_dim=ctx.D,
        n_kv_heads=ctx.Hkv
    ).to(device)
    print(f"Refinement Attention Bridge initialized: {n_params(bridge):,} parameters")

    # 2. Instantiate Standard MLP Adapter baseline for comparison
    mlp_adapter = StandardMLPAdapter(d_model=ctx.d).to(device)
    print(f"Standard MLP Adapter initialized:        {n_params(mlp_adapter):,} parameters")

    # Train Bridge
    print(f"\n--- Training Refinement Attention Bridge ({args.train_steps} steps) ---")
    train_bridge(ctx, bridge, L, data, args, device)

    # Evaluate across evaluation domains
    print(f"\n--- Evaluating Readout Quality across Domains ---")
    for dom, chunks in data.eval.items():
        res = evaluate_readouts(ctx, bridge, mlp_adapter, L, chunks, args.eval_bs, device)
        print(f"\n[Domain: {dom.upper()}] (N={len(chunks)} sequences x {args.seq_len} tokens)")
        print(f"  {'Method':<28s} | {'Forward KL (nats)':<18s} | {'Top-1 Agreement':<18s}")
        print(f"  {'-'*28}-|-{'-'*18}-|-{'-'*18}")
        for method, m_data in res.items():
            kl_str = f"{m_data['kl']:.4f} [{m_data['kl_ci'][1]:.4f}, {m_data['kl_ci'][2]:.4f}]"
            t1_str = f"{m_data['top1']:.2f}%"
            print(f"  {method:<28s} | {kl_str:<18s} | {t1_str:<18s}")

    print("\n[Done] Solution 1 Refinement Attention experiment complete.")


if __name__ == "__main__":
    main()
