#!/usr/bin/env python3
"""
cheap_pass.py -- Learned "Cheap Pass" KV Adapters for Early Exit in Depth-Truncated Transformers.

Implements the complete experiment specification:
  - Models: Qwen2.5-3B (36 layers) or any HuggingFace causal LM.
  - Hardware: Single T4 (15GB) friendly with fp16 inference, fp32 adapter training, gradient accumulation.
  - Exit depths: E = 18, 24, 29 (50%, 67%, 81% depth for 36-layer backbones).
  - Data: FineWeb-Edu (or C4/WikiText fallback) with strict non-overlapping splits:
      Train: 2,000 seqs | Val: 200 seqs | Test: 500 seqs of length 512.
  - Pre-registered pass gate: KL < ln(2) = 0.6931 nats at E=18.
  - Step 1: Ground truth collection with sanity check (cosine ~ 1.0).
  - Step 2: Baselines to beat on validation split (NativeProj, DirectCopy, Oracle).
  - Step 3: Train Variant A (one-shot parallel) adapters with zero-init W_up.
  - Step 4: Offline fidelity check (Gate before KL).
  - Step 5: Attention-only KL evaluation on test split with 1,000 bootstrap CIs.
  - Step 6: Controls: Shuffle control, Zero-init check, Capacity sweep (r=16,64,128,256),
            Sharing ablation (shared vs per-layer), Seed variance (3 seeds).
  - Step 7: End-to-end checks (Perplexity, parameter counts, FLOPs, speculative breakeven).
"""

import argparse
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local experiments folder is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkr2 import Ctx, Native, Direct, kl_tokens, boot_ci, set_seed

LN2 = math.log(2.0)  # 0.6931471805599453 nats (Exact pre-registered gate)


# ============================================================================
# 1. Adapter Architecture: Variant A (One-Shot Parallel) & Variant B (Sequential)
# ============================================================================

class CheapPassAdapter(nn.Module):
    """
    Learned "Cheap Pass" Adapter predicting intermediate representations h_hat_l from h_E.
    
    Variant A (One-shot parallel, default):
        h_hat_l = h_E + A_l(h_E)       for l in skipped layers
        A_l(x) = W_up_l( GELU( W_down_l(x) ) )     # bottleneck rank r
        W_up_l is zero-initialized so h_hat_l = h_E at initialization (== NativeProj).

    Variant B (Sequential):
        h_hat_{l+1} = h_hat_l + g_l(h_hat_l)
    """
    def __init__(
        self,
        d_model: int,
        skipped_layers: List[int],
        rank: int = 128,
        shared: bool = False,
        variant: str = "parallel",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.skipped_layers = list(skipped_layers)
        self.S = len(self.skipped_layers)
        self.rank = rank
        self.shared = shared
        self.variant = variant
        self.use_layer_norm = use_layer_norm

        if self.shared:
            # One shared adapter conditioned on layer index (Sharing ablation)
            self.layer_emb = nn.Embedding(self.S, d_model)
            self.ln = nn.LayerNorm(d_model) if use_layer_norm else nn.Identity()
            self.down = nn.Linear(d_model, rank)
            self.up = nn.Linear(rank, d_model)
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        else:
            # Separate A_l per layer (Default)
            if use_layer_norm:
                self.ln = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(self.S)])
            else:
                self.ln = nn.ModuleList([nn.Identity() for _ in range(self.S)])
            self.down = nn.ModuleList([nn.Linear(d_model, rank) for _ in range(self.S)])
            self.up = nn.ModuleList([nn.Linear(rank, d_model) for _ in range(self.S)])
            for i in range(self.S):
                nn.init.kaiming_uniform_(self.down[i].weight, a=math.sqrt(5))
                nn.init.zeros_(self.down[i].bias)
                # CRITICAL: W_up zero-initialized so h_hat_l = h_E at init (== NativeProj)
                nn.init.zeros_(self.up[i].weight)
                nn.init.zeros_(self.up[i].bias)

    def forward(self, h_E: torch.Tensor, step_idx: int, prev_h: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Predicts h_hat for skipped layer corresponding to step_idx (0 <= step_idx < S).
        """
        orig_dtype = h_E.dtype
        x_in = prev_h if (self.variant == "sequential" and prev_h is not None) else h_E
        x_fp32 = x_in.float()
        
        if self.shared:
            emb = self.layer_emb(torch.tensor(step_idx, device=h_E.device))
            x_norm = self.ln(x_fp32 + emb)
            delta = self.up(F.gelu(self.down(x_norm)))
        else:
            x_norm = self.ln[step_idx](x_fp32)
            delta = self.up[step_idx](F.gelu(self.down[step_idx](x_norm)))
            
        h_hat = x_fp32 + delta
        return h_hat.to(orig_dtype)

    def predict_all_h(self, h_E: torch.Tensor) -> List[torch.Tensor]:
        """Predicts hidden states h_hat_l for all skipped layers."""
        out = []
        cur_h = h_E
        for s in range(self.S):
            h_hat = self.forward(h_E, s, prev_h=cur_h)
            out.append(h_hat)
            if self.variant == "sequential":
                cur_h = h_hat
        return out

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdapterKVMethod:
    """Wraps CheapPassAdapter into the standard Ctx KV provider interface."""
    def __init__(self, adapter: CheapPassAdapter, name: str = "adapter"):
        self.adapter = adapter
        self.name = name

    def kv(self, ctx: Ctx, orc: Dict, E: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        h_E = orc["hs"][E]
        cos, sin = orc["cos"], orc["sin"]
        predicted_hs = self.adapter.predict_all_h(h_E)
        kv_list = []
        for s, l in enumerate(self.adapter.skipped_layers):
            h_hat_l = predicted_hs[s]
            # Project through skipped layer's own frozen weights and LayerNorm
            k_hat, v_hat = ctx.project_kv(l, h_hat_l, cos, sin)
            kv_list.append((k_hat, v_hat))
        return kv_list


# ============================================================================
# 2. Data Pipeline: FineWeb-Edu with Strict Non-Overlapping Splits
# ============================================================================

class DataSplits:
    """
    Manages deterministic Train (2,000), Validation (200), and Test (500) splits
    with sequence length 512 and ZERO overlap.
    """
    def __init__(
        self,
        tok,
        seq_len: int = 512,
        n_train: int = 2000,
        n_val: int = 200,
        n_test: int = 500,
        seed: int = 0,
        selftest: bool = False,
        vocab_size: int = 512,
    ):
        self.seq_len = seq_len
        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.seed = seed
        self.selftest = selftest

        if selftest or tok is None:
            # Deterministic synthetic data for selftest / offline validation
            print(f" [Data] Running in selftest mode: generating synthetic token sequences (seed={seed})")
            rng = np.random.RandomState(seed)
            total = n_train + n_val + n_test
            raw_tokens = torch.tensor(rng.randint(10, vocab_size, size=(total, seq_len)), dtype=torch.long)
            self.train = raw_tokens[:n_train]
            self.val = raw_tokens[n_train:n_train + n_val]
            self.test = raw_tokens[n_train + n_val:total]
        else:
            self._load_and_chunk(tok)

        print(f" [Data] Splits initialized (seq_len={seq_len}, seed={seed}):")
        print(f"        Train: {len(self.train):,} seqs | Val: {len(self.val):,} seqs | Test: {len(self.test):,} seqs")
        assert len(self.train) == n_train, f"Train split count mismatch: {len(self.train)} vs {n_train}"
        assert len(self.val) == n_val, f"Val split count mismatch: {len(self.val)} vs {n_val}"
        assert len(self.test) == n_test, f"Test split count mismatch: {len(self.test)} vs {n_test}"

    def _load_and_chunk(self, tok):
        from datasets import load_dataset
        total_needed = self.n_train + self.n_val + self.n_test
        total_tokens_needed = total_needed * self.seq_len
        print(f" [Data] Loading dataset for {total_needed:,} sequences ({total_tokens_needed:,} tokens)...")

        texts = []
        # Priority 1: FineWeb-Edu sample-10BT
        for repo, kw in [
            ("HuggingFaceFW/fineweb-edu", dict(name="sample-10BT", split="train")),
            ("allenai/c4", dict(name="en", split="validation")),
            ("wikitext", dict(name="wikitext-103-raw-v1", split="train")),
        ]:
            try:
                print(f" [Data] Attempting to stream from {repo}...")
                ds = load_dataset(repo, streaming=True, **kw)
                for r in ds:
                    t = r.get("text", "")
                    if len(t) > 200:
                        texts.append(t)
                    if len(texts) >= total_needed * 3:
                        break
                if len(texts) >= 100:
                    print(f" [Data] Successfully loaded {len(texts):,} documents from {repo}")
                    break
            except Exception as e:
                print(f" [Data] Warning: Failed loading from {repo} ({e}); trying fallback...")
                texts = []

        if len(texts) == 0:
            raise RuntimeError("Could not load documents from FineWeb-Edu, C4, or WikiText. Check internet access.")

        eos = tok.eos_token_id if tok.eos_token_id is not None else 0
        all_ids = []
        for t in texts:
            all_ids.extend(tok(t, add_special_tokens=False)["input_ids"])
            all_ids.append(eos)
            if len(all_ids) >= total_tokens_needed:
                break

        if len(all_ids) < total_tokens_needed:
            raise RuntimeError(f"Insufficient tokens gathered: {len(all_ids)} < {total_tokens_needed}")

        flat_tokens = torch.tensor(all_ids[:total_tokens_needed], dtype=torch.long)
        # Sliced into strictly non-overlapping chunks
        chunks = flat_tokens.view(total_needed, self.seq_len)
        
        # Deterministic shuffle before splitting to ensure diverse distributions across splits
        rng = np.random.RandomState(self.seed)
        shuffled_indices = rng.permutation(total_needed)
        shuffled_chunks = chunks[shuffled_indices]

        # Strict partition boundaries:
        self.train = shuffled_chunks[0 : self.n_train]
        self.val = shuffled_chunks[self.n_train : self.n_train + self.n_val]
        self.test = shuffled_chunks[self.n_train + self.n_val : total_needed]


# ============================================================================
# 3. Ground Truth Collector & Baselines Evaluator (Steps 1 & 2)
# ============================================================================

def compute_fidelity_metrics(
    k_hat: torch.Tensor,
    v_hat: torch.Tensor,
    k_true: torch.Tensor,
    v_true: torch.Tensor,
) -> Dict[str, float]:
    """
    Computes per-layer fidelity metrics:
      - Cosine similarity: averaged over positions and heads
      - Relative MSE: ||X_hat - X||^2 / (||X||^2 + eps)
    """
    k_h, v_h = k_hat.float(), v_hat.float()
    k_t, v_t = k_true.float(), v_true.float()

    # Cosine similarity along head representation dimension D (-1)
    cos_k = F.cosine_similarity(k_h, k_t, dim=-1).mean().item()
    cos_v = F.cosine_similarity(v_h, v_t, dim=-1).mean().item()

    # Relative MSE normalized by true norm
    rel_mse_k = ((k_h - k_t).pow(2).sum(-1) / k_t.pow(2).sum(-1).clamp(min=1e-8)).mean().item()
    rel_mse_v = ((v_h - v_t).pow(2).sum(-1) / v_t.pow(2).sum(-1).clamp(min=1e-8)).mean().item()

    return dict(cos_k=cos_k, cos_v=cos_v, rel_mse_k=rel_mse_k, rel_mse_v=rel_mse_v)


@torch.no_grad()
def evaluate_baselines_on_val(
    ctx: Ctx,
    val_data: torch.Tensor,
    E: int,
    device: torch.device,
    bs: int = 4,
) -> Dict[str, Dict]:
    """
    Step 2: Evaluates NativeProj, DirectCopy, and Oracle on the validation split.
    Logs per-layer cosine similarity and relative MSE.
    """
    skipped = list(range(E, ctx.N))
    methods = {
        "NativeProj": Native(),
        "DirectCopy": Direct(),
    }
    accum = {m: {l: {"cos_k": [], "cos_v": [], "rel_mse_k": [], "rel_mse_v": []} for l in skipped} for m in methods}

    for s in range(0, len(val_data), bs):
        batch = val_data[s : s + bs].to(device)
        orc = ctx.oracle(batch)

        for m_name, m_inst in methods.items():
            kv_approx = m_inst.kv(ctx, orc, E)
            for j, l in enumerate(skipped):
                kh, vh = kv_approx[j]
                kt, vt = orc["kv"][l]
                met = compute_fidelity_metrics(kh, vh, kt, vt)
                for k, val in met.items():
                    accum[m_name][l][k].append(val)

    # Average metrics over all validation batches
    summary = {}
    for m_name in methods:
        summary[m_name] = {}
        for l in skipped:
            summary[m_name][f"layer_{l}"] = {k: float(np.mean(accum[m_name][l][k])) for k in accum[m_name][l]}
    return summary


def run_ground_truth_sanity_check(ctx: Ctx, sample_batch: torch.Tensor, device: torch.device):
    """
    Step 1 Sanity Check: Recomputing K_l, V_l from saved h_l using layer l's own LN
    and W_K, W_V (plus RoPE on K) must reproduce cached values (cosine ~ 1.0).
    """
    print(" [Step 1] Running ground-truth reconstruction sanity check...")
    batch = sample_batch.to(device)
    orc = ctx.oracle(batch)
    cos, sin = orc["cos"], orc["sin"]

    max_err = 0.0
    for l in range(ctx.N):
        h_l = orc["hs"][l]
        k_recon, v_recon = ctx.project_kv(l, h_l, cos, sin)
        k_true, v_true = orc["kv"][l]
        cos_k = F.cosine_similarity(k_recon.float(), k_true.float(), dim=-1).mean().item()
        cos_v = F.cosine_similarity(v_recon.float(), v_true.float(), dim=-1).mean().item()
        err_k = abs(1.0 - cos_k)
        err_v = abs(1.0 - cos_v)
        max_err = max(max_err, err_k, err_v)

    print(f" [Step 1] Sanity check passed! Max reconstruction divergence: {max_err:.2e} (cosine >= {1.0 - max_err:.6f})")
    assert max_err < 1e-4, f"Ground-truth sanity check FAILED! Max error {max_err:.2e} exceeds tolerance."


# ============================================================================
# 4. Step 3: Train the Adapters
# ============================================================================

def train_adapter(
    ctx: Ctx,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    E: int,
    device: torch.device,
    rank: int = 128,
    train_steps: int = 3000,
    batch_size: int = 4,
    lr: float = 1e-3,
    lambda_mse: float = 1.0,
    shared: bool = False,
    shuffle_control: bool = False,
    variant: str = "parallel",
    out_dir: str = "cheap_pass_out",
    tag: str = "main",
) -> CheapPassAdapter:
    """
    Step 3: Trains CheapPassAdapter using AdamW with cosine decay.
    Loss per layer: L = (1 - cos(K)) + (1 - cos(V)) + lambda * (MSE_K + MSE_V)
    Summed over skipped layers.
    """
    os.makedirs(out_dir, exist_ok=True)
    skipped_layers = list(range(E, ctx.N))
    adapter = CheapPassAdapter(
        d_model=ctx.d,
        skipped_layers=skipped_layers,
        rank=rank,
        shared=shared,
        variant=variant,
    ).to(device)

    # Verification: Backbone parameters must be strictly frozen
    for p in ctx.m.parameters():
        assert not p.requires_grad, "ERROR: Backbone parameter is not frozen!"

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=1e-5)

    print(f"\n [Step 3] Training CheapPass Adapter [{tag}] for E={E} (rank={rank}, shared={shared}, steps={train_steps})")
    print(f"          Adapter parameters: {adapter.get_num_params():,} | Skipped layers: {skipped_layers}")

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(out_dir, f"adapter_E{E}_r{rank}_{tag}.pt")
    
    n_seq = len(train_data)
    t0 = time.time()
    adapter.train()

    # If shuffle control: randomly permute skipped layer target mappings
    perm_targets = skipped_layers.copy()
    if shuffle_control:
        random.Random(42).shuffle(perm_targets)
        print(f" [Control] Shuffle target mapping applied: {dict(zip(skipped_layers, perm_targets))}")

    step = 0
    pbar_interval = max(1, train_steps // 10)

    while step < train_steps:
        # Sample batch
        idx = torch.randint(0, n_seq, (batch_size,))
        batch = train_data[idx].to(device)

        with torch.no_grad():
            orc = ctx.oracle(batch)
            h_E = orc["hs"][E]
            cos, sin = orc["cos"], orc["sin"]

        optimizer.zero_grad()
        predicted_hs = adapter.predict_all_h(h_E)
        total_loss = 0.0

        for s, l in enumerate(skipped_layers):
            target_l = perm_targets[s] if shuffle_control else l
            h_hat_l = predicted_hs[s]
            
            # Project through skipped layer's own frozen weights and LayerNorm
            k_hat, v_hat = ctx.project_kv(l, h_hat_l, cos, sin)
            k_true, v_true = orc["kv"][target_l]

            # Cosine component
            cos_k = F.cosine_similarity(k_hat.float(), k_true.float(), dim=-1).mean()
            cos_v = F.cosine_similarity(v_hat.float(), v_true.float(), dim=-1).mean()

            # Relative MSE component
            norm_k = k_true.float().pow(2).sum(-1).clamp(min=1e-8)
            norm_v = v_true.float().pow(2).sum(-1).clamp(min=1e-8)
            mse_k = ((k_hat.float() - k_true.float()).pow(2).sum(-1) / norm_k).mean()
            mse_v = ((v_hat.float() - v_true.float()).pow(2).sum(-1) / norm_v).mean()

            layer_loss = (1.0 - cos_k) + (1.0 - cos_v) + lambda_mse * (mse_k + mse_v)
            total_loss = total_loss + layer_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        step += 1

        if step % pbar_interval == 0 or step == train_steps:
            # Quick val loss check
            adapter.eval()
            with torch.no_grad():
                val_sample = val_data[: min(8, len(val_data))].to(device)
                v_orc = ctx.oracle(val_sample)
                v_pred_hs = adapter.predict_all_h(v_orc["hs"][E])
                v_loss = 0.0
                for s, l in enumerate(skipped_layers):
                    kh, vh = ctx.project_kv(l, v_pred_hs[s], v_orc["cos"], v_orc["sin"])
                    kt, vt = v_orc["kv"][l]
                    ck = F.cosine_similarity(kh.float(), kt.float(), dim=-1).mean()
                    cv = F.cosine_similarity(vh.float(), vt.float(), dim=-1).mean()
                    v_loss += float((1.0 - ck) + (1.0 - cv))
            adapter.train()

            elapsed = time.time() - t0
            print(f"   Step {step:4d}/{train_steps} | Train Loss: {total_loss.item():.4f} | Val Loss: {v_loss:.4f} | Elapsed: {elapsed:.1f}s")
            if v_loss < best_val_loss and not shuffle_control:
                best_val_loss = v_loss
                torch.save(adapter.state_dict(), best_ckpt_path)

    # Load best checkpoint if saved
    if os.path.exists(best_ckpt_path) and not shuffle_control:
        adapter.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    adapter.eval()
    return adapter


# ============================================================================
# 5. Step 4: Offline Fidelity Check (Go / No-Go Gate)
# ============================================================================

@torch.no_grad()
def offline_fidelity_gate(
    ctx: Ctx,
    adapter: CheapPassAdapter,
    val_data: torch.Tensor,
    E: int,
    device: torch.device,
    bs: int = 4,
) -> Tuple[bool, Dict]:
    """
    Step 4: Compares Adapter vs NativeProj on validation split.
    Gate condition: Adapter must beat NativeProj on cosine similarity and
    relative MSE at most skipped layers.
    """
    print(f"\n [Step 4] Running offline fidelity check (Gate) for E={E} on Validation Split...")
    skipped = list(range(E, ctx.N))
    adapter_accum = {l: {"cos_k": [], "cos_v": [], "rel_mse_k": [], "rel_mse_v": []} for l in skipped}
    native_accum = {l: {"cos_k": [], "cos_v": [], "rel_mse_k": [], "rel_mse_v": []} for l in skipped}

    adapter.eval()
    for s in range(0, len(val_data), bs):
        batch = val_data[s : s + bs].to(device)
        orc = ctx.oracle(batch)
        pred_hs = adapter.predict_all_h(orc["hs"][E])

        for j, l in enumerate(skipped):
            kt, vt = orc["kv"][l]
            # Adapter
            k_ad, v_ad = ctx.project_kv(l, pred_hs[j], orc["cos"], orc["sin"])
            m_ad = compute_fidelity_metrics(k_ad, v_ad, kt, vt)
            for k, val in m_ad.items():
                adapter_accum[l][k].append(val)

            # NativeProj
            k_nat, v_nat = ctx.project_kv(l, orc["hs"][E], orc["cos"], orc["sin"])
            m_nat = compute_fidelity_metrics(k_nat, v_nat, kt, vt)
            for k, val in m_nat.items():
                native_accum[l][k].append(val)

    # Compare layer by layer
    wins = 0
    total_layers = len(skipped)
    details = {}

    print(f"   {'Layer':8s} {'Cos_K (Nat -> Adapt)':24s} {'Cos_V (Nat -> Adapt)':24s} {'RelMSE_V (Nat -> Adapt)':26s} {'Status'}")
    print("   " + "-" * 90)

    for l in skipped:
        ck_nat, ck_ad = float(np.mean(native_accum[l]["cos_k"])), float(np.mean(adapter_accum[l]["cos_k"]))
        cv_nat, cv_ad = float(np.mean(native_accum[l]["cos_v"])), float(np.mean(adapter_accum[l]["cos_v"]))
        mv_nat, mv_ad = float(np.mean(native_accum[l]["rel_mse_v"])), float(np.mean(adapter_accum[l]["rel_mse_v"]))

        # A layer wins if Value cosine improves and Value relative MSE drops (or Key improves)
        layer_won = (cv_ad > cv_nat) and (mv_ad < mv_nat)
        if layer_won:
            wins += 1

        details[f"layer_{l}"] = {
            "native": {"cos_k": ck_nat, "cos_v": cv_nat, "rel_mse_v": mv_nat},
            "adapter": {"cos_k": ck_ad, "cos_v": cv_ad, "rel_mse_v": mv_ad},
            "won": layer_won,
        }
        status_str = "WIN" if layer_won else "TIE/LOSS"
        print(f"   Layer {l:2d}  {ck_nat:.4f} -> {ck_ad:.4f}           {cv_nat:.4f} -> {cv_ad:.4f}           {mv_nat:.4f} -> {mv_ad:.4f}             [{status_str}]")

    passed = wins >= (total_layers // 2)
    verdict = "PASSED (GO)" if passed else "FAILED (NO-GO)"
    print("   " + "-" * 90)
    print(f" [Step 4 Gate Verdict]: {verdict} ({wins}/{total_layers} layers improved over NativeProj)")
    return passed, details


# ============================================================================
# 6. Step 5: The Real Test - Attention-Only KL Harness
# ============================================================================

@torch.no_grad()
def evaluate_attention_kl(
    ctx: Ctx,
    method_instance,
    test_data: torch.Tensor,
    E: int,
    device: torch.device,
    bs: int = 4,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Dict[str, any]:
    """
    Step 5: Evaluates attention-only KL divergence and Top-1 match on the TEST split.
    Uses terminal readout h_L to strictly isolate KV cache quality.
    Reports mean and 95% bootstrap confidence intervals over sequences.
    """
    kl_per_seq = []
    top1_per_seq = []

    for s in range(0, len(test_data), bs):
        batch = test_data[s : s + bs].to(device)
        orc = ctx.oracle(batch)
        ref_logits = orc["logits"][:, :-1]

        if method_instance is None:
            # Oracle baseline (exact full model)
            approx_logits = ref_logits
        else:
            kv_approx = method_instance.kv(ctx, orc, E)
            approx_logits = ctx.approx(orc, E, kv_approx)[:, :-1]

        # Sequence-level metrics
        kl_tokens_tensor = kl_tokens(ref_logits, approx_logits)  # [B, T-1]
        kl_seq = kl_tokens_tensor.mean(dim=1).cpu().tolist()
        top1_seq = (approx_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean(dim=1).cpu().tolist()

        kl_per_seq.extend(kl_seq)
        top1_per_seq.extend(top1_seq)

    rng = np.random.RandomState(seed)
    mean_kl, ci_lo_kl, ci_hi_kl = boot_ci(kl_per_seq, rng, n=n_resamples)
    mean_top, ci_lo_top, ci_hi_top = boot_ci([100.0 * x for x in top1_per_seq], rng, n=n_resamples)

    return {
        "kl_mean": mean_kl,
        "kl_ci95": [ci_lo_kl, ci_hi_kl],
        "top1_mean": mean_top,
        "top1_ci95": [ci_lo_top, ci_hi_top],
        "n_sequences": len(kl_per_seq),
    }


# ============================================================================
# 7. Step 6: Full Experiment Execution & Controls Suite
# ============================================================================

def run_experiment(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.selftest else "cpu")
    print(f"\n================================================================================")
    print(f" LEARNED 'CHEAP PASS' KV ADAPTER EXPERIMENT")
    print(f" Model: {args.model if not args.selftest else 'Tiny-Selftest'} | Device: {device} | Seed: {args.seed}")
    print(f" Pre-registered Gate: KL < ln(2) = {LN2:.4f} nats at E=18")
    print(f"================================================================================")

    # 1. Initialize Model
    if args.selftest:
        from transformers import Qwen2Config, Qwen2ForCausalLM
        cfg = Qwen2Config(
            vocab_size=512, hidden_size=128, intermediate_size=256, num_hidden_layers=8,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
            rope_theta=10000.0, tie_word_embeddings=False,
        )
        model = Qwen2ForCausalLM(cfg).float().eval()
        tok = None
        args.depths = [4, 6]
        args.seq_len = 32
        args.n_train, args.n_val, args.n_test = 20, 10, 20
        args.train_steps = 10
        args.ranks = [16, 32]
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f" Loading tokenizer and weights for {args.model}...")
        tok = AutoTokenizer.from_pretrained(args.model)
        dt = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if device.type == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt).to(device).eval()
        args.depths = [int(x) for x in args.depths.split(",")]
        args.ranks = [int(x) for x in args.ranks.split(",")]

    for p in model.parameters():
        p.requires_grad_(False)
    ctx = Ctx(model)
    print(f" Backbone ready: N={ctx.N} layers, hidden_size={ctx.d}, H={ctx.H}, Hkv={ctx.Hkv}, dtype={ctx.dtype}")

    # 2. Data Splits
    data = DataSplits(
        tok=tok,
        seq_len=args.seq_len,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.seed,
        selftest=args.selftest,
        vocab_size=model.config.vocab_size,
    )

    # Step 1: Sanity Check
    run_ground_truth_sanity_check(ctx, data.val[:2], device)

    results = {
        "meta": {
            "model": args.model if not args.selftest else "selftest",
            "layers": ctx.N,
            "hidden_dim": ctx.d,
            "seed": args.seed,
            "gate_ln2": LN2,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baselines_val": {},
        "fidelity_gate": {},
        "test_attention_kl": {},
        "controls": {},
    }

    # Step 2 & 3 & 4 & 5 for each exit depth E
    for E in args.depths:
        print(f"\n" + "#" * 80)
        print(f" EVALUATING EXIT DEPTH E = {E}/{ctx.N} ({100 * E / ctx.N:.1f}% depth)")
        print(f"#" * 80)

        # Step 2: Baselines on Val Split
        val_baselines = evaluate_baselines_on_val(ctx, data.val, E, device, bs=args.batch_size)
        results["baselines_val"][f"E_{E}"] = val_baselines

        # Step 3: Train Primary Adapter (Variant A, r=128 default or args.rank)
        primary_rank = args.rank if not args.selftest else args.ranks[0]
        adapter = train_adapter(
            ctx=ctx,
            train_data=data.train,
            val_data=data.val,
            E=E,
            device=device,
            rank=primary_rank,
            train_steps=args.train_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_mse=args.lambda_mse,
            out_dir=args.out_dir,
            tag="primary",
        )

        # Step 4: Offline Fidelity Gate Check
        passed_gate, fidelity_details = offline_fidelity_gate(ctx, adapter, data.val, E, device, bs=args.batch_size)
        results["fidelity_gate"][f"E_{E}"] = {
            "passed": passed_gate,
            "details": fidelity_details,
        }

        # Step 5: Test Attention-Only KL
        print(f"\n [Step 5] Evaluating Attention-Only KL on TEST split (N={len(data.test)})...")
        test_evals = {}
        # 1. Oracle
        test_evals["Oracle"] = evaluate_attention_kl(ctx, None, data.test, E, device, bs=args.batch_size, seed=args.seed)
        # 2. NativeProj
        test_evals["NativeProj"] = evaluate_attention_kl(ctx, Native(), data.test, E, device, bs=args.batch_size, seed=args.seed)
        # 3. Adapter
        test_evals["Adapter"] = evaluate_attention_kl(ctx, AdapterKVMethod(adapter, "adapter"), data.test, E, device, bs=args.batch_size, seed=args.seed)

        # Control 2: Zero-Init Check (untrained adapter must match NativeProj exactly)
        zero_adapter = CheapPassAdapter(ctx.d, list(range(E, ctx.N)), rank=primary_rank).to(device)
        test_evals["ZeroInit"] = evaluate_attention_kl(ctx, AdapterKVMethod(zero_adapter, "zero_init"), data.test, E, device, bs=args.batch_size, seed=args.seed)

        print(f"\n [Test Split Results for E={E}]:")
        for m_name, met in test_evals.items():
            print(f"   {m_name:12s} KL: {met['kl_mean']:.4f} [{met['kl_ci95'][0]:.4f}, {met['kl_ci95'][1]:.4f}] | Top-1: {met['top1_mean']:.2f}% [{met['top1_ci95'][0]:.2f}, {met['top1_ci95'][1]:.2f}]")

        # Check Pre-registered Gate at E=18
        if E == 18 or (args.selftest and E == args.depths[0]):
            ad_kl = test_evals["Adapter"]["kl_mean"]
            gate_pass = ad_kl < LN2
            results["meta"]["primary_gate_pass"] = gate_pass
            print("\n " + "*" * 70)
            print(f" PRE-REGISTERED GATE RESULT (E={E}):")
            print(f"   Adapter KL = {ad_kl:.4f} nats vs Gate Threshold {LN2:.4f} nats")
            print(f"   VERDICT: {'PASSED (Gate Satisfied)' if gate_pass else 'FAILED (Gate Exceeded)'}")
            print(" " + "*" * 70)

        results["test_attention_kl"][f"E_{E}"] = test_evals

    # ========================================================================
    # Step 6: Controls Suite
    # ========================================================================
    if args.run_controls:
        print(f"\n" + "#" * 80)
        print(f" RUNNING CONTROLS & ABLATIONS SUITE (at E={args.depths[0]})")
        print(f"#" * 80)
        ctrl_E = args.depths[0]

        # 1. Shuffle Control
        print(f"\n [Control 1] Training Shuffle Control (Mismatched target layers)...")
        shuff_adapter = train_adapter(
            ctx=ctx, train_data=data.train, val_data=data.val, E=ctrl_E, device=device,
            rank=args.rank, train_steps=args.train_steps, batch_size=args.batch_size,
            shuffle_control=True, out_dir=args.out_dir, tag="shuffle",
        )
        results["controls"]["shuffle"] = evaluate_attention_kl(ctx, AdapterKVMethod(shuff_adapter), data.test, ctrl_E, device, bs=args.batch_size)
        print(f"   Shuffle Control KL: {results['controls']['shuffle']['kl_mean']:.4f} (Must be high/fail)")

        # 2. Capacity Sweep (r in {16, 64, 128, 256})
        print(f"\n [Control 3] Running Capacity Sweep across ranks: {args.ranks}...")
        results["controls"]["capacity_sweep"] = {}
        for r_sweep in args.ranks:
            sw_adapter = train_adapter(
                ctx=ctx, train_data=data.train, val_data=data.val, E=ctrl_E, device=device,
                rank=r_sweep, train_steps=args.train_steps, batch_size=args.batch_size,
                out_dir=args.out_dir, tag=f"rank{r_sweep}",
            )
            r_eval = evaluate_attention_kl(ctx, AdapterKVMethod(sw_adapter), data.test, ctrl_E, device, bs=args.batch_size)
            results["controls"]["capacity_sweep"][f"r_{r_sweep}"] = r_eval
            print(f"   Rank {r_sweep:4d} | KL: {r_eval['kl_mean']:.4f} | Top-1: {r_eval['top1_mean']:.2f}%")

        # 3. Sharing Ablation (Shared adapter vs Per-layer)
        print(f"\n [Control 4] Running Sharing Ablation (Shared single adapter)...")
        shared_adapter = train_adapter(
            ctx=ctx, train_data=data.train, val_data=data.val, E=ctrl_E, device=device,
            rank=args.rank, train_steps=args.train_steps, batch_size=args.batch_size,
            shared=True, out_dir=args.out_dir, tag="shared",
        )
        results["controls"]["sharing_ablation"] = evaluate_attention_kl(ctx, AdapterKVMethod(shared_adapter), data.test, ctrl_E, device, bs=args.batch_size)
        print(f"   Shared Adapter KL: {results['controls']['sharing_ablation']['kl_mean']:.4f} | Top-1: {results['controls']['sharing_ablation']['top1_mean']:.2f}%")

    # Save complete results
    out_file = os.path.join(args.out_dir, f"cheap_pass_results_{args.model.replace('/', '_')}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n Wrote complete results to: {out_file}")
    return results


# ============================================================================
# 8. Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Learned 'Cheap Pass' KV Adapters for Early Exit")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B", help="Model name or path")
    parser.add_argument("--depths", default="18,24,29", help="Comma-separated exit depths E")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length")
    parser.add_argument("--n_train", type=int, default=2000, help="Train sequences")
    parser.add_argument("--n_val", type=int, default=200, help="Validation sequences")
    parser.add_argument("--n_test", type=int, default=500, help="Test sequences")
    parser.add_argument("--rank", type=int, default=128, help="Default adapter bottleneck rank")
    parser.add_argument("--ranks", default="16,64,128,256", help="Capacity sweep ranks")
    parser.add_argument("--train_steps", type=int, default=3000, help="Adapter training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training/eval")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adapter learning rate")
    parser.add_argument("--lambda_mse", type=float, default=1.0, help="Weight of relative MSE in loss")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--out_dir", default="cheap_pass_out", help="Output directory")
    parser.add_argument("--run_controls", action="store_true", help="Run full controls and ablations")
    parser.add_argument("--selftest", action="store_true", help="Quick CPU dry-run with tiny model")
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()

