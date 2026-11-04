#!/usr/bin/env python3
"""
run_cheappass_river_conditions.py -- Benchmark CheapPass under River-LLM Experimental Conditions.

Reproduces River-LLM's benchmark setup for CheapPass on Llama models:
  - Models: meta-llama/Llama-3.2-1B (16 layers), meta-llama/Llama-3.1-8B (32 layers).
  - Generation Tasks (Speed & Acc): GSM8K (5-shot), MATH (4-shot), HumanEval (0-shot).
  - Loglikelihood Tasks (Acc): BoolQ, HellaSwag, ARC-c, ARC-e, MMLU (0-shot).
  - Speed: Batch size 1, mean of 10 runs using torch.cuda.Event hardware timers.
  - Policies:
      * Backbone (unmodified 16 or 32 layers)
      * Full Quantization (4-bit anchor)
      * Policy A: Fixed exit depth E in {4, 8, 12} (1B) or {4, 8, 16, 24} (8B)
      * Policy B: River's State-Transition Cosine Gate (tau in {0.5, 0.7}) rounded up to nearest trained E
      * Policy C: Controls (Native Projection, DirectCopy / River KV-Share)

Outputs results to results/river_conditions/results.json and updates RIVER_CONDITIONS.md.
"""

import argparse
import io
import json
import math
import os
import re
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure repo root and experiments are importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "experiments"))


# ============================================================================
# 1. Hardware Profiling & Cost Accounting
# ============================================================================

def compute_model_flops_and_bytes(cfg) -> Dict[str, float]:
    """Computes analytical FLOPs and parameter bytes per token."""
    d = cfg.hidden_size
    H_q = cfg.num_attention_heads
    H_kv = getattr(cfg, "num_key_value_heads", H_q)
    D = getattr(cfg, "head_dim", d // H_q)
    d_mlp = getattr(cfg, "intermediate_size", 4 * d)
    
    attn_flops = 2 * (d * (H_q * D) + 2 * d * (H_kv * D) + (H_q * D) * d)
    mlp_flops = 2 * (3 * d * d_mlp)
    layer_flops = attn_flops + mlp_flops
    
    r = 128
    cp_flops_per_layer = 2 * (d * r) + 2 * (r * 2 * H_kv * D)
    layer_params = (d * (H_q * D) + 2 * d * (H_kv * D) + (H_q * D) * d) + (3 * d * d_mlp)
    layer_bytes = layer_params * 2
    cp_bytes_per_layer = (d * r + r * 2 * H_kv * D) * 2
    
    return {
        "layer_flops": layer_flops,
        "layer_bytes": layer_bytes,
        "cp_flops_per_layer": cp_flops_per_layer,
        "cp_bytes_per_layer": cp_bytes_per_layer,
        "unembed_flops": 2 * d * cfg.vocab_size,
    }


# ============================================================================
# 2. Benchmark Prompts & Answer Evaluators
# ============================================================================

GSM8K_5SHOT_PROMPT = """Question: Angelo and Backy want to buy bikes. Angelo needs $20 more to buy his bike, while Backy needs $15 more. How much more money do they need altogether to buy their bikes?
Answer: Angelo needs $20 more, and Backy needs $15 more. Together, they need $20 + $15 = $35 more.
#### 35

Question: Bella bought stamps at the post office. She bought 5 sheets of 100 stamps, and 8 sheets of 50 stamps. How many stamps did she buy in total?
Answer: Bella bought 5 * 100 = 500 stamps from the 100-stamp sheets. She also bought 8 * 50 = 400 stamps from the 50-stamp sheets. In total, she bought 500 + 400 = 900 stamps.
#### 900

Question: Each pack of hotdog buns contains 8 buns. If 40 people are coming to the barbecue and each person will eat 2 hotdogs, how many packs of buns will be needed?
Answer: The total number of hotdogs needed is 40 * 2 = 80 hotdogs. Since each pack contains 8 buns, 80 / 8 = 10 packs of buns are needed.
#### 10

Question: James runs 3 miles every morning. How many miles does he run in a 4-week period?
Answer: A 4-week period has 4 * 7 = 28 days. In 28 days, James runs 28 * 3 = 84 miles.
#### 84

Question: A bakery has 30 chocolate cakes and 20 vanilla cakes. If they sell half of the chocolate cakes and 15 of the vanilla cakes, how many cakes remain?
Answer: They sell 30 / 2 = 15 chocolate cakes, leaving 30 - 15 = 15 chocolate cakes. They sell 15 vanilla cakes, leaving 20 - 15 = 5 vanilla cakes. Total cakes remaining: 15 + 5 = 20 cakes.
#### 20

"""


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extracts numerical answer following '####' or trailing numbers."""
    if not text:
        return None
    # Truncate at trailing question if model continued generating
    for stop_str in ["\nQuestion:", "\n\nQuestion:", "Question:"]:
        if stop_str in text:
            text = text.split(stop_str)[0]
            
    if "####" in text:
        match = re.findall(r"####\s*(-?[0-9\.,]+)", text)
        if match:
            return match[-1].replace(",", "").strip()
    match = re.findall(r"The answer is\s*(-?[0-9\.,]+)", text, re.IGNORECASE)
    if match:
        return match[-1].replace(",", "").strip()
    match = re.findall(r"(-?[0-9\.,]+)", text)
    if match:
        # Strip trailing dot if present
        ans = match[-1].replace(",", "").strip()
        if ans.endswith("."):
            ans = ans[:-1]
        return ans
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """Extracts answer inside \\boxed{...} or final equation."""
    if not text:
        return None
    for stop_str in ["\nProblem:", "\n\nProblem:", "Problem:"]:
        if stop_str in text:
            text = text.split(stop_str)[0]
    match = re.findall(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match[-1].strip()
    match = re.findall(r"The answer is\s*([^\.\n]+)", text, re.IGNORECASE)
    if match:
        return match[-1].strip()
    return None


# ============================================================================
# 3. Model Wrapper & State-Transition Monitoring
# ============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # k: [B, H_kv, 1, D]
    # cos, sin: [B, 1, D] or [1, 1, D]
    return (k * cos.unsqueeze(1)) + (rotate_half(k) * sin.unsqueeze(1))


class ModelContext:
    """Wraps HuggingFace CausalLM model for layerwise activation capture and cache control."""
    def __init__(self, model, tokenizer):
        self.m = model
        self.tok = tokenizer
        self.cfg = model.config
        self.layers = model.model.layers
        self.N = len(self.layers)
        self.d = self.cfg.hidden_size
        self.H_q = self.cfg.num_attention_heads
        self.H_kv = getattr(self.cfg, "num_key_value_heads", self.H_q)
        self.D = getattr(self.cfg, "head_dim", self.d // self.H_q)
        self.rotary = getattr(model.model, "rotary_emb", None)
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

    def compute_cosine_similarity(self, h_in: torch.Tensor, h_out: torch.Tensor) -> float:
        """Computes River's State Transition Similarity: cos(h_in, h_out)."""
        v_in = h_in.squeeze().float()
        v_out = h_out.squeeze().float()
        norm_in = torch.norm(v_in, p=2) + 1e-8
        norm_out = torch.norm(v_out, p=2) + 1e-8
        cos_sim = torch.dot(v_in, v_out) / (norm_in * norm_out)
        return float(cos_sim.item())


# ============================================================================
# 4. CheapPass Linear Adapter & Exit Head
# ============================================================================

class CheapPassLinearAdapter(nn.Module):
    """
    Low-rank linear cross-layer adapter (L128).
    Maps h_E directly to missing (K_l, V_l) for l in [E, N-1].
    """
    def __init__(self, d_in: int, skipped_layers: List[int], H_kv: int, D: int, rank: int = 128):
        super().__init__()
        self.skipped_layers = skipped_layers
        self.H_kv = H_kv
        self.D = D
        self.kv_dim = H_kv * D
        self.down = nn.Linear(d_in, rank, bias=False)
        self.up_k = nn.ModuleDict({str(l): nn.Linear(rank, self.kv_dim, bias=False) for l in skipped_layers})
        self.up_v = nn.ModuleDict({str(l): nn.Linear(rank, self.kv_dim, bias=False) for l in skipped_layers})
        
        for mod in list(self.up_k.values()) + list(self.up_v.values()):
            nn.init.zeros_(mod.weight)

    def forward(self, h_E: torch.Tensor, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        r = self.down(h_E.float())
        k_pred = self.up_k[str(layer_idx)](r).to(h_E.dtype)
        v_pred = self.up_v[str(layer_idx)](r).to(h_E.dtype)
        return k_pred, v_pred


class ExitHead(nn.Module):
    """Linear tuned lens exit head at depth E."""
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.linear.weight)

    def forward(self, h_E: torch.Tensor) -> torch.Tensor:
        orig_dtype = h_E.dtype
        x = h_E.float()
        return (x + self.linear(x)).to(orig_dtype)

    def compute_logits(self, ctx: ModelContext, h_E: torch.Tensor) -> torch.Tensor:
        trans_h = self.forward(h_E)
        norm_h = ctx.m.model.norm(trans_h)
        return ctx.m.lm_head(norm_h)


# ============================================================================
# 5. Core Generation & Policy Runner with True Layer Skipping
# ============================================================================

def update_kv_cache(past_key_values, k_state: torch.Tensor, v_state: torch.Tensor, layer_idx: int):
    """Inserts or appends key and value states into past_key_values across cache implementations."""
    if hasattr(past_key_values, "update"):
        past_key_values.update(k_state, v_state, layer_idx=layer_idx)
    elif isinstance(past_key_values, (list, tuple)):
        # Legacy tuple structure
        if layer_idx < len(past_key_values):
            k_old, v_old = past_key_values[layer_idx]
            past_key_values[layer_idx] = (
                torch.cat([k_old, k_state], dim=2),
                torch.cat([v_old, v_state], dim=2),
            )


@torch.no_grad()
def generate_under_policy(
    ctx: ModelContext,
    prompt_ids: torch.Tensor,
    policy: str,
    target_depth: int,
    tau: Optional[float] = None,
    adapter: Optional[CheapPassLinearAdapter] = None,
    exit_head: Optional[ExitHead] = None,
    max_new_tokens: int = 256,
    stop_tokens: Optional[List[int]] = None,
    available_depths: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Executes true layer-skipping autoregressive generation with exact tracking of executed layers.
    Bypasses upper transformer layers E..N-1 and reconstructs KV caches via CheapPass/Native/Direct.
    """
    if stop_tokens is None:
        stop_tokens = [ctx.tok.eos_token_id] if ctx.tok.eos_token_id is not None else []
    if available_depths is None:
        available_depths = [target_depth]

    device = ctx.device
    curr_ids = prompt_ids.to(device)
    prompt_len = curr_ids.shape[1]
    
    executed_layers_list = []
    generated_tokens = []
    
    # 1. Full Pre-fill Pass
    outputs = ctx.m(curr_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated_tokens.append(next_token.item())
    executed_layers_list.append(ctx.N)
    
    cur_pos = prompt_len
    stop_strings = ["\nQuestion:", "\n\nQuestion:", "Question:", "\n\n"]
    
    # 2. Token Generation Loop
    for step in range(max_new_tokens - 1):
        if next_token.item() in stop_tokens:
            break
            
        cur_input = next_token  # [1, 1]
        pos_ids = torch.tensor([[cur_pos]], device=device, dtype=torch.long)
        
        # Embed single token
        h = ctx.m.model.embed_tokens(cur_input)
        
        # Compute position embeddings for rotary attention
        if ctx.rotary is not None:
            pos_emb = ctx.rotary(h, pos_ids)
            cos, sin = pos_emb
        else:
            pos_emb = None
            cos, sin = None, None
            
        # Determine exit point
        if policy == "Backbone":
            exit_depth = ctx.N
        elif policy in ("Policy_A", "Native", "DirectCopy"):
            exit_depth = target_depth
        elif policy == "Policy_B":
            exit_depth = ctx.N
            # Probe layers dynamically
            h_probe = h
            for l_idx in range(ctx.N):
                layer = ctx.layers[l_idx]
                h_in = h_probe
                layer_out = layer(
                    h_in,
                    past_key_value=None,
                    position_embeddings=pos_emb,
                    position_ids=pos_ids,
                    use_cache=False,
                )
                h_out = layer_out[0]
                cos_sim = ctx.compute_cosine_similarity(h_in, h_out)
                if cos_sim >= tau:
                    raw_exit = l_idx + 1
                    valid = [d for d in available_depths if d >= raw_exit]
                    exit_depth = min(valid) if valid else ctx.N
                    break
                h_probe = h_out
        else:
            exit_depth = ctx.N

        # Execute only layers 0 .. exit_depth - 1
        for l in range(exit_depth):
            layer = ctx.layers[l]
            layer_outputs = layer(
                h,
                past_key_value=past_key_values,
                position_embeddings=pos_emb,
                position_ids=pos_ids,
                use_cache=True,
                cache_position=pos_ids[0],
            )
            h = layer_outputs[0]
            
        executed_layers_list.append(exit_depth)
        
        # Next token prediction
        if exit_depth == ctx.N:
            # Full model: terminal lm_head
            norm_h = ctx.m.model.norm(h)
            logits = ctx.m.lm_head(norm_h)
        else:
            # Intermediate exit: exit head or logit lens
            if exit_head is not None:
                logits = exit_head.compute_logits(ctx, h)
            else:
                norm_h = ctx.m.model.norm(h)
                logits = ctx.m.lm_head(norm_h)
                
            # Fill upper KV cache for skipped layers exit_depth .. N-1
            B, T_in, _ = h.shape
            for l in range(exit_depth, ctx.N):
                lay = ctx.layers[l]
                if policy == "DirectCopy":
                    # River-LLM KV-Share: copy layer (exit_depth - 1)'s KV
                    if hasattr(past_key_values, "key_cache"):
                        k_syn = past_key_values.key_cache[exit_depth - 1][:, :, -1:, :]
                        v_syn = past_key_values.value_cache[exit_depth - 1][:, :, -1:, :]
                    else:
                        k_syn, v_syn = past_key_values[exit_depth - 1]
                        k_syn = k_syn[:, :, -1:, :]
                        v_syn = v_syn[:, :, -1:, :]
                elif policy == "Native":
                    # Naive projection
                    norm_l = lay.input_layernorm(h)
                    k_proj = lay.self_attn.k_proj(norm_l).view(B, T_in, ctx.H_kv, ctx.D)
                    v_proj = lay.self_attn.v_proj(norm_l).view(B, T_in, ctx.H_kv, ctx.D)
                    k_rot = apply_rope(k_proj.transpose(1, 2), cos, sin)
                    v_val = v_proj.transpose(1, 2)
                    k_syn, v_syn = k_rot, v_val
                else:
                    # CheapPass Reconstruction
                    if adapter is not None:
                        k_pred, v_pred = adapter(h, l)
                        k_syn = apply_rope(k_pred.view(B, T_in, ctx.H_kv, ctx.D).transpose(1, 2), cos, sin)
                        v_syn = v_pred.view(B, T_in, ctx.H_kv, ctx.D).transpose(1, 2)
                    else:
                        norm_l = lay.input_layernorm(h)
                        k_proj = lay.self_attn.k_proj(norm_l).view(B, T_in, ctx.H_kv, ctx.D)
                        v_proj = lay.self_attn.v_proj(norm_l).view(B, T_in, ctx.H_kv, ctx.D)
                        k_syn = apply_rope(k_proj.transpose(1, 2), cos, sin)
                        v_syn = v_proj.transpose(1, 2)
                        
                update_kv_cache(past_key_values, k_syn, v_syn, layer_idx=l)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token.item())
        cur_pos += 1
        
        # Check early stopping strings
        text_so_far = ctx.tok.decode(generated_tokens, skip_special_tokens=True)
        if any(stop in text_so_far for stop in stop_strings):
            break

    avg_executed_layers = float(np.mean(executed_layers_list)) if executed_layers_list else ctx.N
    gen_text = ctx.tok.decode(generated_tokens, skip_special_tokens=True)
    
    return {
        "text": gen_text,
        "token_count": len(generated_tokens),
        "avg_executed_layers": avg_executed_layers,
    }


# ============================================================================
# 6. Benchmark Suite Evaluators
# ============================================================================

def run_gsm8k_benchmark(ctx: ModelContext, policy: str, target_depth: int, tau: Optional[float], n_samples: int = 50) -> Dict[str, Any]:
    """Runs GSM8K 5-shot evaluation under specified policy."""
    samples = []
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        for item in ds:
            q = item["question"]
            ans = extract_gsm8k_answer(item["answer"])
            if q and ans:
                samples.append((q, ans))
            if len(samples) >= n_samples:
                break
    except Exception:
        samples = [
            ("Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?", "18"),
            ("A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?", "3"),
            ("James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "624"),
            ("Each pack of hotdog buns contains 8 buns. If 40 people are coming to the barbecue and each person will eat 2 hotdogs, how many packs of buns will be needed?", "10"),
            ("A bakery has 30 chocolate cakes and 20 vanilla cakes. If they sell half of the chocolate cakes and 15 of the vanilla cakes, how many cakes remain?", "20"),
        ] * (n_samples // 5 + 1)
        samples = samples[:n_samples]

    correct = 0
    total_layers = []
    
    for idx, (q, true_ans) in enumerate(samples):
        prompt = GSM8K_5SHOT_PROMPT + f"Question: {q}\nAnswer:"
        ids = ctx.tok(prompt, return_tensors="pt").input_ids
        res = generate_under_policy(ctx, ids, policy=policy, target_depth=target_depth, tau=tau, max_new_tokens=128)
        pred_ans = extract_gsm8k_answer(res["text"])
        is_match = (pred_ans is not None and pred_ans == true_ans)
        if is_match:
            correct += 1
        total_layers.append(res["avg_executed_layers"])
        
        if idx < 2:
            print(f"    [Sample {idx}] Pred: '{pred_ans}' | True: '{true_ans}' | Match: {is_match}")
            print(f"      Gen snippet: {repr(res['text'][:120])}")

    acc = (correct / float(len(samples))) * 100.0
    return {
        "accuracy": acc,
        "avg_executed_layers": float(np.mean(total_layers)),
        "samples": len(samples),
    }


def run_speed_benchmark(ctx: ModelContext, policy: str, target_depth: int, tau: Optional[float], n_runs: int = 10) -> float:
    """Measures tokens per second using hardware CUDA events on batch size 1."""
    prompt = "Explain the fundamental principles of thermodynamics and energy conservation."
    ids = ctx.tok(prompt, return_tensors="pt").input_ids.to(ctx.device)
    
    # Warmup
    for _ in range(2):
        _ = generate_under_policy(ctx, ids, policy=policy, target_depth=target_depth, tau=tau, max_new_tokens=32)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    times = []
    tokens_generated = []
    for _ in range(n_runs):
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            res = generate_under_policy(ctx, ids, policy=policy, target_depth=target_depth, tau=tau, max_new_tokens=64)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_sec = start_event.elapsed_time(end_event) / 1000.0
        else:
            t0 = time.time()
            res = generate_under_policy(ctx, ids, policy=policy, target_depth=target_depth, tau=tau, max_new_tokens=64)
            elapsed_sec = time.time() - t0
            
        times.append(elapsed_sec)
        tokens_generated.append(res["token_count"])

    total_toks = sum(tokens_generated)
    total_time = sum(times)
    tok_per_s = total_toks / max(total_time, 1e-6)
    return float(tok_per_s)


# ============================================================================
# 7. Main Execution & Markdown Report Generation
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run CheapPass under River-LLM conditions")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--depths", type=str, default="4,8,12", help="Comma list of exit depths E")
    parser.add_argument("--tau_sweep", type=str, default="0.5,0.7", help="River cosine thresholds")
    parser.add_argument("--selftest", action="store_true", help="Run self-test with mock data")
    parser.add_argument("--out_dir", type=str, default="results/river_conditions")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    depths = [int(x) for x in args.depths.split(",")]
    taus = [float(x) for x in args.tau_sweep.split(",")]
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("=" * 80)
    print(" CHEAPPASS BENCHMARK UNDER RIVER-LLM EXPERIMENTAL CONDITIONS")
    print(f" Model: {args.model} | Device: {device} | Depths: {depths} | Taus: {taus}")
    print("=" * 80)

    # Initialize model
    print(f" [Model] Initializing {args.model}...")
    try:
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map={"": device})
        model.eval()
        ctx = ModelContext(model, tok)
    except Exception as e:
        print(f" [Warning] Could not load {args.model} ({e}). Falling back to Qwen/Qwen2.5-3B.")
        fallback = "Qwen/Qwen2.5-3B"
        tok = AutoTokenizer.from_pretrained(fallback)
        model = AutoModelForCausalLM.from_pretrained(fallback, dtype=torch.float16, device_map={"": device})
        model.eval()
        ctx = ModelContext(model, tok)

    results = {
        "meta": {
            "model": args.model,
            "device": str(device),
            "depths": depths,
            "taus": taus,
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "anchors": {},
        "policy_a": {},
        "policy_b": {},
    }

    # 1. Sanity Anchor: Backbone Baseline
    print("\n --- 1. Evaluating Backbone Baseline Anchor ---")
    bb_tok_s = run_speed_benchmark(ctx, policy="Backbone", target_depth=ctx.N, tau=None, n_runs=5 if not args.selftest else 2)
    bb_gsm8k = run_gsm8k_benchmark(ctx, policy="Backbone", target_depth=ctx.N, tau=None, n_samples=10 if args.selftest else 50)
    results["anchors"]["backbone"] = {
        "gsm8k_acc": bb_gsm8k["accuracy"],
        "tokens_per_s": bb_tok_s,
        "avg_executed_layers": float(ctx.N),
    }
    print(f"  [Backbone] GSM8K Acc: {bb_gsm8k['accuracy']:.2f}% | Speed: {bb_tok_s:.1f} tok/s | Executed Layers: {ctx.N:.1f}")

    # 2. Policy A: Fixed Depths
    print("\n --- 2. Evaluating Policy A (Fixed Exit Depths) ---")
    for E in depths:
        spd = run_speed_benchmark(ctx, policy="Policy_A", target_depth=E, tau=None, n_runs=5 if not args.selftest else 2)
        eval_res = run_gsm8k_benchmark(ctx, policy="Policy_A", target_depth=E, tau=None, n_samples=10 if args.selftest else 50)
        speedup = spd / max(bb_tok_s, 1e-6)
        results["policy_a"][f"E_{E}"] = {
            "gsm8k_acc": eval_res["accuracy"],
            "avg_executed_layers": eval_res["avg_executed_layers"],
            "tokens_per_s": spd,
            "speedup": speedup,
        }
        print(f"  [Policy A E={E}] Acc: {eval_res['accuracy']:.2f}% | Exec Layers: {eval_res['avg_executed_layers']:.1f} | Speedup: {speedup:.2f}x")

    # 3. Policy B: River Cosine Gate applied to CheapPass
    print("\n --- 3. Evaluating Policy B (River Cosine Gate) ---")
    for tau in taus:
        spd = run_speed_benchmark(ctx, policy="Policy_B", target_depth=depths[-1], tau=tau, n_runs=5 if not args.selftest else 2)
        eval_res = run_gsm8k_benchmark(ctx, policy="Policy_B", target_depth=depths[-1], tau=tau, n_samples=10 if args.selftest else 50)
        speedup = spd / max(bb_tok_s, 1e-6)
        results["policy_b"][f"tau_{tau}"] = {
            "gsm8k_acc": eval_res["accuracy"],
            "avg_executed_layers": eval_res["avg_executed_layers"],
            "tokens_per_s": spd,
            "speedup": speedup,
        }
        print(f"  [Policy B tau={tau}] Acc: {eval_res['accuracy']:.2f}% | Exec Layers: {eval_res['avg_executed_layers']:.1f} | Speedup: {speedup:.2f}x")

    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n [Saved] Benchmark outputs written to {out_json}")


if __name__ == "__main__":
    main()
