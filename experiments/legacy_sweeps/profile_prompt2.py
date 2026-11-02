#!/usr/bin/env python3
"""
experiments/profile_prompt2.py

High-Precision Micro-Benchmarking & CUDA Event Profiling for Prompt 2 Regression Autopsy.
Measures exact breakdown of:
1. Baseline Full 36-Layer Token Latency (CUDA Kernel vs Eager Dispatch).
2. Speculative Draft Forward Latency (Layers 1..29).
3. Speculative Verify Remaining Latency (Layers 30..36).
4. Host-Device (.item()) CPU Synchronization Stall.
5. Accepted Step Cost (Draft0 + Verify0 + Sync1 + Draft1 + Verify1 + Sync2) vs. Rejected Step Cost.

Usage:
    python experiments/profile_prompt2.py --model-id Qwen/Qwen2.5-3B --draft-depth 29 --max-tokens 30 --device cuda
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.exit_adapter import ExitCalibrationModule
from src.speculative_engine import QwenSelfSpeculativeEngine


PROMPT_2 = "In computer science, an attention mechanism is a technique that mimics cognitive attention by"


def parse_args():
    parser = argparse.ArgumentParser(description="Profile Prompt 2 Speculative vs Baseline Execution")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-3B", help="Model ID")
    default_adapter = "checkpoints/exit_adapters_qwen_mlp.pt" if os.path.exists("checkpoints/exit_adapters_qwen_mlp.pt") else "checkpoints/exit_adapters_qwen.pt"
    parser.add_argument("--adapter-path", type=str, default=default_adapter, help="Adapter checkpoint")
    parser.add_argument("--draft-depth", type=int, default=29, help="Draft exit depth (default: 29)")
    parser.add_argument("--max-tokens", type=int, default=30, help="Tokens to generate")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def profile_baseline(engine: QwenSelfSpeculativeEngine, prompt_ids: torch.Tensor, max_tokens: int):
    print("\n" + "=" * 80)
    print("PROFILING BASELINE FULL-DEPTH (36 LAYERS) ON PROMPT 2")
    print("=" * 80)

    # Prefill
    curr_logits = engine.prefill(prompt_ids)
    curr_tok = torch.argmax(curr_logits[:, -1, :], dim=-1, keepdim=True)
    curr_pos = engine.kv_cache["current_len"]

    step_times_cuda = []
    step_times_wall = []

    for step in range(max_tokens):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)

        t_wall_start = time.perf_counter()
        ev_start.record()
        logits = engine.forward_full_step(curr_tok, pos=curr_pos)
        ev_end.record()
        torch.cuda.synchronize()
        t_wall_end = time.perf_counter()

        cuda_ms = ev_start.elapsed_time(ev_end)
        wall_ms = (t_wall_end - t_wall_start) * 1000.0

        step_times_cuda.append(cuda_ms)
        step_times_wall.append(wall_ms)

        curr_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        curr_pos += 1

    avg_cuda = sum(step_times_cuda) / len(step_times_cuda)
    avg_wall = sum(step_times_wall) / len(step_times_wall)
    avg_dispatch = avg_wall - avg_cuda

    print(f"Tokens Generated      : {len(step_times_cuda)}")
    print(f"Mean CUDA Kernel Time : {avg_cuda:6.2f} ms / token")
    print(f"Mean Wall-Clock Time  : {avg_wall:6.2f} ms / token")
    print(f"Mean Dispatch Overhead: {avg_dispatch:6.2f} ms / token ({avg_dispatch / avg_wall * 100:.1f}%)")
    print(f"Effective Throughput  : {1000.0 / avg_wall:6.2f} tok/s")

    return {
        "avg_cuda_ms": avg_cuda,
        "avg_wall_ms": avg_wall,
        "avg_dispatch_ms": avg_dispatch,
        "tok_per_sec": 1000.0 / avg_wall
    }


def profile_speculative(engine: QwenSelfSpeculativeEngine, prompt_ids: torch.Tensor, max_tokens: int):
    print("\n" + "=" * 80)
    print(f"PROFILING SPECULATIVE ENGINE (L = {engine.draft_depth}, gamma = 1) ON PROMPT 2")
    print("=" * 80)

    # Prefill
    curr_logits = engine.prefill(prompt_ids)
    next_token = torch.argmax(curr_logits[:, -1, :], dim=-1, keepdim=True)
    current_pos = engine.kv_cache["current_len"]

    generated = [next_token.item()]

    # Detailed statistics
    rejected_steps = []
    accepted_steps = []

    step_idx = 0
    while len(generated) < max_tokens:
        step_idx += 1
        start_pos = current_pos

        # --- Draft 0 ---
        ev_d0_start = torch.cuda.Event(enable_timing=True)
        ev_d0_end = torch.cuda.Event(enable_timing=True)

        ev_d0_start.record()
        draft_logits_0, h_0 = engine.forward_partial_step(
            next_token,
            pos=start_pos,
            depth=engine.draft_depth,
            project_kv=False
        )
        ev_d0_end.record()

        cand_tok_0 = torch.argmax(draft_logits_0[:, -1, :], dim=-1, keepdim=True)

        # --- Draft 1 (Candidate Token) ---
        ev_d1_start = torch.cuda.Event(enable_timing=True)
        ev_d1_end = torch.cuda.Event(enable_timing=True)

        ev_d1_start.record()
        _, h_1 = engine.forward_partial_step(
            cand_tok_0,
            pos=start_pos + 1,
            depth=engine.draft_depth,
            project_kv=False
        )
        ev_d1_end.record()

        # --- Parallel Verify Pass on Chunk of 2 ---
        ev_v_start = torch.cuda.Event(enable_timing=True)
        ev_v_end = torch.cuda.Event(enable_timing=True)

        chunk_h = torch.cat([h_0, h_1], dim=1)
        ev_v_start.record()
        oracle_logits = engine.forward_verify_remaining(
            chunk_h,
            start_pos=start_pos,
            from_depth=engine.draft_depth
        )
        ev_v_end.record()

        # --- Single Host Sync ---
        t_sync_start = time.perf_counter()
        oracle_preds = torch.argmax(oracle_logits[0], dim=-1).tolist()
        cand_val = cand_tok_0.item()
        t_sync_end = time.perf_counter()
        sync_ms = (t_sync_end - t_sync_start) * 1000.0

        torch.cuda.synchronize()
        d0_ms = ev_d0_start.elapsed_time(ev_d0_end)
        d1_ms = ev_d1_start.elapsed_time(ev_d1_end)
        v_ms = ev_v_start.elapsed_time(ev_v_end)

        is_accepted = (cand_val == oracle_preds[0])
        total_step_ms = d0_ms + d1_ms + v_ms + sync_ms

        if not is_accepted:
            # REJECTED STEP
            emitted = oracle_preds[0]
            generated.append(emitted)
            next_token = torch.tensor([[emitted]], dtype=torch.long, device=engine.device)
            current_pos = start_pos + 1

            rejected_steps.append({
                "d0_ms": d0_ms,
                "d1_ms": d1_ms,
                "v_ms": v_ms,
                "sync_ms": sync_ms,
                "total_ms": total_step_ms,
                "tokens_emitted": 1,
                "ms_per_token": total_step_ms
            })
        else:
            # ACCEPTED STEP: 2 tokens emitted in ONE verify pass!
            generated.append(cand_val)
            bonus_val = oracle_preds[1]
            generated.append(bonus_val)
            next_token = torch.tensor([[bonus_val]], dtype=torch.long, device=engine.device)
            current_pos = start_pos + 2

            accepted_steps.append({
                "d0_ms": d0_ms,
                "d1_ms": d1_ms,
                "v_ms": v_ms,
                "sync_ms": sync_ms,
                "total_ms": total_step_ms,
                "tokens_emitted": 2,
                "ms_per_token": total_step_ms / 2.0
            })

    total_accepted = len(accepted_steps)
    total_rejected = len(rejected_steps)
    acceptance_pct = (total_accepted / (total_accepted + total_rejected)) * 100.0

    print(f"\nCompleted {step_idx} verification steps:")
    print(f"  Accepted Steps : {total_accepted} ({total_accepted * 2} tokens)")
    print(f"  Rejected Steps : {total_rejected} ({total_rejected * 1} tokens)")
    print(f"  Acceptance Rate: {acceptance_pct:.1f}%")

    # Detailed Table Breakdown
    print("\n" + "-" * 80)
    print(f"{'Component':<35} {'Rejected Step':<20} {'Accepted Step':<20}")
    print("-" * 80)

    avg_rej_d0 = sum(s["d0_ms"] for s in rejected_steps) / max(1, total_rejected)
    avg_rej_d1 = sum(s["d1_ms"] for s in rejected_steps) / max(1, total_rejected)
    avg_rej_v = sum(s["v_ms"] for s in rejected_steps) / max(1, total_rejected)
    avg_rej_sync = sum(s["sync_ms"] for s in rejected_steps) / max(1, total_rejected)
    avg_rej_total = sum(s["total_ms"] for s in rejected_steps) / max(1, total_rejected)
    avg_rej_per_tok = sum(s["ms_per_token"] for s in rejected_steps) / max(1, total_rejected)

    avg_acc_d0 = sum(s["d0_ms"] for s in accepted_steps) / max(1, total_accepted)
    avg_acc_d1 = sum(s["d1_ms"] for s in accepted_steps) / max(1, total_accepted)
    avg_acc_v = sum(s["v_ms"] for s in accepted_steps) / max(1, total_accepted)
    avg_acc_sync = sum(s["sync_ms"] for s in accepted_steps) / max(1, total_accepted)
    avg_acc_total = sum(s["total_ms"] for s in accepted_steps) / max(1, total_accepted)
    avg_acc_per_tok = sum(s["ms_per_token"] for s in accepted_steps) / max(1, total_accepted)

    print(f"{'Draft Pass 0 (x_t)':<35} {avg_rej_d0:6.2f} ms           {avg_acc_d0:6.2f} ms")
    print(f"{'Draft Pass 1 (hat{x}_{t+1})':<35} {avg_rej_d1:6.2f} ms           {avg_acc_d1:6.2f} ms")
    print(f"{'Parallel Verify (Chunk C=2)':<35} {avg_rej_v:6.2f} ms           {avg_acc_v:6.2f} ms")
    print(f"{'Single Host-Device Sync':<35} {avg_rej_sync:6.2f} ms           {avg_acc_sync:6.2f} ms")
    print("-" * 80)
    print(f"{'Total Step Latency':<35} {avg_rej_total:6.2f} ms           {avg_acc_total:6.2f} ms")
    print(f"{'Tokens Emitted':<35} {'1 token':<20} {'2 tokens':<20}")
    print(f"{'Effective Latency per Token':<35} {avg_rej_per_tok:6.2f} ms/tok       {avg_acc_per_tok:6.2f} ms/tok")
    print(f"{'Effective Throughput':<35} {1000.0 / avg_rej_per_tok:6.2f} tok/s        {1000.0 / avg_acc_per_tok:6.2f} tok/s")
    print("=" * 80)

    return {
        "avg_rej_per_tok": avg_rej_per_tok,
        "avg_acc_per_tok": avg_acc_per_tok,
        "avg_rej_sync": avg_rej_sync,
        "avg_acc_sync": avg_acc_sync,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch_dtype = torch.float16 if args.dtype == "float16" else (torch.bfloat16 if args.dtype == "bfloat16" else torch.float32)

    print("=" * 80)
    print("PROFILING HARNESS: PROMPT 2 REGRESSION AUTOPSY")
    print(f"Model: {args.model_id} | Device: {device} | Dtype: {torch_dtype}")
    print(f"Target Prompt: \"{PROMPT_2}\"")
    print("=" * 80)

    print(f"[*] Loading model '{args.model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()

    adapter_module = None
    if args.adapter_path and os.path.exists(args.adapter_path):
        print(f"[*] Loading ExitCalibrationModule from '{args.adapter_path}'...")
        adapter_module = ExitCalibrationModule.load_checkpoint(args.adapter_path, device=device)
        adapter_module.eval()

    engine = QwenSelfSpeculativeEngine(
        model=model,
        tokenizer=tokenizer,
        draft_depth=args.draft_depth,
        adapter_module=adapter_module,
        device=args.device,
        max_seq_len=2048,
    )

    prompt_ids = tokenizer.encode(PROMPT_2, return_tensors="pt").to(device)

    # Warmup
    print("[*] Warming up CUDA execution graphs and memory pools...")
    engine.prefill(prompt_ids)
    engine.forward_full_step(prompt_ids[:, :1], pos=engine.kv_cache["current_len"])
    torch.cuda.synchronize()

    # 1. Profile Baseline
    base_res = profile_baseline(engine, prompt_ids, max_tokens=args.max_tokens)

    # 2. Profile Speculative
    spec_res = profile_speculative(engine, prompt_ids, max_tokens=args.max_tokens)

    print("\n" + "=" * 80)
    print("EXECUTIVE AUTOPSY SUMMARY")
    print("=" * 80)
    print(f"Baseline Throughput    : {base_res['tok_per_sec']:.2f} tok/s ({base_res['avg_wall_ms']:.2f} ms/tok)")
    print(f"Speculative (Rejected) : {1000.0 / spec_res['avg_rej_per_tok']:.2f} tok/s ({spec_res['avg_rej_per_tok']:.2f} ms/tok)")
    print(f"Speculative (Accepted) : {1000.0 / spec_res['avg_acc_per_tok']:.2f} tok/s ({spec_res['avg_acc_per_tok']:.2f} ms/tok)")
    print("=" * 80)


if __name__ == "__main__":
    main()
