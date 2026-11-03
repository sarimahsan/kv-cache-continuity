#!/usr/bin/env python3
"""
rollout_cheap_pass.py -- End-to-End Autoregressive Rollouts & Speculative Drafting for CheapPass.

Evaluates the compounding impact of synthetic KV caches in real autoregressive decoding:
  1. Multi-Step Greedy Continuation Rollout (N=64 prompts, prompt_len=32, new_tokens=128):
     - Compares Oracle, DirectCopy, NativeProj, and CheapPass.
     - Reports Match@16, Match@64, Match@128, First-Divergence Position, and Gen-NLL under Full Model.
  2. Self-Speculative Drafting Simulation (gamma in {2, 3, 4, 5}):
     - Simulates drafting at layer E with synthetic KV insertion, verified by terminal layer L.
     - Reports mean acceptance rate (alpha) and expected tokens per forward pass (tau).
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cheap_pass import Ctx, CheapPassAdapter, AdapterKVMethod
from vkr2 import Native, Direct, rollout_metrics, gen_nll

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ============================================================================
# 1. Autoregressive Rollout Engine with KV Cache Injection
# ============================================================================

@torch.no_grad()
def execute_rollout(
    ctx: Ctx,
    method_instance,
    E: int,
    prompts: torch.Tensor,
    n_new: int = 128,
    device: torch.device = torch.device("cpu"),
    desc: str = "Rollout",
) -> List[torch.Tensor]:
    """
    Executes greedy autoregressive decoding where:
      - Prompt tokens (0 .. plen-1) have exact full-depth KV states.
      - Generated tokens (plen .. plen + n_new - 1) exit at layer E,
        and their KV cache for layers E..L-1 is populated by method_instance.
      - The query token continues through layers E..L-1 to compute logits.
    """
    outs = []
    pbar = tqdm(prompts, desc=desc, dynamic_ncols=True, leave=False) if tqdm else prompts

    for p in pbar:
        cur = p.unsqueeze(0).to(device)
        plen = cur.shape[1]

        for _ in range(n_new):
            orc = ctx.oracle(cur)
            if method_instance is None:
                # Pure Oracle baseline (no early exit)
                lg = orc["logits"][:, -1]
            else:
                # Mask: Prompt tokens are exact, newly generated tokens use synthetic KV
                mask = torch.arange(cur.shape[1], device=device) < plen
                kv_hat = method_instance.kv(ctx, orc, E)
                lg = ctx.approx(orc, E, kv_hat, exact_mask=mask, last_only=True)[:, -1]

            next_token = lg.argmax(-1, keepdim=True)
            cur = torch.cat([cur, next_token], dim=1)

        # Append only the newly generated continuation tokens
        outs.append(cur[0, plen:].cpu())

    return outs


# ============================================================================
# 2. Speculative Drafting Acceptance Rate Simulation
# ============================================================================

@torch.no_grad()
def evaluate_speculative_acceptance(
    ctx: Ctx,
    method_instance,
    E: int,
    prompts: torch.Tensor,
    gammas: List[int] = [2, 3, 4, 5],
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict[str, float]]:
    """
    Simulates speculative drafting where gamma draft tokens are generated at layer E
    with synthetic KV cache, and verified in parallel by the full model at layer L.
    Computes empirical acceptance rate alpha and effective tokens per forward pass tau.
    """
    results = {}
    m_name = getattr(method_instance, "name", "Oracle") if method_instance is not None else "Oracle"

    for gamma in gammas:
        accepted_counts = []
        total_drafts = []

        for p in prompts[:32]:  # Evaluate across 32 prompts for speed
            cur = p.unsqueeze(0).to(device)
            plen = cur.shape[1]

            # 1. Draft gamma tokens at layer E
            drafted_tokens = []
            draft_cur = cur.clone()
            for _ in range(gamma):
                orc_d = ctx.oracle(draft_cur)
                if method_instance is None:
                    lg_d = orc_d["logits"][:, -1]
                else:
                    mask = torch.arange(draft_cur.shape[1], device=device) < plen
                    kv_d = method_instance.kv(ctx, orc_d, E)
                    lg_d = ctx.approx(orc_d, E, kv_d, exact_mask=mask, last_only=True)[:, -1]
                t_next = lg_d.argmax(-1, keepdim=True)
                drafted_tokens.append(t_next.item())
                draft_cur = torch.cat([draft_cur, t_next], dim=1)

            # 2. Parallel verification by Full Target Model
            verif_orc = ctx.oracle(draft_cur)
            verif_logits = verif_orc["logits"][:, plen - 1 : plen + gamma - 1]  # [1, gamma, V]
            target_greedy = verif_logits.argmax(-1).squeeze(0).tolist()  # [gamma]

            # 3. Count accepted tokens before first rejection
            n_acc = 0
            for d_tok, tgt_tok in zip(drafted_tokens, target_greedy):
                if d_tok == tgt_tok:
                    n_acc += 1
                else:
                    break

            accepted_counts.append(n_acc)
            total_drafts.append(gamma)

        alpha = sum(accepted_counts) / max(1, sum(total_drafts))
        # Effective speedup / compression ratio: 1 bonus token on rejection + accepted
        tau = 1.0 + (sum(accepted_counts) / len(accepted_counts))
        results[f"gamma_{gamma}"] = {
            "acceptance_rate": alpha,
            "tokens_per_pass": tau,
            "mean_accepted": float(np.mean(accepted_counts)),
        }

    return results


# ============================================================================
# 3. Main Rollout Experiment Suite
# ============================================================================

def run_rollout_suite(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("=" * 80)
    print(" END-TO-END AUTOREGRESSIVE ROLLOUT & SPECULATIVE DRAFTING SUITE")
    print(f" Model: {args.model} | Device: {device} | Depths: {args.depths}")
    print(f" Prompts: N={args.n_prompts}, Prompt Len={args.prompt_len}, New Tokens={args.n_new}")
    print("=" * 80)

    # 1. Load Model and Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    ctx = Ctx(model)
    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]

    # 2. Prepare Prompts from FineWeb-Edu (or fallback)
    from datasets import load_dataset
    print(f"\n [Data] Streaming {args.n_prompts} prompts from HuggingFaceFW/fineweb-edu...")
    texts = []
    try:
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
        for r in ds:
            t = r.get("text", "")
            if len(t) > 200:
                texts.append(t)
            if len(texts) >= args.n_prompts * 2:
                break
    except Exception as e:
        print(f" [Data] Warning: Dataset stream failed ({e}); using synthetic prompts.")
        rng = np.random.RandomState(args.seed)
        prompts = torch.tensor(rng.randint(50, tok.vocab_size - 100, size=(args.n_prompts, args.prompt_len)), dtype=torch.long)
        texts = []

    if texts:
        prompts_list = []
        for t in texts:
            toks = tok(t, add_special_tokens=False)["input_ids"]
            if len(toks) >= args.prompt_len:
                prompts_list.append(toks[:args.prompt_len])
            if len(prompts_list) >= args.n_prompts:
                break
        prompts = torch.tensor(prompts_list, dtype=torch.long)

    print(f" [Data] Prepared {len(prompts)} prompts of length {args.prompt_len} tokens.")

    # 3. Oracle Reference Generation (Full 36-layer model)
    print("\n [Step 1] Generating Reference Continuations with Full 36-Layer Oracle...")
    ref_continuations = execute_rollout(ctx, None, 0, prompts, n_new=args.n_new, device=device, desc="Oracle Rollout")
    ref_nll = gen_nll(ctx, prompts, ref_continuations, device)
    mean_ref_nll = float(np.mean(ref_nll))
    print(f"          Oracle Reference NLL: {mean_ref_nll:.4f} (Perplexity: {math.exp(mean_ref_nll):.2f})")

    results = {
        "meta": {
            "model": args.model,
            "depths": depths,
            "n_prompts": args.n_prompts,
            "prompt_len": args.prompt_len,
            "n_new": args.n_new,
            "oracle_nll": mean_ref_nll,
            "oracle_ppl": math.exp(mean_ref_nll),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "rollout_results": {},
        "speculative_results": {},
    }

    # 4. Evaluate Across Depths
    for E in depths:
        print("\n" + "#" * 80)
        print(f" EVALUATING ROLLOUT AT EXIT DEPTH E = {E}/{ctx.N} (Skipping {ctx.N - E} layers)")
        print("#" * 80)

        # Load CheapPass Adapter (Search candidate directories)
        fname = f"adapter_E{E}_r{args.rank}_primary.pt"
        candidate_dirs = [
            args.ckpt_dir,
            "cheap_pass_out",
            "cheap_pass_checkpoints",
            "../cheap_pass_out",
            "../cheap_pass_checkpoints",
            "/kaggle/working/cheap_pass_out",
            "/kaggle/working/cheap_pass_checkpoints",
            "/kaggle/working/kv-cache-continuity/cheap_pass_out",
            "/kaggle/working/kv-cache-continuity/cheap_pass_checkpoints",
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
            print(f" [Warning] Checkpoint '{fname}' not found in any candidate directory!")
            print(f"           Searched: {candidate_dirs}")
            print(f"           (Evaluating untrained adapter)")
        adapter.eval()

        methods = {
            "DirectCopy": Direct(),
            "NativeProj": Native(),
            "CheapPass": AdapterKVMethod(adapter, "cheap_pass"),
        }

        results["rollout_results"][f"E_{E}"] = {}
        results["speculative_results"][f"E_{E}"] = {}

        for m_name, m_inst in methods.items():
            print(f"\n   Running multi-step rollout for: {m_name:12s} [E={E}]...")
            gen_tokens = execute_rollout(ctx, m_inst, E, prompts, n_new=args.n_new, device=device, desc=f"Rollout [{m_name}]")
            
            # Compute Match@k metrics
            m16 = rollout_metrics(ref_continuations, gen_tokens, 16)
            m64 = rollout_metrics(ref_continuations, gen_tokens, min(64, args.n_new))
            m128 = rollout_metrics(ref_continuations, gen_tokens, args.n_new)
            
            # Compute Gen-NLL under full model
            g_nll = gen_nll(ctx, prompts, gen_tokens, device)
            mean_nll = float(np.mean(g_nll))
            delta_nll = mean_nll - mean_ref_nll

            res_entry = {
                "match_at_16": m16["match"],
                "match_at_64": m64["match"],
                "match_at_128": m128["match"],
                "first_divergence_pos": m128["first_div"],
                "gen_nll": mean_nll,
                "delta_nll": delta_nll,
                "gen_ppl": math.exp(mean_nll),
            }
            results["rollout_results"][f"E_{E}"][m_name] = res_entry

            print(f"   {m_name:12s} | Match@16: {m16['match']:5.1f}% | Match@64: {m64['match']:5.1f}% | Match@128: {m128['match']:5.1f}% | FirstDiv: {m128['first_div']:4.1f} tok | Delta-NLL: {delta_nll:+.4f}")

            # Speculative Acceptance Simulation
            spec_res = evaluate_speculative_acceptance(ctx, m_inst, E, prompts, gammas=[2, 3, 4, 5], device=device)
            results["speculative_results"][f"E_{E}"][m_name] = spec_res

    # Save complete JSON
    out_dir = os.path.dirname(args.out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n [Done] Saved complete rollout and speculative results to: {args.out_file}")

    # Generate Markdown Summary
    md_file = args.out_file.replace(".json", ".md")
    with open(md_file, "w") as f:
        f.write("# End-to-End Autoregressive Rollout & Speculative Drafting Report\n\n")
        f.write("## 1. Multi-Step Autoregressive Continuation (New Tokens = 128)\n\n")
        f.write("| Depth $E$ | Method | Match@16 (%) | Match@64 (%) | Match@128 (%) | First Divergence (tok) | $\\Delta\\text{NLL}$ | Gen PPL |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for e_k, m_dict in results["rollout_results"].items():
            e_val = e_k.replace("E_", "")
            for m_name, d in m_dict.items():
                f.write(f"| $E={e_val}$ | **{m_name}** | {d['match_at_16']:.1f}% | {d['match_at_64']:.1f}% | {d['match_at_128']:.1f}% | {d['first_divergence_pos']:.1f} | {d['delta_nll']:+.4f} | {d['gen_ppl']:.2f} |\n")
        
        f.write("\n## 2. Speculative Drafting Acceptance ($\gamma = 3$ and $\gamma = 5$)\n\n")
        f.write("| Depth $E$ | Method | $\\alpha$ ($\gamma=3$) | $\\tau$ ($\gamma=3$) | $\\alpha$ ($\gamma=5$) | $\\tau$ ($\gamma=5$) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: |\n")
        for e_k, s_dict in results["speculative_results"].items():
            e_val = e_k.replace("E_", "")
            for m_name, d in s_dict.items():
                a3, t3 = d["gamma_3"]["acceptance_rate"] * 100, d["gamma_3"]["tokens_per_pass"]
                a5, t5 = d["gamma_5"]["acceptance_rate"] * 100, d["gamma_5"]["tokens_per_pass"]
                f.write(f"| $E={e_val}$ | **{m_name}** | {a3:.1f}% | {t3:.2f}× | {a5:.1f}% | {t5:.2f}× |\n")
    print(f" [Done] Saved formatted rollout report to: {md_file}")


def main():
    p = argparse.ArgumentParser(description="End-to-End Autoregressive Rollouts for CheapPass")
    p.add_argument("--model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--depths", default="18,24,29")
    p.add_argument("--rank", type=int, default=128)
    p.add_argument("--n_prompts", type=int, default=64, help="Number of prompts to evaluate")
    p.add_argument("--prompt_len", type=int, default=32, help="Prompt token length")
    p.add_argument("--n_new", type=int, default=128, help="Newly generated continuation tokens")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt_dir", default="cheap_pass_out", help="Directory where trained adapter weights are stored")
    p.add_argument("--out_file", default="results/cheap_pass_rollout/rollout_results.json")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    run_rollout_suite(args)


if __name__ == "__main__":
    main()

