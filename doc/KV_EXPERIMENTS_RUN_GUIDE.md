# Master Experiment Execution Guide: KV-Cache Continuity Paper

**Project:** *KV-Cache Continuity and Intermediate Representation Limits in Depth-Truncated Transformers*  
**Scope:** **Exclusively experiments for the KV Continuity Paper** (Reviewer Revisions, Missing Baselines, Rigor Diagnostics, and the Scheduled Speculative Decoding Track). No graph-paper systems experiments are included here.  
**Target Environments:** Kaggle (Tesla P100 / T4 x2), Google Colab, or Local CUDA / CPU.

---

## Live Experiment Progress Dashboard (12 of 14 Completed — 86%)

| # | Experiment Focus | Key Paper Claim | Status | Generated Output File |
|---|---|---|:---:|---|
| **A1** | Standardized Table 1 ($N=30$, 3 seeds) | Native Proj crushes Cache Omission & Direct Copy | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_table1/standardized_table1_headline.csv` |
| **A5** | Multi-Seed Shallow Llama Sweep | Pinpoints exact 1-bit boundary at $L=8$ ($50\%$) | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_llama_shallow/llama_multiseed_sweep_meta-llama_Llama-3.2-1B.csv` |
| **A10**| Centered Cosine & Attn Error | Proves high Key cosine is real (not channel bias) | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_cosine_attn/centered_cosine_and_attn_err_L18.csv` |
| **A9** | Causal K/V Swap on Qwen ($L=14\dots 30$) | Value Dominance = $68.6\% - 78.3\%$ across depths | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_kv_swap_qwen/robust_swap_and_metrics_Qwen_Qwen2.5-3B.csv` |
| **A9** | Causal K/V Swap on Llama ($L=6\dots 10$) | Cross-architecture Value Dominance = $64.5\% - 82.1\%$ | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_kv_swap_llama/robust_swap_and_metrics_meta-llama_Llama-3.2-1B.csv` |
| **A3** | Autoregressive Generation (256 tok) | Evaluates open-loop compounding error | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_autoregressive/autoregressive_gen_Qwen_Qwen2.5-3B.csv` |
| **A2** | Extra Fill Baselines (Zero, Mean, Linear) | Beats all heuristic fills by $+38.3\%$ Top-1 | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_kv_baselines/kv_baselines_Qwen_Qwen2.5-3B.csv` |
| **A7** | Downstream Tasks (HellaSwag/ARC/LAMBADA) | 100% accuracy parity with Oracle at $L=24$ ($\Delta=0$) | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_downstream/downstream_benchmarks_Qwen_Qwen2.5-3B.csv` |
| **A11**| Entropy-Matched Quintiles (Wiki/C4/Code) | Error remains below $\ln 2$ across all entropy bins | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_entropy/entropy_domain_comparison_Qwen_Qwen2.5-3B.csv` |
| **A11**| Domain & Context Length Scaling | Verified across WikiText, C4, Code up to $T=512$ | **COMPLETE** | `results/canonical_sweeps/kv_paper/revision_entropy/paper_table_length_and_domain.tex` |
| **Trk**| 60-Prompt Multi-Domain Speculative | $96.67\%$ (58/60) exact token equivalence | **COMPLETE** | `results/canonical_sweeps/kv_paper/speculative_track/experiment_4_tuned_lens_60_prompt.json` |
| **Trk**| Speculative Trade-Off Curves | Publication curves for acceptance vs depth | **COMPLETE** | `results/canonical_sweeps/kv_paper/speculative_track/experiment_8_tuned_lens_depth_tradeoff.pdf` |
| **A12**| Multi-Depth Adapter Training (18, 22, 24, 30)| Prepares matched checkpoints on WikiText-2 | *Running / Pending* | `checkpoints/multi_depth_r512/final_exit_adapters_qwen_mlp_r512.pt` |
| **A4** | Joint Early-Exit Condition Sweep | Error decomposition $\Delta$ with matching adapter | *Pending A12* | `results/canonical_sweeps/kv_paper/revision_joint_condition/joint_condition_decomposition_Qwen_Qwen2.5-3B.csv` |

---

## 0. Four Critical Rules (Read Before Spending GPU Hours)

1. **Joint Condition Needs Matching Adapters (Fix 1):**  
   Never evaluate the joint condition using an adapter trained at a single depth ($L=18$) or on an out-of-domain dataset (TinyStories). Train the multi-depth in-domain adapter covering all candidate depths ($L \in \{18, 22, 24, 30\}$) on WikiText-2 first (Phase 3), then evaluate the joint condition at those identical depths.
2. **Calibrated Hypothesis Phrasing (Fix 2):**  
   If the multi-depth adapter reaches an empirical plateau near $\approx 82\%$ Top-1 agreement, do **not** claim an "intrinsic manifold limitation". Instead, report it precisely as:  
   > *"An empirical representation ceiling for this specific adapter family (bottleneck MLP with residual connection) under the given parameter budget ($r=512$), rather than an absolute theoretical bound."*
3. **Strict Precision & Dtype Consistency (Fix 3):**  
   NVIDIA Tesla P100 GPUs have **no native bfloat16 hardware support**, and T4 GPUs emulate bfloat16 slowly. Always pass `--dtype float16` across all scripts on Kaggle. **Never mix dtypes within the same table** (e.g., baseline in fp16 and early-exit in bf16). Every row in a comparison table must share identical precision.
4. **Gated Llama-3.2 Authentication (Fix 4):**  
   `meta-llama/Llama-3.2-1B` is a gated Hugging Face repository. Add your Hugging Face access token as a Kaggle Secret named `HF_TOKEN` (via *Add-ons -> Secrets*) and ensure *Notebook Settings -> Internet* is turned **ON**, otherwise model downloads will fail.

---

## 1. Environment Setup & Pre-Flight Smoke Checks

### 1.1. Kaggle / Colab Initial Setup
```bash
# Verify GPU accelerator
!nvidia-smi

# Install required dependencies
!pip install -q torch transformers datasets accelerate sentencepiece

# Authenticate Hugging Face for Llama-3.2
import os
from huggingface_hub import login
if "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    print("[OK] Hugging Face authenticated successfully.")
else:
    print("[!] Warning: HF_TOKEN secret not found. Llama-3.2 download will fail if not authenticated.")
```

Set repository directory and run unit tests:
```bash
%cd /kaggle/working/kv_continuity
export PYTHONPATH=.

# Verify that all 11 unit tests pass
!python -m unittest discover tests
```

### 1.2. Pre-Flight Smoke Test Suite (< 3 minutes)
Before launching multi-hour runs, verify every script end-to-end with minimal samples (`--num-samples 2`, single seed):

```bash
# Smoke test Table 1 headline script
python experiments/run_tier1_interventions.py --model-type qwen --model-id Qwen/Qwen2.5-3B --dtype float16 --num-samples 2 --seeds 42 --sweep-depths 18 --skip-swap --skip-cosine --device cuda --output-dir results/smoke_test

# Smoke test Llama shallow sweep
python experiments/run_multiseed_llama_sweep.py --model-id meta-llama/Llama-3.2-1B --depths 4 --seeds 42 --num-samples 2 --device cuda --output-dir results/smoke_test

# Smoke test Causal KV swap
python experiments/run_robust_kv_swap_and_metrics.py --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 --device cuda --output-dir results/smoke_test

# Smoke test Autoregressive generation
python experiments/run_autoregressive_generation.py --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 --max-new-tokens 16 --device cuda --output-dir results/smoke_test

# Smoke test Extra fill baselines
python experiments/run_kv_baselines.py --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 --device cuda --output-dir results/smoke_test
```

---

## 2. Priority Phase 1: Core Reviewer Objections (Run First)

### 2.1. Standardized Table I ($N=30$, 3 Seeds) with Cache Omission (A1)
* **Goal:** Standardizes the headline evaluation protocol on WikiText-2 across $N=30$ sequences (1,890 tokens per seed) with 3 independent seeds (`42, 123, 7`). Updates the headline baseline numbers ($\text{PPL}^* = 5.76$) and adds the Cache Omission baseline.
* **Evaluated conditions:** Full-Depth Oracle ($L=36$), Readout-Only (Zero-Shot), Attn-Only (**Cache Omission**: skipped slots masked out), Attn-Only (Direct Copy), and Attn-Only (Native Projection).
* **Estimated Runtime:** ~12–15 mins on T4/P100; ~4 mins on A100.

```bash
python experiments/run_tier1_interventions.py \
  --model-type qwen \
  --model-id Qwen/Qwen2.5-3B \
  --dtype float16 \
  --num-samples 30 \
  --seeds 42,123,7 \
  --sweep-depths 18,24,30 \
  --skip-swap \
  --skip-cosine \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_table1
```
* **Output:** `results/canonical_sweeps/revision_table1/standardized_table1_headline.csv`

---

### 2.4. Shallow Llama-3.2-1B Depth Sweep ($L \in \{4, 5, 6, 7, 8, 10, 12, 14\}$) (A5)
* **Goal:** Sweeps shallow exit depths on `meta-llama/Llama-3.2-1B` across 3 seeds (`42, 123, 7`) to pinpoint the exact empirical crossing point below the $\ln 2 \approx 0.6931$ nats boundary.
* **Estimated Runtime:** ~8–12 mins on T4/P100; ~3 mins on A100.

```bash
python experiments/run_multiseed_llama_sweep.py \
  --model-id meta-llama/Llama-3.2-1B \
  --depths 4,5,6,7,8,10,12,14 \
  --seeds 42,123,7 \
  --num-samples 30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_llama_shallow
```
* **Output:** `results/canonical_sweeps/revision_llama_shallow/llama_multiseed_sweep_meta-llama_Llama-3.2-1B.csv`

---

### 2.3. Centered Cosine Similarity & Direct Attention-Output Error (A10)
* **Goal:** Eliminates the Qwen QKV-bias / constant channel mean confound by computing centered cosine similarity, alongside the direct attention-output Frobenius error: $\|\text{Attn}(K^*, V^*) - \text{Attn}(\hat{K}, \hat{V})\|_F$.
* **Estimated Runtime:** ~5–7 mins on T4/P100; ~2 mins on A100.

```bash
python experiments/run_tier1_interventions.py \
  --model-type qwen \
  --model-id Qwen/Qwen2.5-3B \
  --dtype float16 \
  --swap-depths 18 \
  --num-samples 30 \
  --skip-table1 \
  --skip-swap \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_cosine_attn
```
* **Output:** `results/canonical_sweeps/revision_cosine_attn/centered_cosine_and_attn_err_L18.csv`

---

### 2.2. Expanded Causal Key/Value Swap Across Depths & Architectures (A9)
* **Goal:** Measures symmetric Shapley Value Dominance ($\% S_V = \frac{S_V}{S_V + S_K} \times 100\%$) across multiple exit depths on Qwen ($L \in \{14, 18, 22, 24, 30\}$) and cross-architecture replication on Llama ($L \in \{6, 8, 10\}$).
* **Estimated Runtime:** ~15–20 mins on T4/P100; ~5 mins on A100.

```bash
# Qwen2.5-3B (Depths 14, 18, 22, 24, 30):
python experiments/run_robust_kv_swap_and_metrics.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 14,18,22,24,30 \
  --num-samples 30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_kv_swap_qwen

# Llama-3.2-1B (Depths 6, 8, 10):
python experiments/run_robust_kv_swap_and_metrics.py \
  --model-id meta-llama/Llama-3.2-1B \
  --depths 6,8,10 \
  --num-samples 30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_kv_swap_llama
```
* **Outputs:**  
  `results/canonical_sweeps/revision_kv_swap_qwen/robust_swap_and_metrics_Qwen_Qwen2.5-3B.csv`  
  `results/canonical_sweeps/revision_kv_swap_llama/robust_swap_and_metrics_meta-llama_Llama-3.2-1B.csv`

---

## 3. Priority Phase 2: Autoregressive Generation & Extra Baselines

### 4.1. Autoregressive (Non-Teacher-Forced) Generation (A3)
* **Goal:** Directly resolves the biggest reviewer objection by measuring compounding error in real, open-loop autoregressive generation over 256 tokens per sequence. Measures PPL degradation, token divergence curves, and Top-1 match rates at tokens 64, 128, and 256.
* **Estimated Runtime:** ~15–20 mins on T4/P100; ~5 mins on A100.

```bash
python experiments/run_autoregressive_generation.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 18,24,30 \
  --num-samples 30 \
  --max-new-tokens 256 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_autoregressive
```
* **Outputs:**  
  `results/canonical_sweeps/revision_autoregressive/autoregressive_gen_Qwen_Qwen2.5-3B.csv`  
  `results/canonical_sweeps/revision_autoregressive/qualitative_samples_Qwen_Qwen2.5-3B.txt`

---

### 4.2. Extra Fill Baselines (A2, A6)
* **Goal:** Evaluates Zero Fill, Mean K/V, Previous-Token Copy, Learned Linear Predictor, and CALM propagation to confirm Native Projection decisively outperforms all trivial and heuristic alternatives.
* **Estimated Runtime:** ~10–12 mins on T4/P100; ~3 mins on A100.

```bash
python experiments/run_kv_baselines.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 18,24,30 \
  --num-samples 30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_kv_baselines
```
* **Output:** `results/canonical_sweeps/revision_kv_baselines/kv_baselines_Qwen_Qwen2.5-3B.csv`

---

## 4. Priority Phase 3: Multi-Depth Adapter Training & Joint Condition

### 3.2. Multi-Depth In-Domain Adapter Training (A12)
* **Goal:** Trains matching bottleneck adapters ($r=512$) covering all candidate depths ($L \in \{18, 22, 24, 30\}$) on WikiText-2. Saves periodic checkpoints every 1,000 steps.
* **Capacity Test:** Run 10,000 steps first (~10M tokens through the 3B teacher). If you have sufficient GPU hours, continue to 25,000 steps to test the adapter capacity plateau.
* **Estimated Runtime (10k steps):** ~2.5 hours on T4/P100; ~45 mins on A100.

```bash
python experiments/train_exit_adapters.py \
  --model-type qwen \
  --model-id Qwen/Qwen2.5-3B \
  --candidate-depths 18,22,24,30 \
  --adapter-type mlp \
  --rank 512 \
  --steps 10000 \
  --batch-size 8 \
  --seq-len 128 \
  --eval-every 500 \
  --device cuda \
  --output-dir checkpoints/multi_depth_r512
```
* **Checkpoint:** `checkpoints/multi_depth_r512/final_exit_adapters_qwen_mlp_r512.pt`

---

### 3.1. Joint Early-Exit Condition Evaluation (A4)
* **Goal:** Evaluates the combined early-exit system at matched depths ($L=18, 22, 24, 30$) using the checkpoint trained above. Directly tests error additivity vs. interaction term $\Delta = \mathcal{E}_{\text{joint}} - (\mathcal{E}_{\text{readout}} + \mathcal{E}_{\text{attn}})$.
* **Estimated Runtime:** ~6–8 mins on T4/P100; ~2 mins on A100.

```bash
python experiments/run_joint_condition_sweep.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 18,22,24,30 \
  --adapter-path checkpoints/multi_depth_r512/final_exit_adapters_qwen_mlp_r512.pt \
  --num-samples 30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_joint_condition
```
* **Output:** `results/canonical_sweeps/revision_joint_condition/joint_condition_decomposition_Qwen_Qwen2.5-3B.csv`

---

## 5. Priority Phase 4: Extended Generalization, Robustness & Downstream

### 5.1. Long-Context Scaling ($T \in \{1024, 2048, 4096\}$) (A8)
* **Goal:** Demonstrates that Native Projection error does not accumulate over extended sequence contexts up to $4{,}096$ tokens.
* **Estimated Runtime:** ~15–20 mins on T4/P100; ~5 mins on A100.

```bash
python experiments/run_long_context_scaling.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 18,24,30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_long_context
```
* **Output:** `results/canonical_sweeps/revision_long_context/long_context_scaling_Qwen_Qwen2.5-3B.csv`

---

### 5.2. Downstream Zero-Shot Benchmarks (A7)
* **Goal:** Evaluates accuracy on HellaSwag, ARC-Easy, and LAMBADA under Native Projection at exit depths $L \in \{22, 24, 30\}$.
* **Estimated Runtime:** ~20–25 mins on T4/P100; ~6 mins on A100.

```bash
python experiments/run_downstream_benchmarks.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 22,24,30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_downstream
```
* **Output:** `results/canonical_sweeps/revision_downstream/downstream_benchmarks_Qwen_Qwen2.5-3B.csv`

---

### 5.3. Entropy & Domain Breakdown (A11)
* **Goal:** Bins tokens by model predictive entropy into 5 quintiles across WikiText-2, C4, and Python code, testing whether high-entropy tokens are more sensitive to Key vs. Value approximation.
* **Estimated Runtime:** ~10–12 mins on T4/P100; ~3 mins on A100.

```bash
python experiments/run_entropy_domain_comparison.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 18,24 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_entropy
```
* **Output:** `results/canonical_sweeps/revision_entropy/entropy_domain_comparison_Qwen_Qwen2.5-3B.csv`

---

### 5.4. Large Model Scaling (7B–8B with Quantization) (A13)
* **Goal:** Validates that Value-dominance and Native Projection scaling replicate on `Qwen/Qwen2.5-7B` or `meta-llama/Llama-3.1-8B` using 8-bit/4-bit quantization on a single GPU.
* **Estimated Runtime:** ~15–20 mins on T4/P100; ~5 mins on A100.

```bash
python experiments/run_large_model_sweep.py \
  --model-id Qwen/Qwen2.5-7B \
  --depths 14,20,24 \
  --load-in-8bit \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_7b
```
* **Output:** `results/canonical_sweeps/revision_7b/large_model_sweep_Qwen_Qwen2.5-7B.csv`

---

### 5.5. Active Speculative Decoding with Cascading Upper-Layer Reads (A14)
* **Goal:** Evaluates speculative drafting ($\gamma \ge 2$) where Draft Token 2 actively reads Native Projection upper-layer slots created by Draft Token 1, proving upper slots are functional and usable.
* **Estimated Runtime:** ~10–12 mins on T4/P100; ~3 mins on A100.

```bash
python experiments/run_active_speculative_decoding.py \
  --model-id Qwen/Qwen2.5-3B \
  --d1 18 \
  --d2 24 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_active_spec
```
* **Output:** `results/canonical_sweeps/revision_active_spec/active_speculative_decoding_report.json`

---

## 6. Priority Phase 5: Scheduled Speculative Decoding & Tuned-Lens Track

*The scheduled experimental track targeting practical super-unity wall-clock throughput via lightweight low-rank calibration.*

### 6.1. Multi-Token Speculative Smoke Benchmark ($\gamma=2$)
* **Goal:** Validates multi-token verification with the existing MLP adapter, confirming lossless exact matches without CUDA OOM or NaN.
* **Estimated Runtime:** ~3–4 mins on T4/P100.

```bash
python experiments/benchmark_speculative_decoding.py \
  --model-id Qwen/Qwen2.5-3B \
  --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \
  --draft-depth 24 \
  --gamma 2 \
  --max-tokens 40 \
  --dtype float16 \
  --device cuda
```

---

### 6.2. Fast Tuned-Lens Calibration Training (< 3 Minutes on GPU)
* **Goal:** Exploits the Key-stable / Value-evolving asymmetry by training an ultra-lightweight ($<0.3\text{M}$ parameter, rank-$64$) low-rank affine tuned lens directly on the intermediate readout direction:
  $$h_{\text{cal}} = h_{d_t} + \frac{\alpha}{r} W_{\text{up}} W_{\text{down}} \operatorname{LN}(h_{d_t})$$
* **Estimated Runtime:** ~2–3 mins on T4/P100; < 1 min on A100.

```bash
python experiments/train_tuned_lens.py \
  --model-id Qwen/Qwen2.5-3B \
  --target-depths 14,18,22,24,28 \
  --adapter-type lora \
  --rank 64 \
  --steps 400 \
  --batch-size 2 \
  --seq-len 128 \
  --lr 1e-3 \
  --distill-temp 2.0 \
  --output-path checkpoints/exit_adapters_qwen_tuned_lens.pt \
  --device cuda
```
* **Checkpoint:** `checkpoints/exit_adapters_qwen_tuned_lens.pt`

---

### 6.3. Systematic Speculative Depth Sweep ($d_t \in \{14, 18, 22, 24, 28, 30\}$)
* **Goal:** Measures empirical draft acceptance rate $\alpha(d_t)$, naive verification speedup, partial-layer verification speedup ($1+\alpha$), and real measured GPU throughput.
* **Estimated Runtime:** ~8–10 mins on T4/P100; ~3 mins on A100.

```bash
# Evaluate baseline MLP adapter:
python experiments/sweep_speculative_depths.py \
  --model-id Qwen/Qwen2.5-3B \
  --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \
  --depths 14,18,22,24,28,30 \
  --gamma 1 \
  --num-prompts 12 \
  --max-tokens 30 \
  --dtype float16 \
  --device cuda \
  --output-dir results/canonical_sweeps/qwen2.5_3b

# Evaluate newly calibrated Tuned-Lens adapter:
python experiments/sweep_speculative_depths.py \
  --model-id Qwen/Qwen2.5-3B \
  --adapter-path checkpoints/exit_adapters_qwen_tuned_lens.pt \
  --depths 14,18,22,24,28,30 \
  --gamma 1 \
  --num-prompts 12 \
  --max-tokens 30 \
  --dtype float16 \
  --device cuda \
  --output-dir results/canonical_sweeps/qwen2.5_3b
```
* **Outputs:** `speculative_depth_sweep.csv` and `speculative_depth_tradeoff.pdf`

---

### 6.4. Publication-Scale 60-Prompt Multi-Domain Speculative Benchmark
* **Goal:** Measures the multi-domain speedup across $N=60$ prompts (Math, Code, Facts, Language) when pairing Native Projection with the tuned-lens readout head.
* **Estimated Runtime:** ~15–20 mins on T4/P100; ~5 mins on A100.

```bash
python experiments/benchmark_large_scale.py \
  --model-id Qwen/Qwen2.5-3B \
  --adapter-path checkpoints/exit_adapters_qwen_tuned_lens.pt \
  --draft-depth 24 \
  --gamma 1 \
  --num-prompts-per-domain 15 \
  --max-tokens 30 \
  --device cuda
```
* **Output:** `results/canonical_sweeps/qwen2.5_3b/benchmark_tuned_lens_results.json`

---

## 7. Master Execution Script & Time Summary

### 7.1. Wall-Clock Runtime Summary

| Phase | Description | Est. Time (T4 / P100) | Est. Time (A100) |
| :--- | :--- | :--- | :--- |
| **Pre-Flight** | Unit Tests & Smoke Tests | **~2 – 3 mins** | ~1 min |
| **Phase 1** | Standardized Table 1, Llama Sweep, Cosine & Attn Error, Causal Swap | **~45 – 55 mins** | ~15 – 18 mins |
| **Phase 2** | Autoregressive Generation & Extra Fill Baselines | **~25 – 32 mins** | ~8 – 10 mins |
| **Phase 3** | Multi-Depth Adapter Training (10k steps) & Joint Condition | **~2.5 – 3.0 hours** | ~45 – 55 mins |
| **Phase 4** | Long-Context, Downstream, Entropy, 7B Sweep, Active Speculative | **~60 – 75 mins** | ~20 – 25 mins |
| **Phase 5** | Speculative Decoding & Tuned-Lens Scheduled Track | **~25 – 30 mins** | ~8 – 10 mins |
| **TOTAL** | **Entire KV Paper Experimental Evidence Suite** | **~5.0 – 6.0 hours** | **~1.8 – 2.2 hours** |

### 7.2. Recommended Two-Session Kaggle Plan

To prevent losing work from browser disconnects or session timeouts (9-hour limit):

#### Session 1: All Inference & Evaluation Experiments (~1.5 to 2 hours)
Execute Phases 1, 2, 4, and 5 in order. Then zip and download the results:
```python
import shutil
shutil.make_archive("kv_inference_results", "zip", "results")
print("[OK] Session 1 results ready for download in Kaggle Output pane!")
```

#### Session 2: Adapter Training & Joint Condition (~2.5 to 3 hours)
Execute Phase 3:
1. Train the multi-depth adapter (`train_exit_adapters.py`, 10k steps).
2. Evaluate the joint condition (`run_joint_condition_sweep.py`).
3. Zip and download checkpoints and results:
```python
import shutil
shutil.make_archive("kv_adapter_checkpoints", "zip", "checkpoints")
shutil.make_archive("kv_joint_results", "zip", "results")
print("[OK] Session 2 checkpoints and joint condition results ready for download!")
```

---

## 8. Artifact Registry & Experiment Completion Status

| Experiment | Script | Output Artifact File | Status | Key Validated Result |
| :--- | :--- | :--- | :---: | :--- |
| **Table I Standardized ($N=30$, 3 seeds) (A1)** | `run_tier1_interventions.py` | `results/canonical_sweeps/kv_paper/revision_table1/standardized_table1_headline.csv` | **COMPLETE** | Native Proj (59.4% Top-1 @ L18) crushes Cache Omission (20.9%) & Direct Copy (11.6%) |
| **Shallow Llama-3.2 Sweep (A5)** | `run_multiseed_llama_sweep.py` | `results/canonical_sweeps/kv_paper/revision_llama_shallow/llama_multiseed_sweep_meta-llama_Llama-3.2-1B.csv` | **COMPLETE** | Pinpointed exact $\ln 2$ boundary at L=8 ($D_{\text{KL}}=0.5898 \le \ln 2$, 63.3% Top-1) |
| **Centered Cosine & Attn Error (A10)** | `run_tier1_interventions.py` | `results/canonical_sweeps/kv_paper/revision_cosine_attn/centered_cosine_and_attn_err_L18.csv` | **COMPLETE** | Centered $\cos_K \approx$ Uncentered ($<0.001$ diff), ruling out channel-bias confound |
| **Causal Swap on Qwen (A9)** | `run_robust_kv_swap_and_metrics.py` | `results/canonical_sweeps/kv_paper/revision_kv_swap_qwen/robust_swap_and_metrics_Qwen_Qwen2.5-3B.csv` | **COMPLETE** | Value Dominance = 68.6% - 78.3% across L14-L30; Keys alone give 82.6%-90.5% Top-1 |
| **Causal Swap on Llama (A9)** | `run_robust_kv_swap_and_metrics.py` | `results/canonical_sweeps/kv_paper/revision_kv_swap_llama/robust_swap_and_metrics_meta-llama_Llama-3.2-1B.csv` | **COMPLETE** | Cross-architecture replication: Value Dominance = 64.5% - 82.1% across L6-L10 |
| **Autoregressive Generation (A3)** | `run_autoregressive_generation.py` | `results/canonical_sweeps/kv_paper/revision_autoregressive/autoregressive_gen_Qwen_Qwen2.5-3B.csv` | **COMPLETE** | 256-token compounding error evaluated across depths, resolving #1 reviewer objection |
| **Extra Fill Baselines (A2)** | `run_kv_baselines.py` | `results/canonical_sweeps/kv_paper/revision_kv_baselines/kv_baselines_Qwen_Qwen2.5-3B.csv` | **COMPLETE** | Native Proj (61.0% Top-1) beats Zero (22.7%), Mean (22.7%), Copy (11.4%) |
| **Downstream Tasks (A7)** | `run_downstream_benchmarks.py` | `results/canonical_sweeps/kv_paper/revision_downstream/downstream_benchmarks_Qwen_Qwen2.5-3B.csv` | **COMPLETE** | 100% parity with 36L Oracle on HellaSwag/ARC/LAMBADA (75.0% acc) at L24 ($\Delta=0.0\%$) |
| **Entropy & Domain Match (A11)** | `run_entropy_domain_comparison.py` | `results/canonical_sweeps/kv_paper/revision_entropy/entropy_domain_comparison_Qwen_Qwen2.5-3B.csv` | **COMPLETE** | $D_{\text{KL}} \le \ln 2$ across all entropy quintiles (WikiText, C4, Code) at L24 |
| **Domain & Length Scaling (A11)** | `benchmark_length_and_domain.py` | `results/canonical_sweeps/kv_paper/revision_entropy/paper_table_length_and_domain.tex` | **COMPLETE** | Scaled across WikiText-2, C4, and Python code up to T=512 tokens |
| **60-Prompt Speculative Benchmark** | `benchmark_large_scale.py` | `results/canonical_sweeps/kv_paper/speculative_track/experiment_4_tuned_lens_60_prompt.json` | **COMPLETE** | 96.67% (58/60) exact match rate across 4 domains, 21.7% acceptance |
| **Speculative Depth Trade-off Curves** | `sweep_speculative_depths.py` | `results/canonical_sweeps/kv_paper/speculative_track/experiment_8_tuned_lens_depth_tradeoff.pdf` | **COMPLETE** | Publication plots produced for empirical acceptance vs draft depth |
| **Matching Multi-Depth Adapter (A12)** | `train_exit_adapters.py` | `checkpoints/multi_depth_r512/final_exit_adapters_qwen_mlp_r512.pt` | *Pending* | Train with batch size 4 (steps 3000) before running Joint Condition |
| **Joint Condition Sweep (A4)** | `run_joint_condition_sweep.py` | `results/canonical_sweeps/kv_paper/revision_joint_condition/joint_condition_decomposition_Qwen_Qwen2.5-3B.csv` | *Pending* | Run immediately after adapter checkpoint completes |
| **Long-Context Scaling T=4096 (A8)** | `run_long_context_scaling.py` | `results/canonical_sweeps/kv_paper/revision_long_context/long_context_scaling_Qwen_Qwen2.5-3B.csv` | *Optional* | Tests extreme context scaling up to T=4096 |
| **7B Model Scaling (A13)** | `run_large_model_sweep.py` | `results/canonical_sweeps/kv_paper/revision_7b/large_model_sweep_Qwen_Qwen2.5-7B.csv` | *Optional* | 8-bit quantized 7B model cross-check |
| **Active Speculative Decoding (A14)** | `run_active_speculative_decoding.py` | `results/canonical_sweeps/kv_paper/revision_active_spec/active_speculative_decoding_report.json` | *Optional* | Multi-token cascading upper-layer draft read |
