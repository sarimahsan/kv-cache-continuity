# Kaggle & Colab Master Execution Guide: Manuscript Revision & Experiments

This document provides the complete, authoritative execution guide for running all revision experiments on **Kaggle** (P100 or T4 x2) or **Google Colab**, alongside local CUDA/CPU execution.

It incorporates all **reviewer objections (Part A: A1–A14)**, all **CUDA-Graph systems benchmarks (Part B: B1–B16)**, and the **old scheduled experiments track (speculative decoding & tuned-lens calibration)**.

---

## 0. Four Critical Rules Before Spending GPU Hours

1. **Joint Condition Needs Matching Adapters (Fix 1):**  
   Never evaluate the joint condition (3.1) using an adapter trained at a different depth ($L=18$ only) or on an out-of-domain dataset (e.g. TinyStories). Train the multi-depth in-domain adapter covering all candidate depths ($L \in \{18, 22, 24, 30\}$) on WikiText-2 first (3.2), then evaluate the joint condition at those identical depths.
2. **Calibrated Hypothesis Phrasing (Fix 2):**  
   If the multi-depth adapter reaches an empirical plateau near $\approx 82\%$ Top-1 agreement, do **not** claim an "intrinsic manifold limitation". Instead, report it precisely as:  
   > *"An empirical representation ceiling for this specific adapter family (bottleneck MLP with residual connection) under the given parameter budget ($r=512$), rather than an absolute theoretical bound."*
3. **Strict Precision & Dtype Consistency (Fix 3):**  
   NVIDIA Tesla P100 GPUs have **no native bfloat16 hardware support**, and T4 GPUs emulate bfloat16 slowly. Always pass `--dtype float16` across all scripts on Kaggle. **Never mix dtypes within the same table** (e.g., full-depth baseline in fp16 and early-exit in bf16). Every row in a comparison table must share identical precision.
4. **Gated Llama-3.2 Authentication (Fix 4):**  
   `meta-llama/Llama-3.2-1B` is a gated Hugging Face repository. Add your Hugging Face access token as a Kaggle Secret named `HF_TOKEN` (via *Add-ons -> Secrets*) and ensure *Notebook Settings -> Internet* is turned **ON**, otherwise weights cannot be downloaded.

---

## 1. Environment Setup & Smoke Checks (Run First)

In your first Kaggle notebook code cell, install dependencies and log in:

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

Set the repository directory and verify all unit tests pass:

```bash
%cd /kaggle/working/kv_continuity
export PYTHONPATH=.

# Verify that all 11 unit tests pass
!python -m unittest discover tests
```

### Pre-Flight Smoke Test Suite (Takes < 2 minutes)
Before launching multi-hour runs, verify every script end-to-end with minimal samples (`--num-samples 2`, single seed):

```bash
# Smoke test 2.1 (Table 1)
python experiments/run_tier1_interventions.py --model-type qwen --model-id Qwen/Qwen2.5-3B --dtype float16 --num-samples 2 --seeds 42 --sweep-depths 18 --skip-swap --skip-cosine --device cuda --output-dir results/smoke_test

# Smoke test 2.4 (Llama shallow sweep)
python experiments/run_multiseed_llama_sweep.py --model-id meta-llama/Llama-3.2-1B --depths 4 --seeds 42 --num-samples 2 --device cuda --output-dir results/smoke_test

# Smoke test 2.3 (Centered cosine & attention error)
python experiments/run_tier1_interventions.py --model-type qwen --model-id Qwen/Qwen2.5-3B --dtype float16 --swap-depths 18 --num-samples 2 --skip-table1 --skip-swap --device cuda --output-dir results/smoke_test

# Smoke test 2.2 (Causal KV swap)
python experiments/run_robust_kv_swap_and_metrics.py --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 --device cuda --output-dir results/smoke_test

# Smoke test 4 (Autoregressive generation)
python experiments/run_autoregressive_generation.py --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 --max-new-tokens 16 --device cuda --output-dir results/smoke_test

# Smoke test 5 (Extra fill baselines)
python experiments/run_kv_baselines.py --model-id Qwen/Qwen2.5-3B --depths 18 --num-samples 2 --device cuda --output-dir results/smoke_test
```

---

## 2. Priority Phase 1: Cheap & Moderate Headline Experiments

### 2.1. Standardized Table I ($N=30$, 3 Seeds) with Cache Omission (A1)
*Changes the headline numbers (the baseline PPL\* of 5.76 updates with the larger 1,890-token slice) and establishes the cache-omission baseline.*

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
* **Evaluated conditions:** Full-Depth Oracle ($L=36$), Readout-Only (Zero-Shot), Attn-Only (**Cache Omission**: skipped slots masked out), Attn-Only (Direct Copy), and Attn-Only (Native Projection).
* **Output:** `results/canonical_sweeps/revision_table1/standardized_table1_headline.csv`

---

### 2.4. Shallow Llama-3.2-1B Depth Sweep ($L \in \{4, 5, 6, 7, 8, 10, 12, 14\}$) (A5)
*Pinpoints the exact empirical crossing point below the $\ln 2 \approx 0.6931$ nats boundary across 3 seeds.*

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
*Removes the Qwen QKV-bias / constant channel mean confound and computes direct attention-output Frobenius error: $\|\text{Attn}(K^*, V^*) - \text{Attn}(\hat{K}, \hat{V})\|_F$.*

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
*Measures symmetric Shapley Value Dominance ($\% S_V = \frac{S_V}{S_V + S_K} \times 100\%$) across multiple exit depths on Qwen and cross-architecture on Llama.*

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

## 3. Priority Phase 2: Multi-Depth Adapter Training & Joint Condition

### 3.2. Multi-Depth In-Domain Adapter Training (Train Matching Depths First!)
*Trains matching bottleneck adapters at all four candidate depths ($L \in \{18, 22, 24, 30\}$) on WikiText-2. To preserve GPU hours, runs 10,000 steps first with regular checkpoint saving every 1,000 steps.*

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
* **Checkpoints:** `checkpoints/multi_depth_r512/final_exit_adapters_qwen_mlp_r512.pt`

---

### 3.1. Joint Early-Exit Condition Evaluation (A4)
*Evaluates the complete early-exit system at matching depths ($L=18, 22, 24, 30$) using the multi-depth in-domain adapter trained above, testing error additivity vs. interaction term $\Delta$.*

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

## 4. Priority Phase 3: Autoregressive Generation & Extra Baselines

### 4.1. Autoregressive (Non-Teacher-Forced) Generation (A3)
*Addresses the biggest reviewer objection in the paper by testing open-loop compounding error over 256 generated tokens.*

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

### 4.2. Extra Fill Baselines (A2)
*Evaluates Zero Fill, Mean K/V, Previous-Token Copy, Learned Linear Predictor, and CALM propagation.*

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

### 4.3. Long-Context Scaling (A8) & Downstream Zero-Shot Benchmarks (A7)
```bash
# Long Context Scaling (T in {1024, 2048, 4096}):
python experiments/run_long_context_scaling.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 18,24,30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_long_context

# Downstream Benchmarks (HellaSwag, ARC-Easy, LAMBADA):
python experiments/run_downstream_benchmarks.py \
  --model-id Qwen/Qwen2.5-3B \
  --depths 22,24,30 \
  --device cuda \
  --output-dir results/canonical_sweeps/revision_downstream
```

---

## 5. Priority Phase 4: CUDA-Graph Paper Experiments (B-Series on 125M)

*Run these to validate the systems claims for the CUDA-Graph paper. These run rapidly on either local CUDA/CPU or Kaggle.*

```bash
# B1 & B7: Quality per bucket and audit whether earlier figures ignored the KV hole
python experiments/run_graph_quality_per_bucket.py --device cuda

# B3 & B4: Fill cost inside captured graph & Fused batched GEMM fill speedup
python experiments/run_graph_fill_cost.py --device cuda

# B5: Sync-free routing profiler trace audit (verifies 0 DtoH copies, 0 .item() calls)
python experiments/run_graph_sync_free_check.py --device cuda

# B6: Main throughput table across batch sizes B in {1, 2, 4, 8, 16, 32, 64}
python experiments/run_graph_main_throughput.py --batch-sizes 1,2,4,8,16,32,64 --repeats 5 --device cuda

# B8: Multi-token batch handling policies (Max-Depth vs. Sub-Batching)
python experiments/run_graph_batch_policies.py --batch-sizes 2,4,8,16,32,64 --device cuda
```

---

## 6. Old Scheduled Experiments: Speculative Decoding & Tuned-Lens Track

*The existing scheduled experimental track exploring practical super-unity wall-clock throughput via lightweight calibration.*

### 6.1. Multi-Token Speculative Decoding Smoke Benchmark ($\gamma=2$)
*Checks lossless multi-token verification without crashes or NaN.*

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
*Trains an ultra-lightweight ($<0.3\text{M}$ parameter, rank-$64$) low-rank affine tuned lens directly on the intermediate readout direction.*

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

---

### 6.3. Systematic Speculative Depth Sweep ($d_t \in \{14, 18, 22, 24, 28, 30\}$)
*Measures draft acceptance rate $\alpha(d_t)$, naive verification speedup, and measured GPU throughput.*

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

---

### 6.4. Publication-Scale 60-Prompt Multi-Domain Speculative Benchmark
*Evaluates across 4 distinct domains (Math, Code, Facts, Language; 15 prompts/domain) with rigorous 95% bootstrap confidence intervals.*

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

## 7. Recommended End-to-End Execution Pipeline

To maximize GPU time efficiency on Kaggle/Colab, execute in this order:

```bash
# Step 1: Pre-flight unit tests & smoke tests
python -m unittest discover tests

# Step 2: Cheap & fast headline experiments (2.1, 2.4, 2.3)
python experiments/run_tier1_interventions.py --model-type qwen --model-id Qwen/Qwen2.5-3B --dtype float16 --num-samples 30 --seeds 42,123,7 --sweep-depths 18,24,30 --skip-swap --skip-cosine --device cuda --output-dir results/canonical_sweeps/revision_table1
python experiments/run_multiseed_llama_sweep.py --model-id meta-llama/Llama-3.2-1B --depths 4,5,6,7,8,10,12,14 --seeds 42,123,7 --num-samples 30 --device cuda --output-dir results/canonical_sweeps/revision_llama_shallow
python experiments/run_tier1_interventions.py --model-type qwen --model-id Qwen/Qwen2.5-3B --dtype float16 --swap-depths 18 --num-samples 30 --skip-table1 --skip-swap --device cuda --output-dir results/canonical_sweeps/revision_cosine_attn

# Step 3: Moderate Causal Swap (2.2)
python experiments/run_robust_kv_swap_and_metrics.py --model-id Qwen/Qwen2.5-3B --depths 14,18,22,24,30 --num-samples 30 --device cuda --output-dir results/canonical_sweeps/revision_kv_swap_qwen
python experiments/run_robust_kv_swap_and_metrics.py --model-id meta-llama/Llama-3.2-1B --depths 6,8,10 --num-samples 30 --device cuda --output-dir results/canonical_sweeps/revision_kv_swap_llama

# Step 4: Quick 125M CUDA Graph Systems Runs (B-Series)
python experiments/run_graph_quality_per_bucket.py --device cuda
python experiments/run_graph_fill_cost.py --device cuda
python experiments/run_graph_sync_free_check.py --device cuda
python experiments/run_graph_main_throughput.py --batch-sizes 1,2,4,8,16,32,64 --repeats 5 --device cuda
python experiments/run_graph_batch_policies.py --batch-sizes 2,4,8,16,32,64 --device cuda

# Step 5: Autoregressive Generation & Extra Baselines (4.1, 4.2)
python experiments/run_autoregressive_generation.py --model-id Qwen/Qwen2.5-3B --depths 18,24,30 --num-samples 30 --max-new-tokens 256 --device cuda --output-dir results/canonical_sweeps/revision_autoregressive
python experiments/run_kv_baselines.py --model-id Qwen/Qwen2.5-3B --depths 18,24,30 --num-samples 30 --device cuda --output-dir results/canonical_sweeps/revision_kv_baselines

# Step 6: Multi-Depth In-Domain Adapter Training (10k steps) & Joint Condition (3.2, 3.1)
python experiments/train_exit_adapters.py --model-type qwen --model-id Qwen/Qwen2.5-3B --candidate-depths 18,22,24,30 --adapter-type mlp --rank 512 --steps 10000 --batch-size 8 --seq-len 128 --eval-every 500 --device cuda --output-dir checkpoints/multi_depth_r512
python experiments/run_joint_condition_sweep.py --model-id Qwen/Qwen2.5-3B --depths 18,22,24,30 --adapter-path checkpoints/multi_depth_r512/final_exit_adapters_qwen_mlp_r512.pt --num-samples 30 --device cuda --output-dir results/canonical_sweeps/revision_joint_condition

# Step 7: Old Scheduled Speculative Decoding Track
python experiments/benchmark_speculative_decoding.py --model-id Qwen/Qwen2.5-3B --adapter-path checkpoints/exit_adapters_qwen_mlp.pt --draft-depth 24 --gamma 2 --max-tokens 40 --dtype float16 --device cuda
python experiments/train_tuned_lens.py --model-id Qwen/Qwen2.5-3B --target-depths 14,18,22,24,28 --adapter-type lora --rank 64 --steps 400 --batch-size 2 --seq-len 128 --lr 1e-3 --distill-temp 2.0 --output-path checkpoints/exit_adapters_qwen_tuned_lens.pt --device cuda
python experiments/sweep_speculative_depths.py --model-id Qwen/Qwen2.5-3B --adapter-path checkpoints/exit_adapters_qwen_tuned_lens.pt --depths 14,18,22,24,28,30 --gamma 1 --num-prompts 12 --max-tokens 30 --dtype float16 --device cuda --output-dir results/canonical_sweeps/qwen2.5_3b
python experiments/benchmark_large_scale.py --model-id Qwen/Qwen2.5-3B --adapter-path checkpoints/exit_adapters_qwen_tuned_lens.pt --draft-depth 24 --gamma 1 --num-prompts-per-domain 15 --max-tokens 30 --device cuda
```

---

## 8. Archiving Results for Kaggle Download

At the conclusion of your Kaggle session, create zip archives to download all generated artifacts via the Kaggle Output pane:

```python
import shutil
shutil.make_archive("revision_results", "zip", "results")
shutil.make_archive("revision_checkpoints", "zip", "checkpoints")
print("[OK] Results and checkpoints zipped successfully!")
```
