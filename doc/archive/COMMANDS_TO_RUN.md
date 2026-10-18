# Master Execution Guide & Commands to Run

This document provides the exact commands to execute the next experimental phase (Action Plan for $>1.0\times$ Real Speedup) and tracks all previously completed empirical milestones.

---

## 🚀 Part 1: Action Plan — Unlocking Real Speedup ($>1.0\times$)

These newly implemented experiments directly target the **readout bottleneck** ($\mathcal{E}_{\text{readout}}$) and operational trade-offs to transition self-speculative decoding from sub-unity to practical super-unity wall-clock throughput.

### 1. Fast Tuned-Lens Calibration (< 3 Minutes on GPU)
* **Goal:** Exploits the **Key-stable / Value-evolving asymmetry**. Instead of a heavy full-manifold MLP, it trains an ultra-lightweight ($<0.3\text{M}$ parameter, rank-$64$) low-rank affine tuned lens directly on the intermediate readout direction:
  $$h_{\text{cal}} = h_{d_t} + \frac{\alpha}{r} W_{\text{up}} W_{\text{down}} \operatorname{LN}(h_{d_t})$$
* **Inference Overhead:** Only $\approx 0.5\text{ MFLOPs}$ per token ($< 0.02\%$ of the $2{,}000\text{ MFLOPs}$ saved by skipping 12 layers).

```bash
python experiments/train_tuned_lens.py \
    --model-id Qwen/Qwen2.5-3B \
    --target-depths 14,18,22,24,28 \
    --adapter-type lora \
    --rank 64 \
    --steps 400 \
    --output-path checkpoints/exit_adapters_qwen_tuned_lens.pt \
    --device cuda
```

---

### 2. Systematic Depth vs. Acceptance & Speedup Sweep
* **Goal:** Sweeps $d_t \in \{14, 18, 22, 24, 28, 30\}$ to systematically map:
  1. Empirical draft acceptance rate: $\alpha(d_t)$
  2. Naive verification speedup: $\mathcal{S}_{\text{naive}}(d_t) = \frac{1+\alpha}{c+1}$ (break-even requires $\alpha > c$)
  3. Partial-layer verification speedup: $\mathcal{S}_{\text{partial}}(d_t) = \frac{1+\alpha}{c + (1 - c)} = \mathbf{1 + \alpha}$
  4. Real-world measured GPU throughput ($\text{tokens/second}$)
* **Output:** Generates `results/canonical_sweeps/qwen2.5_3b/speculative_depth_sweep.csv` and publication plot `speculative_depth_tradeoff.pdf`.

```bash
python experiments/sweep_speculative_depths.py \
    --model-id Qwen/Qwen2.5-3B \
    --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \
    --depths 14,18,22,24,28,30 \
    --gamma 1 \
    --num-prompts 12 \
    --max-tokens 30 \
    --device cuda
```

*(To evaluate with the newly trained tuned-lens head instead, swap `--adapter-path` to `checkpoints/exit_adapters_qwen_tuned_lens.pt`).*

---

### 3. End-to-End Speculative Decoding with Tuned-Lens Readout
* **Goal:** Measures the multi-domain speedup across $N=60$ prompts when pairing Native Projection with the fast tuned-lens readout head:

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

---

## ✅ Part 2: Previously Completed & Validated Experiments

These experiments have already completed execution and their artifacts are verified in the repository:

### 1. Canonical Reproducibility Suite (WikiText-2)
* **Script:** `experiments/run_colab_reproducibility.py`
* **Artifacts:** Saved in [`results/canonical_wikitext2_eval/`](file:///e:/CUDA-Graph/kv_continuity/results/canonical_wikitext2_eval/)
  * `real_wikitext_disentanglement_sweep.csv` ($L=18, 24, 29, 32$)
  * `real_cossim_decay_L18.csv` (Key stability vs. Value decay)
  * `experiment_c_kv_swap_test.csv` (Mechanistic K/V cross-swap)
  * `real_adapter_capacity_L18.csv` ($r=64 \dots 2048$ held-out capacity ablation)

### 2. Continuous 11-Depth Sweep Curve ($L = 6 \dots 34$)
* **Script:** `experiments/isolate_kv_ablation.py`
* **Artifacts:** Saved in [`results/canonical_sweeps/qwen2.5_3b/`](file:///e:/CUDA-Graph/kv_continuity/results/canonical_sweeps/qwen2.5_3b/)
  * `kv_ablation_qwen_depth_sweep.csv` (11 depths, 66 data rows)
  * `kv_layerwise_cosine_drift.csv`
  * `paper_table_qwen_L6.md`

### 3. LLaMA-3.2 Cross-Architecture Verification ($L=8 \dots 14$)
* **Script:** `experiments/isolate_kv_ablation.py`
* **Artifacts:** Saved in [`results/canonical_sweeps/llama3_family/`](file:///e:/CUDA-Graph/kv_continuity/results/canonical_sweeps/llama3_family/)
  * `kv_ablation_llama_depth_sweep.csv`
  * `cossim_teacher_forced_llama.csv`
  * `paper_table_llama_L8.md`

### 4. Baseline Multi-Domain Speculative Benchmark ($N = 60, \gamma = 1$)
* **Script:** `experiments/benchmark_large_scale.py`
* **Artifact:** Saved in [`results/canonical_sweeps/qwen2.5_3b/benchmark_large_scale_results.json`](file:///e:/CUDA-Graph/kv_continuity/results/canonical_sweeps/qwen2.5_3b/benchmark_large_scale_results.json)
  * Mean Speedup: $0.748 \pm 0.021\times$
  * Acceptance Rate $\alpha$: $24.70\% \pm 3.35\%$
  * Lossless Equivalence: $96.67\%$ ($58/60$ exact trajectory matches)

---

## 💻 Part 3: Kaggle / Colab Notebook Helper Cells

### To Pull the Latest Scripts & Checkpoints on Kaggle / Colab:
```bash
!cd /kaggle/working/kv-cache-continuity && git pull
```

### To Zip and Download Results in Kaggle:
```python
import shutil
from IPython.display import FileLink

# Zip results directory
shutil.make_archive("/kaggle/working/latest_results", "zip", "/kaggle/working/kv-cache-continuity/results")
print("✓ Created archive")

# Display clickable link
FileLink("latest_results.zip")
```

### To Recompile Manuscript Figures Locally:
```bash
python manuscript/generate_figures.py
```
*(Outputs all 6 vector PDFs and 300 DPI PNGs to `manuscript/figures/` and `results/figures/`).*
