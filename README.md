# Separating Attention Memory from Readout Error in Depth-Adaptive Transformers
### Preserving KV-Cache Continuity via Native State Projection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Official research codebase and canonical empirical results for:  
**"Separating Attention Memory from Readout Error in Depth-Adaptive Transformers: Preserving KV-Cache Continuity via Native State Projection"**  
*Syed Muhammad Sarim Ahsan (Sir Syed University of Engineering and Technology)*  
Repository: [https://github.com/sarimahsan/kv-cache-continuity](https://github.com/sarimahsan/kv-cache-continuity)

---

## 1. Overview & Core Findings

Depth-adaptive inference (early exiting, layer skipping, and self-speculative decoding) reduces latency by terminating token computation before the final layer. Standard evaluations report only end-to-end task metrics, which conflate two distinct failure modes:
1. **Attention Memory Poisoning ($\mathcal{E}_{\text{attn}}$)**: Inaccurate Key-Value (KV) cache entries stored by shallow exits corrupt downstream attention in future tokens.
2. **Unembedding Manifold Mismatch ($\mathcal{E}_{\text{readout}}$)**: Intermediate hidden states reside off the terminal unembedding manifold, degrading immediate logit readout.

This repository implements a **teacher-forced causal disentanglement protocol** that isolates $\mathcal{E}_{\text{attn}}$ from $\mathcal{E}_{\text{readout}}$ on frozen base foundation models (`Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`, `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.1-8B`) with zero base-model weight modification.

```
                          Teacher-Forced Disentanglement Protocol
  ┌───────────────────────┬──────────────────────┬───────────────────────┐
  │      Condition        │   KV Cache Fed To    │ Readout Logits From   │
  ├───────────────────────┼──────────────────────┼───────────────────────┤
  │ 1. Oracle (Full)      │ True Full KV         │ Full Final Layer L    │
  │ 2. Isolated Readout   │ True Full KV         │ Exit Depth dt (W_U)   │
  │ 3. Isolated Attn Mem  │ Projected KV at dt   │ Full Final Layer L    │
  │ 4. Joint (Mixed-Exit) │ Mixed Exit/Full KV   │ Respective Exit/Full  │
  └───────────────────────┴──────────────────────┴───────────────────────┘
```

### Key Empirical Findings

1. **Native KV State Projection Preserves Attention Continuity**:
   By projecting intermediate representations through native, pre-trained Key/Value projection matrices ($W_K, W_V + \mathrm{LN}_1$) of skipped layers, attention cache error falls below the 1-bit mutual information threshold ($\bar{D}_{\mathrm{KL}} \le \ln 2 \approx 0.6931\text{ nats}$) at:
   - **Qwen2.5-3B**: Depth $L \ge 24$ ($66.7\%$ depth, $\bar{D}_{\mathrm{KL}} = 0.571 \pm 0.025\text{ nats}$ across 3 seeds).
   - **Llama-3.2-1B**: Depth $L \ge 8$ ($50.0\%$ depth, $\bar{D}_{\mathrm{KL}} = 0.590 \pm 0.015\text{ nats}$ across 3 seeds).
   - **Qwen2.5-7B**: Depth $L = 24$ ($85.7\%$ depth, $\bar{D}_{\mathrm{KL}} = 0.303\text{ nats}$).
   The mechanism scales stably to $T=4{,}096$ tokens context length without drift explosion.

2. **Readout Mismatch Dominates Aggregate Error**:
   $\mathcal{E}_{\text{readout}}$ accounts for over $90\%$ of total divergence at early layers. Intermediate representations cannot be reliably mapped to the vocabulary via linear heads alone.

3. **Value Cache Drift Substantially Exceeds Key Cache Drift**:
   Layer-wise Shapley causal swap ablations reveal that Value cache errors account for $64.5\%\text{--}82.1\%$ of attention corruption across architectures under all-position evaluation, confirming theoretical bounds on softmax attention stability. Under historical-only substitution (`self_exact = True`), Value share approaches parity at deeper exits while off-diagonal Key routing error dominates on Llama-3.2-1B.

4. **Linear & Affine Residual Structure**:
   Cross-layer KV transport is predominantly linear. Closed-form per-channel affine calibration ($S3$/$S4$, 18.4K–36.9K parameters, no gradient descent) recovers $62\%\text{--}71\%$ of a rank-128 adapter's gain, and closed-form ridge regression achieves $99.9\%\text{--}102.1\%$ gap closure in seconds.

5. **Multi-Task Controlled Evaluation vs. River-LLM**:
   On Llama-3.2-1B and Llama-3.1-8B across 8 standard benchmarks (MMLU, ARC-c, ARC-e, HellaSwag, BoolQ, GSM8K, MATH, HumanEval), Native Projection and CheapPass prevent catastrophic attention collapse, recovering $48.7\%\text{--}50.7\%$ on 8B MMLU at 50% depth where Direct Copy (River-LLM's KV-Share) falls to chance ($22.7\%$).

---

## 2. Repository Structure

```text
kv_cache_continuity/
├── doc/                             # Technical documentation & formal specifications
│   ├── README.md                    # Documentation index
│   ├── protocol.md                  # Standardized experimental protocol definition
│   └── SELF_SPECULATIVE_DECODING_MATHEMATICAL_SPEC.md # Cost model & breakeven laws
├── experiments/                     # Executable experimental engines & evaluation suites
│   ├── isolate_kv_ablation.py       # Primary 3-condition causal disentanglement runner
│   ├── run_cheappass_river_conditions.py # Controlled 8-task River-LLM condition benchmark
│   ├── cheap_pass.py                # Learned CheapPass KV adapter engine
│   ├── vkr2_unified_repair.py       # Unified KVC vs. HSM repair with bootstrap CIs
│   ├── mixed_depth_eval.py          # Confidence-gated mixed-depth decoding profiler
│   ├── run_autoregressive_generation.py # Multi-prompt free-running generation profiler
│   └── legacy_sweeps/               # Standalone diagnostic & self-speculative benchmarks
├── results/                         # Canonical empirical benchmarks, logs & figures
│   ├── canonical_sweeps/            # Verified depth, length (T=4096), and domain sweeps
│   │   ├── kv_paper/                # Core revision sweeps & Shapley causal swap logs
│   │   ├── llama3_family/           # Llama-3.2-1B multi-seed runs
│   │   └── qwen2.5_3b/              # Qwen2.5-3B diagnostic sweeps & speculative logs
│   ├── figures/                     # Publication figures (PDF & high-res PNG, Figs. 1–7)
│   ├── kv-cache-continuity-results/ # Track 1 closed-form affine & exit-head baselines
│   ├── mixed_depth/                 # Track 2A confidence-gated mixed-depth Pareto results
│   └── river_conditions/            # Track 2B 8-task River-LLM comparative benchmark
├── src/                             # Core inference models, KV managers & projection layers
│   ├── configs.py                   # Model configurations & depth presets
│   ├── eval_dataset.py              # FineWeb-Edu, WikiText-2, C4, GSM8K dataset loaders
│   ├── exit_adapter.py              # Exit calibration heads & tuned lens modules
│   └── kv_cache.py                  # Static & native-projected KV cache managers
├── tests/                           # Unit tests and audit integrity suites
│   ├── test_audit_integrity.py      # Mathematical formula & data integrity audit
│   └── test_kv_continuity.py        # KV cache continuity tests
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md
```

---

## 3. Quickstart & Installation

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/sarimahsan/kv-cache-continuity.git
cd kv-cache-continuity

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### Run Integrity Unit Tests
```bash
python -m unittest discover tests
```
*All audit and mathematical integrity tests pass with `OK`.*

---

## 4. Reproducing Results

All 1B and 3B benchmarks execute on a single 15 GiB consumer GPU (`fp16`/`bf16`, peak VRAM $\le 5.86\text{ GiB}$). 7B, 8B, and 14B evaluations run on 24–40 GiB GPUs.

### A. Primary Causal Disentanglement
```bash
python experiments/isolate_kv_ablation.py \
    --model-id Qwen/Qwen2.5-3B \
    --device cuda
```

### B. Controlled River-LLM 8-Task Benchmark
```bash
python experiments/run_cheappass_river_conditions.py \
    --model-name meta-llama/Llama-3.1-8B \
    --tasks all \
    --device cuda
```

### C. Confidence-Gated Mixed-Depth Pareto Sweep
```bash
python experiments/mixed_depth_eval.py \
    --model-id Qwen/Qwen2.5-3B \
    --device cuda
```

---

## 5. Canonical Empirical Results

Canonical pre-computed results are tracked in [`results/`](results/):

- **Core Disentanglement**: [`results/canonical_sweeps/kv_paper/revision_table1/standardized_table1_headline.csv`](results/canonical_sweeps/kv_paper/revision_table1/standardized_table1_headline.csv)
- **Multi-Seed Qwen2.5-3B Sweep**: [`results/canonical_sweeps/qwen2.5_3b/multiseed_depth_sweep.csv`](results/canonical_sweeps/qwen2.5_3b/multiseed_depth_sweep.csv)
- **Shallow Llama-3.2-1B Sweep**: [`results/canonical_sweeps/kv_paper/revision_llama_shallow/llama_multiseed_sweep_meta-llama_Llama-3.2-1B.csv`](results/canonical_sweeps/kv_paper/revision_llama_shallow/llama_multiseed_sweep_meta-llama_Llama-3.2-1B.csv)
- **Qwen2.5-7B Depth Sweep**: [`results/canonical_sweeps/kv_paper/revision_7b/large_model_sweep_Qwen_Qwen2.5-7B.csv`](results/canonical_sweeps/kv_paper/revision_7b/large_model_sweep_Qwen_Qwen2.5-7B.csv)
- **Shapley Causal Swap Attribution**: [`results/canonical_sweeps/kv_paper/revision_kv_swap_qwen/robust_swap_and_metrics_Qwen_Qwen2.5-3B.csv`](results/canonical_sweeps/kv_paper/revision_kv_swap_qwen/robust_swap_and_metrics_Qwen_Qwen2.5-3B.csv)
- **Closed-Form Affine & Ridge Decomposition**: [`results/kv-cache-continuity-results/baselines_linear/track1_results.json`](results/kv-cache-continuity-results/baselines_linear/track1_results.json)
- **Mixed-Depth Pareto Benchmark**: [`results/mixed_depth/mixed_depth_results.json`](results/mixed_depth/mixed_depth_results.json)
- **River-LLM Comparative Benchmark**: [`results/river_conditions/results.json`](results/river_conditions/results.json) & [`results/river_conditions/RIVER_CONDITIONS.md`](results/river_conditions/RIVER_CONDITIONS.md)

Publication figures are located in [`results/figures/`](results/figures/) (in both vector `.pdf` and high-resolution `.png` formats, Figures 1–7).

---

## 6. Citation

If you use this protocol, codebase, or canonical results in your research, please cite:

```bibtex
@inproceedings{ahsan2026kvcontinuity,
  title={Separating Attention Memory from Readout Error in Depth-Adaptive Transformers: {KV}-Cache Continuity via Native Projection and Affine Repair},
  author={Ahsan, Syed Muhammad Sarim},
  booktitle={IEEE Conference Proceedings},
  year={2026},
  url={https://github.com/sarimahsan/kv-cache-continuity}
}
```

---

## 7. License

This repository is distributed under the [MIT License](LICENSE). Base model weights belong to their respective authors (`Qwen` by Alibaba Cloud, `Llama-3` by Meta AI).
