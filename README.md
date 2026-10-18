# Separating Attention Memory from Readout Error in Depth-Adaptive Transformers
### Preserving KV-Cache Continuity via Native State Projection

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

Official research code and canonical empirical results for the paper:  
**"Separating Attention Memory from Readout Error in Depth-Adaptive Transformers: Preserving KV-Cache Continuity via Native State Projection"**  
*Syed Muhammad Sarim Ahsan (Sir Syed University of Engineering and Technology)*  
Repository: [https://github.com/sarimahsan/kv-cache-continuity](https://github.com/sarimahsan/kv-cache-continuity)

---

## 1. Overview & Core Findings

Depth-adaptive inference (early exiting, layer skipping, and self-speculative decoding) reduces latency by terminating token computation before the final layer. Standard evaluations report only end-to-end task metrics, which conflate two distinct failure modes:
1. **Attention Memory Poisoning ($\mathcal{E}_{\text{attn}}$)**: Inaccurate Key-Value (KV) cache entries stored by shallow exits corrupt downstream attention in future tokens.
2. **Unembedding Manifold Mismatch ($\mathcal{E}_{\text{readout}}$)**: Intermediate hidden states reside off the terminal unembedding manifold, degrading immediate logit readout.

This repository implements a **teacher-forced causal disentanglement protocol** that isolates $\mathcal{E}_{\text{attn}}$ from $\mathcal{E}_{\text{readout}}$ on frozen base foundation models (`Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, `meta-llama/Llama-3.2-1B`) with zero base-model weight modification.

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
   - **Qwen2.5-3B**: Depth $L \ge 22$ ($61.1\%$ depth, $\bar{D}_{\mathrm{KL}} = 0.538\text{ nats}$).
   - **Llama-3.2-1B**: Depth $L \ge 8$ ($50.0\%$ depth, $\bar{D}_{\mathrm{KL}} = 0.590 \pm 0.015\text{ nats}$ across 3 seeds).
   - **Qwen2.5-7B**: Depth $L = 24$ ($85.7\%$ depth, $\bar{D}_{\mathrm{KL}} = 0.303\text{ nats}$).
   The mechanism scales stably to $T=4{,}096$ tokens context length without drift explosion.

2. **Readout Mismatch Dominates Aggregate Error**:
   $\mathcal{E}_{\text{readout}}$ accounts for over $90\%$ of total divergence at early layers. Intermediate representations cannot be reliably mapped to the vocabulary via linear heads alone.

3. **Value Cache Drift Substantially Exceeds Key Cache Drift**:
   Layer-wise Shapley causal swap ablations reveal that Value cache errors account for $60.3\%\text{--}82.1\%$ of attention corruption across architectures, confirming theoretical bounds on softmax attention stability.

4. **10,000-Step Adapter Ceiling on Open-Domain WikiText-2**:
   Bottleneck adapters ($r=512$, GELU, trained for 10,000 steps / 1.28M tokens) plateau between $41.5\%$ and $55.0\%$ top-1 agreement across depths $18\dots30$, illustrating the fundamental gap between canonical prompt memorization and genuine out-of-sample vocabulary prediction.

5. **Mixed-Exit Joint Execution & Rollouts**:
   In multi-prompt autoregressive generation ($N=64$ sequences), Native Projection retains $68.4\%$ top-1 agreement and $64.8\%$ exact match rate, compared to complete collapse under Cache Omission ($0\%$ exact match, perplexity $> 10^5$).

---

## 2. Repository Structure

```text
kv_cache_continuity/
├── checkpoints/              # Trained lightweight adapters & tuned lens heads
│   ├── exit_adapters_qwen_mlp.pt         # 3,000-step MLP bottleneck adapter (12.6 MB)
│   ├── exit_adapters_qwen_tuned_lens.pt  # Tuned-lens linear calibration head (5.3 MB)
│   ├── exit_adapters_qwen.pt             # Linear baseline adapter (25.2 MB)
│   └── multi_depth_r512/                 # 10,000-step multi-depth adapter suite
│       └── training_log_qwen_mlp_r512.csv
├── doc/                      # Detailed technical documentation and run guides
│   ├── KV_EXPERIMENTS_RUN_GUIDE.md       # Master experiment execution guide
│   ├── RESULTS_PLACEMENT_GUIDE.md        # File mapping and directory reference
│   ├── STATE.md                          # Empirical state & numerical audit tables
│   ├── motivation.md                     # Mathematical derivations and proofs
│   ├── evaluation_metrics.txt            # Formal metric definitions
│   ├── generate_guide_pdf.py             # Script to generate PDF execution guide
│   └── archive/                          # Historical reviewer rebuttals and scratch scripts
├── experiments/              # Executable Python experiment scripts
│   ├── isolate_kv_ablation.py            # Primary 3-condition causal disentanglement harness
│   ├── run_joint_condition_sweep.py      # Mixed-exit joint execution sweep
│   ├── run_autoregressive_generation.py  # 64-prompt autoregressive rollout with bootstrap CIs
│   ├── run_robust_kv_swap_and_metrics.py # Shapley causal KV swap ablations
│   ├── run_multiseed_llama_sweep.py      # Llama-3.2-1B multi-seed depth sweep
│   ├── run_large_model_sweep.py          # Qwen2.5-7B scaling experiment
│   ├── run_long_context_scaling.py       # Context scaling up to T=4096
│   ├── run_entropy_domain_comparison.py  # Domain transfer (WikiText-2, C4, Code)
│   ├── run_downstream_benchmarks.py      # Downstream reasoning benchmarks (GSM8K, ARC, MMLU)
│   ├── benchmark_large_scale.py          # 60-prompt multi-domain wall-clock benchmark
│   ├── train_exit_adapters.py            # Multi-depth bottleneck adapter training harness
│   └── verify_kv_continuity.py           # Self-contained verification script
├── results/                  # Canonical empirical data & figures
│   ├── canonical_sweeps/                 # Paper-ready results
│   │   ├── kv_paper/                     # Active publication sweeps (Table 1, swaps, CIs)
│   │   ├── autoregressive_generation/    # 64-prompt generation CSVs and summaries
│   │   ├── joint_condition/              # Mixed-exit and static joint condition sweeps
│   │   ├── qwen2.5_3b/                   # Comprehensive Qwen-3B sweeps
│   │   ├── llama3_family/                # Llama-3.2-1B multi-seed data
│   │   └── graph_quality/                # Speculative execution quality data
│   ├── figures/                          # Publication-grade vector plots (PDF + PNG)
│   └── archive/                          # Comprehensive lineage archive of past runs
├── src/                      # Core modules & model engines
│   ├── qwen_dynamic.py                   # Dynamic depth-truncated Qwen model engine
│   ├── kv_cache.py                       # Static & native-projected KV cache managers
│   ├── exit_adapter.py                   # SingleExitAdapter & ExitCalibrationModule
│   ├── speculative_engine.py             # Zero-recompute speculative decoding engine
│   ├── cuda_graph_engine.py              # Bucketed CUDA Graph inference runtime
│   ├── eval_dataset.py                   # WikiText-2, C4, and Code dataset split loaders
│   ├── dataset.py                        # Tokenizers and sequence batching utilities
│   ├── alignment.py                      # Centered cosine, Procrustes, and error metrics
│   └── configs.py                        # Model configurations & depth presets
├── tests/                    # Unit tests & verification
│   ├── test_audit_integrity.py           # Mathematical formula and data integrity audit
│   └── test_kv_continuity.py             # KV cache continuity tests
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
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
*All 11 audit and mathematical integrity tests should pass with `OK`.*

---

## 4. Reproducing Paper Results

All scripts support execution on a single consumer GPU ($\le 15\text{ GB}$ VRAM, `bfloat16`).

### A. Primary Causal Disentanglement (Table I)
To evaluate the 3-condition decoupled ablation on `Qwen/Qwen2.5-3B`:
```bash
python experiments/isolate_kv_ablation.py \
    --model-id Qwen/Qwen2.5-3B \
    --device cuda
```

### B. Mixed-Exit Joint Condition Sweep (Table XI)
```bash
python experiments/run_joint_condition_sweep.py \
    --model-id Qwen/Qwen2.5-3B \
    --exit-policy mixed_50 \
    --device cuda
```

### C. 64-Prompt Autoregressive Rollout with Bootstrap CIs (Table XII)
```bash
python experiments/run_autoregressive_generation.py \
    --model-id Qwen/Qwen2.5-3B \
    --num-samples 64 \
    --max-new-tokens 32 \
    --device cuda
```

### D. Layer-Wise Shapley Causal KV Swap (Table V & VI)
```bash
python experiments/run_robust_kv_swap_and_metrics.py \
    --model-id Qwen/Qwen2.5-3B \
    --depth 18 \
    --device cuda
```

### E. Multi-Seed Shallow Llama-3.2-1B Sweep (Table III)
```bash
python experiments/run_multiseed_llama_sweep.py \
    --model-id meta-llama/Llama-3.2-1B \
    --device cuda
```

### F. Context-Length & Domain Scaling (Table IV)
```bash
python experiments/run_entropy_domain_comparison.py \
    --model-id Qwen/Qwen2.5-3B \
    --device cuda
```

---

## 5. Canonical Empirical Results

Pre-computed canonical outputs are stored directly in `results/canonical_sweeps/`:
- `results/canonical_sweeps/kv_paper/revision_table1/standardized_table1_headline.csv`: Core Table I.
- `results/canonical_sweeps/kv_paper/revision_autoregressive/autoregressive_gen_Qwen_Qwen2.5-3B.csv`: 64-prompt generation rollout tokens.
- `results/canonical_sweeps/kv_paper/revision_autoregressive/autoregressive_summary_ci_Qwen_Qwen2.5-3B.csv`: Summary metrics with 95% bootstrap confidence intervals.
- `results/canonical_sweeps/kv_paper/revision_joint_condition/joint_decomposition_mixed_50_zeroshot_Qwen_Qwen2.5-3B.csv`: Table XI mixed-exit decomposition.
- `results/canonical_sweeps/kv_paper/revision_llama_shallow/llama_multiseed_sweep_meta-llama_Llama-3.2-1B.csv`: Table III shallow Llama sweep.
- `results/canonical_sweeps/kv_paper/revision_kv_swap_qwen/robust_swap_and_metrics_Qwen_Qwen2.5-3B.csv`: Table V Shapley causal swaps.

Publication figures can be viewed in `results/figures/` (available in both vector `.pdf` and high-resolution `.png` formats):
- `fig1_disentanglement_schematic`: Protocol schematic.
- `fig2_kl_vs_depth`: KL divergence decay across exit depths.
- `fig3_key_vs_value_drift`: Key vs. Value cosine similarity & Frobenius error.
- `fig4_adapter_saturation`: Evaluation perplexity across adapter training budgets.
- `fig5_cross_architecture`: Multi-seed cross-architecture comparison.
- `fig6_speculative_benchmark`: Empirical wall-clock speedup vs. theoretical ceiling.

---

## 6. Citation

If you use this protocol, code, or results in your research, please cite:

```bibtex
@inproceedings{ahsan2025kvcontinuity,
  title={Separating Attention Memory from Readout Error in Depth-Adaptive Transformers: Preserving {KV}-Cache Continuity via Native State Projection},
  author={Ahsan, Syed Muhammad Sarim},
  booktitle={IEEE Conference Proceedings},
  year={2025},
  url={https://github.com/sarimahsan/kv-cache-continuity}
}
```

---

## 7. License

This repository is distributed under the [MIT License](LICENSE). Base model weights belong to their respective authors (`Qwen` by Alibaba Cloud, `Llama-3` by Meta AI).
