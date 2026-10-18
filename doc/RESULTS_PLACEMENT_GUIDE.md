# Directory Structure & Results Placement Guide

**Repository:** `kv_continuity` (KV-Cache Continuity Paper)  
**Purpose:** Single-source directory layout reference. When you finish running an experiment on Kaggle, Colab, or locally, use this document to know **exactly where each result file and checkpoint belongs**.

---

## 1. Cleaned Repository Root Layout

```text
kv_continuity/
├── checkpoints/              # Trained exit adapters & tuned-lens heads (.pt files)
├── doc/                      # Documentation & experiment execution guides
│   ├── KV_EXPERIMENTS_RUN_GUIDE.md  # MASTER execution guide (All KV experiments)
│   ├── RESULTS_PLACEMENT_GUIDE.md   # This file (Where to put what)
│   ├── STATE.md                     # Empirical state and manuscript baseline tables
│   ├── motivation.md                # Scientific core & information-theoretic bounds
│   └── archive/                     # Obsolete proposals & superseded runbooks
├── experiments/              # Executable experiment scripts (Python)
├── manuscript/               # LaTeX paper files (main.tex, sections, references.bib)
├── results/
│   ├── canonical_sweeps/
│   │   └── kv_paper/         # CLEAN, ACTIVE DESTINATIONS FOR ALL KV PAPER EXPERIMENTS
│   ├── figures/              # Generated plots and PDF figures for the paper
│   └── archive/              # Archived previous runs, loose CSVs, and legacy dumps
├── src/                      # Core engine, KV caching, evaluation & adapters
└── tests/                    # Unit tests (run: python -m unittest discover tests)
```

---

## 2. Where to Put What: Results Placement Map

When you download your `zip` archive from Kaggle or Colab, extract the files directly into the corresponding directories below:

### Phase 1: Core Headline Revisions
| Experiment | Output Filename | Destination Directory | Status |
| :--- | :--- | :--- | :---: |
| **2.1 Standardized Table 1 (A1)** | `standardized_table1_headline.csv` | `results/canonical_sweeps/kv_paper/revision_table1/` | [x] Complete |
| **2.4 Shallow Llama Sweep (A5)** | `llama_multiseed_sweep_meta-llama_Llama-3.2-1B.csv` | `results/canonical_sweeps/kv_paper/revision_llama_shallow/` | [x] Complete |
| **2.3 Centered Cosine & Attn (A10)** | `centered_cosine_and_attn_err_L18.csv` | `results/canonical_sweeps/kv_paper/revision_cosine_attn/` | [x] Complete |
| **2.2 Causal Swap (Qwen) (A9)** | `robust_swap_and_metrics_Qwen_Qwen2.5-3B.csv` | `results/canonical_sweeps/kv_paper/revision_kv_swap_qwen/` | [x] Complete |
| **2.2 Causal Swap (Llama) (A9)** | `robust_swap_and_metrics_meta-llama_Llama-3.2-1B.csv` | `results/canonical_sweeps/kv_paper/revision_kv_swap_llama/` | [x] Complete |

---

### Phase 2: Autoregressive Generation & Extra Baselines
| Experiment | Output Filenames | Destination Directory | Status |
| :--- | :--- | :--- | :---: |
| **4.1 Autoregressive Gen (A3)** | `autoregressive_gen_Qwen_Qwen2.5-3B.csv`<br>`qualitative_samples_Qwen_Qwen2.5-3B.txt` | `results/canonical_sweeps/kv_paper/revision_autoregressive/` | [x] Complete |
| **4.2 Extra Fill Baselines (A2)** | `kv_baselines_Qwen_Qwen2.5-3B.csv` | `results/canonical_sweeps/kv_paper/revision_kv_baselines/` | [x] Complete |

---

### Phase 3: Multi-Depth Checkpoints & Joint Condition
| Artifact | Output Filename | Destination Directory | Status |
| :--- | :--- | :--- | :---: |
| **Multi-Depth Checkpoint (A12)** | `final_exit_adapters_qwen_mlp_r512.pt` | `checkpoints/multi_depth_r512/` | [x] Complete |
| **3.1 Joint Condition (A4)** | `joint_condition_decomposition_Qwen_Qwen2.5-3B.csv` | `results/canonical_sweeps/kv_paper/revision_joint_condition/` | [x] Complete |

---

### Phase 4: Extended Generalization, Robustness & Downstream
| Experiment | Output Filename | Destination Directory | Status |
| :--- | :--- | :--- | :---: |
| **5.1 Long-Context Scaling (A8)** | `long_context_scaling_Qwen_Qwen2.5-3B.csv` | `results/canonical_sweeps/kv_paper/revision_long_context/` | [x] Complete |
| **5.2 Downstream Benchmarks (A7)**| `downstream_benchmarks_Qwen_Qwen2.5-3B.csv` | `results/canonical_sweeps/kv_paper/revision_downstream/` | [x] Complete |
| **5.3 Entropy & Domains (A11)** | `entropy_domain_comparison_Qwen_Qwen2.5-3B.csv` | `results/canonical_sweeps/kv_paper/revision_entropy/` | [x] Complete |
| **5.4 7B Model Scaling (A13)** | `large_model_sweep_Qwen_Qwen2.5-7B.csv` | `results/canonical_sweeps/kv_paper/revision_7b/` | [x] Complete |
| **5.5 Active Speculative (A14)** | `active_speculative_decoding_report.json` | `results/canonical_sweeps/kv_paper/revision_active_spec/` | [x] Complete |

---

### Phase 5: Scheduled Speculative Decoding Track
| Experiment | Output Filename | Destination Directory | Status |
| :--- | :--- | :--- | :---: |
| **Tuned-Lens Readout Checkpoint** | `exit_adapters_qwen_tuned_lens.pt` | `checkpoints/` | [x] Complete |
| **Speculative Depth Sweep** | `speculative_depth_sweep.csv`<br>`speculative_depth_tradeoff.pdf` | `results/canonical_sweeps/kv_paper/speculative_track/` | [x] Complete |
| **60-Prompt Publication Benchmark** | `benchmark_tuned_lens_results.json` | `results/canonical_sweeps/kv_paper/speculative_track/` | [x] Complete |

---

## 3. Quick Archive Command on Kaggle

When downloading results from Kaggle or Colab, run this snippet to create two clean zip packages:

```python
import shutil

# 1. Package all experimental CSVs and JSONs
shutil.make_archive("kv_paper_results", "zip", "results/canonical_sweeps/kv_paper")

# 2. Package checkpoints
shutil.make_archive("kv_paper_checkpoints", "zip", "checkpoints")

print("[OK] Download kv_paper_results.zip and kv_paper_checkpoints.zip from the Output pane.")
```

When you download `kv_paper_results.zip` to your local machine, extract its contents directly into:  
`d:\CUDA-Graph\kv_continuity\results\canonical_sweeps\kv_paper\`
