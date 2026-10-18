# Google Colab Execution Guide for Paper Reproducibility

This guide provides the exact commands and code cells to run the full revised benchmark suite on **Google Colab** (using a T4, A100, or L4 GPU).

---

## 1. Setup & Environment (Colab Cell 1)

```bash
# Check GPU and bfloat16 support
!nvidia-smi

# Clone or sync your repository (if not already mounted via Google Drive)
# !git clone https://github.com/your-username/kv-cache-continuity.git
# %cd kv-cache-continuity

# Install required dependencies
!pip install -q torch transformers datasets accelerate tiktoken scipy
```

---

## 2. Quick Smoke Test (Colab Cell 2)

Before running the full 3-hour evaluation sweep, run the fast smoke test to verify dataset downloading, tokenization, model loading, and forward passes:

```bash
!python experiments/run_colab_reproducibility.py \
    --model-id Qwen/Qwen2.5-3B \
    --quick \
    --output-dir results/quick_smoke_test
```

This verifies:
1. **WikiText-2 validation and train split streaming** without errors.
2. **Oracle PPL & Dtype Diagnostic**: verifies whether `bfloat16` works and inspects $L=18$ logit dynamic ranges.
3. **Decoupled Disentanglement**: Runs Readout-Only, Attn-Only (Direct Copy & Native Proj), and Combined on 15 validation sequences.
4. **Experiment B**: Cosine similarity decay curves for Keys and Values.
5. **Experiment C**: K/V swap tests ($\text{Oracle } K + \hat{V}$ vs. $\hat{K} + \text{Oracle } V$).
6. **Authentic Adapter Distillation**: Trains 3 adapter ranks ($r \in \{64, 256, 1024\}$) on the training split and evaluates on held-out validation sequences.

---

## 3. Full Benchmark Run (Colab Cell 3)

For the official numbers to populate the paper's tables:

```bash
!python experiments/run_colab_reproducibility.py \
    --model-id Qwen/Qwen2.5-3B \
    --seq-len 64 \
    --num-samples 100 \
    --exit-depths 18,24,29,32 \
    --adapter-ranks 64,128,256,512,1024,2048 \
    --adapter-steps 1500 \
    --output-dir results/canonical_wikitext2_eval
```

To run with longer context ($T=512$):

```bash
!python experiments/run_colab_reproducibility.py \
    --model-id Qwen/Qwen2.5-3B \
    --seq-len 512 \
    --num-samples 50 \
    --exit-depths 18,24,29,32 \
    --skip-adapters \
    --output-dir results/canonical_wikitext2_T512
```

---

## 4. Generated Artifacts

All outputs will be saved in `results/canonical_wikitext2_eval/`:
- `real_wikitext_disentanglement_sweep.csv` (Updates Tables 1, 2, 5)
- `real_cossim_decay_L18.csv` (Updates Table 6 and Figure 3)
- `experiment_c_kv_swap_test.csv` (Answers Experiment C mechanistic hypothesis)
- `real_adapter_capacity_L18.csv` (Updates Table 3 with authentic generalization metrics)

---

## 5. Tier 1 High-Impact Interventions (Colab Cell 4)

Run the dedicated script that executes **both** Tier 1 experiments in one clean command:
1. **Causal Key/Value Swap Intervention** (4 conditions at $L=18$)
2. **Multi-Seed Statistical Rigor** (3 seeds: `42, 123, 7`)

```bash
# Run on Qwen2.5-3B (Takes ~10-15 minutes on a T4/A100)
!python experiments/run_tier1_interventions.py \
    --model-type qwen \
    --model-id Qwen/Qwen2.5-3B \
    --device cuda \
    --num-samples 30 \
    --seeds 42,123,7 \
    --output-dir results/tier1_revision

# Run on Google Gemma-2-2B (Takes ~10 minutes, uses only ~5.5 GB VRAM):
!python experiments/run_tier1_interventions.py \
    --model-type gemma \
    --model-id google/gemma-2-2b \
    --device cuda \
    --num-samples 30 \
    --seeds 42,123,7 \
    --output-dir results/tier1_revision_gemma

# Or quick smoke-test (1-2 minutes):
!python experiments/run_tier1_interventions.py --quick --device cuda
```

### Artifacts Produced:
- `results/tier1_revision/causal_kv_swap_L*.csv` & `.json` (Prints Table IV for Section VII-C)
- `results/tier1_revision/multiseed_depth_sweep.csv` & `.json` (Prints mean ± std for Tables I & II)

