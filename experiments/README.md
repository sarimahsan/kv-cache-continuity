# Experiments Directory Index (`experiments/`)

This directory contains all experimental engines, benchmark harnesses, ablation studies, and plotting scripts.

---

## 1. Core Experimental Engines

| Script | Purpose & Description |
| :--- | :--- |
| **[`paper.py`](paper.py)** | **Unified Paper Verification Engine:** One-script pipeline reproducing Table 1, fill controls, free-running rollouts, and measured latency across depths. |
| **[`cheap_pass.py`](cheap_pass.py)** | **Cheap-Pass KV Adapters:** Trains lightweight parallel bottleneck adapters ($r=128$, zero-initialized $W_{\text{up}}$) to predict skipped-layer inputs $\hat{h}_l \approx h_l$ from $h_E$. Enforces pre-registered $\text{KL} < \ln 2$ gate. |
| **[`plot_cheap_pass.py`](plot_cheap_pass.py)** | Generates publication figures (`kl_vs_depth.png`, `fidelity_curves.png`, `capacity_sweep.png`) and markdown reports from `cheap_pass.py` outputs. |
| **[`plot_rank_sweep.py`](plot_rank_sweep.py)** | Generates publication-ready figures for the capacity-vs-information asymptotic floor sweep. |

---

## 2. VKR Engine Evolution (Phases 1, 2, 3)

The key-value repair pipeline progressed through three distinct versions, each with dedicated canonical scripts and backward-compatible wrappers:

| Canonical Script | Legacy Alias | Key Innovation & Methodology |
| :--- | :---: | :--- |
| **[`vkr1_value_repair.py`](vkr1_value_repair.py)** | `vkr.py` | **Phase 1 (Value-Targeted Repair):** Proved native Key projection + learned low-rank Value repair outperforms unprojected baselines. |
| **[`vkr2_unified_repair.py`](vkr2_unified_repair.py)** | `vkr2.py` | **Phase 2 (Unified Engine):** Established the modular `Ctx` evaluation harness, streaming C4/FineWeb-Edu, KVC (both Keys and Values), and paired 95% bootstrap CIs. |
| **[`vkr3_multi_angle_repair.py`](vkr3_multi_angle_repair.py)** | `vkr3.py` | **Phase 3 (Multi-Angle Diagnostics):** Investigated 7 distinct angles: asymptotic rank sweeps, shadow trunks, temporal convolutions, LoRA co-adaptation, and lazy backfill refresh curves. |

---

## 3. Specialized Diagnostic & Solution Scripts

* **`sol1_refinement_attention.py`**: Solution 1 cross-attention refinement bridge.
* **`sol2_residual_unembed.py`**: Solution 2 residual unembedding adapter.
* **`sol3_value_only_repair.py`**: Solution 3 Value-only KVC repair (demonstrates Values account for ~78% of recoverable attention memory).
* **`run_partial_verification_speculative.py`**: Evaluates speculative decoding with state resumption from $h_E$, achieving 100% mathematical output fidelity.
* **`run_entropy_gated_exit.py`**: Evaluates dynamic entropy-based early exit policies.
* **`harness_diag.py`**: Isolates self-exact vs self-approximate token protocol factors.

