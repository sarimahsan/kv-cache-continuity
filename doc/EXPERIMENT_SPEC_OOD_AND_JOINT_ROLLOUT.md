# Experiment Specification: Out-of-Distribution Transfer & End-to-End Autoregressive Rollouts

**Project:** `kv-cache-continuity`  
**Date:** November 3, 2026  
**Document Purpose:** Formal Experimental Protocols for Reviewer Actions (1) Out-of-Distribution Generalization and (2) End-to-End Autoregressive Decoding  
**Status:** Pre-Registration Spec (Before Execution)  
**Target Architecture:** `Qwen/Qwen2.5-3B` (36 Layers, Frozen fp16 Backbone)  

---

## PART 1: Experiment 1 — Out-of-Distribution (OOD) Manifold Transfer

### 1.1 Objective & Scientific Hypothesis
* **Research Question:** Does the residual manifold trajectory learned by CheapPass ($\hat{h}_l = h_E + \Delta(h_E)$) on English web text (`FineWeb-Edu`) capture universal Transformer feature geometry, or does it overfit to natural language syntax?
* **Hypothesis:** Inter-layer residual shifts in pre-trained LLMs are largely domain-invariant structural transformations (e.g. attention mixing, token disambiguation). A CheapPass adapter trained strictly on natural language will transfer zero-shot to **code** and **mathematical reasoning**, maintaining $\text{KL} < \ln(2)$ without retraining.

### 1.2 Evaluation Datasets & Token Budgets
Zero training is performed; existing checkpoints (`adapter_E18_r128_primary.pt`, `adapter_E24_...`, `adapter_E29_...`) are evaluated strictly in inference mode.

1. **Domain A (Source / In-Domain Baseline):** `FineWeb-Edu` (100 sequences $\times$ 512 tokens).
2. **Domain B (Structured Code):** `bigcode/the-stack-smol` (Python subset) or `codeparrot/github-code-clean` (100 sequences $\times$ 512 tokens).
   * *Characteristic:* High punctuation density, indentation tokens, strict algorithmic dependencies, long-range identifier binding.
3. **Domain C (Math & Reasoning):** `gsm8k` (concatenated chain-of-thought sequences) or `Open-Orca/FLAN` reasoning subset (100 sequences $\times$ 512 tokens).
   * *Characteristic:* Numeric tokens, multi-step deductive chains, low lexical redundancy.

### 1.3 Pre-Registered Pass Gates
For each exit depth $E \in \{18, 24, 29\}$:
* **Gate 1 (OOD Transfer Bound):** Zero-shot test $\text{KL}_{\text{Adapter}} < \ln(2) \approx 0.6931\text{ nats}$ on both Code and Math domains at $E=18$.
* **Gate 2 (Superiority over Native):** $\text{KL}_{\text{Adapter}} < 0.5 \times \text{KL}_{\text{NativeProj}}$ across all domains and depths.

### 1.4 Test Matrix & Reporting Format
For each domain $\times$ depth combination:
$$\begin{array}{lcccc}
\hline
\textbf{Domain} & \textbf{Depth } E & \textbf{Method} & \bar{D}_{\text{KL}}\text{ (nats) [95\% CI]} & \textbf{Top-1 Match (\%) [95\% CI]} \\
\hline
\text{FineWeb-Edu (ID)} & E=18 & \text{NativeProj} & 0.7680\ [0.7516, 0.7838] & 59.65\%\ [59.16, 60.09] \\
\text{FineWeb-Edu (ID)} & E=18 & \textbf{CheapPass} & \mathbf{0.2392\ [0.2306, 0.2485]} & \mathbf{76.70\%\ [76.39, 76.99]} \\
\hline
\text{The Stack (Code)} & E=18 & \text{NativeProj} & \text{[To measure]} & \text{[To measure]} \\
\text{The Stack (Code)} & E=18 & \textbf{CheapPass} & \text{[To measure]} & \text{[To measure]} \\
\hline
\text{GSM8K (Math)} & E=18 & \text{NativeProj} & \text{[To measure]} & \text{[To measure]} \\
\text{GSM8K (Math)} & E=18 & \textbf{CheapPass} & \text{[To measure]} & \text{[To measure]} \\
\hline
\end{array}$$

---

## PART 2: Experiment 2 — End-to-End Joint Decoding & Speculative Rollout

### 2.1 The Two Disentangled Error Regimes
Standard early exit incurs two independent error components:
1. $\mathcal{E}_{\text{readout}}$: The error of predicting token identity $y_\tau$ from shallow state $h_E$ rather than terminal state $h_L$.
2. $\mathcal{E}_{\text{attn}}$: The error induced in later tokens $t > \tau$ when attending to the approximate KV entries of token $\tau$.

Part 2 evaluates two practical deployment configurations:

---

### 2.2 Regime A: Multi-Step Autoregressive Generation (Compounding Rollout)
* **Goal:** Stress-test whether errors compound catastrophically when a model generates long continuations entirely with synthetic KV states.
* **Protocol:**
  * Fixed prompt of length $P = 32$ tokens processed with full-depth exact KV cache.
  * Autoregressive generation of $N_{\text{new}} = 128$ tokens ($64$ prompts from held-out test set).
  * For all generated tokens $\tau \ge P$:
    * Generation uses greedy decoding from the designated readout.
    * Skipped layers $l \in [E, L-1]$ have their KV cache populated via:
      * Condition 1: **Oracle** (Full 36-layer computation; reference).
      * Condition 2: **DirectCopy** (Floor baseline).
      * Condition 3: **NativeProj** (Baseline).
      * Condition 4: **CheapPass Adapter** (Ours).
* **Metrics:**
  1. **Match@$K$ ($K \in \{16, 64, 128\}$):** Percentage of generated tokens identical to the Oracle greedy sequence.
  2. **First Divergence Position ($\bar{\tau}_{\text{div}}$):** The average token index at which the generated sequence first differs from the Oracle continuation (higher is better).
  3. **Generation Perplexity / NLL ($\Delta\text{NLL}$):** Mean negative log-likelihood of the generated continuations evaluated under the untouched full 36-layer model.

---

### 2.3 Regime B: Self-Speculative Decoding (Draft at $E$, Verify at $L$)
* **Goal:** Measure real-world wall-clock speedup and token acceptance rate in a deployment where early exit is used for speculative drafting.
* **Mechanism:**
  * Draft model: Truncated backbone up to layer $E$ with an auxiliary calibrated exit head.
  * Drafter generates $\gamma \in \{2, 3, 4, 5\}$ speculative candidate tokens.
  * As candidates are generated, CheapPass populates their missing KV states for layers $E \dots L-1$.
  * Target model (verifier): Full model executes a single parallel verification forward pass across the $\gamma$ candidates using the synthesized KV cache.
  * Standard rejection sampling or greedy matching is applied.
* **Pre-Registered Metrics:**
  * **Mean Acceptance Rate ($\bar{\alpha}$):** $\frac{\text{Accepted Tokens}}{\text{Drafted Tokens}}$.
  * **Effective Compression Ratio ($\tau$):** Average tokens generated per full verification forward pass.
  * **Speedup Factor ($S$):** Measured end-to-end latency relative to standard autoregressive decoding.

---

## PART 3: Implementation Architecture & Execution Plan

### 3.1 Script Organization
* `experiments/eval_ood_cheap_pass.py`: Dedicated script for Part 1 (OOD evaluation). Reuses `cheap_pass.py` Ctx and adapter loading; streams 100 sequences from The Stack and GSM8K.
* `experiments/rollout_cheap_pass.py`: Dedicated script for Part 2 (multi-step autoregressive rollout and speculative drafting). Interfaces directly with existing `rollout()` harness.

### 3.2 Computational Budget
* **OOD Evaluation (Part 1):** $\sim 100\text{ sequences} \times 2\text{ domains} \times 3\text{ depths} \approx 600\text{ forward passes}$. Wall-clock time on T4: **$< 4\text{ minutes}$**.
* **Autoregressive Rollout (Part 2):** $64\text{ prompts} \times 128\text{ tokens} \times 4\text{ methods} \approx 32{,}768\text{ token steps}$. Wall-clock time on T4: **$\sim 20 - 25\text{ minutes}$**.
