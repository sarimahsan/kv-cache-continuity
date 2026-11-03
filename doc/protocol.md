# Protocol Definition: KV Cache Continuity & Cache-Substitution Evaluation

**Project:** `kv-cache-continuity`  
**Document:** `doc/protocol.md`  
**Date:** November 3, 2026  
**Status:** Pre-Registered Experimental Standard  
**Target Architecture:** `Qwen/Qwen2.5-3B` (36 Layers, $d=2048$, GQA 16Q/2KV heads, $d_k=128$, frozen fp16)

---

## 1. Executive Summary & Purpose

This document provides the formal mathematical and algorithmic specification of the evaluation protocols employed in the `kv-cache-continuity` research project. It reconciles historical differences between early diagnostic harnesses and the current standardized benchmark, specifies exact data splits and masking rules, defines the scoring mechanics under teacher-forcing, and codifies hard assertions for statistical and operational reproducibility.

---

## 2. Formal Protocol Definitions

### 2.1 The Standardized Protocol: Cache-Substitution Under Terminal Readout
The primary benchmark evaluates how well upper-layer attention can function when historical context keys and values are replaced by synthetic approximations, while the current active token runs through the full network.

Let $\mathbf{x} = (x_0, x_1, \dots, x_{T-1})$ be a sequence of $T$ tokens.
For an un-truncated $L$-layer network, the hidden states and attention KV pairs evolve as:
$$h_l = \text{Layer}_l(h_{l-1}, \mathbf{K}_l, \mathbf{V}_l), \quad l \in [0, L-1]$$
where $\mathbf{K}_l, \mathbf{V}_l \in \mathbb{R}^{B \times H_{kv} \times T \times d_k}$ are the true Key and Value tensors.

Under the **Cache-Substitution Protocol** at early-exit depth $E$:
1. **Truncated Context States:** For all context tokens $\tau \le t$, representations at layer $E$ are obtained from the frozen backbone: $h_E(\tau)$.
2. **Synthetic KV Synthesis:** For all skipped downstream layers $l \in [E, L-1]$, Key and Value vectors for context positions are synthesized via an approximation operator $\mathcal{M}$:
   $$\hat{\mathbf{K}}_l = \mathcal{M}_K(h_E, l), \quad \hat{\mathbf{V}}_l = \mathcal{M}_V(h_E, l)$$
3. **Scored Token Forward Pass:** When scoring position $t \in [0, T-2]$, query $\mathbf{q}_{l, t}$ passes through all layers $l \in [0, L-1]$.
4. **Live Diagonal (`self_exact = True`):** In causal self-attention at layer $l \ge E$, position $t$ attends to past positions $\tau < t$ using synthetic cache $(\hat{\mathbf{K}}_{l, \tau}, \hat{\mathbf{V}}_{l, \tau})$, while attending to **itself ($\tau = t$) using its own live, exact Key and Value** $(\mathbf{K}_{l, t}^*, \mathbf{V}_{l, t}^*)$:
   $$\mathbf{A}_{l}(t, \tau) = \begin{cases} 
   \frac{\mathbf{q}_{l, t} \hat{\mathbf{k}}_{l, \tau}^\top}{\sqrt{d_k}} & \tau < t \\
   \frac{\mathbf{q}_{l, t} (\mathbf{k}_{l, t}^*)^\top}{\sqrt{d_k}} & \tau = t \\
   -\infty & \tau > t 
   \end{cases}$$
   $$\text{AttnOut}_l(t) = \sum_{\tau < t} \mathbf{A}_l(t, \tau) \hat{\mathbf{v}}_{l, \tau} + \mathbf{A}_l(t, t) \mathbf{v}_{l, t}^*$$
5. **Terminal Readout Head:** Next-token logits are computed exclusively by projecting the final-layer representation through the terminal unembedding head:
   $$P_{\text{method}}(x_{t+1} \mid x_{\le t}) = \text{Softmax}\left(h_L(t) \mathbf{W}_{\text{unembed}}\right)$$

### 2.2 Scored Positions & Loss Computation
- **Evaluated Sequence Range:** All $T-1$ transitions $t \in [0, T-2]$ predicting target token $x_{t+1}$.
- **Metrics vs Oracle:** Forward KL divergence and Top-1 match relative to the full un-truncated model $P_{\text{oracle}}(x_{t+1} \mid x_{\le t}) = \text{Softmax}(h_L^*(t) \mathbf{W}_{\text{unembed}})$:
  $$D_{\text{KL}}(P_{\text{oracle}} \parallel P_{\text{method}}) = \sum_{v \in \mathcal{V}} P_{\text{oracle}}(v) \ln\left(\frac{P_{\text{oracle}}(v)}{P_{\text{method}}(v)}\right)$$
  $$\text{Top-1 Agreement} = \mathbb{I}\left(\arg\max_v P_{\text{method}}(v) = \arg\max_v P_{\text{oracle}}(v)\right)$$

---

## 3. Harness Reconciliation: Legacy vs Current Benchmark

Earlier project documentation cited Native Projection at $E=18$ with $\text{KL} = 0.7335\text{ nats}$ and $\text{Top-1} = 70.9\%$, whereas the current benchmark reports $\text{KL} = 0.7680\text{ nats}$ and $\text{Top-1} = 59.65\%$. While forward KL diverged by only $0.0345\text{ nats}$, Top-1 agreement shifted by $11.25\text{ percentage points}$. 

The table below delineates the structural differences between the two harnesses:

| Factor | Legacy Harness (`isolate_kv_ablation.py`) | Current Standardized Benchmark (`vkr2` / `cheap_pass.py`) | Impact on Metrics |
| :--- | :--- | :--- | :--- |
| **Context Length ($T$)** | $T = 64$ (and $T=128$) | **$T = 512$** | Longer context ($4\times-8\times$) exposes queries to more corrupted past tokens, depressing Top-1 match from $70.9\% \to 59.6\%$. |
| **Dataset & Diversity** | 6 Canonical test paragraphs (`CANONICAL_TEST_TEXTS`) | **FineWeb-Edu (`sample-10BT`)**, held-out test split | FineWeb-Edu exhibits higher lexical entropy; errors in attention weight distribution more frequently flip greedy argmax tokens. |
| **Self-Token Attention (`self_exact`)** | `self_exact = False` in static cache step runner (token attends to its own projected $\hat{K}, \hat{V}$) | **`self_exact = True`** (token attends to its own live $K^*, V^*$ on diagonal) | `self_exact = True` isolates past context degradation from self-representation corruption. |
| **Scored Range** | Steps $0 \dots T-2$ sequentially in static cache | All tokens $0 \dots T-2$ via parallel causal attention mask | Mathematically equivalent causal receptive field, but $T=512$ compounds error accumulation. |
| **Hardware Batching** | Batch size $B=1$ step-by-step | Micro-batched $B=4$ with 64-token logit chunking | Chunking eliminates OOM while maintaining exact FP32/FP16 numerical parity. |

### Reconciliation Verdict:
The 11.25% shift in Top-1 accuracy is driven primarily by **context length scaling ($T=64 \to T=512$)** on **diverse natural web text**. In short sequences ($T=64$), attention is dominated by recent local tokens and self-attention, masking manifold drift. In long sequences ($T=512$), attention mass is distributed over hundreds of synthetic entries, amplifying Native Projection's Value norm explosion and causing greedy predictions to drift.

---

## 4. Dataset Splits & Provenance

To guarantee zero test-set contamination:
- **Corpus:** `HuggingFaceFW/fineweb-edu` (subset `sample-10BT`, streaming mode).
- **Tokenization:** `Qwen/Qwen2.5-3B` official tokenizer (vocab size $152{,}064$, EOS token ID $151{,}643$).
- **Sequence Length:** Fixed at $T = 512$ tokens per sequence.
- **Split Partitions (strictly non-overlapping):**
  - **Train:** Sequences $0 \dots 1{,}999$ ($2{,}000$ sequences = $1{,}024{,}000$ tokens).
  - **Validation:** Sequences $2{,}000 \dots 2{,}199$ ($200$ sequences = $102{,}400$ tokens), reserved for hyperparameter selection (e.g. ridge $\lambda$).
  - **Test:** Sequences $2{,}200 \dots 2{,}699$ ($500$ sequences = $256{,}000$ tokens), evaluated once for final reporting.
- **Out-of-Distribution (OOD) Test Sets ($N=100$ each, $T=512$, $51{,}200$ tokens):**
  - **Code:** `iamtarun/python_code_instructions_18k_alpaca` (raw Python code and docstrings).
  - **Math:** `openai/gsm8k` (`main` split, multi-step word problems).

---

## 5. Code Assertions & Numerical Integrity Checks

Every evaluation script in this repository must execute and pass the following hard assertions prior to logging results:

1. **Oracle Identity Assertion:**
   $$\text{KL}(P_{\text{oracle}} \parallel P_{\text{oracle}}) \equiv 0.0000 \quad \text{and} \quad \text{Top-1}(P_{\text{oracle}}, P_{\text{oracle}}) \equiv 100.00\%$$
2. **ZeroInit Invariance Assertion:**
   An untrained adapter initialized with zero weights ($\mathbf{W}_{\text{up}} = 0$) must match Native Projection to within numerical fp16 tolerance:
   $$|\text{KL}_{\text{ZeroInit}} - \text{KL}_{\text{NativeProj}}| < 10^{-5}$$
3. **FP16 Noise Floor Calibration:**
   Running the exact same Oracle forward pass under different batch partitionings ($B=1$ vs $B=4$) measures the hardware fp16 non-associative GEMM accumulation noise floor:
   $$\text{KL}(\text{Oracle}_{B=1} \parallel \text{Oracle}_{B=4}) \approx 10^{-6}\text{ nats}$$
4. **Draft-Depth Assertion (Track 2B):**
   In speculative drafting, instrumented execution layer counter must assert:
   $$\text{Executed Layers per Draft Step} \equiv E$$
5. **Cache-Independence Assertion (Track 2B):**
   In self-speculative decoding with a full verification pass, draft tokens and acceptance must be mathematically invariant to upper-layer cache substitution:
   $$\alpha(\text{Draft} \mid \text{CheapPass}) \equiv \alpha(\text{Draft} \mid \text{Native}) \equiv \alpha(\text{Draft} \mid \text{DirectCopy})$$

---

## 6. Pre-Registered Decision Thresholds

All criteria are pre-registered prior to running Track 1 and Track 2 evaluations:

| Target Metric | Pre-Registered Threshold | Theoretical Meaning |
| :--- | :--- | :--- |
| **Attention KL Gate ($E=18$)** | $\text{KL} < \ln(2) \approx 0.6931\text{ nats}$ | Mutual information loss bounded by 1 bit. |
| **Track 1: Scale/Offset Sufficiency** | Best Group S closes $\ge 50\%$ of CP128 gap | Degradation is primarily amplitude/offset; learned adapter is secondary. |
| **Track 1: Closed-Form Linear Sufficiency**| Best Group R closes $\ge 90\%$ of CP128 gap | Closed-form linear regression suffices; iterative gradient training unnecessary. |
| **Track 1: Non-Linearity Necessity** | CP128 beats L128 with paired CI excluding 0 and $\Delta\text{KL} \ge 10\%$ | GELU non-linearity is statistically required. |
| **Track 2A: Mixed-Depth Quality** | PPL matches Oracle within $3\%$ at $p \approx 30\%-50\%$ | Bounded degradation under practical exit frequencies. |
| **Track 2B: Wall-Clock Viability** | Measured speedup lower bound $95\%\text{ CI} > 1.00$ | True wall-clock acceleration achieved over greedy decoding. |

