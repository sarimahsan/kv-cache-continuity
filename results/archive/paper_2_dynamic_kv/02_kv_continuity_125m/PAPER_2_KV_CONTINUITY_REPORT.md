# Technical Report: Dynamic-Depth Autoregressive Decoding with KV-Cache Continuity

**Target Paper:** Paper 2 (Learned Exit-Depth Router with CUDA Graph Bucketing)  
**Track:** Option B (Autoregressive Generation Acceleration with KV-Cache Continuity)  
**Date:** September 21, 2026  
**Status:** Milestone 2 Validated (Mechanics Verified; Quality Decoupling in Progress)  

---

## 1. The Core Problem: The "KV-Cache Hole"

In standard autoregressive Transformers, tokens are generated one by one ($S = 1$ at step $t$). At each layer $l \in \{1, \dots, L\}$, Key and Value vectors are computed and appended to persistent GPU cache buffers:

$$K_l^{(1:t)} = \left[ K_l^{(1:t-1)}, k_l^{(t)} \right] \in \mathbb{R}^{B \times H \times t \times D_h}$$
$$V_l^{(1:t)} = \left[ V_l^{(1:t-1)}, v_l^{(t)} \right] \in \mathbb{R}^{B \times H \times t \times D_h}$$

When dynamic early exit is introduced:
1. Token $t-1$ exits early at depth $d_{t-1} < L$ (e.g., $d_{t-1} = 4$ out of $12$).
2. Layers $l \in \{5, \dots, 12\}$ are **never executed** for token $t-1$ to save compute and bandwidth.
3. As a result, cache slots $K_l^{(t-1)}$ and $V_l^{(t-1)}$ for $l > 4$ are left **uncomputed ("KV-cache holes")**.
4. When subsequent token $t$ requires a deeper layer (e.g., $d_t = 8$), causal self-attention at Layer 8 must attend over all past tokens:

$$\operatorname{Attention}\left(q_8^{(t)}, K_8^{(1:t)}, V_8^{(1:t)}\right) = \operatorname{Softmax}\left( \frac{q_8^{(t)} (K_8^{(1:t)})^T}{\sqrt{D_h}} \right) V_8^{(1:t)}$$

Without a continuity mechanism, column $t-1$ in $K_8$ and $V_8$ is missing or uninitialized memory, leading to memory corruption, attention degradation, or forced monotonic locking ($d_t \ge d_{t-1}$) which degrades FLOP savings to $< 10\%$.

---

## 2. Architecture & State Propagation Mechanics

We implemented a modular **State-Propagation KV Engine**:

```
Step t-1 (Token exits at Layer 4):
  x^(t-1) ---> [Layer 1..4] ---> h_4^(t-1) ---> LM Head ---> Predicts Token t
                                      │
                       [Native State Propagation]
                     Applies LN_1^(l) + W_K^(l), W_V^(l)
                                      │
                                      ▼
                       Fills K_l^(t-1), V_l^(t-1) for l in {5..12}

Step t (Token needs Layer 8):
  x^(t)   ---> [Layer 1..8] ---> Attends seamlessly across 1..t-1 (Zero Holes!)
```

### Components Implemented:

1. **`StaticKVCache` (`src/kv_cache.py`):**
   - Pre-allocated fixed contiguous memory buffers $(B, H, S_{\max}, D_h)$ for $K$ and $V$ per layer, ensuring pointer immutability for downstream **CUDA Graph capture**.
   - **Strategy 1 (Native State Projection):**
     $$K_l^{(t)} = W_K^{(l)} \operatorname{LN}_1^{(l)}\left( h_{d_t}^{(t)} \right), \quad V_l^{(t)} = W_V^{(l)} \operatorname{LN}_1^{(l)}\left( h_{d_t}^{(t)} \right) \quad \forall l > d_t$$
     *(Note: This formulation parallels CALM-style state propagation; the paper's core novelty lies in the learned router and CUDA Graph multi-depth capture).*
   - **Strategy 2 (Direct Copy — Ablation Baseline):**
     $$K_l^{(t)} \leftarrow K_{d_t}^{(t)}, \quad V_l^{(t)} \leftarrow V_{d_t}^{(t)} \quad \forall l > d_t$$

2. **Step Decoding Integration (`src/model.py`):**
   - Added single-token step execution `forward_step` with variable-depth cutoffs $d_t \in [1, L]$ and KV propagation hooks.

---

## 3. Empirical Results on 125M Base Model

### 3.1 Mechanical Continuity & Stress Test
- **Setup:** $200$ adversarial depth jumps alternating between Layer 4 and Layer 12 ($[4, 12, 6, 12, 4, 4, 12, \dots]$).
- **Result:** **$100\%$ PASS**. Zero crashes, zero shape errors, zero NaNs or Infs.

---

### 3.2 State Reconstruction Across Skipped Layers (Exit at Layer 4)

We evaluate cosine similarity $\operatorname{CosSim}(K_l, K_l^*)$ against unpruned full-depth activations:

| Skipped Layer $l$ | Strategy | $\operatorname{CosSim}(K_l, K_l^*)$ | $\operatorname{CosSim}(V_l, V_l^*)$ | Relative Error $\frac{\| K - K^* \|}{\| K^* \|}$ | Notes & Scientific Interpretation |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Layer 5** | **Native Projection** | $\mathbf{1.0000}$ | $\mathbf{1.0000}$ | $\mathbf{0.0000}$ | *Trivial exact match by construction (Layer 5 input is $h_4$)* |
| Layer 5 | Direct Copy | $+0.0085$ | $-0.0618$ | $1.5502$ | Orthogonal cross-layer noise |
| **Layer 6** | **Native Projection** | $\mathbf{0.9717}$ | $\mathbf{0.9436}$ | $\mathbf{0.2377}$ | Minimal semantic drift ($1$ layer skipped) |
| Layer 6 | Direct Copy | $+0.0807$ | $-0.0950$ | $1.6080$ | Orthogonal noise |
| **Layer 7** | **Native Projection** | $\mathbf{0.9219}$ | $\mathbf{0.8393}$ | $\mathbf{0.4173}$ | Moderate drift |
| Layer 7 | Direct Copy | $-0.0574$ | $-0.0112$ | $1.7476$ | Orthogonal noise |
| **Layer 8** | **Native Projection** | $\mathbf{0.8501}$ | $\mathbf{0.7734}$ | $\mathbf{0.5756}$ | Degradation begins |
| Layer 8 | Direct Copy | $+0.0002$ | $+0.0432$ | $1.8296$ | Orthogonal noise |
| **Layer 10** | **Native Projection** | $\mathbf{0.7774}$ | $\mathbf{0.7066}$ | $\mathbf{0.6968}$ | Substantial divergence |
| Layer 10 | Direct Copy | $-0.0493$ | $+0.0190$ | $1.8232$ | Orthogonal noise |
| **Layer 12** | **Native Projection** | $\mathbf{0.6482}$ | $\mathbf{0.6610}$ | $\mathbf{0.8931}$ | **Significant decay to $0.65$ at terminal layer** |
| Layer 12 | Direct Copy | $-0.0376$ | $-0.0015$ | $1.9394$ | Orthogonal noise |

> [!NOTE]
> Layer 5's correlation of $1.0000$ is an identity by definition because $h_4$ is the true input to Layer 5. The critical scientific metric is the gradual decay across subsequent layers, dropping to $\mathbf{0.65}$ at Layer 12. Direct Copy fails completely ($\operatorname{CosSim} \approx 0.0$) because independent weight parameterization renders cross-layer activations orthogonal.

---

### 3.3 Generation Perplexity & Quality Observations

Tested over $50$ validation sequences ($32$ prompt tokens, $64$ generated tokens):

$$\begin{array}{lcccc}
\hline
\textbf{Condition} & \bar{L} \textbf{ (Mean Depth)} & \textbf{Nominal Savings} & \textbf{Net Savings (Post-Proj)} & \textbf{PPL} \\
\hline
\text{Static Full Depth (12L)} & 12.00 & 0.0\% & 0.0\% & \mathbf{1.715} \\
\textbf{Dynamic Early Exit (Native Proj)} & \mathbf{6.57} & \mathbf{45.2\%} & \mathbf{39.4\%} & \mathbf{5.142} \\
\text{Dynamic Early Exit (Direct Copy)} & 6.57 & 45.2\% & 45.2\% & 5.850 \\
\text{SkipDecode Uniform (L=6)} & 6.00 & 50.0\% & 50.0\% & \mathbf{4.040} \\
\hline
\end{array}$$

#### Honest Scientific Assessment:
1. **PPL Gap:** The dynamic exit perplexity ($5.142$) is substantially higher than the full model ($1.715$), far exceeding our design target ($\Delta\text{PPL} \le 0.05$).
2. **Comparison with SkipDecode:** Uniform SkipDecode at $L=6$ achieved $\text{PPL} = 4.040$, outperforming our uncalibrated random depth schedule ($5.142$). This occurs because SkipDecode maintains a static exit manifold throughout, whereas random jumping forces tokens to exit at premature depths ($L=4$) without a trained router.
3. **FLOP Accounting Correction:** On a standard 125M MHA model, $W_K + W_V$ projection represents $\approx 10.0\%\text{--}16.7\%$ of a layer's FLOPs. Thus, skipping $5.43$ layers on average yields a **net compute reduction of $\approx 39.4\%$**, rather than the uncorrected nominal $45.2\%$.
4. **Qualitative Drift:** While the generated text remains grammatical, hallucinations appear (introducing names "Lily", "Timmy" unprompted), confirming semantic drift under deep exits.

---

## 4. CUDA Graph Execution Constraints

Reviewers will scrutinize claims regarding GPU router dispatch:
- Standard `cudaGraphLaunch` is enqueued host-side by the CPU runtime.
- If a GPU router tensor determines the exit depth $d_t$, the CPU must obtain that depth to select the appropriate graph handle.
- **Architectural Solutions:**
  1. **Asynchronous 4-byte D2H Copy:** A single `cudaMemcpyAsync` transfers the depth integer to host memory concurrently with sampling/EOS checking. On 3B models decoding at $\approx 50\text{ ms/token}$, a $5\,\mu\text{s}$ D2H transfer adds $< 0.01\%$ overhead.
  2. **CUDA Conditional Nodes:** In CUDA 12.4+, conditional graph execution (`CUDA_GRAPH_COND_TYPE_IF`) permits device-side graph branching without host intervention.

---

## 5. Next Decisive Milestone

To isolate whether the $\Delta\text{PPL}$ gap originates from the **KV approximation** or from **premature exit-head readout**, we execute the **Teacher-Forced KV-Only Ablation** (`experiments/isolate_kv_ablation.py`).
