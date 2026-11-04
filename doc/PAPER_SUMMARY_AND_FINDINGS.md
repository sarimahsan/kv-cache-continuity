# Paper Summary & Empirical Findings

**Paper Title:** *Separating Attention Memory from Readout Error in Depth-Adaptive Transformers: Preserving KV-Cache Continuity via Native State Projection*  
**Author:** Syed Muhammad Sarim Ahsan  
**Repository:** [`sarimahsan/kv-cache-continuity`](https://github.com/sarimahsan/kv-cache-continuity)  
**Evaluated Backbones:** `Qwen/Qwen2.5-3B`, `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`, and `meta-llama/Llama-3.2-1B` (all on frozen weights, zero base-model fine-tuning).

---

## 1. What Does the Paper Argue? (Core Thesis & Conceptual Framework)

When Transformers terminate execution early (e.g., in early exiting, layer skipping, or self-speculative decoding), prior literature (SkipDecode, CALM, LayerSkip, MoD) evaluates only end-to-end task degradation (perplexity, BLEU, downstream accuracy). This **conflates two structurally distinct failure modes into a single number**:

1. **Attention Memory Poisoning ($\mathcal{E}_{\text{attn}}$):**  
   When a token exits at depth $d_t < L$, its Key and Value states for skipped layers $l \in \{d_t+1, \ldots, L\}$ are missing or approximated. Future tokens attend over this degraded history, potentially triggering cascading attention corruption across the sequence.
2. **Unembedding Manifold Mismatch ($\mathcal{E}_{\text{readout}}$):**  
   The exiting token's intermediate hidden state $h_{d_t}$ resides on an intermediate representation manifold $\mathcal{M}_{d_t}$ that is geometrically and semantically misaligned with the terminal unembedding matrix $W_{\text{unembed}}$ (which was trained solely to project $h_L$).

### Core Claims & Arguments

* **Causal Disentanglement is Required:** To understand why depth-adaptive inference degrades, one must isolate $\mathcal{E}_{\text{attn}}$ from $\mathcal{E}_{\text{readout}}$ under a controlled, teacher-forced ablation protocol before evaluating unconstrained rollouts.
* **The KV Problem Can Be Solved Training-Free:** We do **not** need expensive continual pretraining or complex caching architectures to preserve attention memory. Populating skipped-layer cache slots using the skipped layers' *own* frozen projection weights and pre-layer normalization (**Native State Projection**):
  $$\hat{K}_l = W_K^{(l)} \operatorname{LN}_1^{(l)}(h_{d_t}), \quad \hat{V}_l = W_V^{(l)} \operatorname{LN}_1^{(l)}(h_{d_t}) \quad (\forall l > d_t)$$
  preserves attention memory continuity at negligible compute cost ($1.4\%$ of FLOPs saved).
* **Direct State Repetition Fails Decisively:** Simply repeating the exit layer's states ($\hat{K}_l = K_{d_t}, \hat{V}_l = V_{d_t}$) or filling with zeros/means collapses generation because independently parameterized layers define mutually orthogonal activation subspaces. Native Projection performs the required coordinate transformation.
* **The Readout Bottleneck is the Real Culprit:** Generation degradation in early exit is **dominated by readout error ($\mathcal{E}_{\text{readout}}$), not attention memory ($\mathcal{E}_{\text{attn}}$)**. Intermediate representations possess a fundamental semantic deficit that lightweight adapters cannot bridge on open-domain text under modest distillation budgets.
* **Key/Value Asymmetry in Attention Memory:** Within $\mathcal{E}_{\text{attn}}$, degradation is driven primarily by **Value drift**, not Key drift. Softmax normalization compresses angular perturbations in Keys, whereas Values propagate linearly into attention outputs.
* **The Speculative Breakeven Constraint:** In self-speculative decoding under full verification, wall-clock speedup requires draft acceptance $\alpha > c = d_t / L$. Zero-shot intermediate readout cannot hit this threshold.

---

## 2. The Teacher-Forced Disentanglement Protocol

To isolate failure modes without compounding autoregressive error loops, the paper establishes a 5-condition protocol evaluated against the pristine full-depth oracle $\mathcal{M}^*$ ($h_L$ readout, exact $K^*, V^*$):

| Condition | KV Cache in Skipped Layers ($l > d_t$) | Token Readout State | Evaluates |
| :--- | :--- | :--- | :--- |
| **1. Oracle Baseline ($\mathcal{M}^*$)** | Exact Full-Depth ($K^*, V^*$) | Terminal state $h_L$ | Pristine reference ($\text{PPL}^*$) |
| **2. Readout-Only** | Exact Full-Depth ($K^*, V^*$) | Intermediate state $h_{d_t}$ | Pure Readout Mismatch ($\mathcal{E}_{\text{readout}}$) |
| **3. Attention-Only (Native Proj)** | Projected $(\hat{K}_l, \hat{V}_l)$ via $W_K^{(l)}, W_V^{(l)}$ | Terminal state $h_L$ | Pure Attention Memory ($\mathcal{E}_{\text{attn}}$) |
| **4. Attention-Only (Direct Copy / Fills)** | Direct copy $(K_{d_t}, V_{d_t})$ or Zero/Mean | Terminal state $h_L$ | Negative controls (unprojected floors) |
| **5. Dynamic Mixed-Exit Joint** | Alternating exit: projected KV for exits, oracle for full | Respective $h_{d_t}$ or $h_L$ | Dynamic heterogeneous execution |

**Operating Criterion:** A 1-bit mutual information threshold $\bar{D}_{\mathrm{KL}}(P^* \parallel \hat{P}) \le \ln(2) \approx 0.6931\text{ nats}$ is used as an objective target for cache fidelity.

---

## 3. What Did We Find Till Now? (Empirical Evidence Matrix)

### Finding 1: Native Projection Establishes a Robust Operating Boundary
* **Qwen2.5-3B (36 layers):**
  * At $L=18$ (50% depth): $\bar{D}_{\mathrm{KL}} = 0.914\text{--}0.960\text{ nats}$ (stress-test boundary, above $\ln 2$).
  * At $L=22$ (61% depth): $\bar{D}_{\mathrm{KL}} = 0.615\text{--}0.739\text{ nats}$ (boundary crossing).
  * At $L=24$ (67% depth): $\bar{D}_{\mathrm{KL}} = 0.437\text{ nats} < \ln 2$, Top-1 agreement $= 78.6\%$.
  * At $L=30$ (83% depth): $\bar{D}_{\mathrm{KL}} = 0.105\text{ nats}$, Top-1 agreement $= 89.2\%$.
* **Llama-3.2-1B (16 layers, 3 seeds):**
  * Meets the 1-bit threshold early at $L=8$ (50% depth): $\bar{D}_{\mathrm{KL}} = 0.5898 \pm 0.015\text{ nats}$, Top-1 $= 63.3\%$.
  * At $L=12$ (75% depth): $\bar{D}_{\mathrm{KL}} = 0.3072\text{ nats}$, Top-1 $= 74.9\%$.
* **Qwen2.5-7B (28 layers):**
  * At $L=24$ (85.7% depth): $\bar{D}_{\mathrm{KL}} = 0.3030\text{ nats}$, Top-1 $= 70.6\%$.

### Finding 2: Direct Copy and Heuristic Baselines Collapse Catastrophically
* At $L=18$ on Qwen2.5-3B:
  * **Native Projection:** $\text{PPL} = 13.33\text{--}53.39$, $\bar{D}_{\mathrm{KL}} \approx 0.95\text{ nats}$, $\text{Top-1} = 61.0\%$.
  * **Direct Copy:** $\text{PPL} = 3{,}088\text{--}5{,}434$, $\bar{D}_{\mathrm{KL}} = 5.52\text{--}6.30\text{ nats}$, $\text{Top-1} = 11.4\%$.
  * **Cache Omission (Masked Out):** $\text{PPL} = 1{,}069.8$, $\bar{D}_{\mathrm{KL}} = 3.96\text{ nats}$, $\text{Top-1} = 22.7\%$.
  * **Zero Fill / Running Mean Fill:** Degenerates identically to Cache Omission ($\bar{D}_{\mathrm{KL}} = 3.96\text{ nats}$).
* **Takeaway:** Skipped-layer cache entries cannot be repeated or left empty. Applying layer-specific $W_K^{(l)}, W_V^{(l)}$ matrices and $\text{LN}_1$ performs vital coordinate realignment into the downstream layer's subspace.

### Finding 3: Readout Mismatch is the Primary Bottleneck & Suffers an Adapter Generalization Ceiling
* **Zero-Shot Readout Fails:** Intermediate hidden states decoded via $W_{\text{unembed}}$ yield astronomical perplexities (at $L=18$: $\text{PPL} = 715{,}406$, $\text{Top-1} = 1.3\%$). Even at $L=30$ (83% depth), zero-shot $\text{PPL} = 30{,}369$ and $\text{Top-1} = 19.6\%$.
* **The Memorization Illusion (In-Sample):** Training a 2-layer MLP bottleneck adapter ($r=2048$, 8.4M params) on 6 canonical prompts reached $81.7\%$ Top-1 and matched oracle PPL ($5.74$ vs $5.76$).
* **The Open-Domain Generalization Ceiling (Out-of-Sample):** When trained across **10,000 steps (1.28M tokens)** on WikiText-2 `train` and evaluated on disjoint `validation`:
  * $L=18$ (50% depth): Top-1 plateaus at **$41.52\%$** ($\text{PPL} = 106.5$, $\bar{D}_{\mathrm{KL}} = 2.02$).
  * $L=24$ (67% depth): Top-1 plateaus at **$44.41\%$** ($\text{PPL} = 87.4$, $\bar{D}_{\mathrm{KL}} = 1.79$).
  * $L=30$ (83% depth): Top-1 plateaus at **$55.00\%$** ($\text{PPL} = 55.3$, $\bar{D}_{\mathrm{KL}} = 1.25$).
* **Takeaway:** Readout error accounts for $>90\%$ of total joint divergence at early exits. Intermediate states lack the deep semantic transformations performed by late MLPs.

### Finding 4: Value Drift Dominates Key Drift (Mechanistic & Causal Proof)
* **Layer-wise Drift:** Post-RoPE Key cosine similarity remains high ($\cos(K^*, \hat{K}) \ge 0.73$ through layer 34 on Qwen-3B), whereas Value similarity rapidly collapses ($\cos(V^*, \hat{V}) < 0.28$ in terminal layers). Centered cosine similarity and Frobenius norm confirm this is not a projection bias artifact.
* **Causal Shapley Swaps:** In full $2 \times 2$ factorial interventions across 8 model/depth settings (Qwen2.5-3B and Llama-3.2-1B):
  * **Value attribution ($\%S_V$):** Accounts for **$64.5\%$ to $82.1\%$** of total attention degradation.
  * All 95% bootstrap confidence intervals strictly exclude parity ($50\%$).
* **Theoretical Mechanism:**
  * **Keys:** Pass through inner product $\to$ softmax: $A = \operatorname{Softmax}(Q K^T / \sqrt{d_k})$. Simplex normalization exponentiates differences, compressing angular errors in non-maximal keys.
  * **Values:** Pass linearly into output: $Y = A \cdot V \implies \Delta Y = A \Delta V$. FFNs continuously inject new factual information into residual streams at each layer, making uncomputed Value states directly corrupt attention outputs with full linear weight.

### Finding 5: Context Length ($T=4096$) and Scale Robustness
* **Long Context Scaling:** Tested on Qwen2.5-3B across $T \in \{64, 128, 256, 512, 1024, 2048, 4096\}$ tokens.
  * At $L=30$, expanding context $4\times$ ($1024 \to 4096$) causes $\bar{D}_{\mathrm{KL}}$ to creep only from $0.2670$ to $0.3177\text{ nats}$ (remaining well below $\ln 2$), with Top-1 agreement $= 78.5\%$.
  * Demonstrates that Native Projection does **not** trigger runaway error accumulation over thousands of tokens.
* **Domain Transfer:** Evaluated on encyclopedic text (WikiText-2), web text (C4), and Python code. On C4 and Code, Native Projection meets the $\ln 2$ bound even at 50% depth ($L=18$).

### Finding 6: Dynamic Mixed-Exit Execution Supports Heterogeneous Decoding
* Tested under an alternating 50% exit policy (tokens alternate between early exit at $d_t$ and full-depth execution at $L$):
  * Downstream full-depth tokens ($e_p = 0$) attending over a half-projected cache maintain high fidelity: $\bar{D}_{\mathrm{KL}}^{\text{full}} \le 0.63\text{ nats}$ at $L=18$, dropping to $0.11\text{ nats}$ at $L=30$, with Top-1 agreement reaching $86.4\%$.
  * Proves attention memory continuity effectively supports downstream tokens during dynamic routing.

### Finding 7: Learned KV Repair (VKR / KVC vs ADEPT HSM)
* **Closing the Native Residual:** Sizing a tiny module ($\le 0.04\%$ backbone parameters, rank 64, 1500 steps on frozen backbone):
  * **KVC (Key-Value Correction):** Directly corrects $(\hat{K}_l, \hat{V}_l)$ via per-layer low-rank projections from a shared latent code.
  * **ADEPT HSM (Hidden-State Mapper):** Re-maps exit state $\tilde{h} = h_{d_t} + \text{MLP}(h_{d_t})$ before Native Projection.
  * **Result:** Lowers $\bar{D}_{\mathrm{KL}}$ by **$71\%\text{--}90\%$ on WikiText-2** and **$48\%\text{--}85\%$ on C4** across Qwen-3B/7B/14B and Llama-3.2-1B (e.g., Qwen-3B at $L=18$: $0.936 \to 0.165\text{ nats}$; Qwen-14B at 29% depth: $1.442 \to 0.389\text{ nats}$).
  * **KVC vs HSM:** KVC beats HSM by $5\%\text{--}20\%$ relative KL reduction across 17 of 20 tested configurations at mid-depths.
* **Free-Running Generation Caveat:** In open-loop greedy rollouts ($N=64\text{--}128$ prompts), learned repair substantially lowers generation NLL, but **KVC and HSM are statistically indistinguishable in generation quality (Match% and Gen-NLL)**. Without verification, greedy trajectories still diverge within ~5–15 tokens due to discrete argmax selection.

### Finding 8: Self-Speculative Decoding Breakeven Barrier
* Under full verification (re-executing all $L$ layers), speedup follows:
  $$\mathcal{S} = \frac{1 + \alpha}{c + 1}, \quad \text{requiring draft acceptance } \alpha > c = \frac{d_t}{L}$$
* At $d_t = 24$ on Qwen2.5-3B ($c = 0.667$), breakeven requires $\alpha > 66.7\%$.
* Zero-shot drafting achieved only $\alpha = 24.70\%$, yielding $\mathcal{S} = 0.748\times$ (a $25\%$ wall-clock slowdown, exactly matching the theoretical cost model).
* Tuned-lens lightweight heads raised acceptance to $41.5\%$ at $L=28$, but still fell short of breakeven ($\mathcal{S} = 0.806\times$).
* **Lossless Equivalence:** Validated 96.67% exact match (58/60 prompts) with full-model greedy text.

---

## 4. Summary Table of Key Results Across Models

| Model | Exit Depth $d_t$ | Rel. Depth | Method / Condition | $\bar{D}_{\mathrm{KL}}$ (nats) | PPL ($\Delta\text{PPL}$) | Top-1 Match | 1-bit Bound ($\le \ln 2$) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Qwen2.5-3B** | $L=18$ | 50.0% | Zero-Shot Readout | 11.7805 | $715{,}406$ | 1.3% | Fails |
| | | | Direct Copy | 5.519 | $5{,}434$ | 11.8% | Fails |
| | | | **Native Projection** | **0.914** | **61.3 (+55.5)** | **59.5%** | Stress boundary |
| | | | **Learned KVC (Rank 64)** | **0.165** | **19.1 (+1.7)** | **79.6%** | **Passes decisively** |
| | $L=24$ | 66.7% | Zero-Shot Readout | 10.7481 | $200{,}684$ | 2.4% | Fails |
| | | | Direct Copy | 3.435 | $752$ | 23.1% | Fails |
| | | | **Native Projection** | **0.437** | **8.0 (+2.3)** | **78.6%** | **Passes** |
| | | | **Learned KVC (Rank 64)** | **0.073** | **18.0 (+0.6)** | **87.0%** | **Passes decisively** |
| | $L=30$ | 83.3% | **Native Projection** | **0.105** | **6.5 (+0.7)** | **89.2%** | **Passes decisively** |
| **Llama-3.2-1B**| $L=8$ | 50.0% | Direct Copy | 4.652 | $1{,}025$ | 23.5% | Fails |
| | | | **Native Projection** | **0.590** | **68.4** | **63.3%** | **Passes** |
| | | | **Learned KVC (Rank 64)** | **0.117** | -- | -- | **Passes decisively** |
| | $L=12$ | 75.0% | **Native Projection** | **0.307** | **47.9** | **74.9%** | **Passes** |
| **Qwen2.5-7B** | $L=24$ | 85.7% | **Native Projection** | **0.303** | -- | **70.6%** | **Passes** |
| **Qwen2.5-14B**| $L=14$ | 29.2% | Native Projection | 1.442 | -- | -- | Fails |
| | | | **Learned KVC (Rank 64)** | **0.389** | -- | -- | **Passes** |

---

## 5. Main Takeaway & Practical Recommendations

1. **For System Designers:** Do not discard or zero out intermediate KV slots during early exiting. Native Projection recovers valid attention continuity without modifying base weights or adding parameters.
2. **For Interpretability & Architecture Researchers:** The failure of early exit is not an attention memory failure; it is an unembedding manifold mismatch. Auxiliary exit heads require multi-task co-training or deeper distillation to generalize on high-entropy text.
3. **For Speculative Decoding:** Training-free self-speculation cannot achieve speedups under full-model verification unless draft acceptance exceeds relative depth ($\alpha > c$). To harvest wall-clock gains, one must implement partial-layer verification (reusing $h_{d_t}$) or co-adapt the draft exit head.
4. **For KV Compression / Repair:** Because Value drift causes ~70–80% of attention error, spend capacity primarily on repairing or transmitting Values rather than Keys.
