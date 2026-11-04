# Paper 2 Technical Diagnostic Report: Dynamic Early-Exit on Qwen-2.5-3B

**Authors:** Advanced Systems & Efficient Inference Research  
**Subject:** Diagnostic Evaluation of KV-Cache Propagation vs. Exit-Head Calibration on Foundation Models  
**Model Under Test:** `Qwen/Qwen2.5-3B` ($36$ physical layers, GQA $16$ query heads / $2$ KV heads, $d_{\text{model}} = 2048$, $d_{\text{head}} = 128$)  
**Hardware & Environment:** Google Colab GPU (NVIDIA CUDA, PyTorch $2.5+$, `float16`)

---

## 1. Executive Summary & Diagnostic Framing

This report documents the empirical evaluation of dynamic early-exit decoding on **Qwen-2.5-3B**. 

### 1.1 What Is Mechanically Verified
- **Cache Buffer Integrity:** The pre-allocated static contiguous KV buffer successfully executed dynamic variable-depth execution across all $36$ layers with zero tensor shape mismatches, zero memory corruption, and zero runtime exceptions.
- **Cross-Layer State Injection:** Keys and values generated via Native State Projection ($W_K^{(l)} \operatorname{LN}_1(h_{d_t})$) were successfully injected into upper layers with appropriate GQA head broadcasting and Rotary Position Embeddings (RoPE).

### 1.2 What Remains Unproven (The Tangled Failure Mode)
During open-ended generation with early exit depths $d_t \in \{18, 24, 29, 36\}$, generated text collapsed into repetitive subword artifacts after the very first token. 

> [!IMPORTANT]
> **Scientific Caveat:** Degenerated text generation does **not** prove whether the projected KV states are effective or flawed. In open-ended dynamic generation, two distinct failure modes are conflated:
> 1. **Exit-Head Calibration Failure:** Emitting logits directly from intermediate representations $h_{18}$ or $h_{24}$ using the unadapted terminal unembedding matrix $W_{\text{unembed}}$ (trained exclusively on $h_{36}$).
> 2. **KV Approximation Error:** Distortion in the key/value states populated in skipped layers, degrading subsequent attention queries.
>
> Until these two factors are rigorously decoupled via teacher-forced ablations, **claims of quality preservation or KV continuity validation on 3B foundation models cannot be asserted.**

---

## 2. Experimental Diagnostic & Empirical Observations

### 2.1 The Two-Token Transition

Across all four evaluation prompts, an identical transition pattern emerged:

$$\begin{array}{lcccc}
\hline
\textbf{Prompt} & \textbf{Full 36L Target} & \textbf{Dynamic Step 1 Token} & \textbf{Dynamic Step 2 Token} & \textbf{Subsequent Text} \\
\hline
\text{Prompt 1 (AI Healthcare)} & \text{" 2025..."} & \mathbf{\text{" soon"}} & \text{"wealtherness..."} & \text{Corrupted subwords} \\
\text{Prompt 2 (Cheetah)} & \text{" it can run..."} & \mathbf{\text{" it"}} \text{ (Exact match)} & \text{"們مفاوضاتᐈ..."} & \text{Corrupted subwords} \\
\text{Prompt 3 (Coffee)} & \text{" grind the..."} & \mathbf{\text{" grind"}} \text{ (Exact match)} & \text{"/loadingالطائف..."} & \text{Corrupted subwords} \\
\text{Prompt 4 (Roman Empire)} & \text{" because of..."} & \mathbf{\text{" because"}} \text{ (Exact match)} & \text{"什么原因ᐈ..."} & \text{Corrupted subwords} \\
\hline
\end{array}$$

#### Step 1 (Prefill Phase at $L = 36$):
The prompt was processed through all $36$ physical layers. The initial emitted logit vector originated from $h_{36}^{(T)}$:

$$\hat{y}_1 = \arg\max \operatorname{Softmax}\left(W_{\text{unembed}} \cdot \operatorname{RMSNorm}(h_{36}^{(T)})\right)$$

Because $h_{36}^{(T)}$ lies on the manifold $\mathcal{M}_{36}$ on which $W_{\text{unembed}}$ was trained, the top-1 prediction is coherent and matches the reference model exactly.

#### Step 2 (First Dynamic Step at $d_2 = 18$):
At step $2$, the depth schedule scheduled an early exit at layer $18$ ($50.0\%$ depth). Logits were read directly from:

$$\hat{y}_2 = \arg\max \operatorname{Softmax}\left(W_{\text{unembed}} \cdot \operatorname{RMSNorm}(h_{18}^{(2)})\right)$$

In an unadapted 36-layer foundation model, intermediate representations at layer $18$ have not undergone the remaining $18$ layers of transformation. Projecting $h_{18}$ directly into the untuned vocabulary projection produces high-entropy distributions and out-of-distribution token sampling. In autoregressive decoding, this corrupted token is fed into the context of step $3$, permanently derailing downstream generation.

---

## 3. Critical Methodological Revisions

### 3.1 Accurate FLOP Accounting (Net Savings vs. Projection Overhead)

A standard transformer layer executes Attention ($Q, K, V, O$) and MLP transformations:

$$\text{FLOPs}_{\text{layer}} \approx 2 d_{\text{model}}^2 \left(1 + \frac{2 n_{\text{kv\_heads}}}{n_{\text{heads}}} + 1\right) + 6 d_{\text{model}} \cdot d_{\text{ffn}}$$

When layers $l \ge d_t$ are skipped, Native Projection executes RMSNorm and the $K, V$ projection matrices:

$$\text{FLOPs}_{\text{proj}}^{(l)} \approx 4 \cdot d_{\text{model}} \cdot (n_{\text{kv\_heads}} \cdot d_{\text{head}})$$

$$\begin{array}{lcccc}
\hline
\textbf{Model} & \textbf{Full Layer FLOPs} & \textbf{Projection FLOPs / Layer} & \textbf{Projection Overhead} & \textbf{Effective Savings at 50\% Depth} \\
\hline
\textbf{125M (MHA)} & \approx 11.8\text{ MFLOPs} & \approx 1.18\text{ MFLOPs} & \mathbf{10.0\%\text{--}16.7\%} & \mathbf{41.7\%} \text{ (not } 50.0\%) \\
\textbf{Qwen-2.5-3B (GQA)} & \approx 144.1\text{ MFLOPs} & \approx 2.10\text{ MFLOPs} & \mathbf{1.45\%} & \mathbf{48.5\%} \text{ (not } 50.0\%) \\
\hline
\end{array}$$

On models with Multi-Head Attention (MHA), $K/V$ projections account for over $10\%$ of layer compute; theoretical savings must subtract this overhead. On Grouped-Query Attention (GQA with $2$ KV heads), projection overhead is attenuated to $\approx 1.5\%$.

### 3.2 Throughput Variance & Measurement Protocol

The raw throughput measurements exhibited high variance across prompts under identical depth schedules:
- Prompt 1: $+45.2\%$ speedup ($18.3\text{ vs } 12.6\text{ tok/s}$)
- Prompt 4: $+2.9\%$ speedup ($17.9\text{ vs } 17.4\text{ tok/s}$)

This discrepancy indicates single-run eager execution artifacts (GPU thermal frequency shifts, JIT initialization, and host scheduling latency) rather than consistent execution speedups. 

**Required Protocol:** All latency claims must:
1. Use `torch.cuda.Event(enable_timing=True)` measuring strictly kernel execution time.
2. Include at least $5$ unmeasured warmup iterations.
3. Report the mean and standard deviation over $N \ge 10$ randomized sequences.

### 3.3 CUDA Graph Host-Synchronization Reality

Standard CUDA Graph execution (`cudaGraphLaunch`) is enqueued from the host CPU. A GPU-computed router argmax tensor cannot dynamically dispatch a host-side graph handle without explicit synchronization:
1. **Host-D2H Transfer:** Transferring the 4-byte scalar routing index via `cudaMemcpyAsync` back to host memory requires an asynchronous transfer. On 3B models decoding at $\approx 50\text{ ms/token}$, a $5\,\mu\text{s}$ D2H transfer adds $< 0.01\%$ overhead.
2. **Conditional Graph Nodes:** Utilizing CUDA 12.4+ conditional nodes (`cudaGraphAddNode` with `CUDA_GRAPH_COND_TYPE_IF`) keeps dispatch fully on-device, but requires driver stack support (CUDA Toolkit $\ge 12.4$).

---

## 4. Decisive Experimental Roadmap

To resolve the core research questions with unassailable empirical rigor, the following three teacher-forced experiments are required:

```
                  ┌──────────────────────────────────────────────────┐
                  │ Decoupling Evaluation on Real Validation Corpora │
                  └────────────────────────┬─────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│  Experiment 1:   │             │  Experiment 2:   │             │  Experiment 3:   │
│  KV-Only Test    │             │   Oracle Exit    │             │ Exit-Head Calib  │
├──────────────────┤             ├──────────────────┤             ├──────────────────┤
│• Predict @ L=36  │             │• Perfect router  │             │• True KV in past │
│• Past KV from    │             │• Early matching  │             │• Exit @ depth d  │
│  proj @ depth d  │             │  exit layer      │             │• Tuned lens vs   │
│• Measures pure   │             │• Native Proj KV  │             │  raw logit lens  │
│  KV damage       │             │• True upper bound│             │• Measures pure   │
│                  │             │  on quality      │             │  head mismatch   │
└──────────────────┘             └──────────────────┘             └──────────────────┘
```

### Experiment 1: The KV-Only Ablation (Decisive Test)
- **Protocol:** In teacher-forced mode on validation text, the next token $x_{t+1}$ is predicted exclusively at **full depth $L$** through the trained terminal unembedding head. However, for past tokens $\tau < t$, layers $l \ge d$ read from **projected KV cache** (Native Projection vs. Direct Copy vs. True KV).
- **Decision Rule:**
  - If $\Delta\text{PPL} \le 0.10$ and Top-1 Agreement $\ge 95\%$: The KV projection mechanism is mathematically sound and preserves attention context across depth changes.
  - If $\Delta\text{PPL} \gg 0.10$: The KV projection itself is losing critical state information, requiring learned low-rank corrections or lazy recomputation.

### Experiment 2: Oracle Exit + Native Projection (125M / 3B)
- Each token exits at the earliest layer $d$ whose top-1 prediction matches the final layer ($d_{\text{oracle}} = \min \{l \mid \arg\max z_l = \arg\max z_L\}$).
- KV continuity is populated via Native Projection.
- This isolates router inaccuracy from architectural viability.

### Experiment 3: Exit-Head Calibration on Foundation Models
- Past tokens retain true full-depth KV; current token exits at candidate depths $d \in \{18, 24, 29, 32\}$.
- Evaluate the raw logit lens vs. a lightweight tuned affine map ($P_l \in \mathbb{R}^{d \times d}$) distilled against $z_L$.
- Measures how much of the Qwen-3B degradation is resolved by linear output calibration.
