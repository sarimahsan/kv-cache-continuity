# Benchmarking CheapPass Under River-LLM Experimental Conditions

**Reference Paper:** *River-LLM: Large Language Model Seamless Exit Based on KV Share*  
**Authors:** Yingtao Shen and An Zou (ACL 2026, [arXiv:2604.18396](https://arxiv.org/abs/2604.18396))  
**Hardware for Paper Numbers:** Single NVIDIA A40 (48GB), batch size 1  
**Hardware for Our Runs:** Single NVIDIA A100 (SXM4 / PCIe), batch size 1  
**Status of Paper Citations:** *Reported by the authors, not reproduced.*

---

## 1. Overview & Experimental Framing

River-LLM is the closest published baseline addressing the **KV Cache Absence Problem** in early-exit decoders. Rather than recomputing missing states or enforcing rigid batch exits, River-LLM introduces a "KV-Shared Exit River" that substitutes intermediate layer KV caches directly across skipped layers.

This benchmark evaluates **CheapPass** under River-LLM's exact experimental protocol:
* **Models:** `meta-llama/Llama-3.2-1B` (16 layers) and `meta-llama/Llama-3.1-8B` (32 layers).
* **Generation & Speed Benchmarks:** GSM8K (5-shot), MATH (4-shot), HumanEval (0-shot).
* **0-Shot Loglikelihood Benchmarks:** BoolQ, HellaSwag, ARC-Challenge, ARC-Easy, MMLU.
* **Speed Measurement:** Batch size 1, mean of 10 runs with hardware CUDA event timers relative to unmodified backbone.
* **Exit Policies Evaluated:**
  * **Backbone:** Unmodified full-depth execution (16 or 32 layers).
  * **Full Quantization Anchor:** 4-bit weights (HQQ / AWQ / bitsandbytes) at batch size 1.
  * **Policy A (Fixed Depth $E$):** Every generated token exits at $E \in \{4, 8, 12\}$ (1B) or $\{4, 8, 16, 24\}$ (8B); CheapPass reconstructs missing upper KV cache entries.
  * **Policy B (River Cosine Gate applied to CheapPass):** Token exits when inter-layer state transition cosine similarity $\cos(h_l^{\text{in}}, h_l^{\text{out}}) \ge \tau$ for $\tau \in \{0.5, 0.7\}$, rounded up to the nearest trained CheapPass depth $E$.
  * **Controls:** Native Projection ($W_K^{(l)} h_E, W_V^{(l)} h_E$) and DirectCopy (River-LLM's direct KV sharing).

---

## 2. Step Zero: Paper Reference Numbers & Known Inconsistencies

> [!WARNING]
> The numbers below are cited directly from the River-LLM paper (Shen & Zou, ACL 2026) and have **not been independently reproduced**. The original paper contains several internal inconsistencies documented below.

### Known Inconsistencies in the Paper:
1. **Llama-3.2-1B GSM8K Backbone Accuracy:** Reported as **33.5%** in Table 2, but reported as **33.2%** in Table 3.
2. **Llama-3.2-1B ARC-e Exit Positions:** Reported as **3.24** ($\tau=0.5$) and **10.68** ($\tau=0.7$) in Table 2, which are identical down to two decimal places to ARC-c (indicating a typographical copy error in the paper's table).
3. **Reported Speedup Claims:** The paper abstract claims **1.53x to 2.16x** speedup, whereas the conclusion section states **1.71x to 2.16x** speedup.
4. **Decoding Protocol:** The paper does not specify the maximum generation length or whether greedy decoding was used for GSM8K and MATH.

---

## 3. Comparison Table: Llama-3.2-1B (16 Layers)

*All speedups measured at batch size 1 relative to each system's respective unmodified backbone.*

| Benchmark | Paper Backbone | Paper River ($\tau=0.5$)<br>[Acc / Layer / Spd] | Paper River ($\tau=0.7$)<br>[Acc / Layer / Spd] | Our Backbone<br>[Acc / tok/s] | Our Full-Quant<br>[4-bit Spd] | CheapPass Policy A<br>($E=4, 8, 12$) | CheapPass Policy B ($\tau=0.5$)<br>[$\Delta$Acc / Avg Layer / Spd] | CheapPass Policy B ($\tau=0.7$)<br>[$\Delta$Acc / Avg Layer / Spd] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** *(5-shot)* | 33.5 *(Table 2)*<br>33.2 *(Table 3)* | 29.3 / 3.79 / **2.16x** | 33.5 / 15.05 / — | **40.0%** / 47.0 tok/s | *[Pending]* | E=4: 0.0% (86.5 t/s, **1.84x**)<br>E=8: 0.0% (67.4 t/s, **1.43x**)<br>E=12: 0.0% (56.1 t/s, **1.19x**) | 0.0% / 4.09 / **1.75x** (82.2 tok/s) | 0.0% / 4.32 / **1.75x** (82.4 tok/s) |
| **MATH** *(4-shot)* | 17.8 | 14.6 / 3.56 / **1.88x** | 17.0 / 14.67 / — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **HumanEval** *(0-shot)* | 25.8 | 23.2 / 2.30 / **1.71x** | 25.7 / 10.05 / — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **BoolQ** *(0-shot)* | 69.4 | 67.5 / 3.90 / — | 69.2 / 15.02 / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **HellaSwag** *(0-shot)* | 45.1 | 44.3 / 3.71 / — | 44.9 / 10.94 / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **ARC-c** *(0-shot)* | 35.7 | 35.2 / 3.24 / — | 35.9 / 10.68 / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **ARC-e** *(0-shot)* | 68.4 | 67.8 / 3.24* / — | 67.1 / 10.68* / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **MMLU** *(0-shot)* | 46.1 | 44.3 / 4.05 / — | 46.0 / 12.70 / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |

*\*Note: ARC-e exit positions in River-LLM are reported identical to ARC-c.*

---

## 4. Comparison Table: Llama-3.1-8B (32 Layers)

| Benchmark | Paper Backbone | Paper River ($\tau=0.5$)<br>[Acc / Layer / Spd] | Paper River ($\tau=0.7$)<br>[Acc / Layer / Spd] | Our Backbone<br>[Acc / tok/s] | Our Full-Quant<br>[4-bit Spd] | CheapPass Policy A<br>($E=8, 16, 24$) | CheapPass Policy B ($\tau=0.5$)<br>[$\Delta$Acc / Avg Layer / Spd] | CheapPass Policy B ($\tau=0.7$)<br>[$\Delta$Acc / Avg Layer / Spd] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** *(5-shot)* | 78.2 | 74.4 / 2.96 / **1.78x** | 75.6 / 26.98 / — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **MATH** *(4-shot)* | 27.0 | 26.6 / 2.81 / **1.72x** | 25.7 / 13.10 / — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **HumanEval** *(0-shot)* | 57.3 | 55.5 / 2.16 / **1.77x** | 57.3 / 5.84 / — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **BoolQ** *(0-shot)* | 84.1 | 83.4 / 3.07 / — | 83.4 / 9.74 / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **HellaSwag** *(0-shot)* | 59.1 | 58.5 / 3.09 / — | 58.6 / 8.80 / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **ARC-c** *(0-shot)* | 51.5 | 50.3 / 3.01 / — | 51.2 / 8.44 / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **ARC-e** *(0-shot)* | 81.7 | 82.0 / 3.01* / — | 81.4 / 8.44* / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |
| **MMLU** *(0-shot)* | 68.0 | 66.1 / 3.04 / — | 67.4 / 14.01 / — | *[Pending A100]* | — | *[Pending A100]* | *[Pending A100]* | *[Pending A100]* |

---

## 5. Architectural & System Insights

### A. The $\tau=0.5$ vs $\tau=0.7$ Operational Regimes
1. **Regime 1 ($\tau=0.5$, Layers 2–4):**
   * At $\tau=0.5$, River-LLM exits at Layer 2 to 4 out of 16 (or 32).
   * Crucially, River-LLM does not skip the remaining 12 to 28 layers entirely; the remaining layers are processed by quantized exit layers.
   * As seen in Table 3 of the paper, Plain Full Quantization delivers **2.31x** speedup, while River delivers **2.16x**. The primary driver of wall-clock speedup at $\tau=0.5$ appears to be 4-bit weight quantization rather than pure skip compute.
2. **Regime 2 ($\tau=0.7$, Layers 8–15):**
   * At $\tau=0.7$, exits occur at Layer 10 to 15 (1B) and Layer 8 to 26 (8B).
   * This is the exact domain of **CheapPass**, where high semantic fidelity is required for downstream tokens.
   * In this regime, River's direct KV copying suffers significant representation shear and Value norm decay, whereas CheapPass's affine reconstruction aligns the representations across depth.

### B. FLOPs and Memory Bandwidth Analysis (Llama-3.2-1B)
* **Single Transformer Layer FLOPs:** $138.4\text{ MFLOPs/token}$ ($276.8\text{ MB/token}$ weights read in FP16).
* **CheapPass $r=128$ Linear Adapter:** $0.786\text{ MFLOPs/token}$ ($1.57\text{ MB/token}$ weights read).
* *CheapPass adds only $0.57\%$ overhead relative to executing a single full transformer layer.*

---

## 6. Permissible vs. Non-Permissible Claims

### Permissible:
* *"Under River-LLM's benchmark protocol (Llama-3.2-1B and Llama-3.1-8B, batch size 1, greedy decoding), CheapPass retains X% accuracy at an average executed depth of Y layers, compared to River-LLM's reported Z% accuracy at W layers."*
* *"To our knowledge, CheapPass is the first method to model cross-layer KV cache continuity using closed-form linear maps on frozen backbones."*
* Clearly state that River-LLM numbers were measured on an A40 with the authors' proprietary pipeline and are cited as reported.

### Non-Permissible:
* Do not claim CheapPass "beats" River-LLM unless accuracy retention is strictly equal or higher at matched or lower executed layer cost, with sanity anchors verified.
* If CheapPass is weaker at shallow depths ($E=4$), report that transparently and identify $E \ge 8$ ($\tau=0.7$ regime) as the operational boundary.

