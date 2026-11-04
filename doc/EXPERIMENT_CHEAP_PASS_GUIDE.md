# Experiment Guide: Learned "Cheap Pass" KV Adapters for Early Exit

This guide details the complete implementation and execution protocol for the **Learned "Cheap Pass" KV Adapters for Early Exit** experiment on `Qwen2.5-3B`.

---

## 1. Architectural Formulation & The "Cheap Pass"

### 1.1 The KV Hole & The Native Projection Baseline
When a token exits early at layer $E$, its Key and Value states at skipped layers $l \in [E+1, \dots, L]$ are never computed.
Under **Native Projection**, we project the exit state $h_E$ directly using each skipped layer's own frozen weights:
$$\hat{K}_l^{\text{nat}} = \operatorname{RoPE}\left(W_K^{(l)} \operatorname{LN}_1^{(l)}(h_E)\right), \quad \hat{V}_l^{\text{nat}} = W_V^{(l)} \operatorname{LN}_1^{(l)}(h_E)$$

### 1.2 The Hypothesis
The true input to skipped layer $l$ is $h_l$, not $h_E$. Because intermediate representations undergo layer-by-layer residual accumulation:
$$h_l = h_E + \sum_{i=E}^{l-1} \Delta h_i$$
A compact trained adapter $A_l$ that predicts $\hat{h}_l \approx h_l$ from $h_E$ should produce Key/Value states closer to the ground truth, reducing downstream attention memory degradation.

### 1.3 Adapter Architecture: Variant A (One-Shot Parallel)
For each skipped layer $l \in [E+1, \dots, L]$:
$$\hat{h}_l = h_E + A_l(h_E)$$
$$A_l(x) = W_{\text{up}, l}\left(\operatorname{GELU}(W_{\text{down}, l}(x))\right)$$
* **Bottleneck Rank:** $r = 128$ (default).
* **Zero-Initialization:** $W_{\text{up}, l}$ is initialized to strictly zero. At step 0, $\hat{h}_l \equiv h_E$, meaning the adapter **exactly reproduces Native Projection at initialization**.
* **Frozen Projection:** Projected into KV space using the layer's own frozen weights:
  $$\hat{K}_l = \operatorname{RoPE}\left(W_K^{(l)} \operatorname{LN}_1^{(l)}(\hat{h}_l)\right), \quad \hat{V}_l = W_V^{(l)} \operatorname{LN}_1^{(l)}(\hat{h}_l)$$

---

## 2. Resource & Cost Analysis (Single 15GB T4 GPU)

### Parameter Count (Rank $r=128$, $d=2048$):
* Per skipped layer: $2 \times d \times r = 2 \times 2048 \times 128 = 524{,}288$ parameters ($\approx 0.52\text{M}$).
* For 18 skipped layers ($E=18$): $18 \times 524\text{K} \approx 9.4\text{M}$ parameters total (**$\approx 0.31\%$** of Qwen2.5-3B).

### Computational Cost (FLOPs):
* Adapter forward pass per token per skipped layer: $4 \times d \times r \approx 1.05\text{ MFLOPs}$.
* One full transformer layer in Qwen2.5-3B: $\approx 12 \times d^2 \approx 50.3\text{ MFLOPs}$.
* **Cost Ratio:** The adapter costs only **$\approx 2.1\%$ of one skipped transformer layer**.

---

## 3. Dataset Splits & Pre-Registered Criteria

* **Corpus:** FineWeb-Edu (`HuggingFaceFW/fineweb-edu`, `sample-10BT`) with fallbacks to `allenai/c4` or `wikitext-103`.
* **Sequence Length:** $T = 512$ tokens.
* **Strict Non-Overlapping Partitions:**
  * **Train Split:** 2,000 sequences ($1{,}024{,}000$ tokens).
  * **Validation Split:** 200 sequences ($102{,}400$ tokens). Checkpoint selection only.
  * **Test Split:** 500 sequences ($256{,}000$ tokens). Evaluated strictly after training finishes.
* **Pre-Registered Pass Gate:**
  $$\bar{D}_{\mathrm{KL}}(P^* \parallel \hat{P}_{\text{Adapter}}) < \ln(2) \approx \mathbf{0.6931\text{ nats}} \quad \text{at } E=18$$
  *(Evaluated under the terminal readout $h_L$ to isolate KV fidelity only).*

---

## 4. Execution Commands (Unified or Split in 2 Parts)

### Step 0: Fast Selftest (CPU / Dry-Run, ~5 seconds)
Verifies that all tensor dimensions, projection hooks, loss functions, gate checks, and data chunking run without errors:
```bash
python experiments/cheap_pass.py --selftest
```

---

### Option A: Running in 2 Split Parts (Recommended for Kaggle / Colab)
To avoid session timeouts and keep runs short and manageable, the experiment is split into two independent parts:

#### Part 1: Adapter Training & Offline Fidelity Gate
Trains the adapter checkpoints and checks whether the offline fidelity gate is passed:
```bash
# Can run for E=18 first (the pre-registered gate depth):
python experiments/train_cheap_pass.py --depths 18 --seq_len 512 --train_steps 3000 --batch_size 4

# Or all depths:
python experiments/train_cheap_pass.py --depths 18,24,29 --seq_len 512 --train_steps 3000 --batch_size 4
```

#### Part 2: Attention-Only Test KL & Pre-Registered Gate Check
Loads the saved checkpoints from Part 1 and evaluates on the 500 test sequences with 1,000 bootstrap CIs:
```bash
python experiments/eval_cheap_pass.py --depths 18 --seq_len 512 --batch_size 4
```

---

### Option B: Unified Single-Command Execution
Runs training, fidelity check, and test evaluation in one go:
```bash
python experiments/cheap_pass.py \
  --model Qwen/Qwen2.5-3B \
  --depths 18,24,29 \
  --seq_len 512 \
  --n_train 2000 \
  --n_val 200 \
  --n_test 500 \
  --rank 128 \
  --train_steps 3000 \
  --batch_size 4 \
  --lr 1e-3 \
  --out_dir cheap_pass_out
```

### Step 6: Full Controls Suite
Executes the required controls at $E=18$:
1. **Shuffle Control:** Mismatches layer target assignments during training (must fail).
2. **Zero-Init Check:** Confirms untrained adapter numerically equals NativeProj ($\text{Diff} \approx 0.0000$).
3. **Capacity Sweep:** Evaluates $r \in \{16, 64, 128, 256\}$.
4. **Sharing Ablation:** Evaluates a single shared adapter conditioned on layer index vs. per-layer adapters.
```bash
python experiments/cheap_pass.py \
  --model Qwen/Qwen2.5-3B \
  --depths 18 \
  --seq_len 512 \
  --run_controls \
  --ranks 16,64,128,256 \
  --out_dir cheap_pass_out
```

### Step 7: Plotting & Table Generation
Generates publication-quality figures and markdown summary tables:
```bash
python experiments/plot_cheap_pass.py \
  --json cheap_pass_out/cheap_pass_results_Qwen_Qwen2.5-3B.json \
  --out_dir cheap_pass_out
```

---

## 5. Decision Rules & Interpretations

| Experimental Outcome | Mechanistic Implication | Actionable Next Step |
| :--- | :--- | :--- |
| **No fidelity gain in Step 4** | Intermediate hidden states cannot be predicted token-locally without full context. | Record as a definitive negative result for token-local cheap passes. |
| **Fidelity gain in Step 4, but no KL gain in Step 5** | Per-layer raw K/V distance does not translate directly to attention distribution fidelity. | Inspect Value-drift vs. Key-drift attention entropy. |
| **KL gain in Step 5: $\text{KL} < 0.6931$ at $E=18$** | **Hypothesis Confirmed.** The valid early exit boundary is successfully extended to 50% depth. | Proceed to end-to-end speculative decoding with partial verification. |

