# Implementation Plan: Resolving the KV-Cache Continuity Problem in Dynamic-Depth Decoding

## 1. Problem Definition & Mathematical Foundations

In dynamic-depth autoregressive generation (**Option B** from `PAPER_2_SCOPING_NOTE_V3.pdf`), token generation proceeds sequentially token-by-token ($S = 1$ at step $t$). A routing policy or confidence threshold selects an exit depth $d_t \in \{4, 6, 8, 10, 12\}$ for an $L = 12$ layer model.

### The "KV-Cache Hole" Dilemma
In standard autoregressive Transformers, at each layer $l \in \{1, \dots, L\}$, Key and Value vectors are computed and appended to the persistent cache:
$$K_l^{(1:t)} = \left[ K_l^{(1:t-1)}, k_l^{(t)} \right] \in \mathbb{R}^{B \times H \times t \times D_h}$$
$$V_l^{(1:t)} = \left[ V_l^{(1:t-1)}, v_l^{(t)} \right] \in \mathbb{R}^{B \times H \times t \times D_h}$$

When token $t-1$ exits early at depth $d_{t-1} < L$ (e.g., $d_{t-1} = 4$):
1. Layers $l \in \{1, \dots, d_{t-1}\}$ compute and store $(k_l^{(t-1)}, v_l^{(t-1)})$.
2. Layers $l \in \{d_{t-1} + 1, \dots, L\}$ are **completely bypassed** to save compute and memory bandwidth.
3. Therefore, $K_l^{(t-1)}$ and $V_l^{(t-1)}$ **do not exist** for upper layers $l > d_{t-1}$.

When a subsequent token $t$ requires a deeper layer $d_t > d_{t-1}$ (e.g., $d_t = 8$):
- Layer $l$ (for $d_{t-1} < l \le d_t$) queries the full history:
$$\operatorname{Attention}\left(q_l^{(t)}, K_l^{(1:t)}, V_l^{(1:t)}\right) = \operatorname{Softmax}\left( \frac{q_l^{(t)} (K_l^{(1:t)})^T}{\sqrt{D_h}} \right) V_l^{(1:t)}$$
- **The Breakdown:** Column $t-1$ in $K_l$ and $V_l$ is missing. If left unhandled, this either causes shape assertion failures, attends to uninitialized garbage memory, or forces monotonic depth locking ($d_t \ge d_{t-1}$), which ruins FLOP savings (collapsing savings from $44.6\%$ to $< 10\%$).

---

## 2. Proposed Architecture for KV-Cache Continuity

To eliminate the KV-cache hole without paying the compute cost of running upper Transformer blocks, we implement a **State-Propagation KV Projection Engine**:

```
Step t-1 (Easy Token exits at Layer 4):
  h_0^(t-1) ---> [Layer 1..4] ---> h_4^(t-1) ---> LM Head ---> Predict Token t
                                        │
                         [State-Propagation Module]
                         Fills K_l^(t-1), V_l^(t-1) for l in {5..12}
                                        │
                                        ▼
                         KV-Cache Slots (l=5..12, pos=t-1) Populated!

Step t (Hard Token needs Layer 8):
  h_0^(t)   ---> [Layer 1..8] ---> Attends seamlessly to pos 1..t-1 (No holes!)
```

### State-Propagation Strategies to Implement & Compare

1. **Strategy 1: Direct Native Layer Projection (Zero Parameter, Zero Extra Weights)**
   Use the existing LayerNorm and $W_K^{(l)}, W_V^{(l)}$ matrices of layer $l$:
   $$K_l^{(t)} = W_K^{(l)} \operatorname{LN}_1^{(l)}\left( h_{d_t}^{(t)} \right), \quad V_l^{(t)} = W_V^{(l)} \operatorname{LN}_1^{(l)}\left( h_{d_t}^{(t)} \right) \quad \forall l \in \{d_t + 1, \dots, L\}$$
   *Advantage:* Instantaneous, requires no training or extra parameters, and reuses existing weights.

2. **Strategy 2: Cross-Depth State Copying (Zero Compute)**
   $$K_l^{(t)} \leftarrow K_{d_t}^{(t)}, \quad V_l^{(t)} \leftarrow V_{d_t}^{(t)} \quad \forall l \in \{d_t + 1, \dots, L\}$$
   *Baseline:* Fast memory copy; serves as the lower-bound ablation to measure head misalignment.

3. **Strategy 3: Lightweight Linear Projection Head ($W_{\text{proj}}^{(d_t \to l)}$)**
   A small rank-$r$ or linear map trained to minimize reconstruction error against ground-truth unpruned KV states:
   $$\mathcal{L}_{\text{proj}} = \sum_{l = d_t + 1}^L \left( \| K_l^{(t)} - K_l^{*(t)} \|_2^2 + \| V_l^{(t)} - V_l^{*(t)} \|_2^2 \right)$$

---

## 3. Implementation Steps & File Modifications

### Component 1: Modular KV-Cache Engine ([NEW] `src/kv_cache.py`)
- Implement a static, pre-allocated `StaticKVCache` class supporting:
  - Fixed buffer allocation: $(B, H, S_{\max}, D_h)$ for $K$ and $V$ per layer (CUDA Graph friendly).
  - Incremental token updates: `update(layer_idx, k_val, v_val, current_pos)`.
  - Hole-filling method: `propagate_state(exit_depth, current_pos, h_exit, layers)`.
  - Slice extraction for causal SDPA attention.

### Component 2: Transformer Support for Step Decoding & KV-Cache ([MODIFY] `src/model.py`)
- Enhance `CausalSelfAttention` to accept an optional `kv_cache: Optional[StaticKVCache]` and `pos: Optional[int]`:
  - If `kv_cache` is provided and $T = 1$: compute $q_l^{(t)}, k_l^{(t)}, v_l^{(t)}$, write $k, v$ to cache at index `pos`, query the cached slice $1:pos$, and return single-token output.
- Enhance `TransformerLM`:
  - Add `forward_step(input_ids, pos, kv_cache, exit_depth=None)`:
    - Runs embeddings.
    - Sequentially executes layers up to `exit_depth` ($l \le d_t$).
    - If `exit_depth < L`: triggers KV state propagation for $l \in \{d_t + 1, \dots, L\}$.
    - Passes $h_{d_t}$ through `ln_f` and `lm_head` to emit logits.
  - Add `generate_dynamic(prompt_tokens, max_new_tokens, routing_policy, state_propagation_strategy)` for end-to-end autoregressive generation.

### Component 3: KV-Cache Continuity & Perplexity Verification Benchmark ([NEW] `experiments/verify_kv_continuity.py`)
- Comprehensive test harness addressing Milestone 2 & Milestone 3:
  1. **Correctness & Zero-Crash Assertion:**
     - Generate tokens with alternating/stochastic variable depths (e.g. $[4, 12, 4, 8, 4, 10, \dots]$) and verify:
       - Zero missing key-value access errors.
       - Zero NaN / Inf outputs.
  2. **KV Reconstruction Error Quantification:**
     - For tokens exited early, compute the Frobenius norm and cosine similarity between propagated $(K_l, V_l)$ vs ground-truth full-depth $(K_l^*, V_l^*)$.
  3. **Generation Quality & Perplexity Drift ($\Delta\text{PPL}$):**
     - Compare $100$-token generation sequences:
       - Condition 1: Static Full Depth ($L = 12$)
       - Condition 2: Dynamic Early-Exit with Strategy 1 (Native Projection)
       - Condition 3: Dynamic Early-Exit with Strategy 2 (Direct Copy)
       - Condition 4: Static SkipDecode baseline
     - Measure $\Delta\text{PPL} = \text{PPL}_{\text{dynamic}} - \text{PPL}_{\text{static}}$ to ensure drift is bounded within the publication target ($\Delta\text{PPL} \le 0.05$).

---

## 4. Verification Plan

### Automated Tests
1. **Unit Tests for `StaticKVCache`**:
   `python -m unittest discover tests` or a standalone test script `tests/test_kv_cache.py`:
   - Verify buffer pre-allocation and slice indexing.
   - Verify state propagation fills slots $l > d_t$.
2. **Dynamic Generation Sanity Test**:
   - Run 100-step generation on synthetic and real prompts with variable exit depths $\{4, 6, 8, 10, 12\}$.
   - Confirm tensor shapes, absence of memory leaks, and deterministic reproducibility.
3. **Empirical Quality & Perplexity Sweep**:
   - Run `python experiments/verify_kv_continuity.py --model-size 125m --num-samples 100 --seq-len 100`.

---

## 5. User Review Required

> [!IMPORTANT]
> **Priority Decision on Strategy 1 vs Strategy 3:**
> Strategy 1 (Native Projection via existing $W_K^{(l)}, W_V^{(l)}$ weights) requires **zero training** and **zero extra parameters**, enabling immediate testing on existing checkpoints. Strategy 3 (Learned projection) requires training an adapter head.
> **Recommendation:** Implement Strategy 1 and Strategy 2 first to validate continuity and measure baseline $\Delta\text{PPL}$, while keeping the interface extensible for Strategy 3.
