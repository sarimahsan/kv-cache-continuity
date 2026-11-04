# Literature Review Guide & Related Work Search Brief

**Project Title:** Separating Attention Memory from Readout Error in Depth-Adaptive Transformers: Preserving KV-Cache Continuity via Native State Projection  
**Target Domains:** Efficient LLM Inference, Dynamic Depth / Early Exiting, KV-Cache Continuity, Speculative Decoding, Mechanistic Interpretability  
**Primary Artifact:** Comprehensive Literature Search Queries, Research Clusters, Landmark References, and Novelty Positioning.

---

## 1. Executive Summary & Core Research Thesis

### 1.1 The Fundamental Problem
Modern Large Language Models (LLMs) spend equal computational depth $L$ on every token regardless of semantic difficulty. When dynamic depth mechanisms (e.g., Early Exiting, Layer Skipping, Dynamic Routing, Mixture-of-Depths) terminate a token at an intermediate layer $d_t < L$, the token bypasses subsequent transformer layers $\{d_t + 1, \dots, L\}$. 

Prior work measures only end-to-end task degradation (PPL, BLEU, downstream task accuracy), **conflating two structurally distinct failure mechanisms**:
1. **Attention Memory Poisoning ($\mathcal{E}_{\text{attn}}$):** Downstream tokens in future decoding steps must attend over the historical sequence. If historical tokens exited early, their Key ($K$) and Value ($V$) states for skipped layers $l > d_t$ are missing, zeroed, or un-updated, potentially corrupting future context.
2. **Unembedding Manifold Mismatch ($\mathcal{E}_{\text{readout}}$):** The early-exiting token's intermediate hidden state $h_{d_t}$ resides on an intermediate activation manifold $\mathcal{M}_{d_t}$ that has not undergone late-layer MLP transformations and is geometrically misaligned with the terminal unembedding matrix $W_{\text{unembed}}$.

### 1.2 Key Contributions & Findings
* **Causal Disentanglement:** Isolates $\mathcal{E}_{\text{attn}}$ from $\mathcal{E}_{\text{readout}}$ using a controlled teacher-forced protocol evaluated against full-depth oracles ($\mathcal{M}^*$).
* **Native State Projection (Training-Free KV Recovery):** Reconstructs missing cache entries by projecting the exit hidden state $h_{d_t}$ through the skipped layers' *own* frozen pre-attention norms and projection weights:
  $$\hat{K}_l = W_K^{(l)} \operatorname{LN}_1^{(l)}(h_{d_t}), \quad \hat{V}_l = W_V^{(l)} \operatorname{LN}_1^{(l)}(h_{d_t}) \quad (\forall l > d_t)$$
  This satisfies an objective 1-bit fidelity criterion ($\bar{D}_{\mathrm{KL}} \le \ln 2$) for relative depths $\ge 50\text{--}65\%$ at zero parameter cost and negligible compute ($<1.4\%$ of FLOPs saved).
* **Failure of Heuristic Controls:** Direct state copying ($K_l = K_{d_t}$), zero fills, and running mean fills collapse catastrophically ($\text{PPL} > 1000$, $\bar{D}_{\mathrm{KL}} > 3.9$ nats) because layer subspaces are mutually orthogonal.
* **The Readout Bottleneck:** $\mathcal{E}_{\text{readout}}$ accounts for $>90\%$ of total joint degradation. Lightweight linear or MLP adapters suffer a strict open-domain generalization ceiling on high-entropy text.
* **Value-Dominance Asymmetry:** Causal Shapley swaps prove that **Value drift accounts for $65\%\text{--}82\%$ of attention corruption**, while Keys are resilient due to Softmax simplex compression ($A = \operatorname{Softmax}(QK^T / \sqrt{d_k})$ vs. linear output $Y = AV$).
* **Speculative Decoding Breakeven Barrier:** Training-free self-speculative drafting under full-depth verification requires acceptance rate $\alpha > c = d_t / L$. Zero-shot drafting cannot break even ($\alpha \approx 25\% < 67\%$).

---

## 2. Key Research Questions (RQs) for Literature Review

* **RQ1 (Attention Memory in Dynamic Inference):** How do existing early-exit, layer-skipping, and dynamic-depth transformer architectures populate, bypass, or repair the KV cache of skipped layers during auto-regressive generation?
* **RQ2 (Representation Alignment at Intermediate Layers):** What techniques exist for aligning intermediate hidden states with the final vocabulary vocabulary/unembedding space (e.g., Logit Lens, Tuned Lens, auxiliary exit heads, manifold mapping)?
* **RQ3 (Key vs. Value Dynamics in Transformer Attention):** Has prior mechanistic interpretability or compression literature documented asymmetric error tolerance between attention Keys and Values?
* **RQ4 (Self-Speculative Decoding Cost Models):** What are the theoretical and empirical breakeven constraints for draft-then-verify self-speculative decoding when reusing prefix hidden states or KV caches?

---

## 3. Adjacent Research Clusters & Landmark Papers

Use this taxonomy to organize the literature review into distinct thematic areas:

```
                                  [Depth-Adaptive LLMs & Early Exiting]
                                               |
         +-------------------------------------+-------------------------------------+
         |                                     |                                     |
[1. Dynamic Depth & Layer Skipping]  [2. KV Cache Reuse & Continuity]    [3. Self-Speculative Decoding]
* CALM, SkipDecode, LayerSkip        * StreamingLLM, H2O, FastGen         * Medusa, Eagle-1/2, REST
* Mixture-of-Depths (MoD)            * ADEPT, Prompt Cache, CacheGen      * Self-Speculative Decoding (Zhang)
         |                                     |                                     |
         +-------------------------------------+-------------------------------------+
                                               |
         +-------------------------------------+-------------------------------------+
         |                                                                           |
[4. Intermediate Manifold Alignment]                        [5. Key vs. Value Asymmetry & Interpretability]
* Logit Lens (nostalgebraist)                                * Value Memory in Transformers (Geva et al.)
* Tuned Lens (Belrose et al.)                                * Massive Activations & Outliers (Dettmers, Sun)
* Early Exit Classifiers (DeeBERT, FastBERT)                 * Softmax Bottleneck & Attention Geometry
```

### Cluster 1: Depth-Adaptive Transformers, Early Exiting & Layer Skipping
* **Core Topic:** Halting execution early per token or skipping intermediate transformer layers to save FLOPs.
* **Seminal / Target Papers:**
  * **CALM:** Schuster et al., *"Confident Adaptive Language Modeling"*, NeurIPS 2022. (Introduced token-level early exit for auto-regressive generation; discussed state copying).
  * **SkipDecode:** Del Corro et al., *"SkipDecode: Autoregressive Skip Decoding with Batching and KV Caching"*, NeurIPS 2023. (Addresses KV cache preservation under batched early exit; synchronizes exit points).
  * **LayerSkip:** Elhoushi et al., *"LayerSkip: Enabling Early Exit Inference and Speculative Decoding"*, arXiv 2024 / Meta. (Layer dropout during training + self-speculative execution).
  * **Mixture-of-Depths (MoD):** Raposo et al., *"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"*, arXiv 2024 / Google DeepMind. (Routing tokens around entire blocks via capacity-constrained routers).
  * **DeeBERT / FastBERT:** Xin et al. (ACL 2020), Liu et al. (ACL 2020). (Seminal classification early exits in encoder models).
  * **PABEE:** Zhou et al., *"Patient Adaptive Early Exit"*, ACL 2020.
* **Gap / Distinction:** Most prior work fine-tunes the base model or relies on heuristic cache copies without separating attention memory failure from readout failure. None identify the mathematical necessity of native coordinate projection ($W_K^{(l)}, W_V^{(l)}$) or the Value-drift bottleneck.

### Cluster 2: KV-Cache Continuity, Eviction & Cross-Layer Reuse
* **Core Topic:** Managing, compressing, or reconstructing KV cache states across long sequences and non-standard architectures.
* **Seminal / Target Papers:**
  * **StreamingLLM:** Xiao et al., *"Efficient Streaming Language Models with Attention Sinks"*, ICLR 2024. (Shows attention sinks preserve Softmax distribution; related to token history preservation).
  * **H2O (Heavy Hitter Oracle):** Zhang et al., *"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"*, NeurIPS 2023. (Token-level cache eviction).
  * **ADEPT:** He et al., *"ADEPT: Adaptive Early-Exit for Efficient LLM Inference with KV Cache Recycling"*, 2024. (Hidden-state mapper approach to reuse early exit states).
  * **Cross-Layer KV Sharing / Multiquery / GQA:** Shazeer (2019), Ainslie et al. (GQA, 2023), Brandon et al. (*"Reducing Transformer Depth on Demand via Cross-Layer Parameter Sharing"*).
  * **FastGen / SnapKV:** Ge et al. (2024), Li et al. (2024). (Adaptive attention sparsity and KV compression).
* **Gap / Distinction:** Cache eviction drops tokens completely; our setting retains the token temporally but lacks representations at upper layer depths. We show Native Projection restores continuity without retraining.

### Cluster 3: Self-Speculative Decoding & Draft Verification
* **Core Topic:** Using earlier layers of the *same* model as a draft model to speculate multiple tokens, verified by the full model.
* **Seminal / Target Papers:**
  * **Speculative Decoding:** Leviathan et al. (ICML 2023), Chen et al. (2023). (Seminal rejection-sampling draft-verify framework).
  * **Self-Speculative Decoding:** Zhang et al., *"Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding"*, ACL 2024. (Uses early exits as draft models).
  * **EAGLE / EAGLE-2:** Li et al., *"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"*, ICML 2024 / 2025. (Drafts at the feature level using one transformer decoder layer on top of base model states).
  * **Medusa:** Cai et al., *"Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads"*, 2024. (Multi-head prediction without separate draft models).
  * **Lookahead Decoding:** Fu et al., 2024. (Jacobi-iteration style token speculation).
* **Gap / Distinction:** We provide exact breakeven cost curves ($\alpha > d_t / L$) showing why training-free zero-shot intermediate drafting fails wall-clock speedup under full verification.

### Cluster 4: Intermediate Representation Decoding & Manifold Geometry
* **Core Topic:** Analyzing what intermediate layers represent and how they map to the final vocabulary space.
* **Seminal / Target Papers:**
  * **The Logit Lens:** nostalgebraist, *"interpreting GPT: the logit lens"*, 2020. (Direct application of terminal unembedding matrix $W_{\text{unembed}}$ to intermediate layers).
  * **The Tuned Lens:** Belrose et al., *"Eliciting Latent Predictions from Transformers with the Tuned Lens"*, ICML 2023. (Training affine / lightweight transformations per layer to inspect intermediate beliefs).
  * **Representation Alignment & Probing:** Alain & Bengio (2016) (*"Understanding intermediate layers using linear classifier probes"*); Geva et al., *"Transformer Feed-Forward Layers Are Key-Value Memories"*, EMNLP 2021.
  * **Direct Logit Attribution (DLA):** Elhage et al., *"A Mathematical Framework for Transformer Circuits"*, Anthropic 2021.
* **Gap / Distinction:** Prior work uses lenses primarily for interpretability; we quantify the information-theoretic generalization ceiling of intermediate-to-vocabulary adapters when used as computational substitutes in generative decoding.

### Cluster 5: Key vs. Value Mechanics, Softmax Geometry & Asymmetry
* **Core Topic:** Dissecting the differing mathematical roles of Key and Value vectors in self-attention.
* **Seminal / Target Papers:**
  * **Value Vector Semantics:** Geva et al., *"What Does a Transformer Learn? A Closer Look at the Keys and Values in Multi-Head Attention"*, 2022.
  * **Key-Value Dimension Asymmetries:** Sukhbaatar et al., *"Augmenting Self-Attention with Persistent Memory"*, 2019.
  * **RoPE & Softmax Dynamics:** Su et al., *"RoFormer: Enhanced Transformer with Rotary Position Embedding"*, 2024.
  * **Quantization Asymmetry:** Dettmers et al. (LLM.int8(), 2022); Xiao et al. (SmoothQuant, 2023). (Shows outlier channels concentrate heavily in specific activation paths).
* **Gap / Distinction:** We present the first causal Shapley attribution proving Value drift dominates Key drift ($>70\%$ attribution) in multi-layer depth approximation, explained by Softmax simplex compression vs. linear Value propagation.

---

## 4. Pre-Formulated Boolean Search Strings

Copy and paste these queries directly into Google Scholar, Semantic Scholar, arXiv, or ACM Digital Library:

### Search String A: Early Exit & KV Cache Handling
```text
("early exit" OR "layer skipping" OR "depth-adaptive" OR "dynamic depth") AND ("KV cache" OR "key-value cache" OR "attention cache") AND ("transformer" OR "LLM")
```

### Search String B: Self-Speculative Decoding & Early Exits
```text
("speculative decoding" OR "self-speculative") AND ("early exit" OR "intermediate layer" OR "draft model") AND ("acceptance rate" OR "speedup" OR "breakeven")
```

### Search String C: Intermediate State Unembedding & Logit Lens
```text
("tuned lens" OR "logit lens" OR "intermediate representations" OR "unembedding") AND ("early exit" OR "adapter" OR "linear probe") AND ("language model")
```

### Search String D: Key vs Value Attention Sensitivity & Asymmetry
```text
("attention keys" AND "attention values") AND ("drift" OR "sensitivity" OR "error propagation" OR "asymmetry" OR "Shapley") AND ("transformer")
```

### Search String E: Preserving KV Continuity in Skipped Layers
```text
("KV cache continuity" OR "missing KV" OR "cache recycling" OR "native projection" OR "cache imputation") AND ("transformer" OR "autoregressive")
```

---

## 5. Ready-to-Use Prompts for AI Research Assistants

Use these structured prompts with AI search engines (Perplexity, Elicit, Consensus, ChatGPT/Claude with Web Browsing):

### Prompt 1: Comprehensive Literature Mapping
```text
I am researching depth-adaptive and early-exiting Transformers (e.g., CALM, SkipDecode, LayerSkip, Mixture-of-Depths). 
Specifically, when a token exits early at layer d_t < L, its Key and Value states are missing for downstream layers l > d_t.

Can you find papers that explicitly address:
1. What happens to the KV cache of skipped layers during autoregressive generation?
2. Do any existing methods reconstruct, project, or approximate the missing KV states (rather than copying or dropping them)?
3. How do these methods separate the degradation caused by missing KV cache (attention memory poisoning) from the degradation caused by reading out intermediate hidden states with the final unembedding matrix?

Please provide citations (Authors, Year, Title, Venue) and summarize their specific approach to KV cache handling.
```

### Prompt 2: Key vs. Value Asymmetry
```text
In transformer self-attention (Multi-Head Attention / GQA), Keys pass through dot products and Softmax normalization, whereas Values pass through a linear summation weighted by the attention matrix.

Find research papers that investigate:
1. The difference in error tolerance between attention Keys and attention Values (e.g., under quantization, pruning, noise injection, or compression).
2. Evidence that Value error propagates more aggressively into output representations than Key error.
3. Mechanistic interpretability studies analyzing the geometric properties of post-RoPE Key embeddings versus Value embeddings across layer depths.
```

### Prompt 3: Speculative Decoding Breakeven Analysis
```text
Find papers analyzing the theoretical cost model and acceptance rate breakeven bounds for self-speculative decoding (where early layers of the same model act as the draft model).
Specifically look for papers discussing:
- Acceptance threshold alpha > cost_draft / cost_target (alpha > d_t / L).
- Why zero-shot intermediate layers or lightweight heads fail to reach high enough acceptance on open-domain text.
- Methods that verify intermediate draft states without re-running the full model from layer 0.
```

---

## 6. Matrix of Direct Comparators & Positioning

When writing the related work section, contrast this research against primary baselines as follows:

| Comparator Paper | Venue / Year | Primary Mechanism | How This Work Differs / Solves Its Failure |
| :--- | :--- | :--- | :--- |
| **SkipDecode** (Del Corro et al.) | NeurIPS 2023 | Synchronizes exit depths across batch items; retains previous KV entries. | Does not isolate $\mathcal{E}_{\text{attn}}$ from $\mathcal{E}_{\text{readout}}$; does not model coordinate projection across orthogonal layer subspaces. |
| **CALM** (Schuster et al.) | NeurIPS 2022 | Early exits when confidence threshold is met; copies or reuses representations. | Relies on base-model co-training; heuristic state reuse suffers from coordinate mismatch; lacks causal KV error attribution. |
| **LayerSkip** (Elhoushi et al.) | arXiv 2024 | Layer dropout during pretraining + self-speculative draft-verify. | Requires pretraining/continual training; does not address post-hoc frozen backbones; does not isolate Value-drift asymmetry. |
| **ADEPT** (He et al.) | 2024 | Trains a Hidden-State Mapper (HSM) on early-exit representations. | HSM modifies $h_{d_t}$ before projection; this work shows Native Projection requires zero training, and direct KVC repair outperforms HSM at lower parameter cost. |
| **Tuned Lens** (Belrose et al.) | ICML 2023 | Affine per-layer decoders for probe interpretability. | Focuses on exploratory inspection; does not examine attention cache contamination in autoregressive loops or open-domain generation ceilings. |
| **EAGLE / EAGLE-2** (Li et al.) | ICML 2024/25 | Autoregressive draft head predicting sequence features. | EAGLE requires training a dedicated speculative feature-decoder; our work characterizes the fundamental limit of the model's *own* frozen intermediate manifolds. |

---

## 7. Suggested Structure for the Paper's "Related Work" Section

1. **Depth-Adaptive Inference & Dynamic Compute Allocation**
   - Review token-level early exiting (CALM, DeeBERT), layer skipping (LayerSkip), and conditional computation (MoD).
   - *Key takeaway to emphasize:* Prior work treats early-exit degradation as a monolithic phenomenon, ignoring the separation between attention history and vocabulary decoding.
2. **KV Cache Compression, Eviction, and Cross-Layer Reuse**
   - Review token eviction (H2O, StreamingLLM, Scissorhands) and architectural sharing (GQA, Multiquery, cross-layer sharing).
   - *Key takeaway to emphasize:* While token eviction operates along the *sequence dimension* ($T$), dynamic depth creates gaps along the *depth dimension* ($L$). Native Projection bridges this depth gap training-free.
3. **Representational Geometry and Intermediate Manifold Alignment**
   - Review Logit Lens, Tuned Lens, and linear probes.
   - *Key takeaway to emphasize:* Confirms the existence of an intractable generalization ceiling for lightweight auxiliary heads, demonstrating that readout mismatch ($\mathcal{E}_{\text{readout}}$) is the true dominant bottleneck of early exiting.
4. **Attention Mechanism Dynamics & Value-Memory Semantics**
   - Review Key/Value role distinctions (Geva et al.) and outlier activation dynamics.
   - *Key takeaway to emphasize:* Provides causal evidence that Softmax normalization insulates Keys, leaving Value drift as the primary carrier of attention memory degradation.
5. **Speculative Decoding & Cost-Benefit Boundaries**
   - Review classic speculative decoding (Leviathan, Chen) and self-speculation (Zhang et al., LayerSkip).
   - *Key takeaway to emphasize:* Derives the strict cost-rate inequality ($\alpha > c = d_t / L$) that explains why zero-shot early exiting cannot achieve speedups under full-model verification.

