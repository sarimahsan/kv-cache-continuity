# Track 1 (Main Focus): Causal Disentanglement of KV Continuity & Self-Speculative Decoding

**Paper Focus:** Machine Learning & Representation Geometry (Target: NeurIPS / ICLR / ICML)  
**Model:** `Qwen/Qwen2.5-3B` (and Qwen2 family)  
**Primary Question:** When tokens exit early or skip layers in autoregressive LLMs, does performance collapse due to **Attention Memory Poisoning ($\mathcal{E}_{\text{attn}}$)** or **Unembedding Manifold Mismatch ($\mathcal{E}_{\text{readout}}$)**?

---

## 1. Executive Summary & Key Discoveries

1. **Causal Disentanglement Protocol**:
   Using a teacher-forced, three-condition ablation protocol, we strictly decoupled attention memory corruption from unembedding readout distortion.
2. **Information-Theoretic Continuity Boundary**:
   Under a pre-registered 1-bit mutual information gate ($\bar{D}_{\text{KL}} \le \ln(2) \approx 0.6931\,\text{nats}$):
   - **$L=18$ (50% depth)**: $\bar{D}_{\text{KL}} = 0.7335\,\text{nats} > 0.6931\,\text{nats} \implies \mathbf{FAILS}$ (established as an intentional lower-bound stress test).
   - **$L=24$ (67% depth)**: $\bar{D}_{\text{KL}} = 0.3310\,\text{nats} < 0.6931\,\text{nats} \implies \mathbf{PASSES}$ with comfortable margin ($79.6\%$ Top-1 match).
   - **$L=29$ (81% depth)**: $\bar{D}_{\text{KL}} = 0.0713\,\text{nats} \ll 0.6931\,\text{nats} \implies \mathbf{PASSES}$ with high fidelity ($90.5\%$ Top-1 match).
   - **Verdict**: Frozen Native KV Projection ($W_K, W_V + \operatorname{LN}_1$) preserves attention continuity for depths $\ge 65\%$ ($L \ge 24$).
3. **Formal Negative Result on Standalone Hard Early Exit**:
   Even after training a 3,000-step non-linear bottleneck MLP exit adapter ($r=512$), intermediate perplexity remains at $\text{PPL} = 61.58\text{--}359.39$ against a full-depth oracle baseline of $5.76$. Standalone hard early exit is empirically non-viable without full-model retraining.
4. **The Self-Speculative Breakeven Law & Ceiling Theorem**:
   Under speculative decoding with horizon $\gamma = 1$, theoretical speedup is governed by $\mathcal{S} = \frac{1+\alpha}{c+1}$:
   - **Breakeven Law**: $\mathcal{S} \ge 1.0 \iff \alpha \ge c = \frac{d_t}{L}$.
   - **Ceiling Theorem**: Maximum possible speedup is bounded by $\mathcal{S}_{\max} = \frac{2}{c+1} \approx 1.20\times$ at $L=24$.
5. **Large-Scale Multi-Domain GPU Benchmark ($N=60$ Prompts)**:
   - Measured speedup: **$0.728\times \pm 0.025$** ($95\%$ CI: $[0.702\times, 0.753\times]$), perfectly matching the theoretical prediction of $0.748\times$.
   - Lossless equivalence: **$96.7\%$ (58/60 prompts)**.
   - Peak GPU VRAM: **$5.86\text{ GB} \ll 15.0\text{ GB}$**.

---

## 2. Directory Layout

```
kv_continuity/
├── src/
│   ├── exit_adapter.py          # ExitCalibrationModule (Linear & Non-linear MLP Adapters)
│   ├── kv_cache.py              # Native KV Projection & Continuous Buffer Management
│   ├── speculative_engine.py    # QwenSelfSpeculativeEngine (Zero-Recompute Parallel Chunk Verify)
│   ├── alignment.py             # Layer alignment & metric utilities
│   ├── dataset.py               # Preprocessing & tokenization loaders
│   └── configs.py               # Experiment & model configurations
├── experiments/
│   ├── benchmark_large_scale.py # N=60..100 Multi-domain benchmark (Math, Code, Science, Language)
│   ├── isolate_kv_ablation.py   # Teacher-forced 3-condition causal disentanglement ablation
│   ├── train_exit_adapters.py   # Training script for 3,000-step MLP exit calibration adapters
│   ├── profile_prompt2.py       # Fine-grained CUDA Event hardware profiling harness
│   ├── verify_kv_continuity.py  # Standalone KV continuity validation
│   └── benchmark_speculative_decoding.py # Quick 4-prompt smoke benchmark
├── doc/
│   ├── FINAL_AUTHOR_RESPONSE_AND_EMPIRICAL_RECONCILIATION.md  # Main author response & revision document
│   ├── RESPONSE_AND_PAPER_SECTION_CAUSAL_DISENTANGLEMENT.md    # Camera-ready Section 4 draft
│   ├── SELF_SPECULATIVE_DECODING_MATHEMATICAL_SPEC.md         # Full mathematical specification
│   ├── PAPER_2_CALIBRATED_ABLATION_EVIDENCE.md                # 3,000-step MLP empirical evidence
│   └── PAPER_2_DECOUPLED_KV_ABLATION_REPORT.md                # Decoupled ablation audit report
├── checkpoints/
│   ├── exit_adapters_qwen_mlp.pt # 3,000-step trained MLP bottleneck adapter (Depths 18, 24, 29)
│   └── exit_adapters_qwen.pt     # 500-step linear baseline adapter
└── benchmark_large_scale_results.json # Full N=60 benchmark empirical data
```

---

## 3. How to Run Key Experiments

All commands should be executed from within `kv_continuity/`:

### A. Run Large-Scale Multi-Domain Benchmark ($N=60$)
```bash
python experiments/benchmark_large_scale.py \
    --model-id Qwen/Qwen2.5-3B \
    --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \
    --draft-depth 24 \
    --gamma 1 \
    --num-prompts-per-domain 15 \
    --max-tokens 30 \
    --device cuda
```

### B. Run Teacher-Forced Decoupled KV Ablation (Table 1)
```bash
python experiments/isolate_kv_ablation.py \
    --model-id Qwen/Qwen2.5-3B \
    --device cuda
```

### C. Hardware Profile Prompt 2 Autopsy
```bash
python experiments/profile_prompt2.py \
    --model-id Qwen/Qwen2.5-3B \
    --adapter-path checkpoints/exit_adapters_qwen_mlp.pt \
    --draft-depth 29 \
    --max-tokens 30 \
    --device cuda
```
