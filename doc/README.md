# Technical Documentation & Evaluation Specifications

Welcome to the technical documentation for the **KV-Cache Continuity (`kv-cache-continuity`)** framework.

This directory outlines the formal evaluation protocols, architectural specifications, and mathematical formulations for preserving Key-Value (KV) cache continuity in depth-adaptive transformer inference.

---

## Documentation Index

### 1. [Evaluation Protocol (`protocol.md`)](protocol.md)
The formal experimental protocol defining:
- Teacher-forced causal disentanglement mechanics between attention memory error ($\mathcal{E}_{\text{attn}}$) and readout error ($\mathcal{E}_{\text{readout}}$).
- The historical-only approximation regime (`self_exact = True`) vs. all-position cache approximation (`self_exact = False`).
- Evaluation slices, dataset split provenance (FineWeb-Edu, WikiText-2, C4, GSM8K, Python Alpaca), and statistical confidence interval standards.
- Hard numerical assertions and pre-registered decision thresholds.

### 2. [Self-Speculative Decoding Cost Model (`SELF_SPECULATIVE_DECODING_MATHEMATICAL_SPEC.md`)](SELF_SPECULATIVE_DECODING_MATHEMATICAL_SPEC.md)
Detailed mathematical formulation of latency speedup $\mathcal{S}(\alpha, c, \gamma)$ in early-exit speculative drafting:
- Analytic conditions for wall-clock speedup $\alpha > c$ under full verification.
- Ideal ceiling formulation for partial-layer verification resuming from $h_{\dt}$.
- Empirical parameter validation across 60 multi-domain prompts on Qwen2.5-3B.

### 3. [River-LLM Comparative Benchmark (`../results/river_conditions/RIVER_CONDITIONS.md`)](../results/river_conditions/RIVER_CONDITIONS.md)
Controlled 8-task empirical benchmark comparing Native Projection, CheapPass KV adapters, and Direct Copy (River-LLM's KV-Share) across:
- `meta-llama/Llama-3.2-1B` ($E \in \{4, 8, 12\}$)
- `meta-llama/Llama-3.1-8B` ($E \in \{8, 16, 24\}$)
- Downstream tasks: MMLU, ARC-Challenge, ARC-Easy, HellaSwag, BoolQ, GSM8K, MATH, HumanEval.

---

## Directory Conventions

- **`doc/`**: Public-facing technical specifications and protocols.
- **`doc/internal_notes/`**: Archived internal development guides, scratch memos, and experiment workbooks (local-only, ignored by Git).
- **`doc/archive/`**: Historical review response drafts and prior iteration notes (local-only, ignored by Git).
