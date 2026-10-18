# Experiment 1: Gamma-2 Lossless Smoke Benchmark

**Status:** Complete  
**Run date:** 2026-10-15  
**Run environment:** Kaggle GPU  
**Experiment stage:** Next-experiments runbook, Experiment 1  
**Purpose:** Verify that multi-token self-speculative decoding with `gamma=2`
executes without crashes and preserves token-for-token output equivalence on the
smoke prompts.

## Configuration

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B` |
| Device | CUDA GPU |
| Dtype | `torch.float16` |
| Draft depth | `24/36` layers |
| Speculative horizon | `gamma=2` |
| Adapter | `exit_adapters_qwen_mlp.pt` |
| Adapter source | `/kaggle/input/models/sarimahsan/qwen-mlp/pytorch/default/1/exit_adapters_qwen_mlp.pt` |
| Loaded adapter depths | `18, 24, 29` |
| Prompts | 4 |
| Generation budget | 40 tokens, as configured for the run |

## Observed results

| Prompt | Baseline tok/s | Speculative tok/s | Speedup | Acceptance | Exact match |
|---:|---:|---:|---:|---:|---|
| 1 | 13.97 | 9.08 | 0.65x | 23.7% | Yes |
| 2 | 17.09 | 10.28 | 0.60x | 29.7% | Yes |
| 3 | 17.16 | 10.63 | 0.62x | 32.4% | Yes |
| 4 | 17.19 | 10.24 | 0.60x | 31.6% | Yes |
| **Average** | — | — | **0.62x** | **29.4%** | **4/4** |

Additional measurements:

- **Lossless equivalence:** `4/4` prompts, or `100%` exact token-for-token match.
- **Peak GPU VRAM:** `5.86 GB` of the reported `15.0 GB` limit.
- **Runtime status:** completed successfully without a crash, NaN, shape error, or
  out-of-memory error.

## Interpretation

The implementation passes the Experiment 1 smoke-test objective: `gamma=2`
multi-token drafting completed successfully and matched standard autoregressive
decoding for every tested prompt.

The measured throughput was **0.62x**, so this configuration is a slowdown rather
than an acceleration. The acceptance rate of **29.4%** is not sufficient to
offset the cost of drafting at depth `24/36` while the verifier performs full
model verification.

This result does **not** establish a publication-scale `gamma=2` estimate because
only four prompts were evaluated. It should be reported as a smoke-test result,
not as a replacement for the planned 60-prompt benchmark.

## Warnings and reproducibility notes

- The model was downloaded using unauthenticated Hugging Face requests. This is
  a rate-limit warning, not an experiment failure.
- Transformers reported that `torch_dtype` is deprecated in favor of `dtype`.
  This is also a warning and did not affect completion.
- The benchmark script prints its summary to stdout. This Markdown file records
  the captured console output; it is not generated automatically by the script.
- The run used the existing MLP adapter, not the planned tuned-lens checkpoint.
- Native Projection is not isolated by this benchmark; the measured prototype
  uses the full-verification path described in the manuscript.

## Source console summary

```text
SELF-SPECULATIVE DECODING BENCHMARK (Continuous Projected KV Cache)
Model: Qwen/Qwen2.5-3B | Device: cuda | Dtype: torch.float16
Draft Depth: L = 24 | Speculative Horizon: gamma = 2

Prompt 1: baseline 13.97 tok/s, speculative 9.08 tok/s,
          speedup 0.65x, acceptance 23.7%, exact match YES
Prompt 2: baseline 17.09 tok/s, speculative 10.28 tok/s,
          speedup 0.60x, acceptance 29.7%, exact match YES
Prompt 3: baseline 17.16 tok/s, speculative 10.63 tok/s,
          speedup 0.62x, acceptance 32.4%, exact match YES
Prompt 4: baseline 17.19 tok/s, speculative 10.24 tok/s,
          speedup 0.60x, acceptance 31.6%, exact match YES

AVERAGE SPEEDUP:         0.62x
AVERAGE ACCEPTANCE RATE: 29.4%
LOSSLESS EQUIVALENCE:    4/4 prompts matched 100% token-for-token
PEAK GPU VRAM CONSUMED:  5.86 GB / 15.0 GB limit
```

