# Tuned-Lens Speculative Depth Sweep

**Status:** Complete  
**Run date:** 2026-10-15  
**Model:** `Qwen/Qwen2.5-3B`  
**Adapter:** `checkpoints/exit_adapters_qwen_tuned_lens.pt`  
**Adapter type:** LoRA, rank 64  
**Horizon:** `gamma=1`  
**Prompts:** 12  
**Generation budget:** 30 tokens  
**Dtype:** `float16`  
**Device:** CUDA

## Results

| Draft depth | Relative depth | Acceptance | Measured speedup | Exact match | Baseline tok/s | Speculative tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 38.89% | 1.84% | 0.7656x | 100.00% | 15.86 | 12.14 |
| 18 | 50.00% | 24.43% | 0.8520x | 91.67% | 15.86 | 13.51 |
| 22 | 61.11% | 2.11% | 0.6642x | 100.00% | 15.86 | 10.53 |
| 24 | 66.67% | 30.65% | 0.8081x | 100.00% | 15.86 | 12.82 |
| 28 | 77.78% | 9.37% | 0.6639x | 100.00% | 15.86 | 10.53 |
| 30 | 83.33% | 22.55% | 0.7163x | 100.00% | 15.86 | 11.36 |

## Interpretation

The best measured throughput in this sweep was `0.8520x` at depth 18, which is
still slower than the baseline. Depth 24 had the highest acceptance (`30.65%`)
among the tested depths with full exact-match equivalence, but its measured
throughput was `0.8081x`.

The depth-18 run recorded `91.67%` exact match (`11/12` prompts), so it should
not be treated as lossless. The sweep therefore does not establish a positive
speedup or a universally lossless tuned-lens configuration.

The theoretical partial-verification curve exceeds `1.0x` at several depths, but
the measured GPU speed remains below `1.0x` for every configuration. This
separates the idealized acceptance-based estimate from the current prototype's
measured end-to-end cost.

## Files

- Raw measurements: `tuned_lens_depth_sweep.csv`
- Plot: `tuned_lens_depth_tradeoff.pdf`
- Plot preview: `tuned_lens_depth_tradeoff.png`

