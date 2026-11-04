# Experiment 4: Tuned-Lens 60-Prompt Benchmark

**Status:** Complete  
**Run date:** 2026-10-15  
**Model:** `Qwen/Qwen2.5-3B`  
**Draft depth:** `24/36` layers  
**Horizon:** `gamma=1`  
**Prompts:** `60` across four domains  
**Adapter:** tuned-lens checkpoint  
**Device:** CUDA

## Aggregate results

| Metric | Result |
|---|---:|
| Mean speedup | **0.7348x** |
| 95% speedup CI half-width | **0.0195x** |
| Mean acceptance rate | **21.70%** |
| 95% acceptance CI half-width | **2.90 percentage points** |
| Lossless exact-match rate | **96.67% (58/60)** |
| Peak GPU VRAM | **5.84 GB** |

## Interpretation

The tuned-lens adapter did not produce a measured end-to-end speedup above
`1.0x` on the 60-prompt benchmark. The run achieved `0.7348x` mean speedup and
preserved exact token equivalence for `58/60` prompts.

This is a publication-scale result for the tested configuration, but it should
not be presented as a positive acceleration result. The two non-equivalent
prompts should be investigated before making any claim that the tuned-lens
configuration is universally lossless.

## Files

- Measurements: `experiment_4_tuned_lens_60_prompt.json`
- Profiling utility: `experiments/profile_prompt2.py`
