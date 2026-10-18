# Experiment 8: Tuned-Lens Depth Sweep

**Status:** Complete  
**Run date:** 2026-10-15  
**Model:** `Qwen/Qwen2.5-3B`  
**Adapter:** tuned-lens checkpoint  
**Horizon:** `gamma=1`  
**Depths:** `14, 18, 22, 24, 28, 30`  
**Device:** CUDA

## Results

| Draft depth | Relative depth | Acceptance | Measured speedup | Exact match | Baseline tok/s | Speculative tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 14 | 38.89% | 19.30% | 0.8793x | 100.00% | 16.55 | 14.55 |
| 18 | 50.00% | 19.10% | 0.8121x | 91.67% | 16.55 | 13.44 |
| 22 | 61.11% | 24.35% | 0.7862x | 100.00% | 16.55 | 13.01 |
| 24 | 66.67% | 24.88% | 0.7640x | 100.00% | 16.55 | 12.64 |
| 28 | 77.78% | 41.53% | 0.8062x | 100.00% | 16.55 | 13.34 |
| 30 | 83.33% | 22.55% | 0.6846x | 100.00% | 16.55 | 11.33 |

## Interpretation

The best measured throughput was `0.8793x` at depth 14, so this run remained
slower than the baseline at every tested depth. The highest acceptance rate was
`41.53%` at depth 28, with `100%` exact match for the 12-prompt slice.

Depth 18 recorded `91.67%` exact match (`11/12`) and should not be described as
lossless. The sweep does not establish a positive speedup, but it does provide
the intended depth/acceptance/throughput comparison for Experiment 8.

The idealized partial-verification estimates above `1.0x` do not translate into
measured end-to-end GPU speedup in this prototype.

## Files

- Measurements: `experiment_8_tuned_lens_depth_sweep.csv`
- Plot: `experiment_8_tuned_lens_depth_tradeoff.pdf`
- Plot preview: `experiment_8_tuned_lens_depth_tradeoff.png`
