# Experiment Spec: Learned "Cheap Pass" KV Adapters for Early Exit

## 0. Goal

When a token exits early at layer `E`, its K/V entries at layers `E+1..L` are never computed (the "KV hole"), which corrupts attention for later tokens.

Current method (**Native Projection**): take the exit hidden state `h_E` and apply each skipped layer's own `W_K`, `W_V` (plus that layer's input LayerNorm) directly to it.

**Hypothesis:** The true input to skipped layer `l` is `h_l`, not `h_E`. A small trained adapter that predicts `h_l` from `h_E` (a "cheap pass") will give K/V closer to the true ones, and will reduce KL when later tokens attend to the cache.

**This experiment tests KV fidelity only.** It deliberately excludes the readout problem (the intermediate head's weak next-token prediction) by using the terminal readout `h_L` for the KL measurement.

## 1. Setup

- Model: `Qwen2.5-3B` (36 layers), frozen. fp16 inference, fp32 adapter training.
- Hardware: single T4 (15GB). Use small batches and gradient checkpointing if needed.
- Exit depths to test: `E = 18, 24, 29` (50%, 67%, 81% depth).
- Data: FineWeb-Edu (or the same eval set used in the earlier KL harness). Sequence length 512.
  - Train: 2,000 sequences. Validation: 200 sequences. Test: 500 sequences. No overlap between splits.
  - Fix and log the random seeds.
- Pre-registered pass gate: KL < ln(2) = **0.6931** nats (exact, no rounding). Written down before running.

## 2. Step 1 - Collect ground truth (no training)

Run the full frozen model over all sequences with hooks. For every layer `l` in `E..L-1` and every position, save:
- `h_l`: the true input hidden state to layer `l` (post-residual, pre-LN).
- `K_l`, `V_l`: the true key and value tensors as stored in the cache.

Store per exit depth E only what you need (h_E and the targets for layers E+1..L). Use fp16 on disk, memory-map, and keep shapes documented.

Sanity check: recomputing `K_l`, `V_l` from the saved `h_l` using layer `l`'s own LN and `W_K`, `W_V` (plus RoPE on K) must reproduce the cached values (cosine ~1.0). If not, fix the harness before continuing.

## 3. Step 2 - Baselines to beat

Compute the following on the validation split:

| Baseline | Description |
|---|---|
| `NativeProj` | K/V from `h_E` through each skipped layer's own `W_K`, `W_V` (current method) |
| `DirectCopy` | Copy layer-E's K/V into every skipped layer (floor) |
| `Oracle` | True K/V (upper bound, KL = 0) |

Per skipped layer, log:
- cosine similarity of K and of V vs true, averaged over positions and heads
- relative MSE: `||X_hat - X||^2 / ||X||^2`

Plot the per-layer curves. This shows where drift accumulates.

## 4. Step 3 - Train the adapters

Architecture, variant A (try first, one-shot parallel):

```
h_hat_l = h_E + A_l(h_E)       for l = E+1 .. L
A_l(x) = W_up_l( GELU( W_down_l(x) ) )     # bottleneck rank r
W_up_l zero-initialized so that h_hat_l = h_E at init (== NativeProj)
```

- Then: `K_hat_l = RoPE(W_K_l(LN_l(h_hat_l)))`, `V_hat_l = W_V_l(LN_l(h_hat_l))`, using the frozen layer's own weights.
- Rank r = 128 as the default. Separate `A_l` per layer (no sharing) as the default.
- Optional variant B (sequential): `h_hat_{l+1} = h_hat_l + g_l(h_hat_l)`. Run only if variant A shows gains and you want to compare.

Loss (per skipped layer, summed over layers):

```
L = (1 - cos(K_hat_l, K_l)) + (1 - cos(V_hat_l, V_l))  +  lambda * (MSE_K + MSE_V)
```

Start with `lambda = 1`, with K and V normalized per layer by their true norm so layers with large magnitudes don't dominate.

Training details:
- AdamW, lr 1e-3 (adapters only), cosine schedule, 3,000 steps, batch size by memory.
- Train on the train split only. Pick checkpoints on the validation split. Never touch the test split until the final report.
- Backbone is frozen. Verify no backbone parameter has a gradient.

## 5. Step 4 - Offline fidelity check (gate before anything else)

On the validation split, compare `Adapter` vs `NativeProj` using the same per-layer cosine similarity and relative MSE.

**Go / no-go:** If the adapter does not beat NativeProj on cosine similarity and relative MSE at most skipped layers, stop here and write up the negative result. Do not proceed to Step 5.

## 6. Step 5 - The real test: attention-only KL

This reuses the existing attention-only KL harness.

- For a token at position `t` that "exits" at depth E: its cache entries at layers E+1..L are replaced by the approximate K/V (NativeProj or Adapter).
- Later tokens attend to this cache normally through the full network.
- Readout is the **terminal** readout `h_L` (not the intermediate head). This isolates KV quality.
- Metric: KL(full-model next-token distribution || approximate-cache distribution), and Top-1 agreement with the full model.
- Conditions: `Oracle` (KL = 0), `NativeProj`, `Adapter`, plus the controls in Section 7.
- Run at E = 18, 24, 29. Report on the **test** split.

**Pass criterion (pre-registered):**
- Primary: `Adapter` KL at E=18 is below 0.6931 (NativeProj currently measured 0.7335).
- Secondary: `Adapter` KL is lower than NativeProj at E=24 and E=29 (no regression).

Report mean and 95% bootstrap CI over sequences (resample sequences, 1,000 resamples). Do not report conclusions based on a handful of prompts.

## 7. Controls (required for the result to be believable)

1. **Shuffle control:** train adapters with targets taken from mismatched layers or shifted positions. Must fail (KL near or above NativeProj). If it doesn't, the pipeline leaks something.
2. **Zero-init check:** an untrained adapter must reproduce NativeProj exactly.
3. **Capacity sweep:** r in {16, 64, 128, 256}. Report KL and fidelity vs r.
4. **Sharing ablation:** one shared adapter conditioned on layer index vs per-layer adapters.
5. **Seed variance:** 3 seeds for the main configuration. Report spread.

## 8. Step 6 - Only if Step 5 passes: end-to-end checks

- Perplexity with real early exit applied to every token at depth E, using adapter KV vs NativeProj KV vs full model. Use the terminal readout first, then the calibrated intermediate head separately.
- Cost: measure adapter parameter count and FLOPs per token relative to one full layer. Report the cost ratio c with the adapter included.
- Speed benchmark only after the above, using the existing N=60 multi-domain protocol and the breakeven law `S = (1+alpha)/(c+1)`.

## 9. Decision rules

| Outcome | Conclusion |
|---|---|
| No fidelity gain in Step 4 | Cheap pass does not help. Report as a negative result. |
| Fidelity gain in Step 4, no KL gain in Step 5 | Per-layer K/V error is not what hurts attention. Analyze which layers/positions matter. |
| KL gain in Step 5 at E=18 below the 0.6931 gate | The validated exit-depth range can be extended below 65% depth. |

## 10. Deliverables

- Code: data collection, adapter training, fidelity evaluation, KL harness integration. Configs and seeds committed.
- Results tables: per-layer fidelity, KL and Top-1 at each E with 95% CIs, controls, capacity sweep.
- Plots: per-layer cosine similarity and relative MSE (NativeProj vs Adapter), KL vs E, KL vs rank.
- A short write-up that states each result against its pre-registered criterion, including failures. Do not round thresholds or change the criteria after seeing results.

## 11. Prior art to cite and compare against

- State Propagation: DAT (Elbayad et al.), CALM (Schuster et al., 2022), ELLM.
- KV copying: SkipDecode (Del Corro et al., 2023).
- KV recompute: EE-LLM.
- Shared cache with self-speculation: LayerSkip (Elhoushi et al., 2024).
- Closest recent work: River-LLM (arXiv 2604.18396), lightweight companion layers producing approximate KV for skipped layers. Read it first and add it as a baseline if feasible.