# KV-Cache Continuity in Depth-Adaptive Transformers: Project Map

Author: Syed Muhammad Sarim Ahsan. Status as of 2026-10-25. This file summarizes where the project stands, what is already in the paper, what is new, what is still open, and what to run next.

---

## 1. The problem in one paragraph

In early exit / layer skipping, a token that stops at layer L never computes its Keys and Values for layers L..N-1. Later tokens still attend to those layers, so the cache has a hole. Separately, the exiting token's own prediction must be read out from a middle layer, which the unembedding was never trained for. These are two different failure modes:

| Symbol | Name | Meaning |
|---|---|---|
| E_attn | Attention memory poisoning | Missing or approximate KV in skipped layers corrupts later attention |
| E_readout | Readout mismatch | A middle-layer state decodes poorly through the final head |

## 2. Where we stand (honest summary)

| Question | Status |
|---|---|
| Can the two errors be separated and measured? | **Yes.** The teacher-forced disentanglement protocol is done and in the paper. |
| Does training-free Native Projection preserve the cache? | **Partly.** KL falls below ln 2 at about 50-67% depth; copying (Direct) is far worse. |
| Can a tiny learned module repair the KV hole? | **Mostly reduced, not solved.** KVC removes 71-90% of native KL (WikiText-2) with at most 0.04% extra parameters. |
| Does lower KL give better generated text? | **No.** KVC and the ADEPT-style mapper tie in generation; both drift from the full model within a few dozen tokens. |
| Is readout solved? | **No.** Adapters plateau at 41-55% top-1 agreement; zero-shot draft acceptance is about 3%. |
| Is there a measured speedup? | **No.** Best measurement is 0.95x (slower than baseline). |

Neither of the two parts is fully solved. The KV hole is largely mitigated on a frozen model. Readout is open.

## 3. What is already in the paper (source: `8c913f3c-attachment.txt`)

- Teacher-forced disentanglement protocol with five conditions (Attention-only, Readout-only, Direct Copy, omission/heuristic fills, dynamic joint).
- Native Projection results: Qwen2.5-3B, Llama-3.2-1B (shallow sweep, multi-seed), Qwen2.5-7B, context scaling to T = 4,096.
- Free-running rollout (Direct vs Native diverge within about 3-10 tokens).
- Adapter study: readout plateaus at 41.5-55.0% top-1 (1.28M-token budget).
- Cosine/Frobenius analysis and causal Shapley swaps (Value drift reported as 64.5-82.1%).
- Self-speculative cost model and a prototype.
- Known pre-existing source bugs fixed in `kv_continuity_merged.tex`: `\Ejoint` was undefined; `\kl_{\text{full}}` caused double-subscript errors.

## 4. What was added this session

New draft section **"Results: Learned KV Repair"** is in `learned_kv_repair_section.tex` and merged into `kv_continuity_merged.tex`. It contains three tables (learned repair, key/value error split, rollouts) plus patches to the abstract, contributions, related work, discussion, limitations, conclusion, reproducibility, and four new references.

### 4.1 Methods compared (frozen backbone, teacher-forced, attention-only)

| Method | Idea | Params (Qwen-3B, L24) |
|---|---|---|
| Native | Project exit state through each skipped layer's own K/V weights | 0 |
| Direct | Copy the K/V of layer L-1 into all skipped layers | 0 |
| ADEPT-style HSM | Shared MLP corrects the hidden state, then native projection (our reimplementation, not ADEPT's code) | 338,001 |
| KVC | Low-rank corrections to both Keys (pre-RoPE) and Values per skipped layer | 332,296 |

Training: WikiText-2 train, 1,500 steps (1,000 for 7B/14B), batch 4, T = 128, lr 1e-3, loss = KL + 0.5 x normalized V-MSE. Evaluation: WikiText-2 validation and C4, N = 120 sequences (60 for 7B/14B), paired bootstrap 95% CIs. 3 training seeds for Qwen-3B (L18, L24) and Llama-1B; single seed elsewhere.

### 4.2 Headline results (mean KL in nats, WikiText-2)

| Model | Exit L (depth) | Native | HSM | KVC | KVC vs HSM | KVC reduction vs Native |
|---|---|---|---|---|---|---|
| Qwen2.5-3B | 18 (50%) | 0.9358 | 0.1804 | 0.1649 | better (all seeds) | 82.4% |
| Qwen2.5-3B | 24 (67%) | 0.5750 | 0.0876 | 0.0730 | better | 87.3% |
| Qwen2.5-3B | 30 (83%) | 0.1442 | 0.0193 | 0.0196 | tie | 86.4% |
| Llama-3.2-1B | 8 (50%) | 0.4069 | 0.1301 | 0.1167 | better | 71.3% |
| Llama-3.2-1B | 10 (62%) | 0.2227 | 0.0507 | 0.0482 | better | 78.4% |
| Llama-3.2-1B | 12 (75%) | 0.1888 | 0.0177 | 0.0183 | HSM slightly better / tie | 90.3% |
| Qwen2.5-7B | 14 (50%) | 0.6523 | 0.1738 | 0.1510 | better (1 seed) | 76.8% |
| Qwen2.5-7B | 21 (75%) | 0.3216 | 0.0404 | 0.0355 | better (1 seed) | 89.0% |
| Qwen2.5-14B | 14 (29%) | 1.4424 | 0.4370 | 0.3888 | better (1 seed) | 73.0% |
| Qwen2.5-14B | 21 (44%) | 0.8686 | 0.2476 | 0.2192 | better (1 seed) | 74.8% |

KVC beat HSM in 17 of 20 model/depth/dataset cells (relative KL reduction 5-20%). It ties at the late exits.

### 4.3 Generation (free-running, 64 new tokens, 32-token prompts)

KVC and HSM are statistically indistinguishable in generation: all paired intervals for match rate and full-model NLL include zero. Both are much better than Native/Direct, but match with the full model stays low (at most 57% on Qwen-3B, at most 50% on Llama-1B). **KL gains did not convert into generation gains.**

### 4.4 Key vs Value error split (oracle substitution, Value share of error)

| Model | Value share by depth |
|---|---|
| Qwen2.5-3B | 69.7% (L18), 66.1% (L24), 51.3% (L30) |
| Llama-3.2-1B | 57.4% (L8), 43.1% (L10), 28.0% (L12) |
| Qwen2.5-7B | 66.5% (L14), 44.5% (L21) |
| Qwen2.5-14B | 80.6% (L14), 68.0% (L21) |

Pattern: model- and depth-dependent. Values dominate shallow exits on Qwen; Keys dominate deeper exits on Llama.

## 5. Open problems and things to fix before submitting

1. **Numbers disagree with the paper's older tables** (marked `[TODO]` in red in the draft):
   - Value share / Shapley: new 51% vs paper 75.8% (Qwen-3B L30); new 57% vs 76.2% (Llama L8); new 43% vs 82.1% (Llama L10). This conflicts with the abstract claim "Value dominance in all eight configs".
   - Native KL is lower than the paper's tables for Llama-1B (L8: 0.41 vs 0.59; L10: 0.22 vs 0.42), Qwen-3B L30 (0.14 vs 0.21), and Qwen-7B L14.
   - Rollout direction: new harness has Native at least as good as Direct at late exits; old table has Direct ahead (settings differ: 64 vs 256 tokens, prompt length, N).
   - Leading hypothesis (untested): the new harness keeps the current token's own K/V exact; the older protocol may not. Diagnostic: run vkr2 at `--seq_len 64 --eval_n 30 --train_steps 1 --methods kvc` on Llama L8/10/12, Qwen-3B L30, Qwen-7B L14 and compare.
2. **Verify citations** (FREE, EE-LLM, ADEPT, River-LLM one-line descriptions; River-LLM author list may be incomplete). River-LLM runs the skipped layers in 4-bit rather than skipping them, so that sentence needs refining.
3. **Paper cost model:** the equation claiming partial-layer verification "lifts the ceiling to 1+alpha" is likely wrong (see section 7). Reword as a conjecture or fix it.
4. **Compile in IEEEtran:** only checked with a stand-in `article` class here.

## 6. What we are testing next: `vkr3.py`

`vkr3.py` imports `vkr2.py` (keep them in the same folder). It passed a tiny-random-model self-test (exact history reproduces the full model to 1e-8; shadow module equals native at init; LoRA damage check works). **It has not yet been run on a real model.**

| Angle | Method / flag | Question it answers |
|---|---|---|
| Capacity vs information | `--exp rank` | Does KL keep falling with more parameters, or is a token-local map information-limited? |
| Trajectory | `shadow`, `shadowS`, `shadowX`, `shadowSXC` | Does a tiny network that walks the residual stream from L to N beat a one-shot map from h_L? |
| Context | `kvcx` | Does a causal conv over previous tokens help KVC? |
| One module, all depths | `kvcm` | Does a single depth-conditioned generator match per-depth modules? |
| Co-adaptation | `coadapt` | Does LoRA on upper layers, trained jointly with KVC, close the gap the frozen model cannot? (SwiftKV-style) |
| Full conversion | `joint`, `exithead` | Can one conversion fix KV and readout together (exit head + LoRA on all layers)? |
| Staleness / backfill | `--exp refresh` | How fast does error grow if history is exact only beyond R tokens? Protect the first k tokens? |
| Mixed exits | `--exp mix`, `--p_exact` | What happens when only some tokens exit? |
| On-policy training | `--onpolicy_n` | Does training on the model's own samples close the KL-to-generation gap? |
| Diagnostics | `--exp diag` | Per-layer K/V error and KL by token position |

Run order (on the 3B model; send outputs back for interpretation):

```bash
# 1) capacity vs information
python vkr3.py --model Qwen/Qwen2.5-3B --depths 18 --exp rank --ranks 16,64,256,1024
# 2) new frozen-backbone repairs vs v2 champion (reuses kvc/adept_hsm checkpoints via --ckpt_dir)
python vkr3.py --model Qwen/Qwen2.5-3B --depths 18 --methods adept_hsm,kvc,kvcx,shadow,shadowS,shadowX --ckpt_dir vkr_out --exp main,refresh,diag
# 3) the main bet: co-adaptation and joint conversion
python vkr3.py --model Qwen/Qwen2.5-3B --depths 18 --methods kvc,coadapt,joint,exithead --train_steps 2000 --batch 2 --exp main --rollout_n 32
```

How to read results:
- `dKL vs kvc` below 0 with a CI excluding 0 means the method beats the v2 champion.
- `selfcheck` (coadapt/joint) must stay near 0, otherwise the conversion damaged the model.
- A repair counts as a solution only if rollout Gen-NLL approaches the full model's, not just KL.

### Decision gate

After steps 1-3 (roughly two weeks of work): if KL and Gen-NLL both improve clearly with co-adaptation, commit to the conversion-recipe project (exit heads plus KV generator for a pretrained model). If not, fold the findings into the current paper and move on.

## 7. Self-speculative decoding: where it fits

Self-speculative decoding is a way **around** the KV hole, not a fix. Every token's upper layers are eventually run, so the cache stays exact and output is lossless. The speed gain depends on draft acceptance, which depends on readout quality.

Measured (Qwen2.5-3B, draft depth 24 of 36, 12 prompts, 384 tokens, batch 1):

| Strategy | Throughput | Speedup | Draft acceptance | Lossless |
|---|---|---|---|---|
| Standard baseline | 17.37 tok/s | 1.00x | n/a | yes |
| Naive full verifier | 10.37 tok/s | 0.60x | 3.0% | yes |
| Partial-layer verification | 16.42 tok/s | 0.95x | 3.0% | yes |

Split-pass overhead (draft + resumed verification vs one full forward) is only +2%.

**Correction to the earlier projection.** The projected table (1.22x at 25% acceptance up to 1.96x at 100%) used speedup = (1+alpha)/1.02, which treats the bonus token as free. Counting the lower-layer pass the bonus token needs, one round costs about 1 + c where c = draft depth / total depth (0.67 here), so for one draft token per round:

speedup = (1 + alpha) / (1 + c)

| alpha | Corrected | Earlier projection |
|---|---|---|
| 25% | 0.75x | 1.22x |
| 50% | 0.90x | 1.47x |
| 75% | 1.05x | 1.71x |
| 100% | 1.20x | 1.96x |

Break-even is alpha > c (matching the paper's own naive condition). With longer draft chains the ceiling is about 1/c = 1.5x. Likely reason partial verification reached 0.95x rather than 0.60x: it probably does not produce bonus tokens, so it cannot exceed about 1.0x. **Check:** force the draft to equal the true next token (alpha = 100%) and measure throughput. About 1.0x means no gain mechanism; about 1.2x means the corrected formula holds. Realistic gains need a shallower draft with high acceptance, which means trained exit heads (LayerSkip-style).

## 8. Items from an earlier exploration doc (c87d7ef4) that should NOT go in the paper as written

- **"Value-Only KVC" is not better than full KVC.** At about 431k parameters, Value-only gets KL 0.2017; full KVC at about 433k gets 0.1649 (about 22% better). The "40.6% fewer parameters" compares against a 726k model that scores the same 0.1652. Key-only (0.1996) is slightly better than Value-only.
- **"Never slower than baseline"** is contradicted by the measured 0.95x.
- **"Part 1 + Part 2 form one architecture"** does not hold: when every token is verified, upper-layer KV is computed exactly and KV repair is never used.
- The 1.4-1.96x speedup projections are unmeasured and use the incorrect formula above.

## 9. Literature map: how the field handles the missing-KV problem

| Approach | Examples | Pros | Cons |
|---|---|---|---|
| Fill with an approximation (copy, project, learned) | CALM, GateSkip, ADEPT, MimicNet, our KVC | Cheap, works on frozen models | Never exact; copying can cause repetition loops |
| Defer then recompute exactly | EE-LLM, AdaDecode (lossless, up to 1.73x), N-vium, LayerSkip | Exact cache | No FLOPs saved; gain from batching; N-vium needs pretraining from scratch |
| Avoid creating holes | SkipDecode (monotone exit depth), TIDE (runs all layers) | No hole | Little or no real saving |
| Run skipped layers cheaply | River-LLM (4-bit exit copies, training-free, 1.5-2.2x up to 8B) | KV close to exact | Tuned threshold per model; still runs the layers |
| Architecture removes the dependency | YOCO, LCKV, MoR | Hole disappears by design | Pretraining from scratch |
| Convert a pretrained model | SwiftKV (later layers' KV from one earlier state, distilled; prefill only) | Works on existing models | Prefill only; Qwen degrades more at 50% depth |

Key insight from the literature: a frozen backbone's upper layers were trained to read exact KV, so repair modules hit a ceiling. SwiftKV-style co-adaptation (training the layers to live with generated KV) is the most promising untested route, and a decode-time version combined with an exit-readout loss may be an open gap (not exhaustively searched).

Sources: [River-LLM](https://arxiv.org/abs/2604.18396), [ADEPT](https://arxiv.org/abs/2601.03700), [AdaDecode](https://arxiv.org/abs/2506.03700), [N-vium](https://arxiv.org/abs/2605.13190), [EE-LLM framework paper](https://arxiv.org/abs/2407.20272), [LayerSkip](https://aclanthology.org/2024.acl-long.681.pdf), [SwiftKV](https://arxiv.org/abs/2410.03960), [GateSkip](https://arxiv.org/abs/2510.13876), [Early Exit Is a Natural Capability](https://arxiv.org/abs/2412.01455), [MoR](https://arxiv.org/abs/2507.10524), [YOCO](https://arxiv.org/abs/2405.05254), [LCKV](https://arxiv.org/abs/2405.10637), [TIDE](https://arxiv.org/abs/2603.21365), [MimicNet](https://link.springer.com/chapter/10.1007/978-981-95-1233-1_7), [SkipDecode](https://arxiv.org/abs/2307.02628), [Kangaroo](https://arxiv.org/abs/2404.18911), [FlexiDepth](https://arxiv.org/abs/2503.23798).

Literature caveats: descriptions come from abstracts/HTML versions, not full appendices; several papers have internal numeric inconsistencies; FREE could not be fetched, so its description in the paper draft needs verifying.

## 10. Files

| File | What it is |
|---|---|
| `kv_continuity_merged.tex` | Full paper with the new section and patches merged in |
| `learned_kv_repair_section.tex` | The new section alone (3 tables inlined) |
| `vkr2.py` | v2 harness: Native/Direct/HSM/VKR/KVC, evaluation, rollouts, Gen-NLL |
| `vkr3.py` | v3 experiments (section 6); needs `vkr2.py` alongside |
| `PROJECT_MAP.md` | This file |

## 11. Success criteria for calling it "solved"

- **KV hole:** generation quality close to the full model (Gen-NLL and downstream tasks such as GSM8K, HumanEval) with no verification.
- **Readout:** top-1 agreement high enough (roughly 70-80% or more at 50-67% depth) that early exit pays off.
- **Both together:** measured wall-clock speedup above 1x on real tasks, on at least two model families and sizes, with seeds and confidence intervals.

Fallback if the main bet fails: the disentanglement protocol, the Key/Value error split, and the finding that teacher-forced KL gains do not translate to generation gains are publishable on their own and fit the current paper.