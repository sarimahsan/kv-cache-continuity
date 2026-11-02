# Paper A: KV-Cache Continuity and the Limits of Intermediate Representations in Depth-Truncated Transformers

## 1. Motivation

Depth-adaptive inference (early exit, layer skipping, mixture-of-depths) promises large
compute savings by letting easy tokens exit a Transformer before the final layer. Nearly
all prior work on this either (a) evaluates only end-to-end generation quality, which
conflates multiple possible failure modes into one number, or (b) assumes the KV-cache
"hole" left by an exited token is a solved or negligible problem.

This paper asks a narrower, more falsifiable question than "does early exit work":

> When a token exits early, does the resulting **key/value cache approximation**
> corrupt downstream attention, does the **intermediate-layer readout** fail to align
> with the terminal unembedding manifold, or both — and can either failure mode be
> fixed independently of the other?

Answering this cleanly required building a **teacher-forced, causally disentangled
ablation protocol** that isolates KV-cache corruption from readout mismatch as two
separate, independently measurable error sources — something not present in prior
early-exit literature (SkipDecode, MoD, HELIOS, TIDE, CALM all report only combined
end-to-end effects).

## 2. Core Contributions

1. **Native KV State Projection**: a training-free mechanism that fills the KV-cache
   hole left by an early-exiting token by applying the *skipped layer's own* projection
   weights and normalization to the exit hidden state:
   $$\hat{K}_l = W_K^{(l)}\,\mathrm{LN}_1^{(l)}(h_{d_t}), \quad \hat{V}_l = W_V^{(l)}\,\mathrm{LN}_1^{(l)}(h_{d_t}), \quad \forall l > d_t$$
   Requires zero base-model retraining.

2. **A causal disentanglement ablation protocol** (teacher-forced, three-condition):
   attention-only (pristine readout, projected KV), readout-only (pristine KV,
   intermediate readout), and a direct-copy negative control — cleanly separating
   $\mathcal{E}_{\text{attn}}$ from $\mathcal{E}_{\text{readout}}$.

3. **A precise, information-theoretic operating boundary** for when KV continuity holds:
   the 1-bit mutual-information gate $\bar{D}_{KL} \le \ln(2) \approx 0.6931$ nats,
   validated (not merely asserted) against real measurements at multiple depths.

4. **A clean negative result**: standalone hard early exit (reading the token's own
   output from an intermediate depth) does not work, even with a trained non-linear
   calibration adapter — and the paper explains mechanistically *why* (Value
   representations keep evolving through upper-layer MLPs while Key representations
   stabilize early, so attention routing survives truncation but semantic readout
   doesn't).

## 3. Key Findings So Far (Qwen2.5-3B, 36L, cross-validated on a 125M model)

**KV continuity mechanism (Native Projection) — validated for exit depth ≥ ~65% of network depth:**

| Exit depth | Relative depth | KL divergence (nats) | Top-1 match | vs. ln(2)=0.6931 bound |
|---|---|---|---|---|
| L=18 | 50% | 0.7335 | 70.9% | **fails** (explicit stress-test boundary) |
| L=24 | 67% | 0.3310 | 79.6% | passes, comfortable margin |
| L=29 | 81% | 0.0713 | 90.5% | passes decisively |

Direct-Copy negative control fails completely at every depth (near-zero cosine
similarity between reconstructed and true KV states — independently parameterized
layers produce orthogonal activation spaces), confirming Native Projection is doing
real coordinate alignment, not incidental noise reduction.

**Standalone hard early exit — closed negative result:**

Even after a 3,000-step 2-layer bottleneck MLP calibration adapter (r=512, GELU,
zero-init residual), teacher-forced readout PPL remains far above the full-depth
baseline (5.76) at every tested depth: 359.39 (L=18), 171.40 (L=24), 61.58 (L=29).
Top-1 agreement tops out at 44.2% even at 81% depth. An earlier internal claim of
"Exit-Head Readout: VALIDATED" was retracted once absolute (not just relative)
numbers were checked — a useful methodological cautionary note the paper can include.

## 4. What Would Make This Paper Stronger

Roughly in priority order:

1. **Cross-architecture generalization.** Everything above is Qwen2.5-3B (GQA) plus a
   125M dense model. Add at least one more model family/size (e.g., a Llama-family
   model, or a mid-size MoE) to show the ≥65%-depth boundary and the Key-stable/
   Value-decays asymmetry aren't Qwen-specific artifacts. This is the single biggest
   thing a reviewer will ask for.

2. **Explain *why* the 65% boundary sits where it does, not just that it does.**
   Right now the ≥65% threshold is empirical. A mechanistic story (e.g., tracking
   per-layer effective rank of Value representations, or measuring how many
   subsequent MLP blocks contribute non-trivial "semantic refinement" via
   ablation/patching) would turn an empirical cutoff into a predictive one —
   much stronger for reviewers, since it explains *why* the number is what it is
   rather than reporting it as a coincidence of this model.

3. **Push the Direct-Copy vs. Native-Projection comparison into a proper controlled
   experiment across the full depth sweep**, not just at 2-3 spot depths, so the
   "orthogonal activation space" claim has a continuous curve to point to, not
   three data points.

4. **Report variance, not just point estimates.** All KL/PPL/Top-1 numbers currently
   read as single measurements per depth. Even a handful of random seeds per
   condition (evaluation seed, not necessarily retraining) would let you report
   confidence intervals and pre-empt an obvious "how many sequences was this
   averaged over, and what's the spread?" question.

5. **Tie the negative result to a concrete downstream implication.** The paper
   currently ends the KV story with "hard early exit doesn't work, calibration
   doesn't fix it." Give the reader something actionable: e.g., an explicit
   recommendation for practitioners on what a decoder-adaptivity method should
   check before trusting KV-cache approximation (the disentanglement protocol
   itself, packaged as a diagnostic recipe others can reuse).

6. **Ablate the calibration adapter architecture itself**, not just its training
   budget. You went from linear (500 steps) to a 2-layer MLP (3,000 steps) and
   both plateaued far above baseline — is that a capacity ceiling or a training
   budget ceiling? A short experiment scaling adapter rank/depth further (even if
   it still fails) turns "we tried one bigger adapter and it wasn't enough" into
   "we swept adapter capacity and the failure is structural, not a training
   artifact" — a materially stronger negative-result claim.

7. **Release the disentanglement harness.** Given the protocol itself (three-way
   teacher-forced decomposition) is a reusable contribution independent of your
   specific findings, packaging it as a small released tool/script would make the
   paper more citable and testable by others — reviewers respond well to
   reproducibility artifacts, especially for ablation-heavy papers.

## 5. Related Work to Position Against

> **Caution:** an AI-drafted literature review pass on this project fabricated an
> experimental setup that does not exist — it misread the N=60 *prompt* benchmark as
> a 60-*layer* model and built a "depth dilemma" narrative around that error. That
> framing must never appear in the manuscript. The model in every experiment reported
> so far is Qwen2.5-3B (36 layers); valid draft depths are L=24 (67%) and L=29 (81%)
> of that 36-layer network. Every citation and claim below should still be
> independently verified before use — treat this list as leads, not settled facts.

- **CALM** (Schuster et al., 2022) — closest prior mechanism for state propagation;
  clarify precisely how Native Projection differs (per-layer weight reuse vs.
  CALM's approach) and cite explicitly.
- **SkipDecode** (Del Corro et al., 2023) — used as your uniform-depth quality baseline;
  keep this comparison but make sure the comparison is depth-matched/iso-FLOP, as you
  already corrected for in the systems side of this project.
- **Mixture-of-Depths** (Raposo et al., 2024) — cite for token-level adaptivity framing,
  note structural incompatibility with monotonic exit as already established.
- **Tuned Lens / Logit Lens literature** — your calibration adapter is a Tuned-Lens-style
  mechanism; cite the original Tuned Lens work explicitly if not already done, since
  reviewers in interpretability will expect it.
- **Keys-stable / Values-evolving framing** — a promising, worth-verifying connection:
  Key stability across depth plausibly links to induction-head / in-context-learning
  literature (Olsson et al., 2022 — attention "routing" forming early), while
  continued Value evolution plausibly links to the "Transformer FFNs as key-value
  memories" framing (Geva et al., 2022 — FFNs keep writing into the residual stream
  at every layer). Verify both papers actually support this before citing.
- **DeeBERT / FastBERT / BranchyNet** — classic early-exit-as-classification baselines;
  useful contrast for framing Native Projection as built for *generation continuity*
  rather than an isolated per-layer classification head.
- **H2O (Zhang et al., 2023), StreamingLLM (Xiao et al., 2023)** — adjacent KV-cache
  literature (token eviction, not depth truncation); reasonable to cite as related-but-
  distinct problems.
- **EAGLE, Medusa** — modern speculative decoding methods that use external/distilled
  draft models rather than same-model layer truncation; useful for framing *why* the
  field moved away from naive self-speculation, and ties directly into the "is this
  worth pursuing over a real small draft model" question already raised for this
  project. Verify exact mechanisms before citing.
- **Gromov et al. (2024)** on hidden-state geometry across depth — a plausible reference
  for discussing why intermediate layers occupy a distinct manifold from the terminal
  layer; verify it says what's being attributed to it before relying on it.

## 6. Suggested Paper Structure

1. Introduction — motivate the disentanglement question directly (not "early exit is
   good, here's our method" — lead with "prior work can't tell you *why* early exit
   generation degrades")
2. Related Work
3. Native KV State Projection (mechanism)
4. Causal Disentanglement Protocol (methodology — this is a contribution in itself)
5. Results: KV continuity boundary (attention-only sweep)
6. Results: standalone readout is non-viable (readout-only sweep, calibration attempts)
7. Mechanistic discussion: Key vs. Value stability across depth
8. Discussion: implications for future early-exit / adaptive-depth work
9. Limitations (single-family generalization, adapter capacity ceiling not fully
   characterized, etc. — address the "what would make it stronger" points above
   here directly, even for whichever ones you don't have time to run)

## 7. Current Status

Core empirical results (Sections 5-6 above) are essentially done. Main remaining work
before this is submission-ready is item #1 above (cross-architecture check) and #2
(mechanistic explanation of the 65% boundary) — both are the kind of thing reviewers
ask for in a first-round revision if not addressed pre-emptively.