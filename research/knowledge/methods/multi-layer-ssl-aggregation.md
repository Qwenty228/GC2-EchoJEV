---
id: multi-layer-ssl-aggregation
type: method
sources:
  - configs/ssl_probe.yaml (this repo — the already-specified learned-softmax layer weighting baseline)
  - https://dl.acm.org/doi/10.1145/3664647.3681345 (Zhang, Wen, Hu, "Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier", ACM MM 2024 — abstract/listing page only, full text 403-blocked)
  - https://github.com/QiShanZhang/SLSforASVspoof-2021-DF (official code repo for the SLS paper — README fetched; reports EER numbers, does not document the mechanism in prose)
  - https://arxiv.org/html/2509.03409 (Tran et al., "Multi-level SSL Feature Gating for Audio Deepfake Detection" — fetched; cites SLS in passing while proposing its own gating method, secondary source for what SLS does)
  - DOI 10.1016/j.csl.2026.102044 (Meng, Li et al., "MSLA-XLS-R: Enhancing Hierarchical SSL Representations for Audio Deepfake Detection", Computer Speech & Language, in press 2026 — title/DOI/venue only, abstract paywalled and not reached)
confidence: low
last_verified: 2026-09-21
related: [ssl-layer-probing]
---

# Multi-layer SSL aggregation beyond learned-softmax: SLS and MSLA-XLS-R

## Claim in one sentence
Both **SLS** (Zhang et al., ACM MM 2024) and **MSLA-XLS-R** (Meng et al., Comp. Speech & Lang.
2026) are real, published, citable methods that go beyond the single learned-softmax layer
weighting already specified in `configs/ssl_probe.yaml`, but I could not obtain enough primary
text to describe either mechanism precisely — SLS's own paper (ACM Digital Library) is
paywalled/403-blocked and its official code repo's README states results without documenting
the mechanism in prose; MSLA-XLS-R's full text is paywalled on ScienceDirect and I could reach
only its title, authors, venue and DOI. This note is therefore **low confidence** and should
not be treated as settling the "is it worth scheduling" question on its own.

## Findings

### What I could confirm about SLS (Zhang, Wen, Hu, ACM MM 2024)
- Title: "Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier." Venue:
  Proceedings of the 32nd ACM International Conference on Multimedia, pages 6765–6773, 2024.
  DOI 10.1145/3664647.3681345.
- "SLS" = Sensitive Layer Selection. Per a secondary source (Tran et al. 2509.03409, which
  cites SLS while building its own alternative), SLS performs "layer selection for Transformer
  encoders" via "dynamic channel-wise recalibration of feature maps" — this phrase is the
  clearest mechanistic description I could obtain, and it is **second-hand**, not from the SLS
  paper itself. It suggests SLS does more than a single scalar weight per layer (as in the
  softmax-over-layers baseline already specified in `configs/ssl_probe.yaml`): "channel-wise"
  implies per-feature-dimension gating within and/or across layers, closer to a learned gating/
  attention mechanism than a single scalar mixing coefficient. I could not confirm this from
  the primary paper and it should be treated as a lead, not a fact.
- The official code repo (`QiShanZhang/SLSforASVspoof-2021-DF`) reports, without describing the
  training/architecture prose in the README: ASVspoof2021 DF EER **1.92%**, ASVspoof2021 LA EER
  **2.87%**, In-the-Wild EER **7.46%** (tagged `source: literature` if ever quoted downstream —
  these are the paper authors' own numbers on their own splits, not reproduced by us).
  ASVspoof2019 LA is not reported in either source I reached.
- Whether the underlying XLS-R backbone is frozen or fine-tuned in SLS is **not confirmed** —
  the README references downloading pretrained XLS-R 300M but does not state its trainable
  status.

### What I could confirm about MSLA-XLS-R (Meng, Tang, Sun, Deng, Li, Li)
- Title: "MSLA-XLS-R: Enhancing Hierarchical SSL Representations for Audio Deepfake Detection."
  Venue: Computer Speech & Language, DOI 10.1016/j.csl.2026.102044. Search-engine metadata
  places it "in press"/2026, consistent with `concern.md`'s framing of MSLA-XLS-R as a named,
  real, contemporary-but-not-yet-fully-established method.
- "MSLA" almost certainly expands to **multi-scale layer aggregation** based on the title and
  on adjacent-paper conventions (multiple *other* 2024–2025 papers in this space, e.g. "A
  Spoofing Speech Detection Method Combining Multi-Scale Features and Cross-Layer Information,"
  independently propose combining features at multiple temporal/representational scales across
  transformer layers, which is the same conceptual family) — but I could not fetch MSLA-XLS-R's
  own abstract (ScienceDirect returned HTTP 403) to confirm this is accurate for this specific
  paper rather than inferred from the surrounding literature. **This expansion is not verified
  against the primary source and should not be repeated as fact without a second attempt at the
  paper itself** (e.g. via institutional access, a preprint mirror, or Google Scholar cache).

### What our own spec already does (for comparison, this repo, verified — not literature)
`configs/ssl_probe.yaml` already specifies exactly the mechanism the handoff calls for at
Stage 8: `alpha_l = softmax(a_l)`, `H = sum_l alpha_l H^(l)` — one trainable scalar per layer,
shared across all channels/time-steps/feature-dimensions of that layer, with XLS-R itself
frozen. This is the simplest possible "combine information across layers" mechanism: it can
express "layer 12 matters 3x as much as layer 4" but cannot express "dimension 47 of layer 12
matters differently from dimension 900 of layer 12," and cannot express per-utterance,
input-dependent layer weighting (the weights `a_l` are learned once, globally, not predicted
per input).

## Mechanism / derivation
Not fully available for SLS or MSLA-XLS-R — see confidence caveat above. What can be stated
precisely, because it is our own configuration rather than a literature claim: the baseline
`alpha_l = softmax(a_l)`, `H = sum_l alpha_l H^(l)` is a **single global convex combination**
of per-layer utterance-pooled (or frame-level, pre-pooling) representations, with `L` trainable
scalars total (`L` = number of XLS-R layers, e.g. 25 for the 300M checkpoint including the
embedding layer). Any method that is "more sophisticated" than this must, by construction, add
either (a) per-channel/per-dimension granularity, (b) input-conditioning (the weights depend on
the utterance, not just the layer index), or (c) a learned nonlinear combination rather than a
convex sum — the secondary-source description of SLS ("channel-wise recalibration") is
consistent with (a), but this is not confirmed against SLS's own text.

## Implications for us

- **Do not schedule SLS or MSLA-XLS-R replication as a Wave-level task on the strength of what
  I found here.** The evidence available to me is not strong enough to write an implementation
  spec: I do not know SLS's precise architecture, and I do not have MSLA-XLS-R's abstract at
  all. Scheduling either would mean an implementer guessing at a mechanism from secondary
  sources, which contradicts the project's "cite or delete" discipline as much as scheduling a
  benchmark model without knowing its architecture would.
- **The simple learned-softmax weighting already in `configs/ssl_probe.yaml` remains the
  correct starting point**, and is explicitly what `concern.md` §6 itself says to do first
  ("begin with the simplest interpretable experiment"). SLS/MSLA-style aggregation is,
  on the evidence available, a **later refinement gated on T22 showing the simple weighting
  already helps** (per `tasks/BACKLOG.md` T22's own gate) — not a replacement for it, and not
  something to build in parallel with it. This matches the backlog's existing sequencing; I
  found nothing in this research pass that should change that sequencing.
- If this project later wants to attempt an SLS-style refinement, the actionable next research
  step is **not** more web search — it is obtaining the actual PDF (e.g. through a university
  library proxy, since GC2 is an NUS project and NUS likely has ACM DL / ScienceDirect access
  that this agent's tools do not) before any implementer is asked to reproduce it.

## Open questions / what would change this
- Getting past the ACM DL 403 (SLS) and ScienceDirect 403 (MSLA-XLS-R) paywalls — via
  institutional access, an author's personal page/preprint, or a cached HTML version — would
  directly raise this note's confidence from `low` to at least `medium`, and is the single
  highest-value next step if either method is to be taken further.
- The one open, concrete mechanistic distinction worth re-checking once primary text is
  available: whether SLS's layer combination is per-channel (plausible per the secondary
  source) or per-time-frame (also plausible, and a different, complementary kind of
  sophistication) — these have different implementation costs and different implications for
  what "more sophisticated than learned-softmax" actually buys.
