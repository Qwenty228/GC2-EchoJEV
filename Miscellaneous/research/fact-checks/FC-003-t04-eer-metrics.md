---
id: FC-003-t04-eer-metrics
task: T04
claim_by: T04 implementation agent (metrics.py claim block), original task spec in tasks/TASKS.md
verdict: contradicted
checked_on: 2026-09-20
---

# FC-003 — T04's original task spec required EER interpolation and stated the wrong score polarity, both contradicted by the official ASVspoof scoring code

## Claim as stated
> (original `tasks/TASKS.md` T04 spec) "EER is the primary anti-spoofing metric: find where
> FAR and FRR cross, interpolating between the bracketing operating points. Document the
> convention (higher score = spoof) in the docstring and enforce it with an argument, not a
> comment." … "`per_attack_eer` scores each spoof generator against the shared bonafide
> pool — not against a subset — and returns a macro mean alongside the per-attack dict."
> Claim: "the EER interpolation and the per-attack bonafide-pooling convention are correct."

## Evidence examined
- knowledge: `research/knowledge/methods/eer-and-metrics.md` (this researcher's note),
  citing the ASVspoof 2019 Evaluation Plan §5.2 + footnote 9, and the official
  `eval_metrics.py` (`asvspoof-challenge/2021/eval-package/eval_metrics.py`,
  `compute_det_curve`/`compute_eer`), fetched and read verbatim in that note.
- independent re-verification (performed by the coordinator, per its message): re-fetched
  `eval-package/eval_metrics.py` directly and confirmed verbatim the same two facts —
  `argmin(|frr-far|)` then `mean(frr[i], far[i])` with no interpolation, and *"higher
  detection scores values are assumed to indicate stronger support for the bona fide
  hypothesis."*
- code (current, corrected, verified in this pass): `src/audio_df/metrics.py:1-70` — module
  docstring now states the official (non-interpolated) algorithm explicitly, quotes the same
  evaluation-plan footnote, requires a `higher_is_spoof: bool` argument with no default
  (rather than hardcoding either polarity or inferring it from label order), and implements
  `per_attack_eer` against the full shared bonafide pool.
- code: `src/audio_df/data.py` / `configs/features.yaml` — no dependency on this claim beyond
  what's already covered in FC-004 / `asvspoof-2019-la.md`.

## Verdict
**Contradicted**, on two independent points, in the original task specification (not the
implementing agent's own invention — `tasks/TASKS.md` itself stated both incorrect
conventions, and the implementing agent's claim block asserted conformance to that incorrect
spec). Both defects have since been corrected at the task-spec level and in the shipped code,
and are now **confirmed** against the official source.

1. **EER interpolation — contradicted.** The official ASVspoof scoring package
   (`eval_metrics.py`, imported into the 2021 eval-package "from eval_metrics.py in official
   t-DCF computation package") does not interpolate between bracketing operating points. It
   takes the single index minimising `|FRR - FAR|` over the exact, fully-enumerated empirical
   DET curve, and reports the **mean of FRR and FAR at that one index** — a documented
   simplification, not an oversight: the ASVspoof 2019 Evaluation Plan's own footnote 9 states
   plainly, "You may use θ_EER = argmin_θ |P_fa(θ) − P_miss(θ)| or more advanced methods such
   as EER based on convex hull (ROCCH-EER)... In ASVspoof2019, we use a simplified
   nearest-neighbor type computation demonstrated in the scoring package." A correctly
   interpolated EER is a *different*, more accurate estimator than what ASVspoof scoring
   actually reports; the two will diverge measurably on small score sets (exactly the kind of
   4-point hand-computed test case the task's own test plan specifies), and converge only in
   the large-sample limit.

2. **Score polarity — contradicted.** The original task text asserted "higher score = spoof."
   Both the evaluation plan prose (§5, p.8: "high detection score should indicate bona fide
   and low detection score should indicate spoofing attack") and the official
   `eval_metrics.py` docstring ("higher detection scores values are assumed to indicate
   stronger support for the bona fide hypothesis") state the opposite: **higher score =
   bonafide, lower score = spoof.** Had this been implemented as stated, any EER/AUC computed
   from correctly-signed classifier scores would be `1 - EER` (equivalently, `1 - AUC`)
   relative to the official convention — a quiet, hard-to-notice inversion for a
   near-chance detector (flips ~0.45 to ~0.55, which just looks like "a somewhat weak model"),
   though it would be caught quickly for a strong detector (flips ~0.03 to ~0.97, which looks
   broken).

3. **Per-attack bonafide-pooling convention — confirmed**, no correction needed. This part of
   the original spec matches the official-style breakdown exactly, verified directly against
   `clovaai/aasist/evaluation.py`, which builds `bona_cm` once from the entire bonafide score
   set and reuses it unchanged across every attack in the per-attack loop — see
   `research/knowledge/datasets/asvspoof-2019-la.md` for the exact code excerpt.

## Reasoning
Both defects trace to the task specification itself asserting a plausible-sounding but
unverified convention ("interpolate for a precise crossing" and "higher = spoof, since spoof
scores are usually treated as the target class in some framings") rather than the actual
documented ASVspoof convention, which is counter-intuitive on both points relative to a naive
first guess (a from-scratch implementer might reasonably assume interpolation is more
"correct" and that the positive/spoof class gets the high score, since spoof detection is
often framed as `p(spoof|x)`). This is precisely why T04's claim ("matches the official EER
definition") needed to be checked against the primary source rather than accepted on the
strength of a clear, confident-sounding docstring — the task spec and the implementation were
both internally consistent with each other, and both wrong relative to the actual official
scoring code.

## Required action
Already applied: `tasks/TASKS.md` T04 corrected at source (per the coordinator), and
`src/audio_df/metrics.py` reimplemented to vendor/match the official nearest-neighbour
algorithm exactly, with a conformance test against the official code as the executable spec
(per the coordinator's message). No further action required from this fact-check. Recommend
the conformance test explicitly include the exact 4-point hand-computed case from the task's
own test plan, computed both ways (interpolated vs. official nearest-neighbour), so the two
conventions' numeric divergence is visible in the test suite itself rather than only in this
note.
