---
id: eer-and-metrics
type: method
sources:
  - https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf (§5.2 "Secondary metric: equal error rate (EER)", p.11, and footnote 9)
  - https://raw.githubusercontent.com/asvspoof-challenge/2021/main/eval-package/eval_metrics.py (official `compute_eer`/`compute_det_curve`, fetched raw and read verbatim)
  - https://github.com/clovaai/aasist/blob/main/evaluation.py (official-style per-attack EER breakdown, corroborates bonafide-pooling convention)
confidence: high
last_verified: 2026-09-20
related: [asvspoof-2019-la, asvspoof-2021]
---

# EER and t-DCF as used in official ASVspoof scoring

## Claim in one sentence
The official ASVspoof scoring code computes EER **without interpolation** — it takes the
operating point where `|FRR − FAR|` is smallest and reports the mean of FRR and FAR *at that
discrete point*, not a crossing found by interpolating between bracketing points; the score
polarity convention is **higher score = bonafide, lower score = spoof**; per-attack EER pools
each attack's spoof scores against the full/shared bonafide score set; t-DCF, not EER, is the
official *primary* metric for ASVspoof 2019, with EER kept only as a secondary,
backward-compatible metric.

## Findings

### Score polarity (Evaluation Plan §5, p.8, and `eval_metrics.py` docstring)
Evaluation plan, verbatim: *"high detection score should indicate bona fide and low detection
score should indicate spoofing attack"* — stated explicitly as a continuation of the
convention from the 2015/2017 challenge editions. The official t-DCF/EER code
(`eval_metrics.py`, function `compute_tDCF`, NOTE block) states the same thing independently:
*"In relative terms, higher detection scores values are assumed to indicate stronger support
for the bona fide hypothesis."* Two independent primary sources agree: **bonafide is the
positive/high-score class, spoof is the negative/low-score class.**

### Official EER definition and the interpolation question (CRITICAL)
Evaluation plan §5.2 (p.11): *"EER corresponds to CM threshold s_EER at which the miss and
false alarm rates equal each other, i.e. EER = P_fa^cm(s_EER) = P_miss^cm(s_EER)."*
Footnote 9 on the same page, verbatim: *"One may not find such threshold exactly as P_fa(s)
and P_miss(s) change in discrete steps. You may use
θ_EER = argmin_θ |P_fa^cm(θ) − P_miss^cm(θ)| or more advanced methods such as EER based on
convex hull (ROCCH-EER) implemented in the Bosaris toolkit ...**In ASVspoof2019, we use a
simplified nearest-neighbor type computation demonstrated in the scoring package.**"*

The actual scoring package (`eval_metrics.py`, imported "from eval_metrics.py in official
t-DCF computation package `https://www.asvspoof.org/resources/tDCF_python_v2.zip`") implements
exactly the nearest-neighbour method the footnote describes, with **no interpolation
whatsoever**:

```python
def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]
```

Reading this precisely: `target_scores` = bonafide scores (label 1, i.e. the "positive"/high
class per the polarity convention above), `nontarget_scores` = spoof scores (label 0). The
DET curve is built by sorting **all** scores ascending and walking a cumulative count — this
is a full, exact empirical DET curve at every observed score value, not a coarse grid. The
EER is then: find the single index `i` on this curve minimising `|FRR[i] − FAR[i]|`, and
report `eer = mean(FRR[i], FAR[i])` — the **average of the two nearest empirical error rates
at that index**, not a value obtained by linearly interpolating between the bracketing points
where FRR crosses FAR. This is a deliberate, documented simplification (the footnote's own
words: "simplified nearest-neighbor type computation"), chosen over more accurate
interpolation/convex-hull methods (ROCCH-EER) for backward-compatibility/simplicity with
prior ASVspoof editions.

### Per-attack EER — shared bonafide pool (confirmed, not merely assumed)
`aasist/evaluation.py` (a widely used, ASVspoof-baseline-style implementation) computes the
per-attack breakdown as:
```python
bona_cm = cm_scores[cm_keys == 'bonafide']                                  # ALL bonafide scores
spoof_cm_breakdown = {atk: cm_scores[cm_sources == atk] for atk in attack_types}
eer_cm_breakdown = {atk: compute_eer(bona_cm, spoof_cm_breakdown[atk])[0] for atk in attack_types}
```
`bona_cm` is built once, from the full bonafide set, and reused unchanged across every
`atk` in the per-attack loop. Each attack's EER is therefore that attack's spoof scores
against the entire shared bonafide pool — **confirmed**, not a subset partitioned by attack.
This matches the handoff's and the configs' assumption exactly.

### t-DCF: what it is and why EER is used as primary here anyway
Evaluation plan §5 (p.7) and §5.1 (p.8–10): the **tandem detection cost function (t-DCF)** is
the official *primary* metric for ASVspoof 2019, adopted specifically because ASVspoof 2019
"aligns more closely with the field of automatic speaker verification" — it scores a CM as a
gate placed in front of a fixed ASV system, combining CM miss/false-alarm rates with a
*given* ASV system's own error rates and a set of application-motivated cost/prior parameters
(`C_miss`, `C_fa`, `π_tar`, `π_non`, `π_spoof`; Table 2 values: `π_tar=0.9405, π_non=0.0095,
π_spoof=0.05`, all miss/fa costs = 1/10). The normalized minimum t-DCF,
`t-DCF_norm^min = t-DCF_norm(s*)` with `s* = argmin_s t-DCF_norm(s)`, is the official
challenge-ranking metric. EER is retained only as a **secondary** metric explicitly "to
observe the correlation between the new metric and the equal error rate (EER) metric used in
previous editions, and ... to support applications beyond ASV" (footnote 1, p.2) — i.e. EER
is application-independent (no CM+ASV tandem, no ASV system needed), which is exactly why a
project without a paired/fixed ASV system (like this workshop) would use EER as its *primary*
metric: t-DCF cannot be computed at all without a fixed, organizer-provided ASV system and
its scores, which this project's `configs/`/`tasks/` do not appear to include. EER is the
metric that is actually computable end-to-end from CM scores alone.

## Mechanism / derivation
Given bonafide scores `B = {b_i}` and spoof scores `S = {s_j}`, with the polarity convention
"higher = bonafide":
1. Sort all scores ascending; at each candidate threshold `θ` (taken from the observed score
   values plus one point below the minimum), compute
   `FRR(θ) = P(bonafide scored ≤ θ) = |{b_i ≤ θ}| / |B|` (bonafide wrongly rejected) and
   `FAR(θ) = P(spoof scored > θ) = |{s_j > θ}| / |S|` (spoof wrongly accepted).
2. `EER_threshold_index = argmin_θ |FRR(θ) − FAR(θ)|`.
3. `EER = mean(FRR(θ*), FAR(θ*))` at that index — no interpolation between bracketing
   thresholds where the curves would exactly cross.

## Implications for us

- **`tasks/TASKS.md` T04 explicitly instructs**: *"find where FAR and FRR cross,
  interpolating between the bracketing operating points."* **This contradicts the official
  ASVspoof scoring convention**, which — per the evaluation plan's own footnote 9 and the
  actual `eval_metrics.py` code — deliberately uses a **non-interpolated, nearest-neighbour**
  EER (`argmin |FRR-FAR|`, then average FRR/FAR at that single index). If T04's
  `compute_eer` implements true linear interpolation between bracketing points, its numeric
  output will differ (usually negligibly on large eval sets with many operating points, but
  non-negligibly on small dev sets or synthetic test cases with few points — exactly the kind
  of 4-point hand-computed test case T04's own test plan specifies) from what the official
  ASVspoof tooling would report on identical scores. **This is the single most
  actionable finding in this note**: T04 should either (a) match the official
  nearest-neighbour convention exactly if "matches official ASVspoof scoring" is the goal, or
  (b) explicitly document that it deliberately deviates by using interpolation for smoother/
  more precise EER estimates, and never claim its number is "the official EER" without that
  caveat. The two conventions will diverge measurably on the hand-computed 4-point test case
  T04's own test plan calls for — that test should be checked against both conventions before
  trusting either as "correct."
- T04's stated polarity convention ("higher score = spoof", per `tasks/TASKS.md` line 92) is
  **backwards relative to the official convention** confirmed here twice (evaluation plan
  prose + `eval_metrics.py` docstring): the official convention is **higher score = bonafide,
  lower score = spoof**. If T04's implementation encodes "higher = spoof" as a hardcoded
  assumption rather than an argument, its output will be `1 - EER` relative to official
  scoring on correctly-signed classifier scores, and any code path assuming sklearn-style
  `roc_auc_score(y, scores)` with `y=1` for spoof would also invert the AUC's implied
  direction versus the official convention. T04 does say "enforce it with an argument, not a
  comment," which is the right instinct — but the polarity stated in the task text itself is
  wrong and should be corrected to match the official source before implementation is
  fact-checked.
- The per-attack EER / shared-bonafide-pool convention in T04 and `configs/features.yaml`
  (implicitly) is **confirmed correct** by the `aasist/evaluation.py` excerpt above — no
  correction needed there.
- Because this project has no paired ASV system and no organizer-provided ASV scores,
  reporting t-DCF is not feasible; EER-as-primary is the historically and mechanically
  correct choice for this workshop, not merely a simplification — this should be stated
  explicitly in the final report's metrics section, citing this note.

## Open questions / what would change this
- I read the *2019* evaluation plan and the *2021* eval-package's `eval_metrics.py`; the
  2021 package explicitly says it re-imports from the same official t-DCF package used in
  2019, so I treat the EER algorithm as unchanged between 2019 and 2021 — this is stated by
  the file's own docstring, not independently verified against a 2019-only released
  `eval_metrics.py`, but I judge this low-risk given the exact wording match with the 2019
  evaluation plan's footnote.
- I did not fetch the ROCCH-EER (Bosaris toolkit) definition; if the project ever wants a
  more precise EER for reporting alongside the official one, ROCCH-EER is the named
  alternative the evaluation plan itself points to.
