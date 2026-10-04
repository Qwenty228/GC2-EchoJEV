"""EER and companion scoring metrics for spoofing/deepfake detection.

This module answers "how good are these scores at separating bonafide from spoof",
nothing more. It does not choose an operating threshold from evaluation data —
`prf1_at_threshold` takes a threshold as a required argument, and picking that
threshold from dev-set statistics is task T11's job, not this module's. Calling
any function here with eval-derived thresholds baked in silently would be a
methodology bug, not a metrics bug.

## Score polarity

Every public function here takes `higher_is_spoof: bool` as a required, positional-
or-keyword argument with **no default**. There is no "reasonable default" for score
polarity: a model can be scored either way (probability-of-spoof, or a log-likelihood
ratio that is high for bonafide), and getting the direction backwards silently turns a
good detector's EER into `1 - EER`. For a strong detector (EER ~0.03) that flips to
~0.97, which looks like a broken pipeline and gets caught. For a mediocre detector
(EER ~0.45) it flips to ~0.55 and looks like a slightly-worse-than-chance result that
is easy to shrug off as "the model isn't very good" instead of "the polarity is wrong".
Making the argument required (not a module-level constant, not inferred from label
order) is the guard against that failure mode.

**The ASVspoof/official convention, and this project's default orientation, is
`higher_is_spoof=False`** (higher score = bonafide), per the 2019 evaluation plan
§5 p.8 ("high detection score should indicate bona fide") and the official
`eval_metrics.py` docstring ("higher detection score values are assumed to indicate
stronger support for the bona fide hypothesis") — see
`research/knowledge/methods/eer-and-metrics.md`. This module does not hardcode that
direction anywhere; every function still requires the caller to state it explicitly
for their own scores. The note above is only about which orientation to expect from
models trained/scored the ASVspoof way (e.g. LLR-style CM scores), so a caller isn't
surprised that "positive-facing" scores in this literature usually mean bonafide.

## EER convention

**This module follows the official ASVspoof scoring convention exactly, per
`research/knowledge/methods/eer-and-metrics.md`** (which cites the ASVspoof 2019
evaluation plan §5.2 footnote 9 and the official `eval_metrics.py` scoring package
verbatim). That note supersedes an earlier draft of this task which called for
linear interpolation between bracketing operating points — the official tooling
deliberately does **not** interpolate:

> "One may not find such threshold exactly as P_fa(s) and P_miss(s) change in
> discrete steps. You may use θ_EER = argmin_θ |P_fa(θ) − P_miss(θ)| ... In
> ASVspoof2019, we use a simplified nearest-neighbor type computation."

Concretely, with bonafide treated as the "target"/high-score class in the official
convention (see `compute_eer`'s handling of `higher_is_spoof` for how this project's
required polarity argument maps onto that convention without hardcoding a direction):

1. Sort all scores ascending. At every observed score value (plus one synthetic
   point just below the global minimum, matching the official code's
   `thresholds[0] = min(scores) - 0.001` so that the very first operating point has
   FRR=0, FAR=1 exactly), compute the exact empirical
   `FRR(t) = P(bonafide scored <= t)` (a genuine utterance wrongly rejected) and
   `FAR(t) = P(spoof scored > t)` (a spoofed utterance wrongly accepted).
2. `EER = mean(FRR(t*), FAR(t*))` where `t* = argmin_t |FRR(t) - FAR(t)|` — the
   average of the two nearest empirical error rates at the single best observed
   operating point. **This is deliberately not an interpolated crossing value.**
   On a small hand-computable score set this can differ measurably from an
   interpolated EER; on a large eval set with many operating points the two
   converge in practice.
3. The returned threshold is the actual observed (or one-below-minimum synthetic)
   score value at that operating point, not an interpolated value.

We match the official algorithm's variable roles (`target_scores` = bonafide,
`nontarget_scores` = spoof) via `_compute_det_curve`, which is a direct translation
of the official `compute_det_curve`/`compute_eer` from
`https://raw.githubusercontent.com/asvspoof-challenge/2021/main/eval-package/eval_metrics.py`.
Polarity is handled by sign-flipping the scores fed into that routine (see
`compute_eer`), never by hardcoding which direction "spoof" points.

## Per-attack EER convention

`per_attack_eer` scores every attack/generator against the **entire shared bonafide
pool**, never a per-attack subset of bonafide utterances. Splitting the bonafide pool
per attack would (a) shrink the negative sample for each attack for no reason, since
bonafide utterances are not attack-specific, and (b) make per-attack EERs incomparable
because they'd be computed against different negative distributions. The function
returns the per-attack EER dict, the unweighted (macro) mean across attacks, and the
pooled EER (all spoof attacks combined vs. the shared bonafide pool) for reference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

__all__ = [
    "compute_eer",
    "compute_eer_interpolated",
    "roc_auc",
    "pr_auc",
    "prf1_at_threshold",
    "brier_score",
    "per_attack_eer",
    "min_dcf",
]

_BONAFIDE_STRINGS = {"bonafide", "bona-fide", "genuine", "real"}
_SPOOF_STRINGS = {"spoof", "fake", "synthetic"}


def _normalize_labels(labels: Sequence) -> np.ndarray:
    """Map heterogeneous labels to a boolean `is_spoof` array.

    Accepts, per element:
      - strings `"bonafide"`/`"spoof"` (and a few synonyms), case-insensitive
      - binary `0`/`1` (int, float, numpy scalar, or bool), where `1` means spoof

    Anything else — a third label, a non-binary number, `NaN`, `None` — raises
    `ValueError` immediately. Silently coercing an ambiguous label into a boolean
    is exactly the kind of invisible bug this module exists to prevent.
    """
    labels = list(labels)
    if len(labels) == 0:
        raise ValueError("labels must not be empty")

    out = np.empty(len(labels), dtype=bool)
    for i, raw in enumerate(labels):
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in _BONAFIDE_STRINGS:
                out[i] = False
            elif low in _SPOOF_STRINGS:
                out[i] = True
            else:
                raise ValueError(
                    f"unrecognised string label {raw!r} at index {i}; "
                    f"expected one of {_BONAFIDE_STRINGS | _SPOOF_STRINGS}"
                )
        elif isinstance(raw, (bool, np.bool_)):
            out[i] = bool(raw)
        elif isinstance(raw, (int, np.integer, float, np.floating)):
            if raw == 0:
                out[i] = False
            elif raw == 1:
                out[i] = True
            else:
                raise ValueError(
                    f"ambiguous numeric label {raw!r} at index {i}; "
                    "only binary 0 (bonafide) / 1 (spoof) is accepted"
                )
        else:
            raise ValueError(f"unrecognised label type {type(raw)!r} at index {i}: {raw!r}")
    return out


def _validate_scores_and_labels(
    scores: Sequence, labels: Sequence
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise labels, coerce scores to float array, and enforce two-class input."""
    scores_arr = np.asarray(list(scores), dtype=float)
    is_spoof = _normalize_labels(labels)
    if scores_arr.shape[0] != is_spoof.shape[0]:
        raise ValueError(
            f"scores and labels must have the same length; got {scores_arr.shape[0]} "
            f"and {is_spoof.shape[0]}"
        )
    if scores_arr.size == 0:
        raise ValueError("scores/labels must not be empty")
    if np.any(np.isnan(scores_arr)):
        raise ValueError("scores contain NaN")
    n_bonafide = int(np.sum(~is_spoof))
    n_spoof = int(np.sum(is_spoof))
    if n_bonafide == 0 or n_spoof == 0:
        raise ValueError(
            f"need both classes present to compute a two-class metric; "
            f"got {n_bonafide} bonafide and {n_spoof} spoof"
        )
    return scores_arr, is_spoof


def _compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Direct translation of the official ASVspoof `compute_det_curve`.

    Source (fetched and read verbatim):
    https://raw.githubusercontent.com/asvspoof-challenge/2021/main/eval-package/eval_metrics.py

    `target_scores` is the "higher = this class" class under the official convention
    (bonafide, once `compute_eer` has oriented the scores so higher means bonafide).
    `nontarget_scores` is the other class (spoof).

    `frr[i]` (official name; in the ASVspoof countermeasure convention this is the miss
    rate P_miss — the fraction of *bonafide* scored at or below threshold `thresholds[i]`,
    i.e. wrongly rejected as spoof) and `far[i]` (official name; the false-alarm rate
    P_fa — the fraction of *spoof* scored above `thresholds[i]`, i.e. wrongly accepted as
    bonafide) are returned exactly as the official code computes
    them, including the stable mergesort (tie-handling matters here) and the
    synthetic first threshold `min(all_scores) - 0.001` with `frr[0]=0`, `far[0]=1`.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Stable sort matters: it fixes how exact ties between target/nontarget scores
    # are ordered, which the official algorithm (and this conformance-matching
    # translation) relies on for reproducible tie-breaking.
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )
    return frr, far, thresholds


def compute_eer(scores: Sequence, labels: Sequence, higher_is_spoof: bool) -> tuple[float, float]:
    """Equal error rate and its operating threshold, matching official ASVspoof scoring.

    Returns `(eer, threshold)` using the **official, non-interpolated** algorithm: the
    empirical DET curve is built by sorting every observed score (official
    `compute_det_curve`, stable mergesort, exact cumulative counts — see
    `_compute_det_curve`), then `eer = mean(frr, far)` at the single index minimising
    `|frr - far|`. There is no interpolation between bracketing points; see the
    module docstring for why this matches `research/knowledge/methods/eer-and-metrics.md`
    and the official evaluation-plan footnote. Use `compute_eer_interpolated` if you
    specifically want a smoother, non-official interpolated estimate for internal
    analysis — never for numbers you intend to compare against official baselines.

    The official convention scores bonafide as the high/"target" class
    (`higher_is_spoof=False`). This function honours whatever `higher_is_spoof` you
    pass rather than assuming that convention: when `higher_is_spoof=True`, scores
    are sign-flipped before being handed to `_compute_det_curve` as
    `target=-bonafide, nontarget=-spoof` (so "higher" becomes "bonafide" internally,
    matching the official code's assumption), and the returned threshold is flipped
    back to the original scale. This is what makes flipping `higher_is_spoof` on the
    same scores produce exactly `1 - eer` — see `tests/test_metrics.py`.
    """
    scores_arr, is_spoof = _validate_scores_and_labels(scores, labels)
    bonafide_scores = scores_arr[~is_spoof]
    spoof_scores = scores_arr[is_spoof]

    sign = -1.0 if higher_is_spoof else 1.0
    frr, far, thresholds = _compute_det_curve(sign * bonafide_scores, sign * spoof_scores)

    idx = int(np.argmin(np.abs(frr - far)))
    eer = float(np.mean((frr[idx], far[idx])))
    thr = float(sign * thresholds[idx])
    return eer, thr


def compute_eer_interpolated(
    scores: Sequence, labels: Sequence, higher_is_spoof: bool
) -> tuple[float, float]:
    """Interpolated EER — analysis only, NOT comparable to official ASVspoof numbers.

    The official ASVspoof scoring convention (`compute_eer` above, and
    `research/knowledge/methods/eer-and-metrics.md`) deliberately uses a
    non-interpolated nearest-neighbour EER. This function instead linearly
    interpolates between the two empirical operating points where
    `FAR(t) - FRR(t)` changes sign, giving a smoother (but non-official) estimate:

        FAR(t) = P(bonafide scored >= t)   (higher_is_spoof=True orientation)
        FRR(t) = P(spoof scored  <  t)

    with comparisons mirrored for `higher_is_spoof=False`. Falls back to the
    observed min-|FAR-FRR| point (no interpolation) if no sign change is found
    (e.g. heavily tied scores). Do not report this value as "the EER" alongside
    official ASVspoof baselines without this caveat attached.
    """
    scores_arr, is_spoof = _validate_scores_and_labels(scores, labels)
    bonafide_scores = scores_arr[~is_spoof]
    spoof_scores = scores_arr[is_spoof]

    thresholds = np.unique(np.concatenate([bonafide_scores, spoof_scores]))
    if higher_is_spoof:
        far = np.array([np.mean(bonafide_scores >= t) for t in thresholds])
        frr = np.array([np.mean(spoof_scores < t) for t in thresholds])
    else:
        far = np.array([np.mean(bonafide_scores <= t) for t in thresholds])
        frr = np.array([np.mean(spoof_scores > t) for t in thresholds])
    diff = far - frr

    for i in range(len(thresholds) - 1):
        if diff[i] == 0.0:
            return float(far[i]), float(thresholds[i])
        if diff[i] > 0.0 and diff[i + 1] < 0.0:
            frac = diff[i] / (diff[i] - diff[i + 1])
            thr = thresholds[i] + frac * (thresholds[i + 1] - thresholds[i])
            eer_far = far[i] + frac * (far[i + 1] - far[i])
            eer_frr = frr[i] + frac * (frr[i + 1] - frr[i])
            return float((eer_far + eer_frr) / 2.0), float(thr)
    if diff[-1] == 0.0:
        return float(far[-1]), float(thresholds[-1])

    # Fallback: no sign change found (heavily tied or non-monotonic curve). Report
    # the closest observed operating point without interpolation.
    idx = int(np.argmin(np.abs(diff)))
    return float((far[idx] + frr[idx]) / 2.0), float(thresholds[idx])


def roc_auc(scores: Sequence, labels: Sequence, higher_is_spoof: bool) -> float:
    """ROC-AUC of the scores treating spoof as the positive class.

    Delegates to `sklearn.metrics.roc_auc_score`. When `higher_is_spoof` is False,
    scores are negated first so "higher score = more spoof-like" holds for sklearn's
    positive-class convention.
    """
    scores_arr, is_spoof = _validate_scores_and_labels(scores, labels)
    oriented = scores_arr if higher_is_spoof else -scores_arr
    return float(roc_auc_score(is_spoof.astype(int), oriented))


def pr_auc(scores: Sequence, labels: Sequence, higher_is_spoof: bool) -> float:
    """Precision-recall AUC (average precision) with spoof as the positive class."""
    scores_arr, is_spoof = _validate_scores_and_labels(scores, labels)
    oriented = scores_arr if higher_is_spoof else -scores_arr
    return float(average_precision_score(is_spoof.astype(int), oriented))


def prf1_at_threshold(
    scores: Sequence, labels: Sequence, thr: float, higher_is_spoof: bool
) -> dict:
    """Precision/recall/F1 for spoof-as-positive at a caller-supplied threshold.

    `thr` must come from the caller (e.g. dev-set threshold selection done elsewhere,
    task T11). This function does not search for or optimise a threshold — doing so
    against evaluation data would leak eval information into the reported operating
    point.

    Decision rule: predicted spoof iff `score >= thr` when `higher_is_spoof=True`,
    or iff `score <= thr` when `higher_is_spoof=False`.
    """
    scores_arr, is_spoof = _validate_scores_and_labels(scores, labels)
    pred_spoof = scores_arr >= thr if higher_is_spoof else scores_arr <= thr

    tp = int(np.sum(pred_spoof & is_spoof))
    fp = int(np.sum(pred_spoof & ~is_spoof))
    fn = int(np.sum(~pred_spoof & is_spoof))
    tn = int(np.sum(~pred_spoof & ~is_spoof))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(thr),
    }


def brier_score(probs: Sequence, labels: Sequence) -> float:
    """Mean squared error between predicted spoof-probability and the binary label.

    `brier = mean((p - y)^2)` where `y = 1` for spoof, `0` for bonafide. `probs` are
    assumed to already be calibrated-scale probabilities of the spoof class (i.e.
    polarity is fixed by construction — there is no `higher_is_spoof` argument here
    because a probability's "high" direction is unambiguous by definition, unlike a
    raw score).
    """
    probs_arr = np.asarray(list(probs), dtype=float)
    is_spoof = _normalize_labels(labels)
    if probs_arr.shape[0] != is_spoof.shape[0]:
        raise ValueError(
            f"probs and labels must have the same length; got {probs_arr.shape[0]} "
            f"and {is_spoof.shape[0]}"
        )
    if probs_arr.size == 0:
        raise ValueError("probs/labels must not be empty")
    if np.any(np.isnan(probs_arr)):
        raise ValueError("probs contain NaN")
    return float(np.mean((probs_arr - is_spoof.astype(float)) ** 2))


def min_dcf(
    scores: Sequence,
    labels: Sequence,
    higher_is_spoof: bool,
    c_miss: float = 1.0,
    c_fa: float = 10.0,
    p_spoof: float = 0.05,
    normalize: bool = True,
) -> tuple[float, float]:
    """Minimum (normalised) detection cost function, ASVspoof 5 Track 1 definition.

    Verified against the primary source (arXiv:2601.03944v1, Eq. 1, fetched and
    read directly — see `research/knowledge/methods/min-dcf.md`). The paper defines
    the (already-normalised) cost as:

        DCF(tau) = beta * P_miss^cm(tau) + P_fa^cm(tau)
        beta     = C_miss * (1 - pi_spoof) / (C_fa * pi_spoof)

    which is algebraically `raw_DCF(tau) / (C_fa * pi_spoof)` where
    `raw_DCF(tau) = C_miss*(1-pi_spoof)*P_miss^cm(tau) + C_fa*pi_spoof*P_fa^cm(tau)`.
    **The paper normalises unconditionally by `C_fa * pi_spoof`** — not by
    `min(C_miss*(1-pi_spoof), C_fa*pi_spoof)` (the classical NIST-SRE-style
    "normalized DCF" convention). The two happen to coincide numerically for the
    official ASVspoof 5 parameters (`C_miss=1, C_fa=10, pi_spoof=0.05` gives
    `beta≈1.90 >= 1`, making `C_fa*pi_spoof=0.5` the smaller term), but they
    diverge whenever `beta < 1`. This function implements the paper's actual,
    unconditional normaliser, not the `min(...)` form, and a hand-checked test
    with `beta < 1` params exists specifically to show the two differ generally.

    For a countermeasure (CM) operating at threshold `tau`:
      - `P_miss^cm(tau)` is the fraction of bonafide trials wrongly rejected
        (scored on the "reject" side of `tau`), and
      - `P_fa^cm(tau)` is the fraction of spoof trials wrongly accepted
        (scored on the "accept" side of `tau`),
    swept over every empirically observed operating point (never interpolated,
    matching `compute_eer`'s DET curve). `min_dcf` is `min_tau DCF(tau)`.

    `normalize=False` returns the raw, un-normalised cost
    `raw_DCF(tau) = C_miss*(1-pi_spoof)*P_miss^cm(tau) + C_fa*pi_spoof*P_fa^cm(tau)`
    instead (i.e. `normalize=True` result times `C_fa*pi_spoof`).

    Reuses `_compute_det_curve` (the same official, non-interpolated DET curve
    `compute_eer` uses) via the same sign-flip convention on `higher_is_spoof`,
    so `P_miss^cm` is `frr` (fraction of bonafide/target scored at or below the
    threshold) and `P_fa^cm` is `far` (fraction of spoof/nontarget scored above
    the threshold) at every candidate threshold, including the synthetic
    "accept everything" point below the global minimum score.

    Returns `(min_dcf, threshold)`, where `threshold` is on the original score
    scale (not sign-flipped), analogous to `compute_eer`'s second return value.

    Args:
        c_miss: cost of a miss (a bonafide trial wrongly rejected as spoof).
        c_fa: cost of a false alarm (spoof wrongly accepted as bonafide).
        p_spoof: prior probability of a spoof trial (`pi_spoof` above).
        normalize: divide by the paper's fixed normaliser `C_fa * pi_spoof` when
            True (default); note this is **not** `min(...)` — see docstring above.
    """
    scores_arr, is_spoof = _validate_scores_and_labels(scores, labels)
    bonafide_scores = scores_arr[~is_spoof]
    spoof_scores = scores_arr[is_spoof]

    sign = -1.0 if higher_is_spoof else 1.0
    frr, far, thresholds = _compute_det_curve(sign * bonafide_scores, sign * spoof_scores)

    dcf = c_miss * (1.0 - p_spoof) * frr + c_fa * p_spoof * far
    idx = int(np.argmin(dcf))
    raw_min_dcf = float(dcf[idx])
    thr = float(sign * thresholds[idx])

    if not normalize:
        return raw_min_dcf, thr

    normaliser = c_fa * p_spoof
    if normaliser <= 0.0:
        raise ValueError("c_fa * p_spoof must be positive to normalise the DCF")
    return raw_min_dcf / normaliser, thr


def per_attack_eer(
    scores: Sequence, labels: Sequence, attack_ids: Sequence, higher_is_spoof: bool
) -> dict:
    """EER per spoof attack/generator, each scored against the shared bonafide pool.

    Returns:
        {
          "per_attack": {attack_id: eer, ...},   # one entry per non-bonafide attack_id
          "macro_eer": unweighted mean of per_attack values,
          "pooled_eer": EER using all spoof scores (every attack) vs. the shared
                        bonafide pool, i.e. `compute_eer` on the full input,
        }

    Convention: every attack is evaluated against the **full bonafide pool**, not a
    per-attack subset of it — bonafide utterances are not attack-specific, so
    splitting them per attack would both shrink the negative sample for no reason
    and make per-attack EERs incomparable (see module docstring). `attack_ids` for
    bonafide rows are ignored for grouping purposes but must still be provided
    (any value, including "bonafide", works since only spoof rows are grouped).
    """
    scores_arr, is_spoof = _validate_scores_and_labels(scores, labels)
    attack_ids_arr = np.asarray(list(attack_ids), dtype=object)
    if attack_ids_arr.shape[0] != scores_arr.shape[0]:
        raise ValueError(
            f"attack_ids must align with scores/labels; got {attack_ids_arr.shape[0]} "
            f"vs {scores_arr.shape[0]}"
        )

    bonafide_scores = scores_arr[~is_spoof]
    if bonafide_scores.size == 0:
        raise ValueError("no bonafide scores present; cannot compute per-attack EER")

    spoof_mask = is_spoof
    spoof_attack_ids = attack_ids_arr[spoof_mask]
    unique_attacks = sorted(set(spoof_attack_ids.tolist()))
    if not unique_attacks:
        raise ValueError("no spoof rows present; cannot compute per-attack EER")

    per_attack = {}
    for attack in unique_attacks:
        attack_spoof_scores = scores_arr[spoof_mask][spoof_attack_ids == attack]
        combined_scores = np.concatenate([bonafide_scores, attack_spoof_scores])
        combined_labels = ["bonafide"] * bonafide_scores.size + ["spoof"] * attack_spoof_scores.size
        eer, _ = compute_eer(combined_scores, combined_labels, higher_is_spoof=higher_is_spoof)
        per_attack[attack] = eer

    macro_eer = float(np.mean(list(per_attack.values())))
    pooled_eer, _ = compute_eer(scores_arr, is_spoof.astype(int), higher_is_spoof=higher_is_spoof)

    return {
        "per_attack": per_attack,
        "macro_eer": macro_eer,
        "pooled_eer": pooled_eer,
    }
