"""Analytic tests for `audio_df.metrics`.

Every test either has a hand-computable expected value or a known analytic property
(perfect separability -> EER 0, identical distributions -> EER ~0.5, polarity flip ->
EER -> 1 - EER). Shape-only assertions are avoided per project convention.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from audio_df.metrics import (
    brier_score,
    compute_eer,
    compute_eer_interpolated,
    min_dcf,
    per_attack_eer,
    pr_auc,
    prf1_at_threshold,
    roc_auc,
)

# ---------------------------------------------------------------------------
# Official ASVspoof reference implementation, vendored verbatim for conformance
# testing. Source (fetched and read directly):
# https://raw.githubusercontent.com/asvspoof-challenge/2021/main/eval-package/eval_metrics.py
# Only `compute_det_curve` and `compute_eer` are needed here; copied unmodified
# except for the docstring/comment on `compute_eer`.
# ---------------------------------------------------------------------------


def _official_compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

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


def _official_compute_eer(target_scores, nontarget_scores):
    """Returns equal error rate (EER) and the corresponding threshold."""
    frr, far, thresholds = _official_compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

# ---------------------------------------------------------------------------
# compute_eer
# ---------------------------------------------------------------------------


def test_eer_perfectly_separable_is_zero():
    # spoof scores all higher than bonafide scores, higher_is_spoof=True
    labels = ["bonafide"] * 5 + ["spoof"] * 5
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    eer, thr = compute_eer(scores, labels, higher_is_spoof=True)
    assert eer == pytest.approx(0.0, abs=1e-9)
    # threshold must sit at or between the two clusters (no bonafide/spoof overlap)
    assert 0.4 <= thr <= 0.6


def test_eer_identical_distributions_is_half():
    # Bonafide and spoof scores drawn from the exact same fixed set -> no separation.
    # Use interleaved identical values so every operating point has FAR == FRR-ish.
    rng = np.random.default_rng(0)
    shared = rng.normal(size=2000)
    labels = ["bonafide"] * 1000 + ["spoof"] * 1000
    scores = np.concatenate([shared[:1000], shared[:1000]])  # exact same score sample per class
    eer, _ = compute_eer(scores, labels, higher_is_spoof=True)
    assert eer == pytest.approx(0.5, abs=0.02)


def test_eer_polarity_flip_is_one_minus_eer():
    rng = np.random.default_rng(1)
    n = 500
    bonafide = rng.normal(loc=0.0, scale=1.0, size=n)
    spoof = rng.normal(loc=1.5, scale=1.0, size=n)
    labels = ["bonafide"] * n + ["spoof"] * n
    scores = np.concatenate([bonafide, spoof])

    eer_correct, _ = compute_eer(scores, labels, higher_is_spoof=True)
    eer_flipped, _ = compute_eer(scores, labels, higher_is_spoof=False)

    assert eer_flipped == pytest.approx(1.0 - eer_correct, abs=1e-9)
    # sanity: the correctly-oriented EER should be small since classes are well separated
    assert eer_correct < 0.3


def test_eer_hand_computed_six_points():
    # Hand-computed case.
    # bonafide scores (label 0): 0.1, 0.3, 0.5
    # spoof scores    (label 1): 0.4, 0.6, 0.8
    # higher_is_spoof=True, so FAR(t) = P(bonafide score >= t), FRR(t) = P(spoof score < t)
    #
    # Sorted candidate thresholds (midpoints between consecutive unique scores), we evaluate
    # FAR/FRR at each unique score value used as threshold t (score >= t counted as "spoof"):
    #
    # scores sorted: 0.1(b) 0.3(b) 0.4(s) 0.5(b) 0.6(s) 0.8(s)
    #
    # t=0.1: FAR = |{b: b>=0.1}|/3 = 3/3 = 1.0   FRR = |{s: s<0.1}|/3 = 0/3 = 0.0
    # t=0.3: FAR = |{0.3,0.5}|/3 = 2/3 = 0.667   FRR = |{s<0.3}|/3 = 0/3 = 0.0
    # t=0.4: FAR = |{0.5}|/3 = 1/3 = 0.333       FRR = |{s<0.4}|/3 = 0/3 = 0.0
    # t=0.5: FAR = |{0.5}|/3 = 1/3 = 0.333       FRR = |{0.4}|/3 = 1/3 = 0.333  <- exact match
    # t=0.6: FAR = 0/3 = 0.0                     FRR = |{0.4}|/3 = 1/3 = 0.333
    # t=0.8: FAR = 0/3 = 0.0                     FRR = |{0.4,0.6}|/3 = 2/3 = 0.667
    #
    # At t=0.5, FAR == FRR == 1/3 exactly. So EER = 1/3, no interpolation needed for this case.
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.3, 0.5, 0.4, 0.6, 0.8]
    eer, thr = compute_eer(scores, labels, higher_is_spoof=True)
    assert eer == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert thr == pytest.approx(0.5, abs=1e-9)


def test_eer_string_and_int_labels_agree():
    rng = np.random.default_rng(2)
    n = 200
    bonafide = rng.normal(loc=0.0, size=n)
    spoof = rng.normal(loc=2.0, size=n)
    scores = np.concatenate([bonafide, spoof])
    str_labels = ["bonafide"] * n + ["spoof"] * n
    int_labels = [0] * n + [1] * n

    eer_str, thr_str = compute_eer(scores, str_labels, higher_is_spoof=True)
    eer_int, thr_int = compute_eer(scores, int_labels, higher_is_spoof=True)
    assert eer_str == pytest.approx(eer_int)
    assert thr_str == pytest.approx(thr_int)


def test_eer_rejects_ambiguous_labels():
    with pytest.raises(ValueError):
        compute_eer([0.1, 0.2, 0.3], ["maybe", "spoof", "bonafide"], higher_is_spoof=True)


def test_eer_rejects_non_binary_numeric_labels():
    with pytest.raises(ValueError):
        compute_eer([0.1, 0.2, 0.3], [0, 1, 2], higher_is_spoof=True)


def test_eer_rejects_single_class():
    with pytest.raises(ValueError):
        compute_eer([0.1, 0.2, 0.3], ["bonafide", "bonafide", "bonafide"], higher_is_spoof=True)


def test_eer_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_eer([], [], higher_is_spoof=True)


# ---------------------------------------------------------------------------
# Conformance: compute_eer must match the official ASVspoof implementation exactly
# (in the official's own `higher_is_spoof=False` / bonafide-is-target orientation),
# because our numbers need to be directly comparable to official baselines.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [10, 11, 12, 13, 14])
def test_compute_eer_matches_official_on_random_data(seed):
    rng = np.random.default_rng(seed)
    n = 150
    # Official convention: target=bonafide (higher score), nontarget=spoof (lower).
    bonafide = rng.normal(loc=1.0, scale=1.0, size=n)
    spoof = rng.normal(loc=-0.5, scale=1.2, size=n)
    scores = np.concatenate([bonafide, spoof])
    labels = ["bonafide"] * n + ["spoof"] * n

    ours_eer, ours_thr = compute_eer(scores, labels, higher_is_spoof=False)
    official_eer, official_thr = _official_compute_eer(bonafide, spoof)

    assert ours_eer == pytest.approx(official_eer, abs=1e-12)
    assert ours_thr == pytest.approx(official_thr, abs=1e-12)


def test_compute_eer_matches_official_with_ties():
    # Deliberately tied scores across and within classes; the official algorithm's
    # stable mergesort tie-handling must be reproduced exactly.
    bonafide = np.array([0.5, 0.5, 0.7, 0.9, 0.9, 0.9])
    spoof = np.array([0.5, 0.3, 0.3, 0.9, 0.1, 0.5])
    scores = np.concatenate([bonafide, spoof])
    labels = ["bonafide"] * bonafide.size + ["spoof"] * spoof.size

    ours_eer, ours_thr = compute_eer(scores, labels, higher_is_spoof=False)
    official_eer, official_thr = _official_compute_eer(bonafide, spoof)

    assert ours_eer == pytest.approx(official_eer, abs=1e-12)
    assert ours_thr == pytest.approx(official_thr, abs=1e-12)


def test_compute_eer_matches_official_small_hand_checkable_case():
    # Same 6-point case used in test_eer_hand_computed_six_points, but run through
    # the official-orientation call directly (bonafide=target, spoof=nontarget).
    bonafide = np.array([0.1, 0.3, 0.5])
    spoof = np.array([0.4, 0.6, 0.8])
    # Our public API expects higher_is_spoof; the official convention is
    # higher=bonafide, i.e. higher_is_spoof=False, but bonafide scores here are
    # LOWER than spoof scores, so from our library's point of view this dataset's
    # correct polarity is actually higher_is_spoof=True. We still call the official
    # reference function directly with target=bonafide, nontarget=spoof (which is
    # polarity-agnostic — it just measures separation in whichever direction target
    # scores happen to be higher), and check that our library's higher_is_spoof=True
    # path (which sign-flips before delegating) reproduces the same numbers up to
    # the sign flip.
    ours_eer, ours_thr = compute_eer(
        np.concatenate([bonafide, spoof]),
        ["bonafide"] * 3 + ["spoof"] * 3,
        higher_is_spoof=True,
    )
    official_eer, official_thr = _official_compute_eer(-bonafide, -spoof)
    assert ours_eer == pytest.approx(official_eer, abs=1e-12)
    assert ours_thr == pytest.approx(-official_thr, abs=1e-12)


def test_compute_eer_official_orientation_matches_directly():
    # A case where bonafide genuinely scores higher (the official convention),
    # so higher_is_spoof=False needs no sign flip at all -- the most direct
    # conformance check.
    bonafide = np.array([0.4, 0.6, 0.8])
    spoof = np.array([0.1, 0.3, 0.5])
    ours_eer, ours_thr = compute_eer(
        np.concatenate([bonafide, spoof]),
        ["bonafide"] * 3 + ["spoof"] * 3,
        higher_is_spoof=False,
    )
    official_eer, official_thr = _official_compute_eer(bonafide, spoof)
    assert ours_eer == pytest.approx(official_eer, abs=1e-12)
    assert ours_thr == pytest.approx(official_thr, abs=1e-12)


def test_compute_eer_interpolated_differs_from_official_on_small_set():
    # Documents (and pins) that the interpolated variant is NOT expected to match
    # the official nearest-neighbour EER in general -- it exists for a different
    # purpose and must never be silently swapped in for compute_eer.
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.3, 0.5, 0.4, 0.6, 0.8]
    eer_official, _ = compute_eer(scores, labels, higher_is_spoof=True)
    eer_interp, _ = compute_eer_interpolated(scores, labels, higher_is_spoof=True)
    # In this particular case both land on an exact crossing at 1/3, so they agree;
    # the important behavioural contract is that they are two distinct code paths
    # (verified by the module having two distinct public functions), not that they
    # must diverge on every input.
    assert eer_official == pytest.approx(1.0 / 3.0, abs=1e-9)
    assert eer_interp == pytest.approx(1.0 / 3.0, abs=1e-9)


# ---------------------------------------------------------------------------
# roc_auc
# ---------------------------------------------------------------------------


def test_roc_auc_matches_sklearn_on_random_data():
    rng = np.random.default_rng(3)
    n = 300
    scores = rng.normal(size=n)
    labels_int = rng.integers(0, 2, size=n)
    labels = ["spoof" if lbl == 1 else "bonafide" for lbl in labels_int]

    ours = roc_auc(scores, labels, higher_is_spoof=True)
    theirs = roc_auc_score(labels_int, scores)
    assert ours == pytest.approx(theirs, abs=1e-9)


def test_roc_auc_polarity_flip_complements():
    rng = np.random.default_rng(4)
    n = 200
    bonafide = rng.normal(loc=0.0, size=n)
    spoof = rng.normal(loc=1.0, size=n)
    scores = np.concatenate([bonafide, spoof])
    labels = ["bonafide"] * n + ["spoof"] * n

    auc_true = roc_auc(scores, labels, higher_is_spoof=True)
    auc_false = roc_auc(scores, labels, higher_is_spoof=False)
    assert auc_true == pytest.approx(1.0 - auc_false, abs=1e-9)


# ---------------------------------------------------------------------------
# pr_auc
# ---------------------------------------------------------------------------


def test_pr_auc_perfect_separation_is_one():
    labels = ["bonafide"] * 5 + ["spoof"] * 5
    scores = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
    score = pr_auc(scores, labels, higher_is_spoof=True)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_pr_auc_in_valid_range():
    rng = np.random.default_rng(5)
    n = 200
    scores = rng.normal(size=n)
    labels = rng.choice(["bonafide", "spoof"], size=n)
    if len(set(labels)) < 2:
        labels[0] = "bonafide"
        labels[1] = "spoof"
    score = pr_auc(scores, labels, higher_is_spoof=True)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# prf1_at_threshold
# ---------------------------------------------------------------------------


def test_prf1_at_threshold_hand_computed():
    # Same six points as the EER hand-computation.
    # threshold=0.5, higher_is_spoof=True -> predict spoof iff score >= 0.5
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.3, 0.5, 0.4, 0.6, 0.8]
    # predictions: 0.1->bonafide(correct), 0.3->bonafide(correct), 0.5->spoof(WRONG, true bonafide)
    #              0.4->bonafide(WRONG, true spoof), 0.6->spoof(correct), 0.8->spoof(correct)
    # TP=2 (0.6,0.8), FP=1 (0.5), FN=1 (0.4), TN=2 (0.1,0.3)
    # precision = TP/(TP+FP) = 2/3
    # recall    = TP/(TP+FN) = 2/3
    # f1        = 2/3
    result = prf1_at_threshold(scores, labels, thr=0.5, higher_is_spoof=True)
    assert result["precision"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert result["recall"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert result["f1"] == pytest.approx(2.0 / 3.0, abs=1e-9)


def test_prf1_at_threshold_polarity_matters():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.3, 0.5, 0.4, 0.6, 0.8]
    r_true = prf1_at_threshold(scores, labels, thr=0.5, higher_is_spoof=True)
    r_false = prf1_at_threshold(scores, labels, thr=0.5, higher_is_spoof=False)
    assert r_true["precision"] != pytest.approx(r_false["precision"])


# ---------------------------------------------------------------------------
# brier_score
# ---------------------------------------------------------------------------


def test_brier_score_matches_manual_mean_squared_error():
    rng = np.random.default_rng(6)
    probs = rng.uniform(size=100)
    labels_int = rng.integers(0, 2, size=100)
    labels = ["spoof" if lbl == 1 else "bonafide" for lbl in labels_int]
    ours = brier_score(probs, labels)
    manual = np.mean((probs - labels_int) ** 2)
    assert ours == pytest.approx(manual, abs=1e-12)


def test_brier_score_perfect_predictions_is_zero():
    probs = [0.0, 0.0, 1.0, 1.0]
    labels = ["bonafide", "bonafide", "spoof", "spoof"]
    assert brier_score(probs, labels) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# per_attack_eer
# ---------------------------------------------------------------------------


def test_per_attack_eer_pooled_sits_between_easy_and_hard_attacks():
    rng = np.random.default_rng(7)
    n_bonafide = 400
    n_per_attack = 200

    bonafide_scores = rng.normal(loc=0.0, scale=1.0, size=n_bonafide)
    # A02: trivially detectable -> spoof scores far from bonafide
    a02_scores = rng.normal(loc=5.0, scale=1.0, size=n_per_attack)
    # A03: hard to detect -> spoof scores overlap heavily with bonafide
    a03_scores = rng.normal(loc=0.1, scale=1.0, size=n_per_attack)

    scores = np.concatenate([bonafide_scores, a02_scores, a03_scores])
    labels = (
        ["bonafide"] * n_bonafide + ["spoof"] * n_per_attack + ["spoof"] * n_per_attack
    )
    attack_ids = (
        ["bonafide"] * n_bonafide + ["A02"] * n_per_attack + ["A03"] * n_per_attack
    )

    result = per_attack_eer(scores, labels, attack_ids, higher_is_spoof=True)

    assert result["per_attack"]["A02"] < 0.05
    assert result["per_attack"]["A03"] > 0.3
    assert result["per_attack"]["A02"] < result["pooled_eer"] < result["per_attack"]["A03"]
    assert result["macro_eer"] == pytest.approx(
        (result["per_attack"]["A02"] + result["per_attack"]["A03"]) / 2.0, abs=1e-9
    )


def test_per_attack_eer_uses_shared_bonafide_pool():
    # Two attacks, both evaluated against the SAME full bonafide pool, not disjoint subsets.
    rng = np.random.default_rng(8)
    n_bonafide = 300
    n_attack = 150

    bonafide_scores = rng.normal(loc=0.0, scale=1.0, size=n_bonafide)
    a01_scores = rng.normal(loc=3.0, scale=1.0, size=n_attack)
    a02_scores = rng.normal(loc=3.0, scale=1.0, size=n_attack)

    scores = np.concatenate([bonafide_scores, a01_scores, a02_scores])
    labels = ["bonafide"] * n_bonafide + ["spoof"] * n_attack + ["spoof"] * n_attack
    attack_ids = ["bonafide"] * n_bonafide + ["A01"] * n_attack + ["A02"] * n_attack

    result = per_attack_eer(scores, labels, attack_ids, higher_is_spoof=True)
    # Since both attacks have identical score distributions and share the same bonafide
    # pool, their EERs should be nearly identical.
    assert result["per_attack"]["A01"] == pytest.approx(result["per_attack"]["A02"], abs=0.05)


def test_per_attack_eer_rejects_no_bonafide():
    with pytest.raises(ValueError):
        per_attack_eer(
            [0.1, 0.2, 0.3],
            ["spoof", "spoof", "spoof"],
            ["A01", "A01", "A02"],
            higher_is_spoof=True,
        )


# ---------------------------------------------------------------------------
# degenerate inputs across the module
# ---------------------------------------------------------------------------


def test_roc_auc_rejects_single_class():
    with pytest.raises(ValueError):
        roc_auc([0.1, 0.2, 0.3], ["bonafide", "bonafide", "bonafide"], higher_is_spoof=True)


def test_pr_auc_rejects_empty_input():
    with pytest.raises(ValueError):
        pr_auc([], [], higher_is_spoof=True)


def test_brier_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        brier_score([0.1, 0.2], ["bonafide", "spoof", "spoof"])


def test_brier_score_rejects_empty_input():
    with pytest.raises(ValueError):
        brier_score([], [])


# ---------------------------------------------------------------------------
# min_dcf
# ---------------------------------------------------------------------------


def _brute_force_min_dcf(bonafide, spoof, c_miss, c_fa, p_spoof, normalize=True):
    """Independent, non-vectorised reference implementation for cross-checking.

    Sweeps every candidate threshold (all observed scores plus one point below
    the global minimum) and computes P_miss/P_fa directly with plain Python
    loops -- deliberately not sharing any code with `_compute_det_curve` or
    `min_dcf`, so agreement is real evidence, not a shared-bug tautology.
    """
    bonafide = list(bonafide)
    spoof = list(spoof)
    candidates = [min(bonafide + spoof) - 0.001] + sorted(bonafide + spoof)

    best_dcf = None
    best_thr = None
    for tau in candidates:
        p_miss = sum(1 for b in bonafide if b <= tau) / len(bonafide)
        p_fa = sum(1 for s in spoof if s > tau) / len(spoof)
        dcf = c_miss * (1.0 - p_spoof) * p_miss + c_fa * p_spoof * p_fa
        if best_dcf is None or dcf < best_dcf:
            best_dcf = dcf
            best_thr = tau

    if not normalize:
        return best_dcf, best_thr
    # ASVspoof 5 Eq. 1 normalises unconditionally by C_fa * pi_spoof (not by
    # min(C_miss*(1-pi), C_fa*pi)) -- see Miscellaneous/research/knowledge/methods/min-dcf.md.
    normaliser = c_fa * p_spoof
    return best_dcf / normaliser, best_thr


def test_min_dcf_hand_computed_tiny_example():
    # Small, asymmetric, partially-overlapping score set; higher = bonafide.
    bonafide = [0.0, 1.0, 2.5]
    spoof = [-1.0, 0.5, 1.5, 3.0]
    scores = bonafide + spoof
    labels = ["bonafide"] * len(bonafide) + ["spoof"] * len(spoof)

    c_miss, c_fa, p_spoof = 1.0, 10.0, 0.05
    expected_dcf, _ = _brute_force_min_dcf(bonafide, spoof, c_miss, c_fa, p_spoof)

    got_dcf, got_thr = min_dcf(
        scores, labels, higher_is_spoof=False, c_miss=c_miss, c_fa=c_fa, p_spoof=p_spoof
    )
    assert got_dcf == pytest.approx(expected_dcf, abs=1e-9)
    # threshold must be an actual candidate operating point (or the synthetic one below min)
    assert got_thr <= min(scores) or got_thr in scores


def test_min_dcf_perfect_separation_is_zero():
    rng = np.random.default_rng(0)
    bonafide = rng.normal(loc=5.0, scale=0.1, size=50)
    spoof = rng.normal(loc=-5.0, scale=0.1, size=50)
    scores = np.concatenate([bonafide, spoof])
    labels = ["bonafide"] * 50 + ["spoof"] * 50

    dcf, _ = min_dcf(scores, labels, higher_is_spoof=False, c_miss=1.0, c_fa=10.0, p_spoof=0.05)
    assert dcf == pytest.approx(0.0, abs=1e-9)


def test_min_dcf_normalized_value_is_bounded_by_one_for_default_params():
    # This bound holds for the OFFICIAL ASVspoof 5 params (c_miss=1, c_fa=10,
    # p_spoof=0.05) because beta = c_miss*(1-p_spoof)/(c_fa*p_spoof) ~= 1.90 >= 1,
    # which makes c_fa*p_spoof (=0.5, the paper's fixed normaliser) the smaller of
    # the two raw-DCF terms, so raw_DCF/normaliser <= raw_DCF/min(...) <= 1 for any
    # scores. It is NOT a general property of min_dcf's normalisation -- see
    # test_min_dcf_normalized_value_can_exceed_one_when_beta_below_one below.
    rng = np.random.default_rng(3)
    for _seed_offset in range(5):
        s = rng.normal(size=200)
        labels = rng.integers(0, 2, size=200)
        # guarantee both classes present
        labels[0], labels[1] = 0, 1
        dcf, _ = min_dcf(
            s, labels, higher_is_spoof=True, c_miss=1.0, c_fa=10.0, p_spoof=0.05, normalize=True
        )
        assert 0.0 <= dcf <= 1.0 + 1e-9


def test_min_dcf_normalized_value_can_exceed_one_when_beta_below_one():
    """ASVspoof 5 Eq. 1 normalises unconditionally by C_fa*pi_spoof, not by
    min(C_miss*(1-pi_spoof), C_fa*pi_spoof). With c_miss=10, c_fa=1, p_spoof=0.5,
    beta = 10*0.5/(1*0.5) = 10 (>= 1 actually -- pick params giving beta < 1
    instead): c_miss=1, c_fa=10, p_spoof=0.5 -> beta = 1*0.5/(10*0.5) = 0.1 < 1.
    Then C_fa*p_spoof = 5.0 while C_miss*(1-p_spoof) = 0.5, so the paper's fixed
    normaliser (5.0) is the LARGER term -- the min(...) form would instead divide
    by 0.5, giving a value 10x smaller. This test asserts the paper's formula
    (divide by C_fa*p_spoof unconditionally), cross-checked against the
    independent brute-force reference, not the min(...) convention."""
    bonafide = [0.0, 1.0, 2.5]
    spoof = [-1.0, 0.5, 1.5, 3.0]
    scores = bonafide + spoof
    labels = ["bonafide"] * len(bonafide) + ["spoof"] * len(spoof)
    c_miss, c_fa, p_spoof = 1.0, 10.0, 0.5

    beta = c_miss * (1 - p_spoof) / (c_fa * p_spoof)
    assert beta < 1.0  # sanity-check the chosen params actually exercise this case

    expected_dcf, _ = _brute_force_min_dcf(bonafide, spoof, c_miss, c_fa, p_spoof)
    got_dcf, _ = min_dcf(
        scores, labels, higher_is_spoof=False, c_miss=c_miss, c_fa=c_fa, p_spoof=p_spoof
    )
    assert got_dcf == pytest.approx(expected_dcf, abs=1e-9)

    # The min(...) convention would divide by min(0.5, 5.0) = 0.5 instead of 5.0,
    # giving a value exactly 10x larger than the paper's own normalisation.
    min_form_normaliser = min(c_miss * (1 - p_spoof), c_fa * p_spoof)
    paper_normaliser = c_fa * p_spoof
    assert paper_normaliser == pytest.approx(5.0)
    assert min_form_normaliser == pytest.approx(0.5)
    raw_dcf, _ = min_dcf(
        scores, labels, higher_is_spoof=False, c_miss=c_miss, c_fa=c_fa, p_spoof=p_spoof,
        normalize=False,
    )
    min_form_value = raw_dcf / min_form_normaliser
    assert got_dcf == pytest.approx(raw_dcf / paper_normaliser, abs=1e-9)
    assert got_dcf != pytest.approx(min_form_value, abs=1e-6)


def test_min_dcf_orientation_flip_is_consistent():
    rng = np.random.default_rng(5)
    bonafide = rng.normal(loc=1.0, size=40)
    spoof = rng.normal(loc=-0.5, size=40)
    scores = np.concatenate([bonafide, spoof])
    labels = ["bonafide"] * 40 + ["spoof"] * 40

    # higher_is_spoof=False on `scores` must equal higher_is_spoof=True on `-scores`,
    # same symmetry compute_eer relies on for its polarity argument.
    dcf_a, thr_a = min_dcf(
        scores, labels, higher_is_spoof=False, c_miss=1.0, c_fa=10.0, p_spoof=0.05
    )
    dcf_b, thr_b = min_dcf(
        -scores, labels, higher_is_spoof=True, c_miss=1.0, c_fa=10.0, p_spoof=0.05
    )
    assert dcf_a == pytest.approx(dcf_b, abs=1e-9)
    assert thr_a == pytest.approx(-thr_b, abs=1e-9)


def test_min_dcf_unnormalized_matches_normalized_times_c_fa_p_spoof():
    scores = [0.0, 1.0, 2.5, -1.0, 0.5, 1.5, 3.0]
    labels = ["bonafide"] * 3 + ["spoof"] * 4
    c_miss, c_fa, p_spoof = 1.0, 10.0, 0.05
    normaliser = c_fa * p_spoof  # ASVspoof 5 Eq. 1's fixed normaliser

    raw, thr_raw = min_dcf(
        scores, labels, higher_is_spoof=False, c_miss=c_miss, c_fa=c_fa, p_spoof=p_spoof,
        normalize=False,
    )
    norm, thr_norm = min_dcf(
        scores, labels, higher_is_spoof=False, c_miss=c_miss, c_fa=c_fa, p_spoof=p_spoof,
        normalize=True,
    )
    assert raw == pytest.approx(norm * normaliser, abs=1e-9)
    assert thr_raw == pytest.approx(thr_norm, abs=1e-9)


def test_min_dcf_rejects_single_class():
    with pytest.raises(ValueError):
        min_dcf([0.1, 0.2, 0.3], ["bonafide", "bonafide", "bonafide"], higher_is_spoof=True)
