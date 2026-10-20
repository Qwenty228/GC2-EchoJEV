"""Analytic tests for `audio_df.helpers.research.stats`. No dataset required.

Every test has a hand-computable expected value or a known analytic property:
Cohen's d on unit-variance normals separated by a known mean gap, Cliff's delta on
disjoint/identical samples plus a brute-force O(n^2) cross-check, bootstrap CI
coverage via simulation, single_feature_auc matching metrics.roc_auc exactly, and
per_bin_effect_size isolating a single perturbed bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from audio_df.helpers.metrics import roc_auc
from audio_df.helpers.research.stats import (
    bootstrap_ci,
    cliffs_delta,
    cohens_d,
    describe_by_class,
    per_bin_effect_size,
    single_feature_auc,
)

# ---------------------------------------------------------------------------
# describe_by_class
# ---------------------------------------------------------------------------


def test_describe_by_class_columns_and_values():
    df = pd.DataFrame(
        {
            "feat": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0],
            "label": ["bonafide"] * 3 + ["spoof"] * 3,
        }
    )
    out = describe_by_class(df, "feat", "label")
    assert list(out.columns) == ["mean", "median", "std", "iqr", "n"]
    assert set(out.index) == {"bonafide", "spoof"}
    assert out.loc["bonafide", "n"] == 3
    assert out.loc["bonafide", "mean"] == pytest.approx(2.0)
    assert out.loc["bonafide", "median"] == pytest.approx(2.0)
    assert out.loc["spoof", "mean"] == pytest.approx((4.0 + 10.0 + 20.0) / 3)


def test_describe_by_class_n_equals_1_is_sane():
    df = pd.DataFrame({"feat": [5.0, 1.0, 2.0], "label": ["only", "many", "many"]})
    out = describe_by_class(df, "feat", "label")
    assert out.loc["only", "n"] == 1
    assert out.loc["only", "mean"] == pytest.approx(5.0)
    # std of a single observation is undefined, not fabricated as 0.0
    assert pd.isna(out.loc["only", "std"])
    # IQR of a single point is well-defined: Q1 = Q3 = the value itself.
    assert out.loc["only", "iqr"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cohens_d
# ---------------------------------------------------------------------------


def test_cohens_d_unit_normals_separated_by_one():
    rng = np.random.default_rng(42)
    n = 200_000
    a = rng.normal(loc=1.0, scale=1.0, size=n)
    b = rng.normal(loc=0.0, scale=1.0, size=n)
    d = cohens_d(a, b)
    # Large n -> pooled SD -> ~1.0; tolerance covers Monte Carlo sampling noise only.
    assert d == pytest.approx(1.0, abs=0.02)


def test_cohens_d_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(size=5000)
    assert cohens_d(a, b) == pytest.approx(0.0, abs=0.05)


def test_cohens_d_sign_reflects_direction():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([0.0, 1.0, 2.0])
    assert cohens_d(a, b) > 0
    assert cohens_d(b, a) < 0
    assert cohens_d(a, b) == pytest.approx(-cohens_d(b, a))


def test_cohens_d_hand_computable():
    # a: [1,2,3], mean=2, var(ddof=1)=1; b: [4,5,6], mean=5, var(ddof=1)=1
    # pooled_sd = sqrt((2*1+2*1)/4) = 1; d = (2-5)/1 = -3
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert cohens_d(a, b) == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# cliffs_delta
# ---------------------------------------------------------------------------


def _brute_force_cliffs_delta(a, b):
    n_gt = n_lt = 0
    for x in a:
        for y in b:
            if x > y:
                n_gt += 1
            elif x < y:
                n_lt += 1
    return (n_gt - n_lt) / (len(a) * len(b))


def test_cliffs_delta_disjoint_samples():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    assert cliffs_delta(a, b) == pytest.approx(1.0)
    assert cliffs_delta(b, a) == pytest.approx(-1.0)


def test_cliffs_delta_identical_distributions_near_zero():
    rng = np.random.default_rng(1)
    a = rng.normal(size=3000)
    b = rng.normal(size=3000)
    assert cliffs_delta(a, b) == pytest.approx(0.0, abs=0.05)


def test_cliffs_delta_matches_brute_force_small_case():
    rng = np.random.default_rng(7)
    a = rng.normal(loc=0.3, size=25)
    b = rng.normal(loc=-0.1, size=30)
    fast = cliffs_delta(a, b)
    slow = _brute_force_cliffs_delta(a, b)
    assert fast == pytest.approx(slow)


def test_cliffs_delta_with_ties():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 2.0, 4.0])
    slow = _brute_force_cliffs_delta(a, b)
    assert cliffs_delta(a, b) == pytest.approx(slow)


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_deterministic_under_seed():
    rng = np.random.default_rng(3)
    a = rng.normal(loc=1.0, size=200)
    b = rng.normal(loc=0.0, size=200)
    ci1 = bootstrap_ci(lambda x, y: np.mean(x) - np.mean(y), a, b, n_boot=500, seed=123)
    ci2 = bootstrap_ci(lambda x, y: np.mean(x) - np.mean(y), a, b, n_boot=500, seed=123)
    assert ci1 == ci2


def _run_coverage_simulation(n_trials: int, n_boot: int, n_per_group: int = 60) -> float:
    """Fraction of nominal-95% `bootstrap_ci` intervals that cover the true mean gap.

    Shared by the fast smoke test and the slow, higher-budget real check below.
    """
    true_diff = 0.5
    alpha = 0.05
    hits = 0
    master_rng = np.random.default_rng(2024)

    def mean_diff(x, y):
        return float(np.mean(x) - np.mean(y))

    for trial in range(n_trials):
        a = master_rng.normal(loc=true_diff, scale=1.0, size=n_per_group)
        b = master_rng.normal(loc=0.0, scale=1.0, size=n_per_group)
        lo, hi = bootstrap_ci(mean_diff, a, b, n_boot=n_boot, alpha=alpha, seed=trial)
        if lo <= true_diff <= hi:
            hits += 1

    return hits / n_trials


def test_bootstrap_ci_coverage_smoke():
    """Fast, low-budget sanity check: not the real validation (too few resamples/trials
    for a tight bound, see `test_bootstrap_ci_coverage_simulation` below), just a guard
    against gross breakage (e.g. coverage collapsing to 0 or staying pinned at 1.0)."""
    coverage = _run_coverage_simulation(n_trials=100, n_boot=300)
    assert 0.80 <= coverage <= 1.0, f"observed coverage {coverage:.3f} (smoke test)"


@pytest.mark.slow
def test_bootstrap_ci_coverage_simulation():
    """The test that actually validates the method: does a nominal 95% CI cover the
    true parameter ~95% of the time over repeated synthetic trials?

    Budget: n_trials=400, n_boot=1000. Both matter for how tight a bound is honest:
    - `n_boot` too low makes each individual CI's 2.5/97.5 percentiles noisy, biasing
      the *estimated* coverage (in either direction, but empirically low) independent
      of whether the method itself is correct — this is a harness artifact, not a
      property of the percentile bootstrap.
    - `n_trials` too low makes the *coverage estimate itself* noisy: hits/n_trials is
      a binomial proportion with SE = sqrt(p(1-p)/n_trials). At the nominal p=0.95 and
      n_trials=400, SE = sqrt(0.95*0.05/400) ~= 0.0109 (1.09%). A ~2.3-SE half-width
      (~2.5%) gives a band of roughly [0.925, 0.975] that should contain the true
      coverage with high probability while still catching a genuinely miscalibrated
      interval (e.g. a bug that silently returns [-inf, inf] and covers 100%, or one
      that's too narrow and covers <90%).

    Earlier iteration of this test used n_boot=400/n_trials=200 and a one-sided
    [0.88, 1.0] band; both were too loose (a interval that is always [-inf, inf]
    would have passed) and the low n_boot/n_trials budget produced a biased-low point
    estimate (~0.91-0.93 depending on seed) that was at the time mis-attributed to
    "percentile vs BCa" bias. Cross-checked against `scipy.stats.bootstrap(...,
    method="percentile")` at matched budgets: scipy shows the same low-budget bias
    (~0.915 at n_boot=400/n_trials=200) and the same convergence to ~0.95 at
    n_boot>=1000/n_trials>=2000, confirming the shortfall was a simulation-budget
    artifact, not a defect of the percentile method (nor of this implementation).
    """
    coverage = _run_coverage_simulation(n_trials=400, n_boot=1000)
    assert 0.925 <= coverage <= 0.975, (
        f"observed coverage {coverage:.4f} for nominal 95% CI; "
        "expected within ~2.3 binomial SE of 0.95 at n_trials=400"
    )


# ---------------------------------------------------------------------------
# single_feature_auc
# ---------------------------------------------------------------------------


def test_single_feature_auc_matches_metrics_roc_auc():
    rng = np.random.default_rng(5)
    values = rng.normal(size=500)
    labels = rng.integers(0, 2, size=500)
    expected = roc_auc(values, labels, higher_is_spoof=True)
    assert single_feature_auc(values, labels) == pytest.approx(expected)


def test_single_feature_auc_perfect_separation():
    values = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    labels = np.array([0, 0, 0, 1, 1, 1])  # higher values are labeled spoof (1)
    assert single_feature_auc(values, labels) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# per_bin_effect_size
# ---------------------------------------------------------------------------


def test_per_bin_effect_size_isolates_single_bin():
    rng = np.random.default_rng(9)
    n_a, n_b, n_bins = 300, 300, 16
    spec_a = rng.normal(loc=0.0, scale=1.0, size=(n_a, n_bins))
    spec_b = rng.normal(loc=0.0, scale=1.0, size=(n_b, n_bins))
    perturbed_bin = 4
    spec_a[:, perturbed_bin] += 2.0  # large, known shift in exactly one bin

    d = per_bin_effect_size(spec_a, spec_b)
    assert d.shape == (n_bins,)
    assert abs(d[perturbed_bin]) > 1.5
    other_bins = np.delete(np.arange(n_bins), perturbed_bin)
    assert np.all(np.abs(d[other_bins]) < 0.5)


def test_per_bin_effect_size_matches_cohens_d_per_column():
    rng = np.random.default_rng(11)
    spec_a = rng.normal(size=(50, 5))
    spec_b = rng.normal(loc=0.3, size=(50, 5))
    d = per_bin_effect_size(spec_a, spec_b)
    for col in range(5):
        assert d[col] == pytest.approx(cohens_d(spec_a[:, col], spec_b[:, col]))


def test_per_bin_effect_size_rejects_mismatched_bins():
    a = np.zeros((10, 4))
    b = np.zeros((10, 5))
    with pytest.raises(ValueError):
        per_bin_effect_size(a, b)
