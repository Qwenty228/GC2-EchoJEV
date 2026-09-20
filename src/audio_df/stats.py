"""Effect sizes, bootstrap CIs, and per-feature/per-bin separability, done honestly.

This module answers "how different are the two classes on this feature/spectrum",
using estimators whose bias/denominator conventions are documented explicitly (see
each function's docstring) rather than left implicit — a Cohen's d that silently
picks pooled-vs-unpooled SD or n-vs-n-1 can disagree with another implementation by
a few percent for no scientifically meaningful reason.

## Why `per_bin_effect_size` returns no p-values

Per the handoff (Stage 1, "Statistical comparison"): for high-dimensional spectra we
visualise **mean differences and standardized effect sizes per frequency bin**, and
explicitly avoid treating thousands of uncorrected pointwise significance tests as
scientific evidence. The obvious-looking alternative — run a t-test per bin and mark
"significant" bins — is deliberately not implemented here, for two concrete reasons:

1. **Multiple comparisons.** A spectrogram/spectrum has hundreds to thousands of
   frequency bins. Running one hypothesis test per bin at alpha=0.05 without
   correction (Bonferroni, FDR, ...) manufactures a large number of "significant"
   bins by chance alone even under the null. Adding a correction would fix the
   false-positive rate but still answers the wrong question.
2. **Wrong question, large-n regime.** With the sample sizes this project has
   (thousands of utterances), even a scientifically negligible mean difference
   becomes "p < 0.001" — statistical significance stops tracking practical/scientific
   relevance. What the workshop actually wants to know is "how large and how
   reliable is the difference", which is an effect-size-with-uncertainty question,
   not a significance question. `per_bin_effect_size` returns a standardized
   difference (Cohen's-d-like) per bin, meant to be plotted directly (e.g. as a
   spectrum overlay or heatmap of standardized effect); `bootstrap_ci` is the
   companion tool for attaching uncertainty to any single scalar summary of that
   curve, without pretending pointwise per-bin tests would add rigor they can't
   provide at this scale.

Prefer effect sizes with confidence intervals over significance tests throughout
this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from audio_df.metrics import roc_auc

__all__ = [
    "describe_by_class",
    "cohens_d",
    "cliffs_delta",
    "bootstrap_ci",
    "single_feature_auc",
    "per_bin_effect_size",
]


def describe_by_class(df: pd.DataFrame, feature: str, label_col: str) -> pd.DataFrame:
    """Per-class mean/median/std/IQR/n for one scalar feature.

    Returns a DataFrame indexed by the distinct values of `label_col`, with columns
    `["mean", "median", "std", "iqr", "n"]`. `std` uses the pandas default (ddof=1,
    sample standard deviation) and is `NaN` for a class with `n=1` — that is the
    honest answer (a single observation has no estimable spread), not a fabricated
    0.0. `iqr` (Q3 - Q1, linear interpolation, pandas default) is well-defined and
    0.0 for `n=1` since Q1=Q3=the single value; it is reported without a caveat.
    """
    grouped = df.groupby(label_col)[feature]
    out = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "median": grouped.median(),
            "std": grouped.std(),
            "iqr": grouped.quantile(0.75) - grouped.quantile(0.25),
            "n": grouped.count(),
        }
    )
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d = (mean(a) - mean(b)) / pooled_sd, positive when a > b on average.

    **Denominator convention (explicit, because implementations disagree here):**
    pooled standard deviation with Bessel's correction (n-1) on each group, i.e. the
    classic

        pooled_sd = sqrt( ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2) )

    where `var_a`, `var_b` use ddof=1. This is the "pooled, unbiased-variance" form
    (Cohen 1988's original pooled-SD definition), not Glass's delta (which divides by
    one group's SD only) and not the small-sample Hedges' g bias correction factor
    (`J = 1 - 3/(4*(n_a+n_b)-9)`), which is NOT applied here. Requires `n_a + n_b > 2`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = a.size, b.size
    if n_a + n_b <= 2:
        raise ValueError("cohens_d requires n_a + n_b > 2 to estimate a pooled SD")
    var_a = np.var(a, ddof=1) if n_a > 1 else 0.0
    var_b = np.var(b, ddof=1) if n_b > 1 else 0.0
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0.0:
        return 0.0 if np.mean(a) == np.mean(b) else np.sign(np.mean(a) - np.mean(b)) * np.inf
    return float((np.mean(a) - np.mean(b)) / pooled_sd)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a_i > b_i) - P(a_i < b_i), the non-parametric effect size.

    delta in [-1, 1]. delta = +1 means every element of `a` exceeds every element of
    `b` (fully disjoint, a above b); delta = -1 is the reverse; delta ~ 0 means the
    distributions are stochastically indistinguishable by this measure (e.g. equal
    distributions in expectation).

    Computed in O(n log n) via rank-based counting (sort the pooled sample, use
    cumulative counts to get, for each `b` value, how many `a` values are greater/
    less), not the naive O(n_a * n_b) pairwise double loop — this matters because the
    handoff runs this per frequency bin across a whole spectrum.

    Ties (a_i == b_j) contribute to neither the ">" nor "<" count, matching the
    standard definition delta = (#gt - #lt) / (n_a * n_b).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n_a, n_b = a.size, b.size
    if n_a == 0 or n_b == 0:
        raise ValueError("cliffs_delta requires both samples to be non-empty")

    sorted_a = np.sort(a)
    # For each b_j, count how many a_i < b_j and how many a_i > b_j via binary search
    # (searchsorted), which is O((n_a + n_b) log n_a) overall.
    n_less = np.searchsorted(sorted_a, b, side="left")  # count a_i < b_j
    n_leq = np.searchsorted(sorted_a, b, side="right")  # count a_i <= b_j
    n_greater = n_a - n_leq  # count a_i > b_j

    total_gt = int(np.sum(n_greater))  # pairs where a_i > b_j, summed over j
    total_lt = int(np.sum(n_less))  # pairs where a_i < b_j, summed over j

    return float((total_gt - total_lt) / (n_a * n_b))


def bootstrap_ci(
    stat_fn,
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `stat_fn(a, b)`.

    Resamples `a` and `b` independently, with replacement, at their original sizes,
    `n_boot` times; returns the `(alpha/2, 1 - alpha/2)` percentiles of the resulting
    `stat_fn` values (e.g. a 95% CI for `alpha=0.05`). This is the plain percentile
    method (not BCa) — adequate for the roughly-symmetric effect-size statistics this
    module produces, and simple enough that its coverage behaviour can be checked
    directly by simulation (see `tests/test_stats.py`).

    Deterministic for a fixed `seed`: uses a local `np.random.default_rng(seed)`, no
    global numpy random state is touched.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n_a, n_b = a.size, b.size

    boot_stats = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        resampled_a = a[rng.integers(0, n_a, size=n_a)]
        resampled_b = b[rng.integers(0, n_b, size=n_b)]
        boot_stats[i] = stat_fn(resampled_a, resampled_b)

    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def single_feature_auc(values: np.ndarray, labels) -> float:
    """ROC-AUC of one scalar feature used alone as a spoof detector.

    Delegates to `metrics.roc_auc` rather than reimplementing ROC-AUC. Polarity
    choice: `higher_is_spoof=True` is passed, i.e. this treats a higher raw feature
    value as more spoof-like by convention. This is an arbitrary but explicit choice
    for *unsigned* EDA scalar features (e.g. spectral centroid, ZCR) that have no
    inherent "high = bonafide" direction the way an ASVspoof CM log-likelihood score
    does.

    **This makes the returned AUC direction-dependent, not just a magnitude of
    separability.** A feature that perfectly separates the classes with *low* values
    indicating spoof will report AUC = 0.0 here, not 1.0 — `single_feature_auc` does
    not automatically orient itself toward the more-separating direction. A caller
    building a per-feature AUC table (the Stage 1 EDA table this function exists for)
    must look at `max(auc, 1 - auc)`, or report both `auc` and `1 - auc`, to avoid
    reading a perfectly-separating-in-the-other-direction feature as uninformative
    (AUC near 0 looks "bad" at a glance, but is exactly as separating as AUC near 1).
    """
    return roc_auc(values, labels, higher_is_spoof=True)


def per_bin_effect_size(spec_a: np.ndarray, spec_b: np.ndarray) -> np.ndarray:
    """Standardized mean difference per frequency bin, for plotting only.

    `spec_a`, `spec_b` are `(n_a, n_bins)` and `(n_b, n_bins)` arrays (e.g. rows are
    per-utterance log-power spectra, columns are frequency bins). Returns a
    `(n_bins,)` array of Cohen's-d-style standardized differences, one per bin,
    computed independently per column with the same pooled-SD, n-1 convention as
    `cohens_d` (vectorised here for speed rather than looping `cohens_d` per bin).

    Deliberately returns **no p-values and no significance flags** — see the module
    docstring for why running one hypothesis test per bin (with or without multiple-
    comparison correction) is the wrong tool at this scale, and why standardized
    effect size is the quantity this project visualises instead.
    """
    spec_a = np.asarray(spec_a, dtype=float)
    spec_b = np.asarray(spec_b, dtype=float)
    if spec_a.ndim != 2 or spec_b.ndim != 2:
        raise ValueError("spec_a and spec_b must be 2D (n_samples, n_bins)")
    if spec_a.shape[1] != spec_b.shape[1]:
        raise ValueError(
            f"spec_a and spec_b must have the same number of bins; "
            f"got {spec_a.shape[1]} and {spec_b.shape[1]}"
        )
    n_a, n_b = spec_a.shape[0], spec_b.shape[0]
    if n_a + n_b <= 2:
        raise ValueError("per_bin_effect_size requires n_a + n_b > 2 to estimate a pooled SD")

    mean_a = spec_a.mean(axis=0)
    mean_b = spec_b.mean(axis=0)
    var_a = spec_a.var(axis=0, ddof=1) if n_a > 1 else np.zeros(spec_a.shape[1])
    var_b = spec_b.var(axis=0, ddof=1) if n_b > 1 else np.zeros(spec_b.shape[1])
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    diff = mean_a - mean_b
    zero_diff_fallback = np.where(diff == 0, 0.0, np.inf * np.sign(diff))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(pooled_sd > 0, diff / pooled_sd, zero_diff_fallback)
    return d.astype(float)
