"""Stage 0 leakage audit: can a trivial model predict the label from nuisance metadata alone?

Per `docs/handoff.md` ("Stage 0 - Dataset and leakage audit", "Leakage tests"): before any
modelling, fit tiny classifiers on metadata-only features (duration, sample rate, codec,
RMS/loudness, leading/trailing/total silence, source corpus, speaker identity) and check
whether any of them predicts the bonafide/spoof label "suspiciously well". If one does, that
is a shortcut a real model would learn instead of solving the intended detection problem
(the handoff's own example: "microphone noise -> real" or "clean WAV -> fake").

This module never trains anything for the actual detection task (see `tasks/CONVENTIONS.md`'s
module table) — its classifiers exist only to answer "is a shortcut trivially available",
never to be used downstream.

## Threshold convention (binding: see `research/knowledge/methods/leakage-audit.md`)

The researcher's verified note establishes that **no literature-sourced AUC threshold for
"leakage" exists** for this kind of metadata-shortcut audit. The `threshold=0.65` default
used throughout this module is a **project-local convention**, not a citation. It is chosen
to be conservative-but-not-paranoid: the one directly comparable published number (Müller
et al. 2021, arXiv:2106.12914 — a single leading-silence-duration feature reaching 15.1% EER,
i.e. an AUC well above 0.65 on ASVspoof 2019 LA eval) would clearly still be caught by this
threshold. Never cite 0.65 as if it came from a paper; it did not.

## Categorical encoding choice

`codec` and `source_corpus` are typically low-cardinality (a handful of distinct values across
the whole manifest). `speaker_id` is high-cardinality and is the special case the handoff
calls out: if speakers are wholly disjoint across classes (every speaker's utterances all fall
in one class), that disjointness *is* the leak — a model doesn't need to learn anything
acoustic, it only needs to recognise a voice it has seen before.

All categorical nuisances are encoded the same way, uniformly, via **fold-safe (out-of-fold)
target encoding**: within each cross-validation fold, every category is mapped to the
training-fold's empirical spoof rate for that category; a category unseen in the training
fold (including every row of a held-out category) is mapped to the training-fold's overall
spoof rate (i.e. ~no information, ~0.5-ish depending on class balance). This is chosen over
one-hot encoding for two reasons: (1) it unifies the numeric and categorical code paths — the
encoded value is just another scalar feature handed to the same tiny classifiers — and
(2) it is what makes the speaker-identity case actually work as a *leakage test* rather than
a data-cleaning step: with row-level stratified K-fold and multiple utterances per speaker,
most validation rows' speakers were already seen in that fold's training portion, so their
encoded value is close to 0 or 1 exactly when that speaker is class-pure, which a depth-2 tree
or logistic regression trivially thresholds. A speaker seen only in a held-out fold (never in
that fold's training data) contributes an uninformative ~overall-rate value, which is the
correct behaviour: an audit run on a fold where a speaker is *never* seen in training cannot
possibly detect leakage from that never-seen speaker, honestly reflecting the fold's real
information content rather than overstating it via row-level shuffling artifacts.

The alternative (one-hot encoding for high-cardinality speaker_id) was rejected: it would
either (a) blow up dimensionality for tiny models for no benefit, or (b) assign an all-zero
vector to any speaker not seen in training, which — combined with typical random row-level CV
that does see most speakers in both train and val — behaves almost identically to target
encoding here but is more code for the same answer. Target encoding is simpler and uniform
across every categorical nuisance, so it is used throughout.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from audio_df import audio as audio_mod
from audio_df import metrics

__all__ = [
    "NUISANCE_FEATURES",
    "extract_nuisance_features",
    "audit_single_feature",
    "audit_all",
    "run_audit",
]

# Project-local convention (NOT a literature threshold) — see module docstring.
DEFAULT_LEAKAGE_THRESHOLD = 0.65

_N_BOOTSTRAP = 1000
_N_SPLITS_DEFAULT = 5


def _labels_to_int(labels) -> np.ndarray:
    """Map bonafide/spoof (string or 0/1) labels to an int array, 1 = spoof.

    Mirrors `metrics._normalize_labels`'s accepted vocabulary without importing that private
    helper — this module still delegates the actual AUC/EER *arithmetic* to `metrics.py`;
    this only prepares an integer `y` for sklearn's `.fit()`, which `metrics.py` has no
    equivalent public helper for.
    """
    out = np.empty(len(labels), dtype=int)
    labels = list(labels)
    for i, raw in enumerate(labels):
        if isinstance(raw, str):
            low = raw.strip().lower()
            if low in {"bonafide", "bona-fide", "genuine", "real"}:
                out[i] = 0
            elif low in {"spoof", "fake", "synthetic"}:
                out[i] = 1
            else:
                raise ValueError(f"unrecognised string label {raw!r} at index {i}")
        elif isinstance(raw, (bool, np.bool_)):
            out[i] = int(raw)
        elif isinstance(raw, (int, np.integer, float, np.floating)):
            if raw in (0, 1):
                out[i] = int(raw)
            else:
                raise ValueError(f"ambiguous numeric label {raw!r} at index {i}")
        else:
            raise ValueError(f"unrecognised label type {type(raw)!r} at index {i}")
    return out


def _resolve_audio_path(row: pd.Series, audio_root: str | Path | None) -> Path:
    """Locate a manifest row's audio file, tolerating both manifest path conventions.

    Two conventions exist in this repo and both must work. `data.build_manifest` writes
    `path` with `dataset.root` already joined on (`data/raw/.../LA/..._train/flac/x.flac`),
    while `run_audit` separately passes that same `dataset.root` in as `audio_root`. A
    manifest may also carry paths relative *to* the root. So try the path as written first
    and only fall back to joining `audio_root`; joining unconditionally built `<root>/<root>/
    ...`, and because the caller converts `FileNotFoundError` to NaN, every audio-derived
    feature silently vanished from the report instead of failing.
    """
    p = Path(row["path"])
    if p.exists():
        return p
    if audio_root is not None and not p.is_absolute():
        candidate = Path(audio_root) / p
        if candidate.exists():
            return candidate
    raise FileNotFoundError(str(p))


def _extract_rms(row: pd.Series, audio_root: str | Path | None = None) -> float:
    path = _resolve_audio_path(row, audio_root)
    x, _sr = audio_mod.load_audio(str(path))
    return audio_mod.rms(x)


def _extract_silence(key: str) -> Callable[[pd.Series, Any], float]:
    def _fn(row: pd.Series, audio_root: str | Path | None = None) -> float:
        path = _resolve_audio_path(row, audio_root)
        x, _sr = audio_mod.load_audio(str(path))
        return audio_mod.silence_ratio(x)[key]

    return _fn


# name -> extractor(row, audio_root) -> scalar value.
# The metadata-only extractors read straight from manifest columns produced by
# `data.build_manifest` (see `configs/data.yaml: manifest.columns`); the audio-derived ones
# (rms, silence_*) load the waveform via `audio.py` (owned by another agent, used read-only
# here) and are only computable when the audio file actually exists on disk.
NUISANCE_FEATURES: dict[str, Callable[[pd.Series, Any], Any]] = {
    "duration_s": lambda row, audio_root=None: row["duration_s"],
    "sample_rate": lambda row, audio_root=None: row["sample_rate"],
    "codec": lambda row, audio_root=None: row["codec"],
    "source_corpus": lambda row, audio_root=None: row["source_corpus"],
    "speaker_id": lambda row, audio_root=None: row["speaker_id"],
    "rms": _extract_rms,
    "silence_leading": _extract_silence("leading"),
    "silence_trailing": _extract_silence("trailing"),
    "silence_total": _extract_silence("total"),
}


def extract_nuisance_features(
    manifest: pd.DataFrame, audio_root: str | Path | None = None
) -> pd.DataFrame:
    """Apply every extractor in `NUISANCE_FEATURES` to every row of `manifest`.

    Metadata-only features (duration_s, sample_rate, codec, source_corpus, speaker_id) are
    read straight from manifest columns and always populate. Audio-derived features (rms,
    silence_leading/trailing/total) require the file at `row['path']` (optionally resolved
    under `audio_root`) to exist; when it does not (e.g. synthetic manifests in tests that
    carry no real audio), that cell is left as `NaN` rather than raising, so metadata-only
    audits still run without a dataset present. A column is entirely `NaN` when no row's
    audio could be loaded — `audit_all` drops such columns rather than auditing pure noise.
    """
    records = []
    for _, row in manifest.iterrows():
        rec = {}
        for name, fn in NUISANCE_FEATURES.items():
            try:
                rec[name] = fn(row, audio_root)
            except (FileNotFoundError, OSError, KeyError):
                rec[name] = np.nan
        records.append(rec)
    return pd.DataFrame.from_records(records)


def _is_categorical(values: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(values)


def _target_encode_fold(
    train_values: np.ndarray, train_y: np.ndarray, val_values: np.ndarray
) -> np.ndarray:
    """Fold-safe target encoding: map each category to its training-fold spoof rate.

    Unseen categories (present in `val_values` but not `train_values`) get the training
    fold's overall spoof rate — see module docstring for why this is the right behaviour
    for the speaker-identity leakage case.
    """
    overall_rate = float(np.mean(train_y))
    df = pd.DataFrame({"cat": train_values, "y": train_y})
    rates = df.groupby("cat")["y"].mean().to_dict()
    return np.array([rates.get(v, overall_rate) for v in val_values], dtype=float)


def _fit_and_score(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    """Fit LogisticRegression and a depth-2 DecisionTree; return each model's spoof-prob on val."""
    out = {}

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    out["logreg"] = logreg.predict_proba(X_val)[:, 1]

    tree = DecisionTreeClassifier(max_depth=2, random_state=seed)
    tree.fit(X_train, y_train)
    out["tree"] = tree.predict_proba(X_val)[:, 1]

    return out


def _cross_val_oof_scores(
    X: np.ndarray, y: np.ndarray, is_categorical: bool, seed: int
) -> dict[str, np.ndarray]:
    """Out-of-fold spoof-probability predictions for logreg and tree, via stratified K-fold.

    Evaluating on held-out folds (never on training rows) is required per
    `tasks/CONVENTIONS.md`/the task spec: "a tiny model scoring 1.0 AUC on its own training
    rows proves nothing." `n_splits` shrinks automatically for tiny/imbalanced inputs so this
    also works on the small synthetic manifests the tests use.
    """
    n = len(y)
    min_class_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits = max(2, min(_N_SPLITS_DEFAULT, min_class_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = {"logreg": np.full(n, np.nan), "tree": np.full(n, np.nan)}

    for train_idx, val_idx in skf.split(np.zeros(n), y):
        y_train = y[train_idx]

        if is_categorical:
            raw = X[:, 0]
            enc = _target_encode_fold(raw[train_idx], y_train, raw[val_idx])
            enc_train = _target_encode_fold(raw[train_idx], y_train, raw[train_idx])
            X_train = enc_train.reshape(-1, 1)
            X_val = enc.reshape(-1, 1)
        else:
            X_train = X[train_idx]
            X_val = X[val_idx]

        scores = _fit_and_score(X_train, y_train, X_val, seed)
        for model_name, s in scores.items():
            oof[model_name][val_idx] = s

    return oof


def _bootstrap_ci(
    values: np.ndarray, labels_int: np.ndarray, metric_fn, seed: int, n_boot: int = _N_BOOTSTRAP
) -> tuple[float, float]:
    """Percentile bootstrap CI (2.5, 97.5) for `metric_fn(scores, labels)`, resampling rows."""
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_boot = labels_int[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        stats.append(metric_fn(values[idx], y_boot))
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def _auc_metric(scores: np.ndarray, y_int: np.ndarray) -> float:
    return metrics.roc_auc(scores, y_int, higher_is_spoof=True)


def _eer_metric(scores: np.ndarray, y_int: np.ndarray) -> float:
    eer, _thr = metrics.compute_eer(scores, y_int, higher_is_spoof=True)
    return eer


def audit_single_feature(values, labels, seed: int = 0) -> dict:
    """Fit tiny classifiers (logistic regression, depth-2 decision tree) via cross-validation
    on one nuisance feature and report AUC/EER with bootstrap confidence intervals.

    `values` may be numeric (used directly) or categorical/string (fold-safe target-encoded
    per-fold — see module docstring). Evaluation uses out-of-fold predictions only (never
    scores a model on its own training rows). Of the two tiny models, the one with the higher
    out-of-fold AUC is reported (a shortcut only needs to be found by *one* trivial model to
    be a real risk — see task spec: "the question is 'is this shortcut trivially available'").

    Rows with `NaN` in `values` are dropped before fitting.

    Returns a dict:
        {
          "auc": float, "auc_ci": (lo, hi),
          "eer": float, "eer_ci": (lo, hi),
          "model": "logreg" | "tree",
          "n": int,
        }
    """
    values_arr = pd.Series(values).reset_index(drop=True)
    labels_arr = pd.Series(labels).reset_index(drop=True)

    mask = ~values_arr.isna()
    values_arr = values_arr[mask].reset_index(drop=True)
    labels_arr = labels_arr[mask].reset_index(drop=True)

    y = _labels_to_int(labels_arr)
    is_cat = _is_categorical(values_arr)

    if is_cat:
        X = values_arr.to_numpy().reshape(-1, 1)
    else:
        X = values_arr.to_numpy(dtype=float).reshape(-1, 1)

    oof = _cross_val_oof_scores(X, y, is_cat, seed)

    best_model = max(oof, key=lambda m: _auc_metric(oof[m], y))
    best_scores = oof[best_model]

    auc = _auc_metric(best_scores, y)
    eer = _eer_metric(best_scores, y)
    auc_ci = _bootstrap_ci(best_scores, y, _auc_metric, seed)
    eer_ci = _bootstrap_ci(best_scores, y, _eer_metric, seed + 1)

    return {
        "auc": auc,
        "auc_ci": auc_ci,
        "eer": eer,
        "eer_ci": eer_ci,
        "model": best_model,
        "n": int(len(y)),
    }


def _audit_combined(manifest_features: pd.DataFrame, labels, seed: int) -> dict:
    """Combined-model leakage audit: all nuisance features together in one tiny classifier."""
    labels_arr = pd.Series(labels).reset_index(drop=True)
    feats = manifest_features.reset_index(drop=True)

    # Drop rows with any NaN across the columns actually used, and drop all-NaN columns.
    usable_cols = [c for c in feats.columns if feats[c].notna().any()]
    feats = feats[usable_cols]
    row_mask = feats.notna().all(axis=1)
    feats = feats[row_mask].reset_index(drop=True)
    labels_arr = labels_arr[row_mask].reset_index(drop=True)

    y = _labels_to_int(labels_arr)
    n = len(y)
    min_class_count = int(min(np.sum(y == 0), np.sum(y == 1)))
    n_splits = max(2, min(_N_SPLITS_DEFAULT, min_class_count))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cat_cols = [c for c in usable_cols if _is_categorical(feats[c])]
    num_cols = [c for c in usable_cols if c not in cat_cols]

    oof = {"logreg": np.full(n, np.nan), "tree": np.full(n, np.nan)}

    for train_idx, val_idx in skf.split(np.zeros(n), y):
        y_train = y[train_idx]
        parts_train = []
        parts_val = []
        if num_cols:
            parts_train.append(feats[num_cols].to_numpy(dtype=float)[train_idx])
            parts_val.append(feats[num_cols].to_numpy(dtype=float)[val_idx])
        for c in cat_cols:
            raw = feats[c].to_numpy()
            enc_train = _target_encode_fold(raw[train_idx], y_train, raw[train_idx])
            enc_val = _target_encode_fold(raw[train_idx], y_train, raw[val_idx])
            parts_train.append(enc_train.reshape(-1, 1))
            parts_val.append(enc_val.reshape(-1, 1))

        X_train = np.concatenate(parts_train, axis=1)
        X_val = np.concatenate(parts_val, axis=1)

        scores = _fit_and_score(X_train, y_train, X_val, seed)
        for model_name, s in scores.items():
            oof[model_name][val_idx] = s

    best_model = max(oof, key=lambda m: _auc_metric(oof[m], y))
    best_scores = oof[best_model]

    auc = _auc_metric(best_scores, y)
    eer = _eer_metric(best_scores, y)
    auc_ci = _bootstrap_ci(best_scores, y, _auc_metric, seed)
    eer_ci = _bootstrap_ci(best_scores, y, _eer_metric, seed + 1)

    return {
        "auc": auc,
        "auc_ci": auc_ci,
        "eer": eer,
        "eer_ci": eer_ci,
        "model": best_model,
        "n": int(n),
    }


def audit_all(
    manifest_features: pd.DataFrame,
    labels,
    threshold: float = DEFAULT_LEAKAGE_THRESHOLD,
    seed: int = 0,
) -> pd.DataFrame:
    """Audit every nuisance feature alone, plus all of them together, for label leakage.

    `threshold` is the project-local AUC convention documented in this module's docstring —
    NOT a literature-sourced value. Emits a `UserWarning` reading
    `"LEAKAGE WARNING: feature '<name>' ..."` (unmissable, easy to `pytest.warns`/grep for)
    for every row whose AUC exceeds `threshold`, naming that feature.

    Returns a DataFrame with one row per single feature plus a final `"combined"` row, columns
    `feature, auc, auc_ci_lo, auc_ci_hi, eer, eer_ci_lo, eer_ci_hi, model, n, flagged`.
    Columns that are entirely `NaN` (e.g. audio-derived features with no accessible audio) are
    skipped — auditing pure noise would not be a leakage test.
    """
    rows = []
    for name in manifest_features.columns:
        col = manifest_features[name]
        if col.isna().all():
            continue
        result = audit_single_feature(col, labels, seed=seed)
        rows.append({"feature": name, **result})

    combined = _audit_combined(manifest_features, labels, seed=seed)
    rows.append({"feature": "combined", **combined})

    report = pd.DataFrame(
        [
            {
                "feature": r["feature"],
                "auc": r["auc"],
                "auc_ci_lo": r["auc_ci"][0],
                "auc_ci_hi": r["auc_ci"][1],
                "eer": r["eer"],
                "eer_ci_lo": r["eer_ci"][0],
                "eer_ci_hi": r["eer_ci"][1],
                "model": r["model"],
                "n": r["n"],
            }
            for r in rows
        ]
    )
    report["flagged"] = report["auc"] > threshold

    for _, row in report.iterrows():
        if row["flagged"]:
            warnings.warn(
                f"LEAKAGE WARNING: feature '{row['feature']}' predicts the label with "
                f"AUC={row['auc']:.3f} (threshold={threshold}, a project-local convention, "
                "not a literature value) — this nuisance feature is a shortcut a real "
                "detector could learn instead of solving the intended problem.",
                UserWarning,
                stacklevel=2,
            )

    return report


def run_audit(cfg: dict) -> pd.DataFrame:
    """End-to-end Stage 0 leakage audit driven by a parsed config dict.

    Builds the manifest via `audio_df.data.build_manifest(cfg)`, extracts nuisance features
    (optionally loading audio under `cfg['dataset']['root']` for RMS/silence features),
    audits every feature plus the combined model, writes the report to
    `reports/tables/leakage_audit.csv`, and returns the report DataFrame. `cfg` may set
    `cfg['leakage_audit']['threshold']` (default `DEFAULT_LEAKAGE_THRESHOLD`) and
    `cfg['seed']` (default 0).
    """
    from audio_df.data import build_manifest  # local import: keep leakage.py's import

    # surface unusable to callers that only want extract/audit on a pre-built manifest.
    manifest = build_manifest(cfg)

    audit_cfg = cfg.get("leakage_audit", {})
    threshold = audit_cfg.get("threshold", DEFAULT_LEAKAGE_THRESHOLD)
    seed = cfg.get("seed", 0)
    audio_root = cfg.get("dataset", {}).get("root")

    features = extract_nuisance_features(manifest, audio_root=audio_root)
    report = audit_all(features, manifest["label"], threshold=threshold, seed=seed)

    out_dir = Path(audit_cfg.get("out_dir", "reports/tables"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "leakage_audit.csv"
    report.to_csv(out_path, index=False)

    return report
