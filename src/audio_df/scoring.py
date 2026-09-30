"""Turn an experiment's `scores/*.csv` files into `metrics.json` — and nothing else.

Per the project protocol, metrics are **never** computed inside training code and a
score file is produced once and never re-run in place; this module is the single
place that reads score files and writes metrics. It reimplements no metric: every
number here comes from `audio_df.metrics` (`compute_eer`, `roc_auc`, `min_dcf`,
`prf1_at_threshold`, `per_attack_eer`).

## Score polarity

Every `scores/<testset>.csv` in this project follows the fixed SCORE CONVENTION
(higher score = more bona fide, i.e. `higher_is_spoof=False`). This module hardcodes
that polarity when calling `audio_df.metrics` — it is not a per-call guess, it is the
project-wide file format contract score files are written under (`exp_io.write_scores`
validates the `label` column but the score *direction* is a convention documented in
`experiments/README.md` and `exp_io`'s module docstring, not re-validated numerically
here since there is no way to check "is this score oriented correctly" from the
numbers alone).

## Threshold selection

The operating threshold used for F1/precision/recall on every test set is the EER
threshold computed once on `dev` scores — this experiment's own `scores/dev.csv`, or
another experiment's via `threshold_from` (for dev-less subset runs, e.g. `jev_dev500`,
`eff_eval1k`). It is never fit on eval/itw/etc. If neither is available, F1 is
reported as `null` with an explicit reason string rather than silently omitted.

Decision rule (tested explicitly in `tests/test_scoring.py`, not just documented):
predicted spoof iff `score <= threshold` (project convention `higher_is_spoof=False`;
this is exactly `audio_df.metrics.prf1_at_threshold`'s built-in rule for that polarity,
not a separate reimplementation here).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from audio_df.metrics import compute_eer, min_dcf, per_attack_eer, prf1_at_threshold, roc_auc

__all__ = ["score_experiment"]

_HIGHER_IS_SPOOF = False  # fixed project score-file convention; see module docstring


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _hash_scoring_config(cfg: dict[str, Any]) -> str:
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_scoring_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    for required in ("min_dcf", "threshold", "f1"):
        if required not in cfg:
            raise ValueError(f"scoring config {path} missing required section {required!r}")
    return cfg


def _load_run_summary(exp_dir: Path) -> dict[str, Any] | None:
    """Read `exp_dir/run_summary.json` (peak RSS, swap, wall time, ...) if present.

    Copied verbatim into `metrics.json`'s `"resources"` key (see
    `exp_io.format_resources` for the keys this dict is expected to carry and how
    the registry renders it compactly). Returns `None` (not an error) when the
    file does not exist — most experiments will not have measured resource usage
    yet, and that is a normal, expected state, not a scoring failure.
    """
    path = exp_dir / "run_summary.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _load_score_files(exp_dir: Path) -> dict[str, pd.DataFrame]:
    scores_dir = exp_dir / "scores"
    if not scores_dir.is_dir():
        raise FileNotFoundError(f"{scores_dir} does not exist; nothing to score")
    out = {}
    for path in sorted(scores_dir.glob("*.csv")):
        out[path.stem] = pd.read_csv(path)
    if not out:
        raise FileNotFoundError(f"no score CSVs found under {scores_dir}")
    return out


def _load_manifest_attack_map(manifest_paths: list[str | Path]) -> dict[str, str]:
    """Build `{clip_id: attack_id}` from one or more manifest CSVs.

    Manifests use `utt_id` as their row key (`configs/data.yaml:manifest.columns`);
    score files use `clip_id`. This project's convention is that `clip_id` values are
    exactly the manifest's `utt_id` values, so the join key is `utt_id`.
    """
    mapping: dict[str, str] = {}
    for p in manifest_paths:
        df = pd.read_csv(p)
        if "utt_id" not in df.columns or "attack_id" not in df.columns:
            raise ValueError(f"manifest {p} must have 'utt_id' and 'attack_id' columns")
        mapping.update(dict(zip(df["utt_id"], df["attack_id"], strict=True)))
    return mapping


def _resolve_attack_ids(
    testset_name: str,
    df: pd.DataFrame,
    manifest_attack_map: dict[str, str] | None,
) -> tuple[list[str] | None, str | None]:
    """Returns `(attack_ids_aligned_to_df, reason_if_none)`."""
    if "attack_id" in df.columns:
        return df["attack_id"].astype(str).tolist(), None
    if manifest_attack_map is None:
        return None, "no attack_id column in score file and no manifest_paths provided"
    missing = [c for c in df["clip_id"] if c not in manifest_attack_map]
    if missing:
        return None, (
            f"{len(missing)} clip_id(s) in '{testset_name}' not found in provided manifests "
            f"(e.g. {missing[:5]})"
        )
    return [manifest_attack_map[c] for c in df["clip_id"]], None


def _dev_threshold(
    exp_dir: Path,
    threshold_from: Path | None,
    scores_by_testset: dict[str, pd.DataFrame],
) -> tuple[float | None, dict[str, Any]]:
    """Returns `(threshold_or_None, info_dict)` describing where it came from."""
    if threshold_from is not None:
        dev_path = Path(threshold_from) / "scores" / "dev.csv"
        if not dev_path.exists():
            return None, {
                "exp_dir": str(threshold_from),
                "testset": "dev",
                "value": None,
                "reason": f"threshold_from given but {dev_path} does not exist",
            }
        dev_df = pd.read_csv(dev_path)
        source_dir = str(threshold_from)
    elif "dev" in scores_by_testset:
        dev_df = scores_by_testset["dev"]
        source_dir = str(exp_dir)
    else:
        return None, {
            "exp_dir": None,
            "testset": "dev",
            "value": None,
            "reason": "no scores/dev.csv in this experiment and no threshold_from given",
        }

    eer, thr = compute_eer(
        dev_df["score"], dev_df["label"], higher_is_spoof=_HIGHER_IS_SPOOF
    )
    return thr, {
        "exp_dir": source_dir,
        "testset": "dev",
        "value": float(thr),
        "dev_eer": float(eer),
        "reason": None,
    }


def _score_one_testset(
    name: str,
    df: pd.DataFrame,
    min_dcf_cfg: dict[str, Any],
    threshold: float | None,
    is_dev: bool,
    manifest_attack_map: dict[str, str] | None,
) -> dict[str, Any]:
    scores = df["score"]
    labels = df["label"]
    n = len(df)
    n_bonafide = int((labels == "bonafide").sum())
    n_spoof = int((labels == "spoof").sum())

    result: dict[str, Any] = {"n": n, "n_bonafide": n_bonafide, "n_spoof": n_spoof}

    eer, eer_thr = compute_eer(scores, labels, higher_is_spoof=_HIGHER_IS_SPOOF)
    result["eer"] = float(eer)
    if is_dev:
        result["eer_threshold"] = float(eer_thr)

    dcf, dcf_thr = min_dcf(
        scores,
        labels,
        higher_is_spoof=_HIGHER_IS_SPOOF,
        c_miss=min_dcf_cfg["c_miss"],
        c_fa=min_dcf_cfg["c_fa"],
        p_spoof=min_dcf_cfg["p_spoof"],
        normalize=True,
    )
    result["min_dcf"] = float(dcf)
    result["min_dcf_threshold"] = float(dcf_thr)
    result["min_dcf_params"] = {
        "c_miss": min_dcf_cfg["c_miss"],
        "c_fa": min_dcf_cfg["c_fa"],
        "p_spoof": min_dcf_cfg["p_spoof"],
    }

    result["roc_auc"] = float(roc_auc(scores, labels, higher_is_spoof=_HIGHER_IS_SPOOF))

    if threshold is None:
        result["f1"] = None
        result["precision"] = None
        result["recall"] = None
        result["f1_reason"] = "no dev threshold available (see threshold_source)"
    else:
        prf1 = prf1_at_threshold(scores, labels, thr=threshold, higher_is_spoof=_HIGHER_IS_SPOOF)
        result["f1"] = prf1["f1"]
        result["precision"] = prf1["precision"]
        result["recall"] = prf1["recall"]

    attack_ids, reason = _resolve_attack_ids(name, df, manifest_attack_map)
    if attack_ids is None:
        result["per_attack_eer"] = None
        result["per_attack_eer_reason"] = reason
    else:
        result["per_attack_eer"] = per_attack_eer(
            scores, labels, attack_ids, higher_is_spoof=_HIGHER_IS_SPOOF
        )

    return result


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".metrics-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def score_experiment(
    exp_dir: str | Path,
    scoring_cfg_path: str | Path = "configs/scoring.yaml",
    threshold_from: str | Path | None = None,
    manifest_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Score every `exp_dir/scores/*.csv` and write `exp_dir/metrics.json`.

    See the module docstring for polarity and threshold-selection conventions.
    Returns the same dict that is written to `metrics.json`.
    """
    exp_dir = Path(exp_dir)
    scoring_cfg = _load_scoring_config(scoring_cfg_path)
    scores_by_testset = _load_score_files(exp_dir)

    threshold, threshold_info = _dev_threshold(
        exp_dir, Path(threshold_from) if threshold_from is not None else None, scores_by_testset
    )

    manifest_attack_map = (
        _load_manifest_attack_map(manifest_paths) if manifest_paths else None
    )

    testset_results = {}
    for name, df in scores_by_testset.items():
        testset_results[name] = _score_one_testset(
            name,
            df,
            scoring_cfg["min_dcf"],
            threshold,
            is_dev=(name == "dev"),
            manifest_attack_map=manifest_attack_map,
        )

    score_file_hashes = {
        name: _sha256_of_file(exp_dir / "scores" / f"{name}.csv") for name in scores_by_testset
    }

    payload = {
        "exp_id": exp_dir.name,
        "scoring_config": scoring_cfg,
        "scoring_config_hash": _hash_scoring_config(scoring_cfg),
        "threshold_source": threshold_info,
        "score_file_hashes": score_file_hashes,
        "testsets": testset_results,
        "resources": _load_run_summary(exp_dir),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }

    _atomic_write_json(exp_dir / "metrics.json", payload)
    return payload
