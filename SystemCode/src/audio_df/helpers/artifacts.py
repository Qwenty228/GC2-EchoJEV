"""Trained model artifacts: which systems this machine has, and each system's decision threshold.

Weights live in <root>/artifacts/<system>/<timestamp>/; a COMPLETE file marks a finished one.
Set ADF_ARTIFACTS to use another folder instead (e.g. a volume mounted into a container; the image
ships code only, never weights).

Each system has one artifacts/<system>/threshold.json: the latest dev calibration of the artifact
it names (eer_threshold; platt [a, b] for P(bona fide) = sigmoid(a * score + b)) and its EER on
2021 LA eval (error_rate_pct, None when that run did not test it). The runtime replaces it after
every evaluation; older ones stay in runs/*/calibration.json. Only used for the artifact it names.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from scipy.special import expit

__all__ = ["SYSTEMS", "artifacts_dir", "latest", "available", "platt", "p_bonafide",
           "save_threshold", "threshold", "write_latest_thresholds"]

SYSTEMS = ("a1", "a2", "a4", "d1", "d2", "d3")
THRESHOLD_FILE = "threshold.json"


def artifacts_dir(root: str | Path) -> Path:
    """ADF_ARTIFACTS when set, otherwise <root>/artifacts."""
    override = os.getenv("ADF_ARTIFACTS")
    return Path(override) if override else Path(root) / "artifacts"


def latest(system: str, root: str | Path) -> Path | None:
    """Newest COMPLETE artifact of one system, or None when this machine has none."""
    parent = artifacts_dir(root) / system
    if not parent.is_dir():
        return None
    done = sorted(p for p in parent.iterdir() if (p / "COMPLETE").is_file())
    return done[-1] if done else None


def available(root: str | Path) -> dict[str, Path]:
    """{system: newest complete artifact} for every system present on this machine."""
    found = {s: latest(s, root) for s in SYSTEMS}
    return {s: p for s, p in found.items() if p is not None}


def platt(dev_scores) -> list[float]:
    """[a, b] of a logistic fit on dev scores: P(bona fide) = sigmoid(a * score + b)."""
    from sklearn.linear_model import LogisticRegression

    fit = LogisticRegression().fit(dev_scores[["score"]], dev_scores.label == "bonafide")
    return [float(fit.coef_[0][0]), float(fit.intercept_[0])]


def p_bonafide(record: dict[str, Any] | None, score: float) -> float | None:
    """Calibrated P(bona fide) for a score, or None when the record has no Platt fit."""
    if not record or not record.get("platt"):
        return None
    a, b = record["platt"]
    return float(expit(a * score + b))


def save_threshold(artifact: str | Path, calibration: dict[str, Any], run: str) -> None:
    """Make a run's calibration the system's threshold, replacing the previous one."""
    artifact = Path(artifact)
    record = {"artifact": artifact.name, "run": run, **calibration}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    (artifact.parent / THRESHOLD_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")


def threshold(artifact: str | Path) -> dict[str, Any] | None:
    """The system's stored calibration if it was made for this artifact, else None."""
    artifact = Path(artifact)
    path = artifact.parent / THRESHOLD_FILE
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    return record if record.get("artifact") == artifact.name else None


def write_latest_thresholds(root: str | Path) -> dict[str, str]:
    """threshold.json per system from runs/, for its newest evaluated artifact.

    The threshold comes from that artifact's newest calibration.json, the Platt fit from its newest
    scores/dev.csv, and error_rate_pct from its newest metrics_asvspoof2021_eval.json.
    """
    import pandas as pd

    from audio_df.helpers.research.runs import artifact_key

    runs: dict[str, list[Path]] = {}  # "system/timestamp" -> its runs, oldest first
    for run in sorted((Path(root) / "runs").glob("E-*")):  # run names start with a timestamp
        cfg_path = run / "config.yaml"
        if cfg_path.is_file():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            key = artifact_key(cfg.get("model_artifact", ""))
            if key.split("/")[0] in SYSTEMS:
                runs.setdefault(key, []).append(run)

    def newest(key: str, name: str) -> Path | None:
        return next((r for r in reversed(runs[key]) if (r / name).is_file()), None)

    written = {}
    for system in SYSTEMS:
        own = [k for k in runs if k.startswith(f"{system}/")]
        evaluated = [k for k in own if newest(k, "calibration.json")]
        if not evaluated:
            continue
        key = max(evaluated)
        cal, dev, la = (newest(key, n) for n in
                        ("calibration.json", "scores/dev.csv", "metrics_asvspoof2021_eval.json"))
        record = json.loads((cal / "calibration.json").read_text(encoding="utf-8"))
        record["platt"] = platt(pd.read_csv(dev / "scores" / "dev.csv")) if dev else None
        la_metrics = (la / "metrics_asvspoof2021_eval.json") if la else None
        la_eer = json.loads(la_metrics.read_text())["eer"] if la_metrics else None
        record["error_rate_pct"] = None if la_eer is None else round(100 * la_eer, 1)
        save_threshold(artifacts_dir(root) / key, record, cal.name)
        written[system] = f"{key} from {cal.name}"
    return written


if __name__ == "__main__":
    for system, source in write_latest_thresholds(Path.cwd()).items():
        print(f"{system}: {source}")
