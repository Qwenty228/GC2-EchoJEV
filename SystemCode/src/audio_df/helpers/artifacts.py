"""Trained model artifacts: which systems this machine has, and each one's decision threshold.

Weights live in <root>/artifacts/<system>/<timestamp>/; a COMPLETE file marks a finished one.
Set ADF_ARTIFACTS to use another folder instead (e.g. a volume mounted into a container; the image
ships code only, never weights).

Each evaluated artifact keeps a copy of its latest calibration in threshold.json, written by the
runtime after every evaluation, so the threshold travels with the weights. For artifacts evaluated
before that existed: `uv run python -m audio_df.helpers.artifacts` (see backfill_thresholds).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

__all__ = ["SYSTEMS", "artifacts_dir", "latest", "available", "save_threshold", "threshold",
           "backfill_thresholds"]

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


def save_threshold(artifact: str | Path, calibration: dict[str, Any], run: str) -> None:
    """Store a run's calibration (eer_threshold, source, ...) with the artifact it calibrated."""
    record = {**calibration, "run": run}
    (Path(artifact) / THRESHOLD_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")


def threshold(artifact: str | Path) -> dict[str, Any] | None:
    """The artifact's stored calibration, or None when it was never evaluated."""
    path = Path(artifact) / THRESHOLD_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def backfill_thresholds(root: str | Path) -> dict[str, str]:
    """Write threshold.json from the newest runs/*/calibration.json of each artifact that has none.

    Covers every artifact present on this machine, plus each system's newest evaluated artifact
    even when its weights are elsewhere: that folder then holds only threshold.json (no COMPLETE,
    so it is not available here) and the file joins the weights on the machine that has them.
    """
    from audio_df.helpers.research.runs import artifact_key

    newest_run: dict[str, Path] = {}  # "system/timestamp" -> newest run that calibrated it
    for run in sorted((Path(root) / "runs").glob("E-*")):  # run names start with a timestamp
        cfg_path = run / "config.yaml"
        if (run / "calibration.json").is_file() and cfg_path.is_file():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            key = artifact_key(cfg.get("model_artifact", ""))
            if key.split("/")[0] in SYSTEMS:
                newest_run[key] = run
    newest_artifact = {s: max((k for k in newest_run if k.startswith(f"{s}/")), default=None)
                       for s in SYSTEMS}
    done = {}
    for key, run in sorted(newest_run.items()):
        artifact = artifacts_dir(root) / key
        if (artifact.is_dir() or key in newest_artifact.values()) and threshold(artifact) is None:
            artifact.mkdir(parents=True, exist_ok=True)
            calibration = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
            save_threshold(artifact, calibration, run.name)
            done[key] = run.name
    return done


if __name__ == "__main__":
    for key, run in backfill_thresholds(Path.cwd()).items():
        print(f"{key}: threshold from {run}")
