"""Trained model artifacts: which systems this machine has, and each system's decision threshold.

Weights live in <root>/artifacts/<system>/<timestamp>/; a COMPLETE file marks a finished one.
Set ADF_ARTIFACTS to use another folder instead (e.g. a volume mounted into a container; the image
ships code only, never weights).

Each system has one artifacts/<system>/threshold.json: the latest dev calibration and the artifact
it was made for. The runtime replaces it after every evaluation; older thresholds stay in
runs/*/calibration.json (copy one over to use it). It is only used for the artifact it names.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

__all__ = ["SYSTEMS", "artifacts_dir", "latest", "available", "save_threshold", "threshold",
           "write_latest_thresholds"]

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
    """threshold.json per system from runs/: newest calibration of its newest evaluated artifact."""
    from audio_df.helpers.research.runs import artifact_key

    newest: dict[str, tuple[str, Path]] = {}  # system -> (artifact timestamp, its newest run)
    for run in sorted((Path(root) / "runs").glob("E-*")):  # run names start with a timestamp
        cfg_path = run / "config.yaml"
        if (run / "calibration.json").is_file() and cfg_path.is_file():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            system, _, stamp = artifact_key(cfg.get("model_artifact", "")).partition("/")
            if system in SYSTEMS and stamp >= newest.get(system, ("", None))[0]:
                newest[system] = (stamp, run)
    for system, (stamp, run) in newest.items():
        calibration = json.loads((run / "calibration.json").read_text(encoding="utf-8"))
        save_threshold(artifacts_dir(root) / system / stamp, calibration, run.name)
    return {system: f"{stamp} from {run.name}" for system, (stamp, run) in newest.items()}


if __name__ == "__main__":
    for system, source in write_latest_thresholds(Path.cwd()).items():
        print(f"{system}: {source}")
