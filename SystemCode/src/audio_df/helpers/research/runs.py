"""Collect finished evaluation runs and their score files, for the result notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from audio_df.helpers.metrics import compute_eer


def artifact_key(model_artifact: str) -> str:
    """'<model>/<timestamp>' from a project-relative or old absolute (Windows or POSIX) path."""
    return "/".join(model_artifact.replace("\\", "/").rstrip("/").split("/")[-2:])


def train_audio(cfg: dict, root: Path) -> str:
    """Training audio from config.yaml; older runs: the artifact's preprocessing.json, else raw."""
    prep = root / "artifacts" / artifact_key(cfg.get("model_artifact", "")) / "preprocessing.json"
    return cfg.get("train_preprocessing") or (json.loads(prep.read_text())["train"] if prep.exists() else "raw")


def collect(runs_dir: str | Path, models: list[str]) -> dict[tuple[str, str], dict]:
    """Score files of the newest artifact per (model, training audio).

    Every run of the same artifact is merged, so a later 2021-only evaluation adds its test
    sets to the earlier run; when two runs scored the same set, the later run wins.
    Returns {(model, train): {"artifact", "runs", "scores": {testset: DataFrame}}}.
    """
    by_artifact: dict[str, dict] = {}
    for run in sorted(Path(runs_dir).glob("E-*")):  # run names start with a UTC timestamp
        cfg_path = run / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        if cfg.get("architecture") not in models:
            continue
        entry = by_artifact.setdefault(artifact_key(cfg["model_artifact"]), {
            "model": cfg["architecture"], "train": train_audio(cfg, Path(runs_dir).parent), "runs": [], "scores": {}})
        entry["runs"].append(run.name)
        entry["scores"].update({p.stem: pd.read_csv(p) for p in sorted(run.glob("scores/*.csv"))})
    newest: dict[tuple[str, str], dict] = {}
    for artifact in sorted(by_artifact, key=lambda a: a.split("/")[-1]):  # artifact names are timestamps
        entry = by_artifact[artifact]
        newest[(entry["model"], entry["train"])] = {"artifact": artifact.split("/")[-1], "runs": entry["runs"],
                                                    "scores": entry["scores"]}
    return newest


def eer(scores: pd.DataFrame) -> float:
    """EER in percent; higher score = bona fide."""
    return 100 * compute_eer(scores.score, scores.label, higher_is_spoof=False)[0]


def eer_table(results: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """EER % with one row per (model, train) and one column per test set."""
    table = pd.DataFrame({k: {t: eer(s) for t, s in v["scores"].items()} for k, v in results.items()}).T
    table.index.names = ["model", "train"]
    return table
