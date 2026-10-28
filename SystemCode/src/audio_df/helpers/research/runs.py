"""Collect finished evaluation runs and their score files, for the result notebooks."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from audio_df.helpers.metrics import compute_eer, min_dcf, prf1_at_threshold


def artifact_key(model_artifact: str) -> str:
    """'<model>/<timestamp>' from a project-relative or old absolute (Windows or POSIX) path."""
    return "/".join(model_artifact.replace("\\", "/").rstrip("/").split("/")[-2:])


def train_audio(cfg: dict, root: Path) -> str:
    """Training audio from config.yaml; older runs: the artifact's preprocessing.json, else raw."""
    prep = root / "artifacts" / artifact_key(cfg.get("model_artifact", "")) / "preprocessing.json"
    return cfg.get("train_preprocessing") or (json.loads(prep.read_text())["train"] if prep.exists() else "raw")


def wavlm_layer(cfg: dict, root: Path) -> int | None:
    """D2/D3 WavLM layer from config.yaml, else the artifact's model.json; None for other models.

    Artifacts saved before the layer was configurable used the last layer (12).
    """
    if cfg.get("architecture") not in ("d2", "d3"):
        return None
    if "wavlm_layer" in cfg:
        return cfg["wavlm_layer"]
    meta = root / "artifacts" / artifact_key(cfg.get("model_artifact", "")) / "model.json"
    return json.loads(meta.read_text())["fusion"].get("layer", 12) if meta.exists() else 12


def collect(runs_dir: str | Path, models: list[str]) -> dict[tuple[str, str], dict]:
    """Score files of the newest artifact per (model, training audio). D2/D3 models are labelled
    with their WavLM layer ("d2 L6"), so attention variants are kept apart.

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
        root, layer = Path(runs_dir).parent, wavlm_layer(cfg, Path(runs_dir).parent)
        label = cfg["architecture"] if layer is None else f"{cfg['architecture']} L{layer}"
        entry = by_artifact.setdefault(artifact_key(cfg["model_artifact"]), {
            "model": label, "train": train_audio(cfg, root), "runs": [], "scores": {}})
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


def eer_pct(scores, labels) -> float:
    """EER in percent, rounded to 2 decimals; higher score = bona fide."""
    return round(100 * compute_eer(scores, labels, higher_is_spoof=False)[0], 2)


def metrics_by_testset(scores: dict[str, pd.DataFrame], sets: list[str]) -> dict[str, dict]:
    """{metric: {test set: value}} for one model: EER %, minDCF, and F1 (spoof) at the dev EER threshold."""
    thr = compute_eer(scores["dev"].score, scores["dev"].label, higher_is_spoof=False)[1]
    out: dict[str, dict] = {"EER %": {}, "minDCF": {}, "F1 (spoof)": {}}
    for t in sets:
        if t in scores:
            s = scores[t]
            out["EER %"][t] = eer(s)
            out["minDCF"][t] = min_dcf(s.score, s.label, higher_is_spoof=False)[0]
            f1 = prf1_at_threshold(s.score, s.label, thr, higher_is_spoof=False)["f1"]
            out["F1 (spoof)"][t] = f1
    return out


def eer_table(results: dict[tuple[str, str], dict]) -> pd.DataFrame:
    """EER % with one row per (model, train) and one column per test set."""
    table = pd.DataFrame({k: {t: eer(s) for t, s in v["scores"].items()} for k, v in results.items()}).T
    table.index.names = ["model", "train"]
    return table
