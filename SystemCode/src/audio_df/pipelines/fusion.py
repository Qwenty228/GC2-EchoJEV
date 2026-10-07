"""Fusion pipeline: frame-cache preparation, training and scoring for the main model.

    adf --config configs/fusion_xattn.yaml prepare --testsets train,dev,eval [--silence-mode]
    adf --config configs/fusion_xattn.yaml train --exp e2 [--resume]
    adf --config configs/fusion_xattn.yaml score --model-from runs/<run id> --exp e3 \\
        --silence-mode raw|trimmed --testsets itw

`prepare` builds `<cache_dir>/<cache_tag>/<silence_mode>/<testset>/` (both frame streams
per clip, `features.frames`) under the resource discipline, without a run directory.
`train` trains on the train cache, picks the checkpoint by dev loss and scores dev and
eval; `score` scores more cached test sets with a trained run. Unlike A2, training on
trimmed audio is allowed: set `preprocessing.silence_mode: trimmed` and prepare trimmed
train and dev caches first.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from audio_df.config import load_config
from audio_df.features.frames import FrameCacheReader, build_frame_cache, resolve_layer_weights
from audio_df.features.ssl import load_backbone
from audio_df.models.fusion_detector import FusionDetector, build_fusion_model
from audio_df.pipelines import Command
from audio_df.runtime.device import resolve_device
from audio_df.runtime.exp_io import make_exp_id, write_scores
from audio_df.runtime.lifecycle import experiment_run
from audio_df.runtime.resources import (
    MemoryGuard,
    apply_resource_limits,
    check_can_start,
    heavy_job_lock,
    load_resources,
    memory_status,
)
from audio_df.training.fusion import predict_cache, train_fusion

__all__ = ["COMMANDS", "cache_dir", "cmd_prepare", "cmd_score", "cmd_train"]

_MIX_FILE = "layer_mix.json"


def _manifest_path(cfg: dict[str, Any], testset: str) -> str:
    if testset == "train":
        return str(cfg["data"]["train_manifest"])
    testsets = cfg["data"]["testsets"]
    return str(testsets.get(testset, f"data/manifests/{testset}.csv"))


def cache_dir(cfg: dict[str, Any], silence_mode: str, testset: str) -> Path:
    return Path(cfg["cache_dir"]) / cfg["cache_tag"] / silence_mode / testset


def _is_bonafide(manifest: str) -> dict[str, float]:
    df = pd.read_csv(manifest)
    return {
        str(u): float(lab == "bonafide") for u, lab in zip(df["utt_id"], df["label"], strict=True)
    }


def _scores_frame(ids: list[str], logits: Any, manifest: str) -> pd.DataFrame:
    df = pd.read_csv(manifest)
    df["utt_id"] = df["utt_id"].astype(str)
    meta = df.set_index("utt_id").loc[ids]
    out = pd.DataFrame({"clip_id": ids, "label": meta["label"].to_numpy(), "score": logits})
    if "attack_id" in meta.columns:
        out["attack_id"] = meta["attack_id"].to_numpy()
    return out


def cmd_prepare(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cfg["preprocessing"] = {**cfg["preprocessing"], "silence_mode": args.silence_mode}
    res = load_resources(args.resources)
    apply_resource_limits(res)
    status_fn = getattr(args, "status_fn", memory_status)
    check_can_start(res, status_fn=status_fn)

    model, feature_extractor = getattr(args, "backbone", None) or load_backbone(
        cfg["ssl"]["id"], cfg["ssl"].get("revision")
    )
    device = resolve_device("auto")
    backbone = (model.to(device), feature_extractor)
    weights = resolve_layer_weights(cfg["ssl"]["layer_weights"], int(cfg["ssl"]["n_layers"]))
    tag_dir = Path(cfg["cache_dir"]) / cfg["cache_tag"]
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / _MIX_FILE).write_text(json.dumps([float(w) for w in weights]))

    lock = getattr(args, "lock_path", "data/cache/.heavy_job.lock")
    with heavy_job_lock("fusion-prepare", lock_path=lock):
        for testset in [t.strip() for t in args.testsets.split(",") if t.strip()]:
            df = pd.read_csv(_manifest_path(cfg, testset))
            out = cache_dir(cfg, args.silence_mode, testset)
            build_frame_cache(
                df["utt_id"].astype(str).tolist(),
                df["path"].astype(str).tolist(),
                backbone,
                cfg,
                weights,
                out,
                device,
                chunk_size=int(cfg.get("cache_chunk_size", 256)),
                memory_guard=MemoryGuard(res, status_fn=status_fn),
            )
            print(f"[prepare] {testset} ({args.silence_mode}) -> {out}")
    print(f"next: uv run adf --config {args.config} train --exp e2")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    silence = cfg["preprocessing"]["silence_mode"]
    exp_id = make_exp_id(date.today(), args.exp, "fusion", silence)
    run_cfg = {**cfg, "experiment": args.exp}
    tc = cfg["train"]

    with experiment_run(
        exp_id,
        run_cfg,
        job_name="fusion-train",
        resume=args.resume,
        root=getattr(args, "root", "runs"),
        resources_path=args.resources,
        lock_path=getattr(args, "lock_path", "data/cache/.heavy_job.lock"),
        status_fn=getattr(args, "status_fn", memory_status),
    ) as run:
        device = resolve_device("auto")
        mix_file = Path(cfg["cache_dir"]) / cfg["cache_tag"] / _MIX_FILE
        (run.exp_dir / _MIX_FILE).write_text(mix_file.read_text())
        result = train_fusion(
            build_fusion_model(cfg),
            FrameCacheReader(cache_dir(cfg, silence, "train")),
            _is_bonafide(_manifest_path(cfg, "train")),
            FrameCacheReader(cache_dir(cfg, silence, "dev")),
            _is_bonafide(_manifest_path(cfg, "dev")),
            lr=float(tc["lr"]),
            batch_size=int(tc["batch_size"]),
            max_epochs=int(tc["max_epochs"]),
            patience=int(tc["patience"]),
            pos_weight=None if tc["pos_weight"] == "auto" else float(tc["pos_weight"]),
            device=device,
            seed=int(cfg["seed"]),
        )
        torch.save(result.model.state_dict(), run.exp_dir / "model_state.pt")
        (run.exp_dir / "training_curve.json").write_text(
            json.dumps(
                {
                    "train_loss": result.train_losses,
                    "dev_loss": result.dev_losses,
                    "best_epoch": result.best_epoch,
                    "pos_weight": result.pos_weight,
                },
                indent=2,
            )
        )
        for testset in ("dev", "eval"):
            ids, logits = predict_cache(
                result.model,
                FrameCacheReader(cache_dir(cfg, silence, testset)),
                int(tc["batch_size"]),
                device,
            )
            write_scores(
                run.exp_dir, testset, _scores_frame(ids, logits, _manifest_path(cfg, testset))
            )
        run.note(f"best dev loss at epoch {result.best_epoch} (dev-loss checkpoint selection)")
    print(f"next: uv run python scripts/score_experiment.py {run.exp_dir}")


def cmd_score(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model_from = Path(args.model_from)
    detector = FusionDetector.from_run(model_from)
    exp_id = make_exp_id(date.today(), args.exp, "fusion", args.silence_mode)
    run_cfg = {
        **detector.config,
        "experiment": args.exp,
        "model_from": str(model_from),
        "preprocessing": {**detector.config["preprocessing"], "silence_mode": args.silence_mode},
    }
    testsets = [t.strip() for t in args.testsets.split(",") if t.strip()]
    with experiment_run(
        exp_id,
        run_cfg,
        job_name="fusion-score",
        resume=args.resume,
        root=getattr(args, "root", "runs"),
        resources_path=args.resources,
        lock_path=getattr(args, "lock_path", "data/cache/.heavy_job.lock"),
        status_fn=getattr(args, "status_fn", memory_status),
    ) as run:
        for testset in testsets:
            ids, logits = predict_cache(
                detector.model,
                FrameCacheReader(cache_dir(cfg, args.silence_mode, testset)),
                int(cfg["train"]["batch_size"]),
                detector.device,
            )
            frame = _scores_frame(ids, logits, _manifest_path(cfg, testset))
            write_scores(run.exp_dir, testset, frame, force=args.resume)
    print(
        f"next: uv run python scripts/score_experiment.py {run.exp_dir} "
        f"--threshold-from {model_from}"
    )


COMMANDS: dict[str, Command] = {"prepare": cmd_prepare, "train": cmd_train, "score": cmd_score}
