"""A2 pipeline: frozen SSL -> per-layer pooled cache -> learned layer-weighted head.

Commands (through `audio_df.pipelines.run`; `configs/a2_wavlm_base_plus.yaml` or
`configs/a2_xlsr_300m.yaml`):

    adf --config <a2 config> timing [--n 200]                     builds nothing
    adf --config <a2 config> prepare --testsets train,dev,eval [--silence-mode] [--resume]
                                                                   builds the pooled cache
    adf --config <a2 config> train --exp e1 [--resume]            trains the head, scores
                                                                   dev and eval (raw)
    adf --config <a2 config> score --model-from <raw run> --exp e1|e3 \
        --silence-mode raw|trimmed --testsets eval[,itw,...]      scores more test sets

`prepare` only writes `data/cache/ssl/...`: no run directory and no registry row, but the
full resource discipline (limits, disk check, heavy-job lock, memory guard, resumable
chunks). `train` and `score` go through `runtime.lifecycle.experiment_run` and write
`runs/<id>/scores/<testset>.csv`; metrics come later from `scripts/score_experiment.py`.

Fixed design decisions:

- **Center crop for every split, train included.** The cache is built once, so per-epoch
  random-crop augmentation is not possible; that is the price of not re-running extraction.
- **No trimmed training.** `train` always uses the raw cache; `trimmed` is a diagnostic,
  scoring the raw-trained head on a trimmed cache with the raw dev threshold.
- **Checkpoint by dev loss, not dev EER**, and the loss orientation: see
  `audio_df.training.head`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from audio_df.config import load_config
from audio_df.features.audio import load_audio, preprocess
from audio_df.features.ssl import (
    build_job_config,
    cache_dir_for,
    extract_pooled_hidden_states,
    load_backbone,
    load_cache,
    run_extraction,
)
from audio_df.models.a2_ssl_head import SslHeadDetector, head_from_config
from audio_df.pipelines import Command
from audio_df.runtime.device import resolve_device
from audio_df.runtime.exp_io import make_exp_id, write_scores
from audio_df.runtime.lifecycle import experiment_run
from audio_df.runtime.resources import (
    MemoryGuard,
    MemoryPressureExit,
    apply_resource_limits,
    check_can_start,
    check_free_disk,
    format_timing,
    heavy_job_lock,
    load_resources,
    loader_workers,
    memory_status,
    run_resource_summary,
    timing_run,
)
from audio_df.runtime.seeding import make_generator
from audio_df.runtime.seeding import worker_init_fn as seed_worker_init_fn
from audio_df.training.head import align_cache_with_manifest, train_head

__all__ = [
    "COMMANDS",
    "assert_backbone_matches",
    "cmd_extract",
    "cmd_score",
    "cmd_timing",
    "cmd_train",
    "load_config",
    "resolve_manifest",
    "score_cached",
]

# Extrapolation targets for `timing`. `in_the_wild` is an approximate, UNVERIFIED clip
# count (Miscellaneous/research/fact-checks/FC-plan-2026-09-30.md item 3: only aggregate hours --
# 37.9h total / 17.2h fake / 20.7-20.8h bona fide -- were confirmed against a primary source; no
# clip count was). It is printed as an explicit approximation, never as a fact.
_EXTRAPOLATION_TARGETS: dict[str, int] = {
    "all_raw_splits (train+dev+eval)": 121_461,
    "trimmed_eval": 71_237,
    "in_the_wild (~unverified count)": 31_800,
    "e4_subset (4 x 10k)": 4 * 10_000,
}

# Approximate first-download sizes (HF model card listings), printed once by `timing`.
_MODEL_DOWNLOAD_GB = {
    "microsoft/wavlm-base-plus": 0.38,
    "facebook/wav2vec2-xls-r-300m": 1.27,
}


def resolve_manifest(cfg: dict[str, Any], testset: str) -> str:
    """`"train"` resolves to `cfg["data"]["train_manifest"]`; anything else looks up
    `cfg["data"]["testsets"][testset]` (`dev`, `eval`, `itw`, `eval_mp3_32k`, ...)."""
    if testset == "train":
        return cfg["data"]["train_manifest"]
    testsets = cfg["data"]["testsets"]
    if testset not in testsets:
        raise ValueError(f"unknown testset {testset!r}; known: {['train', *sorted(testsets)]}")
    return testsets[testset]


def assert_backbone_matches(model: nn.Module, cfg: dict[str, Any]) -> None:
    """Fail loudly if the loaded model's shape does not match what the config claims.

    A silent mismatch here would poison the entire cache with the wrong `n_layers`/
    `hidden_dim`, discovered only much later when a head refuses to load.
    """
    # `nn.Module` has no `config`; the Hugging Face backbone does, with untyped fields.
    config: Any = model.config
    n_layers = config.num_hidden_layers + 1
    hidden_dim = config.hidden_size
    expected_layers = cfg["model"]["n_layers"]
    expected_hidden = cfg["model"]["hidden_dim"]
    if n_layers != expected_layers or hidden_dim != expected_hidden:
        raise ValueError(
            f"loaded backbone {cfg['model']['id']!r} has {n_layers} hidden-state tensors x "
            f"{hidden_dim} dims, but config says n_layers={expected_layers}, "
            f"hidden_dim={expected_hidden}"
        )


def score_cached(
    detector: SslHeadDetector, cache_dir: str | Path, manifest_df: pd.DataFrame
) -> pd.DataFrame:
    """Score every cached clip in `cache_dir`, aligned to `manifest_df`'s labels.

    `score` is the head's raw logit (higher = more bona fide, the score-file convention).
    """
    pooled, ordered = align_cache_with_manifest(load_cache(cache_dir), manifest_df)
    df = pd.DataFrame(
        {
            "clip_id": ordered["utt_id"].astype(str),
            "label": ordered["label"],
            "score": detector.score_pooled(pooled),
        }
    )
    if "attack_id" in ordered.columns:
        df["attack_id"] = ordered["attack_id"]
    return df


def cmd_timing(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    res = load_resources()
    apply_resource_limits(res)

    download_gb = _MODEL_DOWNLOAD_GB.get(cfg["model"]["id"])
    if download_gb is not None:
        print(
            f"[timing] first run downloads {cfg['model']['id']} weights from Hugging Face "
            f"(approx. {download_gb} GB)."
        )

    model, feature_extractor = load_backbone(cfg["model"]["id"], cfg["model"]["revision"])
    assert_backbone_matches(model, cfg)
    device = resolve_device("auto")
    model = model.to(device)

    train_manifest = pd.read_csv(cfg["data"]["train_manifest"])
    items = list(zip(train_manifest["utt_id"], train_manifest["path"], strict=True))

    prep = cfg["preprocessing"]

    def _one(item: tuple[str, str]) -> np.ndarray:
        _, path = item
        x, sr = load_audio(path, target_sr=prep["sample_rate"])
        x = preprocess(
            x,
            sr,
            silence_mode=prep["silence_mode"],
            crop_seconds=prep["crop_seconds"],
            crop_mode=prep["crop_mode"],
        )
        return extract_pooled_hidden_states(
            model, feature_extractor, x[None, :], prep["sample_rate"], device
        )

    n = args.n if args.n is not None else res["resources"]["timing_run_clips"]
    result = timing_run(_one, items, n=n, cfg=res)
    print(format_timing(result))

    extrapolations: dict[str, float | None] = {}
    for name, total in _EXTRAPOLATION_TARGETS.items():
        mean_s = result["mean_s_per_item"]
        extrapolations[name] = (total * mean_s) / 3600.0 if mean_s > 0 else None
        print(f"  extrapolated for {name} ({total} clips): {extrapolations[name]:.2f} h")

    out_path = Path("data/cache/timing") / f"a2_{cfg['model']['tag']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"timing": result, "extrapolations_hours": extrapolations}, indent=2)
    )
    print(f"wrote {out_path}")
    print(f"next: uv run adf --config {args.config} prepare --testsets train,dev,eval")


def cmd_extract(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    res = load_resources()
    apply_resource_limits(res)
    status_fn = getattr(args, "status_fn", memory_status)
    check_can_start(res, status_fn=status_fn)

    model, feature_extractor = load_backbone(cfg["model"]["id"], cfg["model"]["revision"])
    assert_backbone_matches(model, cfg)
    device = resolve_device("auto")
    model = model.to(device)
    guard = MemoryGuard(res, status_fn=status_fn)

    testsets = [t.strip() for t in args.testsets.split(",") if t.strip()]
    prep = cfg["preprocessing"]
    log_path = Path(cfg["cache_dir"]) / cfg["model"]["tag"] / "extraction_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with heavy_job_lock(
            "a2-extract", lock_path=getattr(args, "lock_path", "data/cache/.heavy_job.lock")
        ):
            for testset in testsets:
                manifest_path = resolve_manifest(cfg, testset)
                manifest_df = pd.read_csv(manifest_path)
                ids = manifest_df["utt_id"].astype(str).tolist()
                paths = manifest_df["path"].astype(str).tolist()

                out_dir = cache_dir_for(
                    cfg["model"]["tag"], args.silence_mode, testset, root=cfg["cache_dir"]
                )
                need_gb = len(ids) * cfg["model"]["n_layers"] * cfg["model"]["hidden_dim"] * 2 / 1e9
                out_dir.mkdir(parents=True, exist_ok=True)
                check_free_disk(out_dir, need_gb, res)

                job_config = build_job_config(
                    model_id=cfg["model"]["id"],
                    model_tag=cfg["model"]["tag"],
                    n_layers=cfg["model"]["n_layers"],
                    hidden_dim=cfg["model"]["hidden_dim"],
                    silence_mode=args.silence_mode,
                    crop_seconds=prep["crop_seconds"],
                    crop_mode=prep["crop_mode"],
                    sample_rate=prep["sample_rate"],
                    do_normalize=cfg["model"]["do_normalize"],
                    revision=cfg["model"]["revision"],
                )
                num_workers = loader_workers(1, res, kind="ssl")

                run_extraction(
                    ids,
                    paths,
                    model,
                    feature_extractor,
                    out_dir,
                    job_config,
                    device,
                    sample_rate=prep["sample_rate"],
                    silence_mode=args.silence_mode,
                    crop_seconds=prep["crop_seconds"],
                    crop_mode=prep["crop_mode"],
                    batch_size=res["resources"]["ssl_extract_batch_size"],
                    num_workers=num_workers,
                    chunk_size=res["resources"]["chunk_size"],
                    memory_guard=guard,
                    worker_init_fn=seed_worker_init_fn,
                    generator=make_generator(cfg["seed"]),
                )

                summary = run_resource_summary(guard)
                print(f"[extract] {testset} ({args.silence_mode}): {summary}")
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {"testset": testset, "silence_mode": args.silence_mode, **summary}
                        )
                        + "\n"
                    )
    except MemoryPressureExit as exc:
        print(f"[extract] memory pressure critical: {exc.args[0]}; rerun with --resume")
        sys.exit(3)

    print(f"next: uv run adf --config {args.config} train --exp e1")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    exp_id = make_exp_id(date.today(), args.exp, cfg["system"], "raw")
    full_config = {**cfg, "experiment": args.exp}

    with experiment_run(
        exp_id,
        full_config,
        job_name="a2-train",
        resume=args.resume,
        root=getattr(args, "root", "runs"),
        lock_path=getattr(args, "lock_path", "data/cache/.heavy_job.lock"),
        status_fn=getattr(args, "status_fn", memory_status),
    ) as run:
        device = resolve_device("auto")
        tag = cfg["model"]["tag"]
        cache_root = cfg["cache_dir"]

        train_manifest = pd.read_csv(cfg["data"]["train_manifest"])
        dev_manifest = pd.read_csv(cfg["data"]["testsets"]["dev"])
        eval_manifest = pd.read_csv(cfg["data"]["testsets"]["eval"])

        train_cache_dir = cache_dir_for(tag, "raw", "train", root=cache_root)
        dev_cache_dir = cache_dir_for(tag, "raw", "dev", root=cache_root)
        eval_cache_dir = cache_dir_for(tag, "raw", "eval", root=cache_root)

        pooled_train, train_aligned = align_cache_with_manifest(
            load_cache(train_cache_dir), train_manifest
        )
        pooled_dev, dev_aligned = align_cache_with_manifest(load_cache(dev_cache_dir), dev_manifest)

        x_train = torch.tensor(pooled_train.astype(np.float32))
        y_train = torch.tensor((train_aligned["label"] == "bonafide").astype(np.float32).to_numpy())
        x_dev = torch.tensor(pooled_dev.astype(np.float32))
        y_dev = torch.tensor((dev_aligned["label"] == "bonafide").astype(np.float32).to_numpy())

        head = head_from_config(cfg["model"], cfg["head"])
        generator = make_generator(cfg["seed"])
        pw_cfg = cfg["head"]["pos_weight"]
        pos_weight = None if pw_cfg == "auto" else float(pw_cfg)

        result = train_head(
            head,
            x_train,
            y_train,
            x_dev,
            y_dev,
            lr=cfg["head"]["lr"],
            batch_size=cfg["head"]["batch_size"],
            max_epochs=cfg["head"]["max_epochs"],
            patience=cfg["head"]["patience"],
            pos_weight=pos_weight,
            device=device,
            generator=generator,
        )

        torch.save(result.head.state_dict(), run.exp_dir / "head_state_dict.pt")
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
        (run.exp_dir / "layer_weights.json").write_text(
            json.dumps(result.head.layer_weights().detach().cpu().numpy().tolist(), indent=2)
        )

        detector = SslHeadDetector(
            result.head,
            model=cfg["model"],
            head_config=cfg["head"],
            preprocessing=cfg["preprocessing"],
            threshold=float("nan"),  # set later by scripts/score_experiment.py
            model_version=exp_id,
            device=device,
        )
        write_scores(run.exp_dir, "dev", score_cached(detector, dev_cache_dir, dev_manifest))
        write_scores(run.exp_dir, "eval", score_cached(detector, eval_cache_dir, eval_manifest))

        run.note(f"best dev loss at epoch {result.best_epoch} (dev loss checkpoint selection)")

    print(
        f"next: uv run python scripts/score_experiment.py {run.exp_dir} "
        f"&& uv run adf --config {args.config} score --model-from {run.exp_dir} "
        f"--exp {args.exp} --silence-mode raw --testsets itw"
    )


def cmd_score(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model_from = Path(args.model_from)
    detector = SslHeadDetector.from_run(model_from, silence_mode=args.silence_mode)

    exp_id = make_exp_id(date.today(), args.exp, cfg["system"], args.silence_mode)
    full_config = {
        **cfg,
        "experiment": args.exp,
        "model_from": str(model_from),
        # Record the silence mode actually scored (2026-10-03: trimmed runs said "raw").
        "preprocessing": {**cfg["preprocessing"], "silence_mode": args.silence_mode},
    }
    testsets = [t.strip() for t in args.testsets.split(",") if t.strip()]

    # Through experiment_run like every other run, so a score run gets its own registry row,
    # resource summary and failure record (2026-10-01: without a row, score_experiment.py
    # crashed on these runs).
    with experiment_run(
        exp_id,
        full_config,
        job_name="a2-score",
        config_summary=(
            f"a2 score, silence_mode={args.silence_mode}, testsets={','.join(testsets)}, "
            f"model_from={model_from}"
        ),
        resume=args.resume,
        root=getattr(args, "root", "runs"),
        lock_path=getattr(args, "lock_path", "data/cache/.heavy_job.lock"),
        status_fn=getattr(args, "status_fn", memory_status),
    ) as run:
        if args.silence_mode == "trimmed":
            run.note(
                f"Trimmed-silence diagnostic: the raw-trained head from {model_from} scoring "
                "trimmed test audio. The threshold must come from the RAW experiment's dev scores."
            )
        for testset in testsets:
            manifest_df = pd.read_csv(resolve_manifest(cfg, testset))
            cache_dir = cache_dir_for(
                cfg["model"]["tag"], args.silence_mode, testset, root=cfg["cache_dir"]
            )
            df = score_cached(detector, cache_dir, manifest_df)
            write_scores(run.exp_dir, testset, df, force=args.resume)
        exp_dir = run.exp_dir

    testset_list = ",".join(testsets)
    print(
        f"next: uv run python scripts/score_experiment.py {exp_dir} "
        f"--threshold-from {model_from} "
        f"# scores written for {testset_list} ({args.silence_mode})"
    )


COMMANDS: dict[str, Command] = {
    "timing": cmd_timing,
    "prepare": cmd_extract,
    "train": cmd_train,
    "score": cmd_score,
}
