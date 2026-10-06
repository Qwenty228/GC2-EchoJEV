"""A1 pipeline: LFCC + GMM, the official ASVspoof 2019 B02 baseline (`configs/a1_*.yaml`).

Commands (through `audio_df.pipelines.run`):

    adf --config configs/a1_lfcc_gmm.yaml timing [--n 200]
    adf --config configs/a1_lfcc_gmm.yaml train --exp e1 [--resume] [--tag T] [--seed S]
    adf --config configs/a1_lfcc_gmm.yaml score --model-from runs/<raw run id> \
        --exp e1 --silence-mode raw|trimmed --testsets eval[,itw,...] [--resume]

Training (`audio_df.training.gmm`) caches LFCC train frames once and fits two 512-component
diagonal GMMs by chunked EM; scoring builds an `LfccGmmDetector` and scores each test set
from audio (`pipelines.score_audio`). A1 scores whole utterances, as B02 does, not the
project's usual 4 s crop; every run notes this.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audio_df.config import load_config
from audio_df.models.a1_lfcc_gmm import LfccGmmDetector
from audio_df.models.gmm import DiagGMM
from audio_df.pipelines import Command
from audio_df.pipelines.score_audio import score_manifest
from audio_df.runtime.exp_io import make_exp_id, write_scores
from audio_df.runtime.lifecycle import experiment_run
from audio_df.runtime.resources import (
    MemoryPressureExit,
    apply_resource_limits,
    format_timing,
    load_resources,
    memory_status,
    timing_run,
)
from audio_df.training.gmm import (
    N_FEATURES,
    build_frame_memmap,
    build_train_frame_job,
    fit_gmm,
    frames_for_row,
    front_end_config_hash,
    utt_chunk_size,
)

__all__ = ["COMMANDS", "cmd_score", "cmd_timing", "cmd_train"]

# Looked up dynamically (module attribute, not a bound default arg) so tests can
# `monkeypatch.setattr(a1, "_STATUS_FN", fake_status_fn)` to simulate memory
# pressure without touching real `sysctl` state.
_STATUS_FN = memory_status


def _load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"manifest {path} not found; run scripts/prepare_data.py first "
            "(see configs/a1_lfcc_gmm.yaml: data)."
        )
    df = pd.read_csv(path)
    required = {"utt_id", "path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"manifest {path} missing required column(s) {sorted(missing)}")
    return df


def cmd_timing(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    resources_cfg = load_resources(args.resources)
    apply_resource_limits(resources_cfg)

    train_df = _load_manifest(cfg["data"]["train"])
    n = args.n or resources_cfg["resources"]["timing_run_clips"]
    sample_rows = list(train_df.head(n).itertuples(index=False))

    def _fn(row: Any) -> None:
        frames_for_row(row, cfg)

    result = timing_run(
        _fn, sample_rows, n=len(sample_rows), total_items=len(train_df), cfg=resources_cfg
    )
    print("LFCC extraction timing:")
    print(format_timing(result))

    em_estimate: dict[str, Any] = {}
    probe_rows = sample_rows[: min(20, len(sample_rows))]
    if probe_rows:
        subset = np.concatenate([frames_for_row(r, cfg) for r in probe_rows], axis=0)
        n_components = cfg["gmm"]["n_components"]
        if len(subset) >= n_components:
            gmm = DiagGMM(
                n_components=n_components,
                n_features=N_FEATURES,
                variance_floor=cfg["gmm"]["variance_floor"],
            )
            t0 = time.perf_counter()
            gmm.fit_em(
                subset,
                n_iter=1,
                chunk_frames=cfg["gmm"]["chunk_frames"],
                seed=cfg["seed"],
                init_sample_size=len(subset),
            )
            dt = time.perf_counter() - t0
            em_estimate = {"n_frames_probed": int(len(subset)), "seconds_per_em_iter": dt}
            print(
                f"\nEM-iteration time estimate: one iteration over {len(subset)} frames took "
                f"{dt:.2f}s (probe only, not the full train partition; the official baseline "
                f"trains on ~0.85M bonafide + ~7.5M spoof frames for {cfg['gmm']['em_iters']} "
                "iterations each — scale this number accordingly, and expect it to be a lower "
                "bound on a fanless machine under sustained load)."
            )
        else:
            print(
                f"\n(EM-iteration estimate skipped: probe yielded {len(subset)} frames, fewer "
                f"than n_components={n_components})"
            )

    out_dir = Path("data/cache/timing")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "a1.json").write_text(
        json.dumps({"lfcc_extraction": result, "em_estimate": em_estimate}, default=str, indent=2)
    )
    print(f"\nwrote {out_dir / 'a1.json'}. No experiment was created by this command.")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    resources_cfg = load_resources(args.resources)
    cfg["experiment"] = args.exp  # --exp wins over the config file's default
    if args.seed is not None:
        cfg["seed"] = args.seed  # recorded in the saved config (experiment_run saves cfg)

    exp_id = make_exp_id(
        _dt.date.today(), args.exp, "a1", cfg["preprocessing"]["silence_mode"], tag=args.tag
    )
    spacing = cfg["features"].get("spacing", "linear")
    front_end = "mfcc" if spacing == "mel" else "lfcc"
    config_summary = f"a1 {front_end}-gmm train, {cfg['gmm']['n_components']}-component diag GMM x2"

    try:
        with experiment_run(
            exp_id,
            cfg,
            job_name="a1-train",
            need_disk_gb=4.0,  # ~0.2GB bonafide + ~1.8GB spoof frame cache + small model files
            config_summary=config_summary,
            resume=args.resume,
            resources_path=args.resources,
            status_fn=_STATUS_FN,
        ) as run:
            run.note(
                "A1 uses whole utterances (official B02 baseline), not this project's "
                "usual 4s crop (configs/data.yaml: audio.crop_seconds)."
            )

            train_df = _load_manifest(cfg["data"]["train"])
            cache_dir = Path("data/cache/a1") / front_end_config_hash(cfg)
            run.note(f"train-frame cache: {cache_dir} (gitignored, resumable)")

            bona_job = build_train_frame_job(
                train_df[train_df["label"] == "bonafide"],
                "bonafide",
                cache_dir,
                cfg,
                resources_cfg,
                run.guard,
            )
            spoof_job = build_train_frame_job(
                train_df[train_df["label"] == "spoof"],
                "spoof",
                cache_dir,
                cfg,
                resources_cfg,
                run.guard,
            )

            bona_path = cache_dir / "bonafide_frames.f32"
            spoof_path = cache_dir / "spoof_frames.f32"
            n_bona = build_frame_memmap(bona_job, bona_path)
            n_spoof = build_frame_memmap(spoof_job, spoof_path)
            run.note(f"train frames: bonafide={n_bona}, spoof={n_spoof} (dim={N_FEATURES})")

            bona_mm = np.memmap(bona_path, dtype=np.float32, mode="r", shape=(n_bona, N_FEATURES))
            spoof_mm = np.memmap(
                spoof_path, dtype=np.float32, mode="r", shape=(n_spoof, N_FEATURES)
            )

            bona_gmm = fit_gmm(bona_mm, cfg["gmm"], seed=cfg["seed"])
            spoof_gmm = fit_gmm(spoof_mm, cfg["gmm"], seed=cfg["seed"] + 1)

            bona_gmm.save(run.exp_dir / "bona_gmm.npz")
            spoof_gmm.save(run.exp_dir / "spoof_gmm.npz")
            run.note(
                f"bona fide GMM final avg log-lik={bona_gmm.log_likelihood_history[-1]:.3f}, "
                f"spoof GMM final avg log-lik={spoof_gmm.log_likelihood_history[-1]:.3f}"
            )

            detector = LfccGmmDetector(
                bona_gmm,
                spoof_gmm,
                features=cfg["features"],
                preprocessing=cfg["preprocessing"],
                threshold=float("nan"),  # set later by scripts/score_experiment.py
                model_version=exp_id,
            )
            for name, path in cfg["data"]["testsets"].items():
                scores_df = score_manifest(
                    detector,
                    _load_manifest(path),
                    run.exp_dir / "score_jobs" / name,
                    utt_chunk_size(resources_cfg),
                    run.guard,
                )
                write_scores(run.exp_dir, name, scores_df, force=args.resume)

    except MemoryPressureExit:
        print(
            "MemoryPressureExit: memory pressure reached critical between chunks. The "
            "last completed chunk is already checkpointed; rerun with --resume.",
            file=sys.stderr,
        )
        sys.exit(3)

    print(f"\nwrote runs/{exp_id}/scores/{{dev,eval}}.csv (+ more if configured).")
    print(f"Next: uv run python scripts/score_experiment.py runs/{exp_id}")
    print("Then:  uv run python scripts/make_tables.py")


def cmd_score(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    resources_cfg = load_resources(args.resources)
    model_dir = Path(args.model_from)

    detector = LfccGmmDetector.from_run(model_dir, silence_mode=args.silence_mode)

    cfg = dict(cfg)
    cfg["preprocessing"] = {**cfg["preprocessing"], "silence_mode": args.silence_mode}
    cfg["model_from"] = str(model_dir)
    # --exp wins over the config file's default (2026-10-03: In-the-Wild runs were filed
    # under e1 because the config hard-codes `experiment: e1`).
    cfg["experiment"] = args.exp
    if args.seed is not None:
        cfg["seed"] = args.seed

    exp_id = make_exp_id(_dt.date.today(), args.exp, "a1", args.silence_mode, tag=args.tag)
    config_summary = f"a1 score, silence_mode={args.silence_mode}, model_from={model_dir}"
    testset_names = args.testsets.split(",")

    try:
        with experiment_run(
            exp_id,
            cfg,
            job_name="a1-score",
            need_disk_gb=1.0,
            config_summary=config_summary,
            resume=args.resume,
            resources_path=args.resources,
            status_fn=_STATUS_FN,
        ) as run:
            if args.silence_mode == "trimmed":
                run.note(
                    "TRIMMED DIAGNOSTIC: no trimmed training exists for A1 — these are the "
                    f"raw-trained GMMs from {model_dir} scoring trimmed test audio. The "
                    "threshold must come from the RAW experiment's dev scores, not this one's."
                )
            for name in testset_names:
                path = cfg["data"]["testsets"].get(name, f"data/manifests/{name}.csv")
                scores_df = score_manifest(
                    detector,
                    _load_manifest(path),
                    run.exp_dir / "score_jobs" / name,
                    utt_chunk_size(resources_cfg),
                    run.guard,
                )
                write_scores(run.exp_dir, name, scores_df, force=args.resume)
    except MemoryPressureExit:
        print(
            "MemoryPressureExit: memory pressure reached critical between chunks. The "
            "last completed chunk is already checkpointed; rerun with --resume.",
            file=sys.stderr,
        )
        sys.exit(3)

    print(f"\nwrote runs/{exp_id}/scores/*.csv")
    print(
        f"Next: uv run python scripts/score_experiment.py runs/{exp_id} "
        f"--threshold-from {model_dir}"
    )
    print("Then:  uv run python scripts/make_tables.py")


COMMANDS: dict[str, Command] = {"timing": cmd_timing, "train": cmd_train, "score": cmd_score}
