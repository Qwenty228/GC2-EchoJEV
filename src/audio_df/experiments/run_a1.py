"""A1 runner: LFCC + GMM, the official ASVspoof 2019 B02 baseline.

Three subcommands, everything guarded under `if __name__ == "__main__":` per
`tasks/CONVENTIONS.md`:

    uv run python -m audio_df.experiments.run_a1 timing [--n 200]
    uv run python -m audio_df.experiments.run_a1 train --exp e1 [--resume]
    uv run python -m audio_df.experiments.run_a1 score --model-from experiments/<raw exp id> \
        --exp e1 --silence-mode raw|trimmed --testsets eval[,itw,...] [--resume]

## Train-frame cache

`train` extracts LFCC frames for every TRAIN utterance exactly once, per class
(bonafide / spoof), into an on-disk `float32` memmap under
`data/cache/a1/<config_hash>/` (gitignored; `<config_hash>` is a sha256 of the
front-end + preprocessing config, so a changed parameter never silently reuses a stale
cache). This is the *only* feature cache in A1: dev/eval/other test sets are always
recomputed on the fly while scoring (see `_score_manifest`), never cached, since they
are scored once per experiment rather than iterated over ~10 EM passes the way TRAIN
frames are.

Two-stage build, both resumable and memory-bounded:
  1. `ChunkedJob` extracts frames a batch of utterances at a time (chunk size from
     `configs/resources.yaml`, capped small — see `_utt_chunk_size`) into per-chunk
     `.npz` files. Interrupting and rerunning `train --resume` replays no completed
     chunk.
  2. Once the job is complete, `_build_frame_memmap` streams those chunks (one chunk
     in memory at a time) into a single flat memmap file the GMM's chunked EM reads
     directly, never loading the full array into RAM.

## Memory design

Training peaks under ~2 GB RSS by construction: the frame-extraction stage holds at
most one chunk of utterances' frames; the memmap-build stage holds at most one chunk's
`.npz` contents; `models.gmm.DiagGMM.fit_em` holds at most one `[chunk_frames,
n_components]` responsibility array (default 50k x 512 float64 ~= 205 MB) plus O(K*D)
accumulators (512*60*8 bytes ~= 246 KB) — see `models/gmm.py`'s module docstring for
why `sklearn.mixture.GaussianMixture` is not used instead.

## Whole-utterance deviation

A1 deliberately does not use this project's usual 4 s crop
(`configs/data.yaml: audio.crop_seconds`) — the official B02 baseline scores whole
utterances (`configs/a1_lfcc_gmm.yaml: preprocessing.crop_seconds: null`). Every run
logs this via `RunContext.note`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from audio_df.audio import load_audio, preprocess
from audio_df.exp_io import make_exp_id, write_scores
from audio_df.experiments.lifecycle import experiment_run
from audio_df.lfcc_b02 import extract_lfcc_b02
from audio_df.models.gmm import DiagGMM, score_llr
from audio_df.resources import (
    ChunkedJob,
    MemoryGuard,
    MemoryPressureExit,
    apply_resource_limits,
    format_timing,
    load_resources,
    memory_status,
    timing_run,
)

_N_FEATURES = 60

# Looked up dynamically (module attribute, not a bound default arg) so tests can
# `monkeypatch.setattr(run_a1, "_STATUS_FN", fake_status_fn)` to simulate memory
# pressure without touching real `sysctl` state.
_STATUS_FN = memory_status


# ---------------------------------------------------------------------------
# Config / manifest helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        return yaml.safe_load(f)


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


def _front_end_config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(
        {"features": cfg["features"], "preprocessing": cfg["preprocessing"]},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _utt_chunk_size(resources_cfg: dict[str, Any]) -> int:
    """Utterances per frame-extraction chunk: small (<=50), independent of
    `resources.chunk_size` (which is tuned for SSL-style item chunks, not per-utterance
    LFCC extraction batches whose per-item frame count varies with duration)."""
    return min(50, resources_cfg["resources"]["chunk_size"])


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def extract_frames_for_row(row: Any, cfg: dict[str, Any]) -> np.ndarray:
    """Load, preprocess, and extract B02 LFCC for one manifest row. `[T, 60]`."""
    pp = cfg["preprocessing"]
    x, sr = load_audio(row.path, target_sr=pp["sample_rate"])
    x = preprocess(
        x,
        sr,
        silence_mode=pp["silence_mode"],
        crop_seconds=pp.get("crop_seconds"),
    )
    return extract_lfcc_b02(x, sr=sr, cfg=cfg["features"])


def _extract_chunk_frames(
    ids: list[str], id_to_row: dict[str, Any], cfg: dict[str, Any]
) -> np.ndarray:
    arrays = [extract_frames_for_row(id_to_row[i], cfg) for i in ids]
    if not arrays:
        return np.zeros((0, _N_FEATURES), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def build_train_frame_job(
    df: pd.DataFrame,
    label: str,
    cache_dir: Path,
    cfg: dict[str, Any],
    resources_cfg: dict[str, Any],
    memory_guard: MemoryGuard | None = None,
) -> ChunkedJob:
    """Run (or resume) the per-utterance frame-extraction `ChunkedJob` for one class."""
    id_to_row = {row.utt_id: row for row in df.itertuples(index=False)}
    item_ids = sorted(id_to_row)
    job = ChunkedJob(
        out_dir=cache_dir / label,
        item_ids=item_ids,
        chunk_size=_utt_chunk_size(resources_cfg),
        job_config={
            "label": label,
            "features": cfg["features"],
            "preprocessing": cfg["preprocessing"],
        },
        memory_guard=memory_guard,
    )
    for chunk_idx, ids in job.pending():
        frames = _extract_chunk_frames(ids, id_to_row, cfg)
        job.save_chunk(chunk_idx, ids, {"frames": frames})
    return job


def build_frame_memmap(job: ChunkedJob, out_path: Path, n_features: int = _N_FEATURES) -> int:
    """Stream a completed `ChunkedJob`'s per-chunk 'frames' arrays into one on-disk
    `float32` memmap at `out_path`, one chunk in memory at a time. Returns n_frames.

    Rebuilds unconditionally if `out_path` (or its `.json` sidecar recording the frame
    count) is missing or the sidecar's chunk-set fingerprint no longer matches the
    job's — this is a cheap streaming copy, not worth its own separate resumability.
    """
    if not job.is_complete():
        raise RuntimeError(f"cannot build memmap: job at {job.out_dir} is not complete")

    sidecar = out_path.with_suffix(out_path.suffix + ".json")
    fingerprint = {"n_chunks": job.n_chunks, "n_items": len(job.item_ids)}
    if out_path.exists() and sidecar.exists():
        existing = json.loads(sidecar.read_text())
        if existing.get("fingerprint") == fingerprint:
            return int(existing["n_frames"])

    total = 0
    for i in range(job.n_chunks):
        with np.load(job._chunk_path(i)) as data:  # noqa: SLF001 - same-project helper
            total += int(data["frames"].shape[0])

    tmp_path = out_path.with_suffix(out_path.suffix + ".building")
    mm = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(total, n_features))
    offset = 0
    for i in range(job.n_chunks):
        with np.load(job._chunk_path(i)) as data:  # noqa: SLF001
            frames = np.asarray(data["frames"], dtype=np.float32)
            mm[offset : offset + len(frames)] = frames
            offset += len(frames)
    mm.flush()
    del mm
    os.replace(tmp_path, out_path)
    sidecar.write_text(json.dumps({"n_frames": total, "fingerprint": fingerprint}))
    return total


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_chunk(
    ids: list[str],
    id_to_row: dict[str, Any],
    cfg: dict[str, Any],
    bona_gmm: DiagGMM,
    spoof_gmm: DiagGMM,
) -> np.ndarray:
    scores = np.empty(len(ids), dtype=np.float64)
    for i, utt_id in enumerate(ids):
        frames = extract_frames_for_row(id_to_row[utt_id], cfg)
        scores[i] = score_llr(bona_gmm, spoof_gmm, frames)
    return scores


def score_manifest(
    df: pd.DataFrame,
    bona_gmm: DiagGMM,
    spoof_gmm: DiagGMM,
    cfg: dict[str, Any],
    job_dir: Path,
    resources_cfg: dict[str, Any],
    memory_guard: MemoryGuard | None = None,
) -> pd.DataFrame:
    """Score every row of `df` (test-set audio recomputed on the fly, never cached)
    through a resumable `ChunkedJob`, then merge scores back onto `clip_id`/`label`
    (and `attack_id`, if present)."""
    id_to_row = {row.utt_id: row for row in df.itertuples(index=False)}
    item_ids = sorted(id_to_row)
    job = ChunkedJob(
        out_dir=job_dir,
        item_ids=item_ids,
        chunk_size=_utt_chunk_size(resources_cfg),
        job_config={
            "kind": "score",
            "features": cfg["features"],
            "preprocessing": cfg["preprocessing"],
        },
        memory_guard=memory_guard,
    )
    for chunk_idx, ids in job.pending():
        scores = _score_chunk(ids, id_to_row, cfg, bona_gmm, spoof_gmm)
        job.save_chunk(chunk_idx, ids, {"scores": scores})

    result = job.load_all()
    scores_by_id = dict(zip(result["ids"].tolist(), result["scores"].tolist(), strict=True))

    out = df[["utt_id", "label"]].rename(columns={"utt_id": "clip_id"}).copy()
    out["score"] = out["clip_id"].map(scores_by_id)
    if "attack_id" in df.columns:
        out["attack_id"] = df["attack_id"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_timing(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    resources_cfg = load_resources(args.resources)
    apply_resource_limits(resources_cfg)

    train_df = _load_manifest(cfg["data"]["train"])
    n = args.n or resources_cfg["resources"]["timing_run_clips"]
    sample_rows = list(train_df.head(n).itertuples(index=False))

    def _fn(row: Any) -> None:
        extract_frames_for_row(row, cfg)

    result = timing_run(
        _fn, sample_rows, n=len(sample_rows), total_items=len(train_df), cfg=resources_cfg
    )
    print("LFCC extraction timing:")
    print(format_timing(result))

    em_estimate: dict[str, Any] = {}
    probe_rows = sample_rows[: min(20, len(sample_rows))]
    if probe_rows:
        subset = np.concatenate([extract_frames_for_row(r, cfg) for r in probe_rows], axis=0)
        n_components = cfg["gmm"]["n_components"]
        if len(subset) >= n_components:
            gmm = DiagGMM(
                n_components=n_components,
                n_features=_N_FEATURES,
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
    config_summary = (
        f"a1 {front_end}-gmm train, {cfg['gmm']['n_components']}-component diag GMM x2"
    )

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
            cache_dir = Path("data/cache/a1") / _front_end_config_hash(cfg)
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
            run.note(f"train frames: bonafide={n_bona}, spoof={n_spoof} (dim={_N_FEATURES})")

            bona_mm = np.memmap(bona_path, dtype=np.float32, mode="r", shape=(n_bona, _N_FEATURES))
            spoof_mm = np.memmap(
                spoof_path, dtype=np.float32, mode="r", shape=(n_spoof, _N_FEATURES)
            )

            gmm_cfg = cfg["gmm"]
            bona_gmm = DiagGMM(
                n_components=gmm_cfg["n_components"],
                n_features=_N_FEATURES,
                variance_floor=gmm_cfg["variance_floor"],
            )
            bona_gmm.fit_em(
                bona_mm,
                n_iter=gmm_cfg["em_iters"],
                chunk_frames=gmm_cfg["chunk_frames"],
                seed=cfg["seed"],
                init_sample_size=gmm_cfg["init_sample_size"],
            )
            spoof_gmm = DiagGMM(
                n_components=gmm_cfg["n_components"],
                n_features=_N_FEATURES,
                variance_floor=gmm_cfg["variance_floor"],
            )
            spoof_gmm.fit_em(
                spoof_mm,
                n_iter=gmm_cfg["em_iters"],
                chunk_frames=gmm_cfg["chunk_frames"],
                seed=cfg["seed"] + 1,
                init_sample_size=gmm_cfg["init_sample_size"],
            )

            bona_gmm.save(run.exp_dir / "bona_gmm.npz")
            spoof_gmm.save(run.exp_dir / "spoof_gmm.npz")
            run.note(
                f"bona fide GMM final avg log-lik={bona_gmm.log_likelihood_history[-1]:.3f}, "
                f"spoof GMM final avg log-lik={spoof_gmm.log_likelihood_history[-1]:.3f}"
            )

            for name, path in cfg["data"]["testsets"].items():
                test_df = _load_manifest(path)
                scores_df = score_manifest(
                    test_df,
                    bona_gmm,
                    spoof_gmm,
                    cfg,
                    run.exp_dir / "score_jobs" / name,
                    resources_cfg,
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

    print(f"\nwrote experiments/{exp_id}/scores/{{dev,eval}}.csv (+ more if configured).")
    print(f"Next: uv run python scripts/score_experiment.py experiments/{exp_id}")
    print("Then:  uv run python scripts/make_tables.py")


def cmd_score(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    resources_cfg = load_resources(args.resources)
    model_dir = Path(args.model_from)

    bona_gmm = DiagGMM.load(model_dir / "bona_gmm.npz")
    spoof_gmm = DiagGMM.load(model_dir / "spoof_gmm.npz")

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
                test_df = _load_manifest(path)
                scores_df = score_manifest(
                    test_df,
                    bona_gmm,
                    spoof_gmm,
                    cfg,
                    run.exp_dir / "score_jobs" / name,
                    resources_cfg,
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

    print(f"\nwrote experiments/{exp_id}/scores/*.csv")
    print(
        f"Next: uv run python scripts/score_experiment.py experiments/{exp_id} "
        f"--threshold-from {model_dir}"
    )
    print("Then:  uv run python scripts/make_tables.py")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/a1_lfcc_gmm.yaml")
    parser.add_argument("--resources", default="configs/resources.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_timing = sub.add_parser("timing")
    p_timing.add_argument("--n", type=int, default=None)
    p_timing.set_defaults(fn=cmd_timing)

    p_train = sub.add_parser("train")
    p_train.add_argument("--exp", required=True)
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--tag", default=None, help="suffix for the experiment id (slug)")
    p_train.add_argument("--seed", type=int, default=None, help="override cfg['seed']")
    p_train.set_defaults(fn=cmd_train)

    p_score = sub.add_parser("score")
    p_score.add_argument("--model-from", required=True)
    p_score.add_argument("--exp", required=True)
    p_score.add_argument("--silence-mode", choices=["raw", "trimmed"], required=True)
    p_score.add_argument("--testsets", required=True)
    p_score.add_argument("--resume", action="store_true")
    p_score.add_argument("--tag", default=None, help="suffix for the experiment id (slug)")
    p_score.add_argument("--seed", type=int, default=None, help="override cfg['seed']")
    p_score.set_defaults(fn=cmd_score)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
