"""A1 training: a resumable LFCC train-frame cache and two chunked-EM diagonal GMMs.

`train` extracts LFCC frames for every TRAIN utterance exactly once, per class, into an
on-disk `float32` memmap under `data/cache/a1/<config_hash>/` (gitignored; the hash covers
the front-end and preprocessing config, so a changed parameter never reuses a stale cache).
Dev, eval and other test sets are never cached: they are scored once per run, from audio.

Two resumable, memory-bounded stages:

1. `ChunkedJob` extracts frames a batch of utterances at a time into per-chunk `.npz`
   files; rerunning with `--resume` replays no completed chunk.
2. `build_frame_memmap` streams those chunks, one at a time, into one flat memmap that
   `DiagGMM.fit_em` reads in chunks, so the full array is never in RAM.

Training peaks under about 2 GB RSS by construction: one chunk of utterances' frames, or one
`[chunk_frames, n_components]` responsibility array (50k x 512 float64, about 205 MB), plus
O(K*D) accumulators. See `models/gmm.py` for why `sklearn.mixture.GaussianMixture` is not
used. A1 scores whole utterances (no crop), as the official B02 baseline does.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audio_df.features.audio import load_audio
from audio_df.models.a1_lfcc_gmm import lfcc_frames
from audio_df.models.gmm import DiagGMM
from audio_df.runtime.resources import ChunkedJob, MemoryGuard

__all__ = [
    "N_FEATURES",
    "build_frame_memmap",
    "build_train_frame_job",
    "fit_gmm",
    "frames_for_row",
    "front_end_config_hash",
    "utt_chunk_size",
]

N_FEATURES = 60


def front_end_config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(
        {"features": cfg["features"], "preprocessing": cfg["preprocessing"]},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def utt_chunk_size(resources_cfg: dict[str, Any]) -> int:
    """Utterances per frame-extraction chunk: small (<=50), independent of
    `resources.chunk_size` (which is tuned for SSL-style item chunks, not per-utterance
    LFCC extraction batches whose per-item frame count varies with duration)."""
    return min(50, resources_cfg["resources"]["chunk_size"])


def frames_for_row(row: Any, cfg: dict[str, Any]) -> np.ndarray:
    """Load one manifest row's audio and extract its `[T, 60]` LFCC frames."""
    x, _ = load_audio(row.path, target_sr=cfg["preprocessing"]["sample_rate"])
    return lfcc_frames(x, cfg["features"], cfg["preprocessing"])


def extract_chunk_frames(
    ids: list[str], id_to_row: dict[str, Any], cfg: dict[str, Any]
) -> np.ndarray:
    arrays = [frames_for_row(id_to_row[i], cfg) for i in ids]
    if not arrays:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
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
    id_to_row = {str(row.utt_id): row for row in df.itertuples(index=False)}
    item_ids = sorted(id_to_row)
    job = ChunkedJob(
        out_dir=cache_dir / label,
        item_ids=item_ids,
        chunk_size=utt_chunk_size(resources_cfg),
        job_config={
            "label": label,
            "features": cfg["features"],
            "preprocessing": cfg["preprocessing"],
        },
        memory_guard=memory_guard,
    )
    for chunk_idx, ids in job.pending():
        frames = extract_chunk_frames(ids, id_to_row, cfg)
        job.save_chunk(chunk_idx, ids, {"frames": frames})
    return job


def build_frame_memmap(job: ChunkedJob, out_path: Path, n_features: int = N_FEATURES) -> int:
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


def fit_gmm(frames: np.ndarray, gmm_cfg: dict[str, Any], seed: int) -> DiagGMM:
    """Fit one class's diagonal GMM by chunked EM over `frames` (a memmap is fine)."""
    gmm = DiagGMM(
        n_components=gmm_cfg["n_components"],
        n_features=N_FEATURES,
        variance_floor=gmm_cfg["variance_floor"],
    )
    gmm.fit_em(
        frames,
        n_iter=gmm_cfg["em_iters"],
        chunk_frames=gmm_cfg["chunk_frames"],
        seed=seed,
        init_sample_size=gmm_cfg["init_sample_size"],
    )
    return gmm
