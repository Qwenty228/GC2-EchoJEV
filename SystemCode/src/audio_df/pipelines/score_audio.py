"""Score every clip of a manifest from its audio with any detector, resumably.

The generic test-set scorer: each clip is loaded at the detector's sample rate and scored
with `Detector.score_waveform` (the same path `Detector.predict` uses). Work is split into
`ChunkedJob` chunks, so an interrupted run resumes without rescoring finished chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audio_df.features.audio import load_audio
from audio_df.models.base import Detector
from audio_df.runtime.resources import ChunkedJob, MemoryGuard

__all__ = ["score_manifest"]


def score_manifest(
    detector: Detector,
    df: pd.DataFrame,
    job_dir: Path,
    chunk_size: int,
    memory_guard: MemoryGuard | None = None,
) -> pd.DataFrame:
    """Score `df` (columns `utt_id`, `path`, `label`[, `attack_id`]) -> a score frame with
    `clip_id`, `label`, `score`[, `attack_id`], higher = more bona fide."""
    id_to_path = {str(u): str(p) for u, p in zip(df["utt_id"], df["path"], strict=True)}
    job = ChunkedJob(
        out_dir=job_dir,
        item_ids=sorted(id_to_path),
        chunk_size=chunk_size,
        job_config=_job_config(detector),
        memory_guard=memory_guard,
    )
    for chunk_idx, ids in job.pending():
        scores = np.empty(len(ids), dtype=np.float64)
        for i, utt_id in enumerate(ids):
            x, _ = load_audio(id_to_path[utt_id], target_sr=detector.sample_rate)
            scores[i] = detector.score_waveform(x)[0]
        job.save_chunk(chunk_idx, ids, {"scores": scores})

    result = job.load_all()
    scores_by_id = dict(zip(result["ids"].tolist(), result["scores"].tolist(), strict=True))
    out = df[["utt_id", "label"]].rename(columns={"utt_id": "clip_id"}).copy()
    out["clip_id"] = out["clip_id"].astype(str)
    out["score"] = out["clip_id"].map(scores_by_id)
    if "attack_id" in df.columns:
        out["attack_id"] = df["attack_id"].to_numpy()
    return out


def _job_config(detector: Detector) -> dict[str, Any]:
    return {
        "kind": "score",
        "system": detector.system,
        "model_version": detector.model_version,
        "sample_rate": detector.sample_rate,
        "preprocessing": getattr(detector, "preprocessing", None),
    }
