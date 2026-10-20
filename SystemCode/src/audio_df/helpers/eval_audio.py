"""Shared, resumable silence-trimmed evaluation audio; training audio untouched."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from audio_df.helpers.audio import load_audio


class TrimRecipe(TypedDict):
    method: str
    top_db: float
    frame_length: int
    hop_length: int
    sample_rate: int
    version: int


TRIM_RECIPE: Final[TrimRecipe] = {
    "method": "librosa.effects.split",
    "top_db": 30.0,
    "frame_length": 2048,
    "hop_length": 512,
    "sample_rate": 16000,
    "version": 1,
}


def trim_silence(x: np.ndarray) -> np.ndarray:
    """Remove leading/trailing silence while retaining internal pauses."""
    import librosa

    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0 or not np.any(x):
        return x.copy()
    intervals = librosa.effects.split(
        x,
        top_db=TRIM_RECIPE["top_db"],
        frame_length=TRIM_RECIPE["frame_length"],
        hop_length=TRIM_RECIPE["hop_length"],
    )
    if len(intervals) == 0:
        return x.copy()
    return x[int(intervals[0, 0]):int(intervals[-1, 1])].copy()


def trimmed_manifest(df: pd.DataFrame, cache_root: Path, split: str) -> pd.DataFrame:
    """Cache trimmed WAVs in manifest order, preserving labels and all other columns."""
    source = df[["utt_id", "path"]].copy()
    files = []
    for path in source["path"]:
        stat = Path(path).stat()
        files.append((stat.st_size, stat.st_mtime_ns))
    payload = {
        "manifest": source.to_csv(index=False),
        "files": files,
        "recipe": TRIM_RECIPE,
    }
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_root) / "evaluation_audio" / "trimmed" / split / identity[:20]
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / "cache.json"
    if metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8"))["identity"] != identity:
            raise ValueError(f"Trimmed audio cache mismatch: {directory}")
    else:
        temp = directory / "cache.json.tmp"
        temp.write_text(json.dumps({"identity": identity, "recipe": TRIM_RECIPE}, indent=2), encoding="utf-8")
        os.replace(temp, metadata)

    # A complete cache was verified clip-by-clip when first generated. On
    # subsequent runs, avoid opening every WAV header (very slow on Windows).
    # Source file sizes/mtimes are already included in the identity above.
    complete = directory / "COMPLETE"
    if complete.is_file():
        try:
            record = json.loads(complete.read_text(encoding="utf-8"))
            expected = len(source)
            # scandir avoids tens of thousands of separate sf.info() calls.
            with os.scandir(directory) as entries:
                actual = sum(e.is_file() and e.name.endswith(".wav") and
                             len(e.name) == 11 and e.name[:7].isdigit()
                             for e in entries)
            if record == {"identity": identity, "count": expected} and actual == expected:
                result = df.copy()
                result["path"] = [str(directory / f"{i:07d}.wav") for i in range(expected)]
                tqdm.write(f"Trim audio {split}: cache hit ({expected:,} clips)")
                return result
        except (OSError, ValueError, TypeError):
            pass

    paths: list[str] = []
    reused = 0
    for index, path in enumerate(tqdm(source["path"], desc=f"Trim audio {split}", unit="clip")):
        target = directory / f"{index:07d}.wav"
        if target.is_file():
            try:
                info = sf.info(target)
                if info.frames > 0 and info.samplerate == TRIM_RECIPE["sample_rate"]:
                    reused += 1
                    paths.append(str(target.resolve()))
                    continue
            except (OSError, RuntimeError):
                pass
        audio = load_audio(path, TRIM_RECIPE["sample_rate"])
        trimmed = trim_silence(audio)
        if len(trimmed) == 0:
            raise ValueError(f"Empty audio after trimming: {path}")
        temp = directory / f"{index:07d}.tmp.wav"
        sf.write(temp, trimmed, TRIM_RECIPE["sample_rate"], subtype="FLOAT")
        os.replace(temp, target)
        paths.append(str(target.resolve()))
    # Mark complete only after every clip was successfully validated or written.
    marker_tmp = directory / "COMPLETE.tmp"
    marker_tmp.write_text(json.dumps({"identity": identity, "count": len(source)}), encoding="utf-8")
    os.replace(marker_tmp, complete)
    if reused:
        tqdm.write(f"Trim audio {split}: reused {reused:,}/{len(df):,} cached clips")
    result = df.copy()
    result["path"] = paths
    return result
