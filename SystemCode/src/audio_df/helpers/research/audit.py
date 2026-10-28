"""Dataset audit helpers for notebooks 01 and 02: folders, protocol keys, audio checks, silence."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audio_df.data.degrade import clip_seed
from audio_df.data.manifests import _audio_metadata
from audio_df.helpers.audio import load_audio
from audio_df.helpers.signal import silence_ratio

SR = 16000


def tree(d: str | Path, depth: int = 2, level: int = 0) -> None:
    """Print sub-folders with a count of file extensions in each, down to `depth`."""
    for p in sorted(Path(d).iterdir()):
        if p.is_dir():
            exts = pd.Series([f.suffix for f in p.iterdir() if f.is_file()], dtype=str)
            print("  " * level + f"{p.name}/", exts.value_counts().to_dict())
            if level + 1 < depth:
                tree(p, depth, level + 1)


def read_key(path: str | Path, **kw: Any) -> pd.DataFrame:
    """Read a whitespace protocol file; print value counts of every low-cardinality column."""
    k = pd.read_csv(path, sep=r"\s+", header=None, **kw)
    for c in k.columns:
        if k[c].nunique() < 30:
            print(c, k[c].value_counts().to_dict())
    return k


def check_audio(ids, audio_dir: str | Path, ext: str = ".flac", n: int = 200) -> list[Path]:
    """Count missing files and show sample rate / duration / format of `n` sampled clips."""
    from IPython.display import display

    paths = [Path(audio_dir) / f"{u}{ext}" for u in ids]
    print("clips:", len(paths), "| missing audio:", sum(not p.exists() for p in paths))
    info = pd.DataFrame([_audio_metadata(p) for p in pd.Series(paths).sample(n, random_state=0)],
                        columns=["sample_rate", "duration_s", "format"])
    display(info.describe(include="all"))
    return paths


def audio_stats(folder: str | Path) -> tuple[int, float]:
    """(audio files, GB) under a folder, recursive."""
    n, size, stack = 0, 0, [folder]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for e in entries:
                if e.is_dir():
                    stack.append(e.path)
                elif e.name.lower().endswith((".flac", ".wav")):
                    n, size = n + 1, size + e.stat().st_size
    return n, size / 1e9


def e4_noise_bank(cfg: dict, root: str | Path, clip_id: str,
                  condition: str) -> list[np.ndarray] | None:
    """The one MUSAN file scripts/make_degraded.py picks for this clip, or None without MUSAN."""
    noise_dir = Path(root) / cfg["conditions"][condition]["noise_dir"]
    if not noise_dir.is_dir():
        return None
    files = sorted(p for p in noise_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac"})
    if not files:
        return None
    rng = np.random.default_rng(clip_seed(cfg["seed"], clip_id, condition))
    return [load_audio(files[int(rng.integers(len(files)))], SR)]


def edge_silence(df: pd.DataFrame) -> pd.DataFrame:
    """Add lead_s / trail_s: leading / trailing silence in seconds (30 dB rule, as trimming)."""
    rows = []
    for path in df.path:
        x = load_audio(path, SR)
        r, dur = silence_ratio(x), len(x) / SR
        rows.append((r["leading"] * dur, r["trailing"] * dur))
    return df.assign(lead_s=[a for a, _ in rows], trail_s=[b for _, b in rows])
