"""Reusable disk-backed frame cache, without loading all frames into RAM."""
from pathlib import Path
from tqdm.auto import tqdm
import json
import numpy as np
import soundfile as sf
from audio_df2.helpers.parallel import parallel_map
from audio_df2.modules.lfcc import frame_count


def cache_frames(rows, extract, directory: str | Path, key: str, n_features: int):
    """Create/reuse a float32 memmap for one class. `extract(path)` returns [T,D].

    Atomic metadata is written last; interrupted extraction restarts safely.
    Cache key must encode preprocessing, manifest hash and feature configuration.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    file = directory / f"{key}.bin"
    meta = directory / f"{key}.json"
    if file.exists() and meta.exists():
        count = json.loads(meta.read_text())["frames"]
        if file.stat().st_size == count * n_features * 4:
            tqdm.write(f"Cache hit: {file.name} ({count:,} frames)")
            return np.memmap(file, dtype="float32", mode="r", shape=(count, n_features))
    # First pass counts frames (no large array held).
    paths = list(rows)
    count = sum(frame_count(sf.info(path).frames, frame_length=320, hop=160) for path in paths)
    if count == 0:
        raise ValueError("No training frames")
    tmp = directory / f"{key}.building"
    mm = np.memmap(tmp, dtype="float32", mode="w+", shape=(count, n_features))
    offset = 0
    for path, features in zip(paths, parallel_map(extract, paths, workers=4, desc="Extracting LFCC")):
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != n_features:
            raise ValueError(
                f"Wrong feature shape for {path}: {features.shape}")

        mm[offset:offset + len(features)] = features
        offset += len(features)

    mm.flush()
    del mm
    tmp.replace(file)
    meta.write_text(json.dumps({"frames": count, "features": n_features}))
    return np.memmap(file, dtype="float32", mode="r", shape=(count, n_features))
