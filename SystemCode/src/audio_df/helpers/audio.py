"""Load audio without coupling feature modules to datasets or experiments."""
from pathlib import Path
import numpy as np
import soundfile as sf

def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != sample_rate:
        from librosa import resample
        x = resample(x.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
    return np.asarray(x, dtype=np.float32)
