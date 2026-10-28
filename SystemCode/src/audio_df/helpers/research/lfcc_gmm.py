"""A1 (LFCC-GMM) explainability helpers for notebook 03: frame scores and LFCC feature ablations."""
from __future__ import annotations

import copy
from typing import Any

import librosa
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import lfilter

from audio_df.helpers.audio import load_audio
from audio_df.helpers.eval_audio import TRIM_RECIPE
from audio_df.modules.lfcc import _LOG10_EPS, _frame_signal, lfcc_b02_filterbank, official_deltas

SR, HOP = 16000, 160
FILTERBANK = lfcc_b02_filterbank(20, 512, SR)
BAND_CENTRES = np.linspace(0, SR / 2, 22)[1:-1]  # Hz, the 20 linear filters


def frame_llr(detector: Any, path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Waveform, per-frame A1 score, and a speech mask using the same silence rule as trimming."""
    x = load_audio(path, SR)
    f = detector.features(x)
    llr = detector.bonafide.score_log_likelihood(f) - detector.spoof.score_log_likelihood(f)
    speech = np.zeros(len(x), bool)
    for s, e in librosa.effects.split(x, top_db=TRIM_RECIPE["top_db"],
                                      frame_length=TRIM_RECIPE["frame_length"],
                                      hop_length=TRIM_RECIPE["hop_length"]):
        speech[s:e] = True
    return x, llr, speech[np.minimum(np.arange(len(llr)) * HOP + HOP, len(x) - 1)]


def decompose(detector: Any, df: pd.DataFrame) -> pd.DataFrame:
    """Mean frame score per clip over all frames, speech frames and silence frames."""
    rows = []
    for r in df.itertuples():
        _, llr, sp = frame_llr(detector, r.path)
        rows.append({"label": r.label, "all frames": llr.mean(), "speech only": llr[sp].mean(),
                     "silence only": llr[~sp].mean() if (~sp).any() else 0.0,
                     "silence %": 100 * (1 - sp.mean())})
    return pd.DataFrame(rows)


def log_bands(x: np.ndarray) -> np.ndarray:
    """B02 steps 1-5: pre-emphasis, 20 ms Hamming frames, power spectrum, 20 filters, log10."""
    y = lfilter([1.0, -0.97], [1.0], np.asarray(x, np.float64))
    frames = _frame_signal(y, 320, HOP) * np.hamming(320)[None]
    return np.log10(np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2 @ FILTERBANK.T + _LOG10_EPS)


def lfcc_from(bands: np.ndarray) -> np.ndarray:
    """B02 steps 6-8: DCT, delta, delta-delta -> [frames, 60]."""
    static = dct(bands, type=2, axis=-1, norm="ortho")
    delta = official_deltas(static, 3)
    return np.concatenate([static, delta, official_deltas(delta, 3)], axis=1)


def restrict(gmm: Any, dims) -> Any:
    """The diagonal GMM marginalised to `dims` (exact for diagonal covariances; no refit)."""
    out = copy.copy(gmm)
    out.means, out.vars_, out.n_features = gmm.means[:, dims], gmm.vars_[:, dims], len(dims)
    return out


def freeze(bands: np.ndarray, mask) -> np.ndarray:
    """Hold the selected bands at their clip mean: level kept, change over time removed."""
    out = bands.copy()
    out[:, mask] = bands[:, mask].mean(0)
    return out
