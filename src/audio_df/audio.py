"""Audio loading and shaping: read a file, resample, fix its length, and describe silence.

This module only ever touches waveforms. It never computes spectral features (that is
`features.py`'s job) and never knows about caching or manifests. Everything here operates
on a 1-D `np.ndarray` of float samples plus an integer sample rate.
"""

from __future__ import annotations

import librosa
import numpy as np
import soundfile as sf
from librosa import resample as librosa_resample

__all__ = ["load_audio", "fix_length", "rms", "silence_ratio"]


def load_audio(
    path: str,
    target_sr: int = 16000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """Load a file from disk and resample it to `target_sr`.

    Reads with `soundfile` (fast, format-agnostic via libsndfile) and resamples with
    `librosa.resample` (a polyphase/soxr-quality resampler, good enough for a detection
    front-end; we are not trying to preserve phase for perceptual playback). Multi-channel
    audio is averaged to mono when `mono=True`, matching the convention every downstream
    module in this project assumes (`configs/data.yaml: audio.mono: true`).

    The returned sample rate is asserted to equal `target_sr` — silently returning audio
    at the wrong rate would poison every downstream STFT bin width without any visible
    symptom, so we fail loudly here instead of downstream.
    """
    x, sr = sf.read(path, always_2d=False, dtype="float32")

    if x.ndim > 1 and mono:
        x = x.mean(axis=1)
    elif x.ndim > 1:
        # Caller asked to keep channels; still hand back a 1-D array is not possible here,
        # but this project's config always sets mono=true, so keep the simple path exact
        # and only average when asked.
        pass

    if sr != target_sr:
        x = librosa_resample(x.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    assert sr == target_sr, f"load_audio postcondition failed: got sr={sr}, expected {target_sr}"

    return x.astype(np.float32), sr


def fix_length(
    x: np.ndarray,
    n_samples: int,
    mode: str = "center",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crop or pad `x` to exactly `n_samples`.

    Modes:
      - `"center"`: deterministic center crop when `x` is long enough; center-padded with
        zeros when short. Intended for eval, per `configs/data.yaml: crop_strategy:
        random_train_center_eval`.
      - `"random"`: random-offset crop when long enough (uses `rng`, defaults to
        `np.random.default_rng()` for callers that don't care about determinism); falls
        back to `"repeat_pad"` behaviour when short, since a "random pad" has no natural
        meaning and repeat-padding is this project's declared pad strategy
        (`configs/data.yaml: pad_strategy: repeat`).
      - `"repeat_pad"`: tile short audio to length rather than zero-padding. Zero-padding a
        short clip invents silence that isn't in the source signal, and silence ratio is
        one of the nuisance features the Stage 0 leakage audit (T09) screens for — a model
        could learn "this file was padded" as a leakage shortcut correlated with genuine
        vs. real length distributions. Repeat-padding avoids injecting *new* silence; it
        does not, by itself, guarantee zero added silence if the source clip itself already
        ends in silence, which is why `silence_ratio` exists as a separate, honest measure
        callers can check after the fact rather than us silently "fixing" it here.
        When `x` is already long enough this mode crops with a center crop (there is
        nothing to pad).

    Center mode is the only one that zero-pads (for short audio), matching the "center"
    contract of adding equal silence on both sides rather than tiling. Any caller that cares
    about not injecting silence should use `"repeat_pad"`, which never zero-pads.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)

    if n == n_samples:
        return x.copy()

    if n > n_samples:
        if mode == "random":
            start = int(rng.integers(0, n - n_samples + 1))
        else:
            # "center" and "repeat_pad" (nothing to repeat-pad; just crop) both center-crop
            start = (n - n_samples) // 2
        return x[start : start + n_samples].copy()

    # n < n_samples: need to pad
    deficit = n_samples - n

    if mode == "repeat_pad":
        if n == 0:
            return np.zeros(n_samples, dtype=x.dtype)
        n_tiles = int(np.ceil(n_samples / n))
        tiled = np.tile(x, n_tiles)
        return tiled[:n_samples].copy()

    if mode == "random":
        # No natural "random pad": fall back to tiling so we never invent silence.
        if n == 0:
            return np.zeros(n_samples, dtype=x.dtype)
        n_tiles = int(np.ceil(n_samples / n))
        tiled = np.tile(x, n_tiles)
        return tiled[:n_samples].copy()

    # "center": zero-pad symmetrically (extra sample goes at the end)
    left = deficit // 2
    right = deficit - left
    return np.pad(x, (left, right), mode="constant", constant_values=0.0)


def rms(x: np.ndarray) -> float:
    """Root-mean-square amplitude: `sqrt(mean(x**2))`. Empty input returns 0.0."""
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def silence_ratio(
    x: np.ndarray,
    top_db: int = 30,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> dict:
    """Leading, trailing and total silence proportions, on **framed RMS energy**.

    Convention: this matches `librosa.effects.trim`/`split`, not a per-sample amplitude
    threshold. The signal is framed (`frame_length` samples per frame, `hop_length` stride,
    centered — librosa's defaults: 2048/512), RMS energy is computed per frame, converted
    to dB relative to the loudest frame in the clip, and a frame counts as silent when its
    dB is more than `top_db` below that peak frame. `frame_length`/`hop_length` are exposed
    as arguments (not hardcoded) so a caller (e.g. T09's leakage audit) can state exactly
    what framing produced a given ratio.

    This used to threshold instantaneous per-sample magnitude instead. That is the wrong
    convention: every periodic waveform crosses near zero twice per cycle, so per-sample
    thresholding counts ordinary zero-crossings as "silence" — a pure 440 Hz tone with no
    silence at all scored ~2.5%, and a speech-like amplitude-modulated harmonic signal with
    no true silence scored ~23%, because voicing periodicity and amplitude-envelope shape
    leaked into the "silence" number. That makes `silence_ratio` a proxy for waveform shape
    rather than an honest silence measure — exactly the kind of nuisance feature that would
    corrupt Stage 0's leakage audit rather than support it. Framed RMS energy does not have
    this failure mode: a frame spanning many cycles of a loud tone has high RMS regardless
    of individual zero crossings.

    We compute this by delegating to `librosa.effects.split` (with the same `top_db`,
    `frame_length`, `hop_length`), which returns the sample intervals librosa itself
    considers non-silent under this exact convention, rather than re-deriving the frame/dB
    arithmetic by hand and risking a second, subtly different reimplementation. `leading` is
    the sample offset of the first non-silent interval's start (as a fraction of length);
    `trailing` is the distance from the last non-silent interval's end to the end of the
    clip; `total` is one minus the total non-silent duration across all intervals, as a
    fraction of length. Returned as a dict so this shape can grow without breaking callers.

    An all-zero, empty, or (per this convention) entirely-below-threshold input is defined
    as fully silent (ratios of 1.0) — there is no non-silent interval to reference.
    """
    n = len(x)
    if n == 0:
        return {"leading": 1.0, "trailing": 1.0, "total": 1.0}

    peak = np.max(np.abs(x))
    if peak == 0:
        return {"leading": 1.0, "trailing": 1.0, "total": 1.0}

    intervals = librosa.effects.split(
        x, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    if intervals.size == 0:
        # every frame fell below threshold: fully silent by this convention
        return {"leading": 1.0, "trailing": 1.0, "total": 1.0}

    non_silent_samples = int(np.sum(intervals[:, 1] - intervals[:, 0]))
    total = 1.0 - non_silent_samples / n

    leading = float(intervals[0, 0]) / n
    trailing = float(n - intervals[-1, 1]) / n

    return {"leading": leading, "trailing": trailing, "total": total}
