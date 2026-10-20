"""Small signal-analysis helpers for feature notebooks."""
import numpy as np

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


import librosa

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

