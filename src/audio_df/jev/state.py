"""A3's evidence state: turn an A2 score + raw audio into a JSON-serialisable, label-free
description of "the evidence", for the JEV API to reason over via typed questions.

Two variants (`configs/a3_jev.yaml: variants.state`):

- `"numbers"` — raw numeric fields (detector score, confidence, acoustic ratios, duration).
- `"words"` — the same underlying quantities, bucketed into words by thresholds applied in
  code (never learned, never fit to this clip) — `configs/a3_jev.yaml: thresholds`.

**Leakage is a hard requirement, not a style preference.** `build_state` takes `clip_id` only
so callers/tests can assert it never appears in the returned dict; the state itself must never
contain the label, attack_id, speaker, file path, or clip id — the JEV API only ever sees
evidence, never the answer key. See `tests/test_jev_state.py::test_no_label_leakage`.

This module does not know about the JEV client, the cache, or the questions — it only builds
a dict. It reuses `audio_df.audio.silence_ratio` rather than re-deriving silence handling.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from audio_df.audio import silence_ratio

__all__ = ["LEVEL_WORDS", "build_state"]

# Five-way detector level, ordered low (spoof) -> high (bona fide). Index used by
# `audio_df.jev.client.DryRunJevClient` as the sole, deterministic driver of its stub
# answers — see that module's docstring for why a word-level ordinal is enough for a
# clearly-labelled dry run.
LEVEL_WORDS: tuple[str, ...] = (
    "strongly spoof",
    "spoof",
    "borderline",
    "bona fide",
    "strongly bona fide",
)

_DEFAULT_ACOUSTIC_WORDS = ("low", "medium", "high")


def _level_word(score: float, threshold: float, margins: tuple[float, float]) -> str:
    """Map a detector score to one of `LEVEL_WORDS`, relative to `threshold ± margins`.

    `margins = (m1, m2)` with `0 <= m1 <= m2`. `d = score - threshold`:
    `d <= -m2` -> strongly spoof; `-m2 < d <= -m1` -> spoof; `-m1 < d <= m1` -> borderline;
    `m1 < d <= m2` -> bona fide; `d > m2` -> strongly bona fide. Applied in code, never
    learned from data (handoff spec §A3).
    """
    m1, m2 = margins
    d = float(score) - float(threshold)
    if d <= -m2:
        return LEVEL_WORDS[0]
    if d <= -m1:
        return LEVEL_WORDS[1]
    if d <= m1:
        return LEVEL_WORDS[2]
    if d <= m2:
        return LEVEL_WORDS[3]
    return LEVEL_WORDS[4]


def _confidence_from_margin(margin: float, m2: float) -> float:
    """Squash a detector-score margin into a probability-like confidence in `[0, 1)`.

    `confidence = 2 * sigmoid(margin / m2) - 1`, a **fixed, documented transform** — not a
    fitted calibration, and deliberately *not* the raw margin or the raw detector score
    (those are already exposed separately as `detector.score`/implicitly via `level`).
    `m2` is the "strongly bona fide"/"strongly spoof" margin from
    `configs/a3_jev.yaml: thresholds.detector.margins[1]` — the same knob `_level_word`
    uses for its outermost boundary, so `confidence` and `level` are consistent with each
    other. Properties: `margin=0` -> `confidence=0.0`; `margin=m2` (right at the "strongly"
    boundary) -> `confidence≈0.462`; `confidence` -> 1 as `margin` -> infinity; strictly
    increasing in `margin`. `m2<=0` is treated as `m2=1.0` (a threshold config with no
    "strongly" margin has no non-arbitrary scale to squash against, so this falls back to
    an arbitrary-but-fixed unit scale rather than dividing by zero).
    """
    scale = m2 if m2 > 0 else 1.0
    return float(2.0 / (1.0 + np.exp(-margin / scale)) - 1.0)


def _bucket(value: float, thresholds: tuple[float, float], words: tuple[str, str, str]) -> str:
    """Three-way bucket of `value` by `thresholds = (t1, t2)`, `t1 <= t2`."""
    t1, t2 = thresholds
    if value <= t1:
        return words[0]
    if value <= t2:
        return words[1]
    return words[2]


def _high_band_energy_fraction(x: np.ndarray, sr: int, cutoff_hz: float) -> float:
    """Fraction of power-spectrum energy at or above `cutoff_hz`, over the whole clip.

    A single-window power spectrum (`|rfft(x)|**2`) is enough for this coarse evidence
    field — this is not a spectrogram-resolution feature, just a scalar summary of how much
    energy sits above `cutoff_hz`. Deterministic given `x`; an all-zero/empty clip returns
    `0.0` rather than dividing by zero.
    """
    n = len(x)
    if n == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    total = float(spec.sum())
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    high = float(spec[freqs >= cutoff_hz].sum())
    return high / total


def build_state(
    clip_id: str,
    detector_score: float,
    audio: np.ndarray,
    cfg: dict[str, Any],
    variant: str,
) -> dict[str, Any]:
    """Build the JSON-serialisable evidence state for one clip.

    `clip_id` is accepted only to make the "the state never leaks identity" contract
    explicit and testable — it is never written into the returned dict. `detector_score`
    is A2's raw dev-set score for this clip (higher = more bona fide, project convention).
    `audio` is the clip's samples at `cfg.get("sample_rate", 16000)` Hz. `variant` is
    `"numbers"` or `"words"`.

    Fields (both variants): `detector.level` (word, `LEVEL_WORDS`). `numbers` variant adds
    `detector.score` (raw float) and `detector.confidence` — **not** the raw score-to-
    threshold margin (that would just be a re-derivation of `score` and `level`), but the
    margin squashed through the fixed transform `_confidence_from_margin` (module-level
    docstring there) — plus numeric acoustic fields; `words` variant instead buckets
    `detector.confidence` (from the raw margin, via `margins`) and every acoustic field
    into a threshold word from `cfg["thresholds"]`. Deterministic: same inputs ->
    byte-identical JSON.
    """
    if variant not in ("numbers", "words"):
        raise ValueError(f"variant must be 'numbers' or 'words', got {variant!r}")

    thresholds = cfg["thresholds"]
    det_cfg = thresholds["detector"]
    acoustic_cfg = thresholds["acoustic"]
    sr = int(cfg.get("sample_rate", 16000))

    threshold = float(det_cfg["eer_threshold"])
    margins = tuple(det_cfg["margins"])
    level = _level_word(detector_score, threshold, margins)
    margin = abs(float(detector_score) - threshold)

    sil = silence_ratio(np.asarray(audio), top_db=int(acoustic_cfg.get("silence_top_db", 30)))
    high_band = _high_band_energy_fraction(
        np.asarray(audio), sr, float(acoustic_cfg["high_band_hz"])
    )
    duration_s = len(audio) / sr if sr else 0.0

    if variant == "numbers":
        detector_block: dict[str, Any] = {
            "score": float(detector_score),
            "level": level,
            "confidence": _confidence_from_margin(margin, margins[1]),
        }
        acoustic_block: dict[str, Any] = {
            "high_band_energy": float(high_band),
            "silence_leading": float(sil["leading"]),
            "silence_trailing": float(sil["trailing"]),
            "silence_total": float(sil["total"]),
            "duration_s": float(duration_s),
        }
    else:
        confidence_words = tuple(acoustic_cfg.get("confidence_words", _DEFAULT_ACOUSTIC_WORDS))
        detector_block = {
            "level": level,
            "confidence": _bucket(margin, margins, confidence_words),
        }
        high_band_words = tuple(acoustic_cfg.get("high_band_words", _DEFAULT_ACOUSTIC_WORDS))
        silence_words = tuple(acoustic_cfg.get("silence_words", ("little", "some", "a lot")))
        duration_words = tuple(acoustic_cfg.get("duration_words", ("short", "medium", "long")))
        silence_thresholds = tuple(acoustic_cfg["silence"])
        acoustic_block = {
            "high_band_energy": _bucket(
                high_band, tuple(acoustic_cfg["high_band"]), high_band_words
            ),
            "silence_leading": _bucket(sil["leading"], silence_thresholds, silence_words),
            "silence_trailing": _bucket(sil["trailing"], silence_thresholds, silence_words),
            "silence_total": _bucket(sil["total"], silence_thresholds, silence_words),
            "duration_s": _bucket(duration_s, tuple(acoustic_cfg["duration"]), duration_words),
        }

    return {
        "variant": variant,
        "detector": detector_block,
        "acoustic": acoustic_block,
    }
