"""Tests for `audio_df.audio`. All synthetic — no dataset required."""

from __future__ import annotations

import librosa
import numpy as np
import pytest
import soundfile as sf

from audio_df.audio import fix_length, load_audio, preprocess, rms, silence_ratio, trim_silence


def _tone(freq: float, sr: int, duration_s: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _am_harmonics(
    sr: int,
    duration_s: float,
    f0: float = 150.0,
    n_harmonics: int = 5,
    mod_freq: float = 3.0,
    mod_depth: float = 0.3,
) -> np.ndarray:
    """A speech-like signal: several harmonics under a slow amplitude envelope that never
    reaches zero (envelope ranges over [1 - mod_depth, 1]). Has no true silence anywhere,
    but crosses zero every cycle like any periodic waveform — the case that broke a
    per-sample silence threshold (it counted ordinary zero-crossings as "silence")."""
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    envelope = 1.0 - mod_depth * (0.5 * (1.0 + np.sin(2 * np.pi * mod_freq * t)))
    harmonics = sum(np.sin(2 * np.pi * f0 * (k + 1) * t) / (k + 1) for k in range(n_harmonics))
    x = envelope * harmonics
    x = x / np.max(np.abs(x)) * 0.7
    return x.astype(np.float32)


def _expected_ratio_via_librosa(x, top_db, frame_length, hop_length) -> dict:
    """Independent expected-value computation, straight from librosa's public API, used
    only in tests to pin `silence_ratio`'s convention and catch future drift."""
    n = len(x)
    intervals = librosa.effects.split(
        x, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    if intervals.size == 0:
        return {"leading": 1.0, "trailing": 1.0, "total": 1.0}
    non_silent = int(np.sum(intervals[:, 1] - intervals[:, 0]))
    return {
        "leading": float(intervals[0, 0]) / n,
        "trailing": float(n - intervals[-1, 1]) / n,
        "total": 1.0 - non_silent / n,
    }


# ---------------------------------------------------------------------------
# fix_length: crop
# ---------------------------------------------------------------------------


def test_fix_length_center_crop_length():
    x = _tone(440, 16000, 6.0)
    out = fix_length(x, 64000, mode="center")
    assert out.shape == (64000,)


def test_fix_length_center_crop_is_centered():
    # ramp so we can check which slice was taken
    x = np.arange(1000, dtype=np.float32)
    out = fix_length(x, 400, mode="center")
    expected_start = (1000 - 400) // 2
    np.testing.assert_array_equal(out, x[expected_start : expected_start + 400])


def test_fix_length_random_crop_length():
    x = _tone(440, 16000, 6.0)
    rng = np.random.default_rng(0)
    out = fix_length(x, 64000, mode="random", rng=rng)
    assert out.shape == (64000,)


def test_fix_length_random_crop_deterministic_with_seed():
    x = np.arange(10000, dtype=np.float32)
    out1 = fix_length(x, 3000, mode="random", rng=np.random.default_rng(42))
    out2 = fix_length(x, 3000, mode="random", rng=np.random.default_rng(42))
    np.testing.assert_array_equal(out1, out2)


def test_fix_length_random_crop_differs_across_seeds_generally():
    x = np.arange(10000, dtype=np.float32)
    out1 = fix_length(x, 3000, mode="random", rng=np.random.default_rng(1))
    out2 = fix_length(x, 3000, mode="random", rng=np.random.default_rng(2))
    # not a hard guarantee for arbitrary seeds, but for this range it's overwhelmingly true
    assert not np.array_equal(out1, out2)


# ---------------------------------------------------------------------------
# fix_length: pad
# ---------------------------------------------------------------------------


def test_fix_length_repeat_pad_length():
    x = _tone(440, 16000, 1.0)  # 16000 samples
    out = fix_length(x, 64000, mode="repeat_pad")
    assert out.shape == (64000,)


def test_fix_length_center_pad_length():
    x = _tone(440, 16000, 1.0)
    out = fix_length(x, 64000, mode="center")
    assert out.shape == (64000,)


def test_fix_length_repeat_pad_tiles_short_clip_energy():
    """A 1 s clip repeat-padded to 4 s should have ~the same RMS as the original 1 s clip,
    since it is a periodic tiling of the same signal — unlike zero-padding, which would
    roughly halve the RMS (energy diluted by silence)."""
    sr = 16000
    x = _tone(440, sr, 1.0, amplitude=0.5)
    out = fix_length(x, 4 * sr, mode="repeat_pad")

    rms_orig = rms(x)
    rms_tiled = rms(out)
    rms_zero_padded = rms(np.pad(x, (0, 3 * sr), mode="constant"))

    # tiling preserves energy density
    assert abs(rms_tiled - rms_orig) < 1e-3
    # zero-padding would have diluted it to about half (sqrt(1/4) of energy -> half RMS)
    assert rms_zero_padded < rms_tiled * 0.6


def test_fix_length_repeat_pad_is_actual_tile():
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = fix_length(x, 8, mode="repeat_pad")
    np.testing.assert_array_equal(out, np.array([1, 2, 3, 1, 2, 3, 1, 2], dtype=np.float32))


def test_fix_length_no_op_when_already_correct_length():
    x = np.arange(100, dtype=np.float32)
    out = fix_length(x, 100, mode="center")
    np.testing.assert_array_equal(out, x)


# ---------------------------------------------------------------------------
# rms
# ---------------------------------------------------------------------------


def test_rms_matches_analytic_definition():
    rng = np.random.default_rng(7)
    x = rng.normal(size=5000).astype(np.float32)
    assert np.isclose(rms(x), np.sqrt(np.mean(x.astype(np.float64) ** 2)), rtol=1e-5)


def test_rms_zero_signal():
    x = np.zeros(100, dtype=np.float32)
    assert rms(x) == 0.0


def test_rms_empty():
    assert rms(np.array([], dtype=np.float32)) == 0.0


# ---------------------------------------------------------------------------
# silence_ratio
# ---------------------------------------------------------------------------


def test_silence_ratio_half_silent_signal():
    sr = 16000
    tone = _tone(440, sr, 1.0, amplitude=0.8)
    silence = np.zeros(sr, dtype=np.float32)
    x = np.concatenate([tone, silence])
    result = silence_ratio(x, top_db=30)
    assert set(result.keys()) == {"leading", "trailing", "total"}
    assert abs(result["total"] - 0.5) < 0.1
    assert abs(result["trailing"] - 0.5) < 0.1
    assert result["leading"] < 0.05


def test_silence_ratio_all_silence():
    x = np.zeros(1000, dtype=np.float32)
    result = silence_ratio(x)
    assert result["leading"] == 1.0
    assert result["trailing"] == 1.0
    assert result["total"] == 1.0


def test_silence_ratio_no_silence_pure_tone():
    x = _tone(440, 16000, 1.0, amplitude=0.9)
    result = silence_ratio(x, top_db=30)
    assert result["leading"] < 0.01
    assert result["trailing"] < 0.01
    assert result["total"] < 0.05


def test_silence_ratio_empty():
    result = silence_ratio(np.array([], dtype=np.float32))
    assert result == {"leading": 1.0, "trailing": 1.0, "total": 1.0}


# --- Framed-RMS convention: must agree with librosa, not a per-sample threshold ----------
#
# A per-sample amplitude threshold counts every zero-crossing of a periodic waveform as
# "silence", which corrupted this feature (see audio.py docstring for the incident). These
# tests pin agreement with librosa.effects.split directly, on cases designed to expose that
# exact failure mode: a pure tone and a speech-like AM-harmonic signal, neither of which has
# any true silence, plus a signal that is genuinely half zeros.


def test_silence_ratio_agrees_with_librosa_pure_tone():
    sr = 16000
    x = _tone(440, sr, 1.0, amplitude=0.8)
    top_db, frame_length, hop_length = 30, 2048, 512

    got = silence_ratio(x, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    expected = _expected_ratio_via_librosa(x, top_db, frame_length, hop_length)

    for key in ("leading", "trailing", "total"):
        assert abs(got[key] - expected[key]) < 1e-9, key
    # and, concretely: a pure tone with no silence should score near zero, not ~2.5% as
    # the old per-sample implementation did
    assert got["total"] < 0.01


def test_silence_ratio_agrees_with_librosa_am_harmonics_no_true_silence():
    """Speech-like amplitude-modulated harmonics: the envelope never reaches zero, so
    there is no true silence anywhere in the clip. The old per-sample threshold scored
    this around 23% "silent" because it mistook ordinary zero-crossings for silence; the
    framed-RMS convention (and librosa) should score it near zero."""
    sr = 16000
    x = _am_harmonics(sr, duration_s=2.0)
    top_db, frame_length, hop_length = 30, 2048, 512

    got = silence_ratio(x, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    expected = _expected_ratio_via_librosa(x, top_db, frame_length, hop_length)

    for key in ("leading", "trailing", "total"):
        assert abs(got[key] - expected[key]) < 1e-9, key
    assert got["total"] < 0.05


def test_silence_ratio_agrees_with_librosa_half_true_zeros():
    """A signal that is genuinely half true digital silence (real zeros), not just quiet.
    This is the case a silence detector must get right regardless of convention."""
    sr = 16000
    tone = _tone(440, sr, 1.0, amplitude=0.8)
    silence = np.zeros(sr, dtype=np.float32)
    x = np.concatenate([tone, silence])
    top_db, frame_length, hop_length = 30, 2048, 512

    got = silence_ratio(x, top_db=top_db, frame_length=frame_length, hop_length=hop_length)
    expected = _expected_ratio_via_librosa(x, top_db, frame_length, hop_length)

    # Tolerance here is looser than the exact-delegation tests above only in principle —
    # in practice both call the identical librosa.effects.split path, so we still expect
    # near-exact agreement. A larger tolerance (1e-6) is used only to make explicit that
    # this is checking *convention* (frame-boundary-level) agreement, not asserting our
    # function is bit-identical to a hand-maintained duplicate of librosa's internals.
    for key in ("leading", "trailing", "total"):
        assert abs(got[key] - expected[key]) < 1e-6, key
    # sanity: genuinely half the clip is silent
    assert abs(got["total"] - 0.5) < 0.1


# ---------------------------------------------------------------------------
# load_audio: resampling correctness (real numerical test, not shape-only)
# ---------------------------------------------------------------------------


def test_load_audio_resamples_and_preserves_tone_frequency(tmp_path):
    src_sr = 8000
    target_sr = 16000
    freq = 1000.0
    duration_s = 1.0
    x = _tone(freq, src_sr, duration_s, amplitude=0.7)

    wav_path = tmp_path / "tone_8k.wav"
    sf.write(str(wav_path), x, src_sr)

    y, sr = load_audio(str(wav_path), target_sr=target_sr, mono=True)

    assert sr == target_sr
    assert y.dtype == np.float32

    # FFT peak check: dominant frequency should still be ~1000 Hz after resampling
    n = len(y)
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=1.0 / target_sr)
    peak_freq = freqs[np.argmax(spectrum)]

    assert abs(peak_freq - freq) < 20.0  # within 20 Hz tolerance


def test_load_audio_mono_averages_channels(tmp_path):
    sr = 16000
    left = _tone(440, sr, 0.5, amplitude=0.5)
    right = np.zeros_like(left)
    stereo = np.stack([left, right], axis=1)

    wav_path = tmp_path / "stereo.wav"
    sf.write(str(wav_path), stereo, sr)

    y, out_sr = load_audio(str(wav_path), target_sr=sr, mono=True)

    assert out_sr == sr
    assert y.ndim == 1
    # mono average of left+silence should have about half the RMS of left alone
    assert abs(rms(y) - rms(left) / 2) < 1e-2


def test_load_audio_asserts_target_sample_rate(tmp_path):
    sr = 22050
    x = _tone(300, sr, 0.5)
    wav_path = tmp_path / "clip.wav"
    sf.write(str(wav_path), x, sr)

    y, out_sr = load_audio(str(wav_path), target_sr=16000, mono=True)
    assert out_sr == 16000


# ---------------------------------------------------------------------------
# trim_silence
# ---------------------------------------------------------------------------

_TOP_DB, _FRAME_LENGTH, _HOP_LENGTH = 30, 2048, 512


def _clip_with_lead_interior_trail_silence(sr: int = 16000, noise_amp: float = 1e-4):
    """A clip with leading silence, two tone segments separated by interior silence, and
    trailing silence. Silence regions carry tiny noise (not exact zeros) so this exercises
    the framed-RMS-below-top_db convention, not a trivial all-zero check."""
    rng = np.random.default_rng(0)

    def _noise(n_samples: int) -> np.ndarray:
        return (rng.normal(size=n_samples) * noise_amp).astype(np.float32)

    lead = _noise(int(0.5 * sr))
    tone1 = _tone(440, sr, 1.0, amplitude=0.8)
    interior = _noise(int(0.3 * sr))
    tone2 = _tone(600, sr, 1.0, amplitude=0.8)
    trail = _noise(int(0.4 * sr))

    x = np.concatenate([lead, tone1, interior, tone2, trail])
    lead_n, tone1_n, interior_n, tone2_n, trail_n = (
        len(lead),
        len(tone1),
        len(interior),
        len(tone2),
        len(trail),
    )
    return x, sr, (lead_n, tone1_n, interior_n, tone2_n, trail_n)


def test_trim_silence_removes_leading_and_trailing_only():
    x, sr, (lead_n, tone1_n, interior_n, tone2_n, trail_n) = (
        _clip_with_lead_interior_trail_silence()
    )
    trimmed, (start, end) = trim_silence(
        x, sr, top_db=_TOP_DB, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH,
        return_offsets=True,
    )

    # start/end land within one frame_length of the known tone boundaries
    assert abs(start - lead_n) <= _FRAME_LENGTH
    assert abs((len(x) - end) - trail_n) <= _FRAME_LENGTH

    # interior silence is fully preserved: trimmed length is close to
    # tone1 + interior + tone2 (within the leading/trailing rounding above)
    expected_len = tone1_n + interior_n + tone2_n
    assert abs(len(trimmed) - expected_len) <= 2 * _FRAME_LENGTH

    np.testing.assert_array_equal(trimmed, x[start:end])


def test_trim_silence_default_return_is_array_only():
    x, sr, _ = _clip_with_lead_interior_trail_silence()
    out = trim_silence(x, sr)
    assert isinstance(out, np.ndarray)
    assert len(out) < len(x)


def test_trim_silence_idempotent():
    x, sr, _ = _clip_with_lead_interior_trail_silence()
    once = trim_silence(x, sr)
    twice = trim_silence(once, sr)
    np.testing.assert_array_equal(once, twice)


def test_trim_silence_consistent_with_silence_ratio_measure():
    """After trimming, the shared `silence_ratio` convention (the same one
    `leakage.py`'s silence_leading/silence_trailing are computed from) reports ~0 leading
    and trailing silence, within one frame."""
    x, sr, _ = _clip_with_lead_interior_trail_silence()
    trimmed = trim_silence(
        x, sr, top_db=_TOP_DB, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH
    )

    ratio = silence_ratio(
        trimmed, top_db=_TOP_DB, frame_length=_FRAME_LENGTH, hop_length=_HOP_LENGTH
    )
    frame_fraction = _FRAME_LENGTH / len(trimmed)

    assert ratio["leading"] <= frame_fraction
    assert ratio["trailing"] <= frame_fraction


def test_trim_silence_all_silent_returns_unchanged_not_empty():
    x = np.zeros(1000, dtype=np.float32)
    out = trim_silence(x, 16000)
    np.testing.assert_array_equal(out, x)
    assert len(out) == 1000


def test_trim_silence_all_silent_low_noise_returns_unchanged():
    rng = np.random.default_rng(1)
    x = (rng.normal(size=2000) * 1e-6).astype(np.float32)
    out, (start, end) = trim_silence(x, 16000, return_offsets=True)
    np.testing.assert_array_equal(out, x)
    assert (start, end) == (0, len(x))


def test_trim_silence_empty_returns_empty():
    x = np.array([], dtype=np.float32)
    out, (start, end) = trim_silence(x, 16000, return_offsets=True)
    assert len(out) == 0
    assert (start, end) == (0, 0)


def test_trim_silence_no_silence_pure_tone_is_unchanged():
    x = _tone(440, 16000, 1.0, amplitude=0.8)
    out = trim_silence(x, 16000)
    # a tone with essentially no silence trims to (near) its full length
    assert abs(len(out) - len(x)) <= _FRAME_LENGTH


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------


def test_preprocess_raw_no_crop_is_identity_copy():
    x = _tone(440, 16000, 1.0, amplitude=0.7)
    out = preprocess(x, 16000, silence_mode="raw", crop_seconds=None)
    np.testing.assert_array_equal(out, x)
    assert out is not x  # a copy, not the same object


def test_preprocess_trimmed_no_crop_removes_silence():
    x, sr, (lead_n, tone1_n, interior_n, tone2_n, trail_n) = (
        _clip_with_lead_interior_trail_silence()
    )
    out = preprocess(x, sr, silence_mode="trimmed", crop_seconds=None)
    assert len(out) < len(x)
    expected_len = tone1_n + interior_n + tone2_n
    assert abs(len(out) - expected_len) <= 2 * _FRAME_LENGTH


def test_preprocess_invalid_silence_mode_raises():
    x = _tone(440, 16000, 1.0)
    with pytest.raises(ValueError):
        preprocess(x, 16000, silence_mode="bogus", crop_seconds=None)


def test_preprocess_crop_long_clip_center_length():
    x = _tone(440, 16000, 6.0)
    out = preprocess(x, 16000, silence_mode="raw", crop_seconds=4.0, crop_mode="center")
    assert out.shape == (64000,)


def test_preprocess_crop_long_clip_random_length_and_determinism():
    x = _tone(440, 16000, 6.0)
    out1 = preprocess(
        x, 16000, silence_mode="raw", crop_seconds=4.0, crop_mode="random",
        rng=np.random.default_rng(42),
    )
    out2 = preprocess(
        x, 16000, silence_mode="raw", crop_seconds=4.0, crop_mode="random",
        rng=np.random.default_rng(42),
    )
    assert out1.shape == (64000,)
    np.testing.assert_array_equal(out1, out2)


def test_preprocess_short_clip_is_repeat_padded_not_zero_padded():
    sr = 16000
    x = _tone(440, sr, 1.0, amplitude=0.5)  # 1 s, shorter than the 4 s target
    out = preprocess(x, sr, silence_mode="raw", crop_seconds=4.0, crop_mode="center")
    assert out.shape == (4 * sr,)

    # no injected run of exact zeros anywhere near the length center-zero-padding would add
    # (repeat-padding tiles the tone, so there is no long run of exact 0.0 samples)
    is_zero = out == 0.0
    if is_zero.any():
        # isolated exact-zero crossings of the tone are fine; a *long run* would indicate
        # zero-padding was used instead of repeat-padding
        max_run = 0
        run = 0
        for v in is_zero:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        assert max_run < 100

    # repeat-padding preserves energy density (unlike zero-padding, which dilutes it)
    assert abs(rms(out) - rms(x)) < 1e-2


def test_preprocess_short_clip_trimmed_mode_still_repeat_pads():
    sr = 16000
    x, _, _ = _clip_with_lead_interior_trail_silence(sr=sr)
    out = preprocess(x, sr, silence_mode="trimmed", crop_seconds=4.0, crop_mode="center")
    assert out.shape == (4 * sr,)


def test_preprocess_equal_length_no_crop_needed():
    sr = 16000
    x = _tone(440, sr, 4.0)
    out = preprocess(x, sr, silence_mode="raw", crop_seconds=4.0)
    assert out.shape == (4 * sr,)
    np.testing.assert_array_equal(out, x)
