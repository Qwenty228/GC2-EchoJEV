"""Tests for audio_df.degrade — synthetic tones/noise only, no dataset needed."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from audio_df.degrade import (
    add_noise,
    clip_seed,
    degrade,
    mp3,
    opus,
    resample_roundtrip,
)

FFMPEG_MISSING = shutil.which("ffmpeg") is None

SR = 16000


def _tone(
    freq: float = 440.0, duration_s: float = 1.0, sr: int = SR, amp: float = 0.3
) -> np.ndarray:
    t = np.arange(int(duration_s * sr), dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dominant_freq(x: np.ndarray, sr: int = SR) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    return float(freqs[np.argmax(spec)])


def _band_energy(x: np.ndarray, sr: int, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(spec[mask].sum())


# ---------------------------------------------------------------------------
# add_noise
# ---------------------------------------------------------------------------


def test_add_noise_output_length_matches_input():
    x = _tone(duration_s=0.5)
    noise = np.random.default_rng(0).normal(0, 1, size=len(x) * 3).astype(np.float32)
    out = add_noise(x, noise, snr_db=10, rng=np.random.default_rng(1))
    assert len(out) == len(x)


def test_add_noise_measured_snr_within_tolerance():
    rng_noise = np.random.default_rng(42)
    x = _tone(freq=1000, duration_s=1.0, amp=0.5)
    noise = rng_noise.normal(0, 1, size=len(x)).astype(np.float32)

    target_snr = 10.0
    out = add_noise(x, noise, snr_db=target_snr, rng=np.random.default_rng(7))
    added_noise = out - x

    p_signal = float(np.mean(x.astype(np.float64) ** 2))
    p_added_noise = float(np.mean(added_noise.astype(np.float64) ** 2))
    measured_snr = 10.0 * np.log10(p_signal / p_added_noise)
    assert abs(measured_snr - target_snr) < 0.1


def test_add_noise_deterministic_same_seed_same_clip():
    x = _tone(duration_s=0.5)
    noise = np.random.default_rng(0).normal(0, 1, size=len(x) * 4).astype(np.float32)
    out1 = add_noise(x, noise, snr_db=10, rng=np.random.default_rng(123))
    out2 = add_noise(x, noise, snr_db=10, rng=np.random.default_rng(123))
    np.testing.assert_array_equal(out1, out2)


def test_add_noise_different_rng_different_offset_different_output():
    x = _tone(duration_s=0.5)
    noise = np.random.default_rng(0).normal(0, 1, size=len(x) * 4).astype(np.float32)
    out1 = add_noise(x, noise, snr_db=10, rng=np.random.default_rng(1))
    out2 = add_noise(x, noise, snr_db=10, rng=np.random.default_rng(2))
    assert not np.array_equal(out1, out2)


def test_add_noise_loops_short_noise():
    x = _tone(duration_s=1.0)
    short_noise = np.random.default_rng(0).normal(0, 1, size=len(x) // 4).astype(np.float32)
    out = add_noise(x, short_noise, snr_db=10, rng=np.random.default_rng(3))
    assert len(out) == len(x)
    assert not np.allclose(out, x)


def test_add_noise_silent_input_returns_unchanged():
    x = np.zeros(SR, dtype=np.float32)
    noise = np.random.default_rng(0).normal(0, 1, size=SR).astype(np.float32)
    out = add_noise(x, noise, snr_db=10, rng=np.random.default_rng(4))
    np.testing.assert_array_equal(out, x)


def test_add_noise_silent_noise_returns_unchanged():
    x = _tone(duration_s=0.5)
    noise = np.zeros(len(x), dtype=np.float32)
    out = add_noise(x, noise, snr_db=10, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(out, x)


# ---------------------------------------------------------------------------
# resample_roundtrip
# ---------------------------------------------------------------------------


def test_resample_roundtrip_length_matches_input():
    x = _tone(duration_s=0.7)
    out = resample_roundtrip(x, sr=SR, mid_sr=8000)
    assert len(out) == len(x)


def test_resample_roundtrip_removes_content_above_nyquist_of_mid_sr():
    # A 6 kHz tone is above 8 kHz's Nyquist (4 kHz) and should be heavily attenuated by the
    # 16 -> 8 -> 16 kHz round trip's anti-aliasing filter.
    x = _tone(freq=6000, duration_s=0.5, amp=0.5)
    out = resample_roundtrip(x, sr=SR, mid_sr=8000)
    energy_in = _band_energy(x, SR, 5500, 6500)
    energy_out = _band_energy(out, SR, 5500, 6500)
    assert energy_out < 0.05 * energy_in


def test_resample_roundtrip_preserves_low_frequency_content():
    x = _tone(freq=440, duration_s=0.5, amp=0.5)
    out = resample_roundtrip(x, sr=SR, mid_sr=8000)
    assert abs(_dominant_freq(out) - 440.0) < 20.0


# ---------------------------------------------------------------------------
# clip_seed / degrade dispatch
# ---------------------------------------------------------------------------


def test_clip_seed_deterministic_and_id_sensitive():
    a = clip_seed(1337, "LA_E_0000001", "eval_noise_10db")
    b = clip_seed(1337, "LA_E_0000001", "eval_noise_10db")
    c = clip_seed(1337, "LA_E_0000002", "eval_noise_10db")
    d = clip_seed(1337, "LA_E_0000001", "eval_mp3_32k")
    assert a == b
    assert a != c
    assert a != d
    assert isinstance(a, int) and a >= 0


def _cfg():
    return {
        "conditions": {
            "eval_mp3_32k": {"type": "mp3", "bitrate_kbps": 32},
            "eval_opus_16k": {"type": "opus", "bitrate_kbps": 16},
            "eval_noise_10db": {"type": "noise", "snr_db": 10},
            "eval_resample_8k": {"type": "resample", "mid_sr": 8000},
        }
    }


def test_degrade_resample_matches_direct_call():
    x = _tone(duration_s=0.5)
    out = degrade(x, SR, "eval_resample_8k", _cfg(), seed=1337, clip_id="LA_E_1")
    assert len(out) == len(x)


def test_degrade_noise_uses_noise_bank_and_is_deterministic():
    x = _tone(duration_s=0.5)
    bank = [
        np.random.default_rng(i).normal(0, 1, size=len(x) * 2).astype(np.float32)
        for i in range(3)
    ]
    out1 = degrade(x, SR, "eval_noise_10db", _cfg(), seed=1337, clip_id="LA_E_1", noise_bank=bank)
    out2 = degrade(x, SR, "eval_noise_10db", _cfg(), seed=1337, clip_id="LA_E_1", noise_bank=bank)
    out3 = degrade(x, SR, "eval_noise_10db", _cfg(), seed=1337, clip_id="LA_E_2", noise_bank=bank)
    np.testing.assert_array_equal(out1, out2)
    assert not np.array_equal(out1, out3)
    assert len(out1) == len(x)


def test_degrade_noise_without_bank_raises():
    x = _tone(duration_s=0.2)
    with pytest.raises(ValueError, match="noise_bank"):
        degrade(x, SR, "eval_noise_10db", _cfg(), seed=1337, clip_id="LA_E_1")


def test_degrade_unknown_condition_raises():
    x = _tone(duration_s=0.2)
    with pytest.raises(KeyError):
        degrade(x, SR, "does_not_exist", _cfg(), seed=1337, clip_id="LA_E_1")


# ---------------------------------------------------------------------------
# Codec tests — skipped cleanly if ffmpeg is missing.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")
def test_mp3_output_length_matches_input():
    x = _tone(duration_s=0.5)
    out = mp3(x, sr=SR, bitrate_kbps=32)
    assert len(out) == len(x)


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")
def test_mp3_differs_from_input_but_keeps_tone_frequency():
    x = _tone(freq=440, duration_s=1.0, amp=0.5)
    out = mp3(x, sr=SR, bitrate_kbps=32)
    assert not np.allclose(out, x, atol=1e-4)
    assert abs(_dominant_freq(out) - 440.0) < 20.0


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")
def test_mp3_deterministic():
    x = _tone(duration_s=0.5)
    out1 = mp3(x, sr=SR, bitrate_kbps=32)
    out2 = mp3(x, sr=SR, bitrate_kbps=32)
    np.testing.assert_array_equal(out1, out2)


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")
def test_opus_output_length_matches_input():
    x = _tone(duration_s=0.5)
    out = opus(x, sr=SR, bitrate_kbps=16)
    assert len(out) == len(x)


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")
def test_opus_differs_from_input_but_keeps_tone_frequency():
    x = _tone(freq=440, duration_s=1.0, amp=0.5)
    out = opus(x, sr=SR, bitrate_kbps=16)
    assert not np.allclose(out, x, atol=1e-4)
    assert abs(_dominant_freq(out) - 440.0) < 20.0


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")
def test_opus_deterministic():
    x = _tone(duration_s=0.5)
    out1 = opus(x, sr=SR, bitrate_kbps=16)
    out2 = opus(x, sr=SR, bitrate_kbps=16)
    np.testing.assert_array_equal(out1, out2)


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")
def test_degrade_mp3_via_dispatch():
    x = _tone(duration_s=0.3)
    out = degrade(x, SR, "eval_mp3_32k", _cfg(), seed=1337, clip_id="LA_E_1")
    assert len(out) == len(x)


@pytest.mark.skipif(FFMPEG_MISSING, reason="ffmpeg not installed")
def test_degrade_opus_via_dispatch():
    x = _tone(duration_s=0.3)
    out = degrade(x, SR, "eval_opus_16k", _cfg(), seed=1337, clip_id="LA_E_1")
    assert len(out) == len(x)
