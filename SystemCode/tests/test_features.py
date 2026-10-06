"""Tests for `audio_df.features`. All analytic/synthetic — no dataset required.

Per `tasks/CONVENTIONS.md`, numerical code gets a property or an analytic reference, not
just a shape assertion. Every test below either checks against an independent reference
(scipy's own `norm="ortho"` DCT, an analytically-derived filterbank channel index) or a
provable property (orthonormal round-trip, monotone truncation loss, Parseval energy
preservation, constant/zero derivatives of an affine ramp).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import dct as scipy_dct

from audio_df.features import (
    add_deltas,
    apply_filterbank,
    dct_coeffs,
    extract,
    filterbank,
    hz_to_mel,
    mel_to_hz,
    stft_power,
)

SR = 16000
N_FFT = 512
HOP = 160
WIN = 400
N_FILTERS = 20

BASE_CFG = {
    "sr": SR,
    "stft": {
        "n_fft": N_FFT,
        "hop_length": HOP,
        "win_length": WIN,
        "window": "hann",
        "center": True,
    },
    "filterbank": {"n_filters": N_FILTERS, "fmin": 0, "fmax": SR // 2},
    "variants": {
        "feat_a_linear_logfb": {
            "spacing": "linear",
            "log": True,
            "dct": False,
            "n_coeff": None,
            "deltas": False,
        },
        "feat_b_lfcc_full": {
            "spacing": "linear",
            "log": True,
            "dct": True,
            "n_coeff": N_FILTERS,
            "deltas": False,
        },
        "feat_c_lfcc_trunc": {
            "spacing": "linear",
            "log": True,
            "dct": True,
            "n_coeff": 13,
            "deltas": False,
        },
        "feat_d_mel_logfb": {
            "spacing": "mel",
            "log": True,
            "dct": False,
            "n_coeff": None,
            "deltas": False,
        },
        "feat_e_mfcc": {
            "spacing": "mel",
            "log": True,
            "dct": True,
            "n_coeff": 13,
            "deltas": False,
        },
        "feat_f_lfcc_deltas": {
            "spacing": "linear",
            "log": True,
            "dct": True,
            "n_coeff": N_FILTERS,
            "deltas": True,
        },
    },
}

EXPECTED_D = {
    "feat_a_linear_logfb": N_FILTERS,
    "feat_b_lfcc_full": N_FILTERS,
    "feat_c_lfcc_trunc": 13,
    "feat_d_mel_logfb": N_FILTERS,
    "feat_e_mfcc": 13,
    "feat_f_lfcc_deltas": N_FILTERS * 3,
}


def _tone(freq: float, sr: int = SR, duration_s: float = 1.0, amplitude: float = 0.8) -> np.ndarray:
    t = np.arange(int(sr * duration_s), dtype=np.float64) / sr
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(n: int, seed: int = 0, amplitude: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


# ---------------------------------------------------------------------------
# DCT: orthonormal round-trip, matches scipy directly, Parseval energy
# ---------------------------------------------------------------------------


def test_dct_orthonormal_round_trips_via_idct():
    rng = np.random.default_rng(1)
    L = rng.standard_normal((5, N_FILTERS))
    c = dct_coeffs(L, n_coeff=None, orthonormal=True)
    # scipy's own inverse DCT-II (type=3 is the mathematical inverse of type=2) with the
    # matching norm="ortho" must reconstruct L exactly, to floating-point tolerance.
    recon = scipy_dct(c, type=3, axis=-1, norm="ortho")
    np.testing.assert_allclose(recon, L, atol=1e-10)


def test_dct_matches_scipy_ortho_directly():
    rng = np.random.default_rng(2)
    L = rng.standard_normal((7, N_FILTERS))
    ours = dct_coeffs(L, n_coeff=None, orthonormal=True)
    reference = scipy_dct(L, type=2, axis=-1, norm="ortho")
    np.testing.assert_allclose(ours, reference, atol=1e-12)


def test_dct_unnormalised_differs_from_orthonormal():
    rng = np.random.default_rng(3)
    L = rng.standard_normal((4, N_FILTERS))
    ortho = dct_coeffs(L, orthonormal=True)
    raw = dct_coeffs(L, orthonormal=False)
    reference_raw = scipy_dct(L, type=2, axis=-1, norm=None)
    np.testing.assert_allclose(raw, reference_raw, atol=1e-10)
    assert not np.allclose(ortho, raw)


def test_dct_parseval_energy_preserved():
    rng = np.random.default_rng(4)
    L = rng.standard_normal((6, N_FILTERS))
    c = dct_coeffs(L, orthonormal=True)
    energy_time = np.sum(L**2, axis=-1)
    energy_freq = np.sum(c**2, axis=-1)
    np.testing.assert_allclose(energy_freq, energy_time, rtol=1e-10)


def test_dct_truncation_loses_energy_monotonically():
    """More coefficients kept -> smaller reconstruction error (never worse)."""
    rng = np.random.default_rng(5)
    L = rng.standard_normal((3, N_FILTERS))
    c_full = dct_coeffs(L, orthonormal=True)

    errors = []
    for n_coeff in range(1, N_FILTERS + 1):
        c_trunc = np.zeros_like(c_full)
        c_trunc[..., :n_coeff] = c_full[..., :n_coeff]
        recon = scipy_dct(c_trunc, type=3, axis=-1, norm="ortho")
        errors.append(np.sum((recon - L) ** 2))

    errors = np.array(errors)
    assert errors[-1] < 1e-20  # full coefficient set: (near) exact reconstruction
    # monotone non-increasing as more coefficients are kept
    assert np.all(np.diff(errors) <= 1e-12)


# ---------------------------------------------------------------------------
# Filterbank: pure tone lands in the analytically expected channel
# ---------------------------------------------------------------------------


def _expected_channel_linear(freq: float, n_filters: int, fmin: float, fmax: float) -> int:
    edges = np.linspace(fmin, fmax, n_filters + 2)
    centers = edges[1:-1]
    return int(np.argmin(np.abs(centers - freq)))


def _expected_channel_mel(freq: float, n_filters: int, fmin: float, fmax: float) -> int:
    mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
    centers_hz = mel_to_hz(mel_edges[1:-1])
    return int(np.argmin(np.abs(centers_hz - freq)))


@pytest.mark.parametrize("spacing", ["linear", "mel"])
def test_pure_tone_lands_in_expected_filterbank_channel(spacing):
    freq = 3000.0
    x = _tone(freq, duration_s=1.0)
    cfg = {"stft": BASE_CFG["stft"]}
    P = stft_power(x, cfg)
    Psi = filterbank(N_FILTERS, N_FFT, SR, spacing=spacing, fmin=0.0, fmax=SR // 2)
    E = apply_filterbank(P, Psi)

    # average energy per channel over frames away from the edges (avoid STFT edge frames)
    mean_energy = E[2:-2].mean(axis=0)
    observed_channel = int(np.argmax(mean_energy))

    if spacing == "linear":
        expected_channel = _expected_channel_linear(freq, N_FILTERS, 0.0, SR / 2)
    else:
        expected_channel = _expected_channel_mel(freq, N_FILTERS, 0.0, SR / 2)

    assert abs(observed_channel - expected_channel) <= 1


def test_linear_and_mel_filterbanks_differ_but_cover_comparably():
    psi_linear = filterbank(N_FILTERS, N_FFT, SR, spacing="linear", fmin=0.0, fmax=SR // 2)
    psi_mel = filterbank(N_FILTERS, N_FFT, SR, spacing="mel", fmin=0.0, fmax=SR // 2)

    def centers(psi):
        bin_freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1)
        # centre-of-mass frequency per filter as a stand-in for "centre frequency"
        return (psi * bin_freqs).sum(axis=1) / np.clip(psi.sum(axis=1), 1e-12, None)

    c_lin = centers(psi_linear)
    c_mel = centers(psi_mel)
    # different centre-frequency layouts: mel is denser at low freq, so channel 1's centre
    # is materially lower for mel than for linear spacing
    assert c_mel[1] < c_lin[1]
    # comparable total coverage: both partition roughly the full band, summed filter
    # response per bin is a similar order of magnitude and neither collapses to ~0 coverage
    coverage_lin = psi_linear.sum(axis=0)
    coverage_mel = psi_mel.sum(axis=0)
    assert coverage_lin.mean() > 0.3
    assert coverage_mel.mean() > 0.3
    assert np.isclose(coverage_lin.mean(), coverage_mel.mean(), rtol=0.5)


# ---------------------------------------------------------------------------
# extract(): shapes for every variant feat_a-feat_f
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", list(BASE_CFG["variants"].keys()))
def test_extract_shape_per_variant(variant):
    x = _noise(SR * 1, seed=42)
    out = extract(x, variant, BASE_CFG)
    assert out.ndim == 2
    assert out.shape[1] == EXPECTED_D[variant]
    assert out.shape[0] > 0
    assert np.all(np.isfinite(out))


def test_extract_unknown_variant_raises():
    with pytest.raises(KeyError):
        extract(_noise(SR), "Z_not_a_variant", BASE_CFG)


# ---------------------------------------------------------------------------
# add_deltas: linear ramp -> constant first delta, ~zero second delta
# ---------------------------------------------------------------------------


def test_add_deltas_on_linear_ramp():
    t = np.arange(50, dtype=np.float64)
    slope = 2.5
    ramp = (slope * t)[:, None] * np.ones((1, 3))  # [T, D], each column an identical ramp
    out = add_deltas(ramp, order=2)
    assert out.shape == (50, 9)

    static, delta1, delta2 = out[:, :3], out[:, 3:6], out[:, 6:9]
    np.testing.assert_allclose(static, ramp)
    np.testing.assert_allclose(delta1, slope, atol=1e-10)
    np.testing.assert_allclose(delta2, 0.0, atol=1e-8)


def test_add_deltas_order_zero_and_one():
    ramp = np.arange(20, dtype=np.float64)[:, None]
    assert add_deltas(ramp, order=0).shape == (20, 1)
    out1 = add_deltas(ramp, order=1)
    assert out1.shape == (20, 2)
    np.testing.assert_allclose(out1[:, 1], 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# stft_power / apply_filterbank sanity
# ---------------------------------------------------------------------------


def test_stft_power_shape_and_nonnegative():
    x = _tone(440.0, duration_s=0.5)
    cfg = {"stft": BASE_CFG["stft"]}
    P = stft_power(x, cfg)
    expected_k = N_FFT // 2 + 1
    assert P.shape[1] == expected_k
    assert P.shape[0] > 0
    assert np.all(P >= 0.0)


def test_apply_filterbank_shape():
    x = _tone(440.0, duration_s=0.5)
    cfg = {"stft": BASE_CFG["stft"]}
    P = stft_power(x, cfg)
    Psi = filterbank(N_FILTERS, N_FFT, SR, spacing="linear", fmin=0.0, fmax=SR // 2)
    E = apply_filterbank(P, Psi)
    assert E.shape == (P.shape[0], N_FILTERS)
    assert np.all(E >= -1e-9)  # triangular weights are nonnegative, P is nonnegative
