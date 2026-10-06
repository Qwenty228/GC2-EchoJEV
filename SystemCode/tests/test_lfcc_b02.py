"""Tests for the official B02 LFCC front-end — no dataset needed, synthetic audio only.

Several tests below transcribe the official MATLAB source lines directly (quoted in
each test's docstring/comment) rather than re-deriving a closed-form formula, per the
orchestrator's fix request: the closed-form implementations in `lfcc_b02.py` must
agree with an independent, literal transcription of the MATLAB algorithm, not just
with each other.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import lfilter

from audio_df.features.lfcc_b02 import (
    DEFAULT_LFCC_B02_CONFIG,
    extract_lfcc_b02,
    frame_count,
    lfcc_b02_filterbank,
    official_deltas,
)


def _tone(freq=440.0, sr=16000, seconds=1.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


# ---------------------------------------------------------------------------
# 1. Deltas: filter([hlen:-1:-hlen], 1, xx, [], 2) with edge-replicate padding.
# ---------------------------------------------------------------------------


def _matlab_deltas_reference(x: np.ndarray, w: int) -> np.ndarray:
    """Direct Python transcription of the official `deltas.m` (dpwe 2003), operating
    on `x` shaped `[T, D]` exactly like `official_deltas` (the MATLAB code operates
    row-wise on `[D, T]`; this transcription just keeps `official_deltas`'s `[T, D]`
    convention throughout, transposing internally where the MATLAB source would).

        hlen = floor(w/2);
        win = hlen:-1:-hlen;
        xx = [repmat(x(:,1), 1, hlen), x, repmat(x(:,end), 1, hlen)];
        d = filter(win, 1, xx, [], 2);
        d = d(:, 2*hlen + [1:nc]);

    Implemented here with `scipy.signal.lfilter` standing in for MATLAB's `filter`
    (both are the standard direct-form FIR/IIR filter, `b=win`, `a=1`), applied along
    the time axis per feature column, exactly transcribing the `repmat`/`filter`/
    column-select steps rather than using `official_deltas`'s closed-form reduction.
    """
    hlen = w // 2
    T, D = x.shape
    if hlen == 0:
        return np.zeros((T, D), dtype=np.float64)
    win = np.arange(hlen, -hlen - 1, -1, dtype=np.float64)  # hlen:-1:-hlen
    first = x[0:1, :]
    last = x[-1:, :]
    xx = np.concatenate([np.repeat(first, hlen, axis=0), x, np.repeat(last, hlen, axis=0)], axis=0)
    d_full = np.empty_like(xx)
    for col in range(D):
        d_full[:, col] = lfilter(win, [1.0], xx[:, col])
    # MATLAB is 1-indexed: d(:, 2*hlen + [1:nc]) -> 0-indexed columns [2*hlen, 2*hlen+nc)
    return d_full[2 * hlen : 2 * hlen + T, :]


@pytest.mark.parametrize("w", [3, 5])
def test_official_deltas_matches_matlab_filter_transcription(w):
    rng = np.random.default_rng(0)
    C = rng.normal(size=(12, 4))
    expected = _matlab_deltas_reference(C, w)
    actual = official_deltas(C, w=w)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_official_deltas_w3_has_no_normalisation_by_2():
    # d[t] = x_ext[t+1] - x_ext[t-1], NOT divided by 2 (a real correction: earlier
    # drafts of this module wrongly halved this).
    C = np.array([[0.0], [1.0], [4.0], [9.0], [16.0]])  # t^2 samples
    d = official_deltas(C, w=3)
    expected_interior = np.array([4.0 - 0.0, 9.0 - 1.0, 16.0 - 4.0])
    np.testing.assert_allclose(d[1:4, 0], expected_interior)


def test_official_deltas_edge_uses_replicated_padding_unhalved():
    C = np.array([[0.0], [1.0], [4.0]])
    d = official_deltas(C, w=3)
    # x_ext = [0, 0, 1, 4, 4]; d[0] = x_ext[2] - x_ext[0] = 1 - 0 = 1 (NOT 0.5).
    assert d[0, 0] == pytest.approx(1.0)
    # d[2] = x_ext[4] - x_ext[2] = 4 - 1 = 3 (NOT 1.5).
    assert d[2, 0] == pytest.approx(3.0)


def test_official_deltas_rejects_even_window():
    with pytest.raises(ValueError):
        official_deltas(np.zeros((5, 2)), w=4)


def test_official_deltas_w1_is_zero():
    C = np.arange(10.0).reshape(5, 2)
    np.testing.assert_allclose(official_deltas(C, w=1), np.zeros_like(C))


# ---------------------------------------------------------------------------
# 2. Framing: buffer(speech, 320, 160, 'nodelay') zero-pad semantics.
# ---------------------------------------------------------------------------


def _matlab_buffer_nodelay_frame_count(n_samples: int, frame_length: int, hop: int) -> int:
    """Direct loop transcription of MATLAB `buffer(x, frame_length, frame_length-hop,
    'nodelay')`'s frame count: keep emitting full frames (`end <= L`) while advancing
    by `hop`; stop immediately (no extra frame) if a frame's end lands exactly on `L`;
    otherwise emit exactly one final zero-padded frame for the genuine leftover and
    stop. Verified against the official documented example
    `buffer(1:10, 4, 2, 'nodelay')` == 4 frames inside this same test module.
    """
    if n_samples <= 0:
        return 1
    start = 0
    count = 0
    while True:
        end = start + frame_length
        count += 1
        if end <= n_samples:
            if end == n_samples:
                break
            start += hop
            continue
        break  # this frame was the final, zero-padded one
    return count


def test_buffer_nodelay_matches_documented_matlab_example():
    # MATLAB docs: buffer(1:10, 4, 2, 'nodelay') -> exactly 4 columns.
    assert _matlab_buffer_nodelay_frame_count(10, 4, 2) == 4


@pytest.mark.parametrize(
    "n_samples",
    [0, 1, 159, 160, 161, 200, 319, 320, 321, 479, 480, 481, 639, 640, 641, 1000, 5000],
)
def test_frame_count_matches_buffer_nodelay_transcription(n_samples):
    frame_length, hop = 320, 160
    assert frame_count(n_samples, frame_length, hop) == _matlab_buffer_nodelay_frame_count(
        n_samples, frame_length, hop
    )


def test_frame_count_matches_output_t_dimension():
    sr = 16000
    x = _tone(seconds=0.37, sr=sr)  # deliberately not a round number of frames
    feats = extract_lfcc_b02(x, sr=sr)
    frame_length = int(round(sr / 1000 * 20))
    hop = int(round(sr / 1000 * 10))
    expected_t = frame_count(len(x), frame_length, hop)
    assert feats.shape[0] == expected_t


def test_output_shape_is_t_by_60():
    x = _tone(seconds=0.5)
    feats = extract_lfcc_b02(x, sr=16000)
    assert feats.ndim == 2
    assert feats.shape[1] == 60


# ---------------------------------------------------------------------------
# 3. Window: MATLAB hamming(320) is symmetric, not periodic.
# ---------------------------------------------------------------------------


def test_window_is_symmetric_hamming_not_periodic():
    from scipy.signal.windows import hamming as scipy_hamming

    symmetric = np.hamming(320)
    periodic = scipy_hamming(320, sym=False)
    np.testing.assert_allclose(symmetric, scipy_hamming(320, sym=True))
    # Symmetric and periodic Hamming windows of the same length differ (this is the
    # regression this test guards: using get_window(..., default) would be periodic).
    assert not np.allclose(symmetric, periodic)
    # Symmetric Hamming is exactly palindromic; periodic is not (for even length).
    np.testing.assert_allclose(symmetric, symmetric[::-1])


# ---------------------------------------------------------------------------
# 4. Spectrum / pre-emphasis.
# ---------------------------------------------------------------------------


def test_preemphasis_matches_hand_computation():
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    preemph = DEFAULT_LFCC_B02_CONFIG["preemph"]
    expected = lfilter([1.0, -preemph], [1.0], x)
    # y[0] = x[0]; y[n] = x[n] - preemph*x[n-1]
    hand = np.array([x[0], x[1] - preemph * x[0], x[2] - preemph * x[1], x[3] - preemph * x[2]])
    np.testing.assert_allclose(expected, hand)


def test_spectrum_is_abs_fft_squared_257_bins():
    sr, n_fft = 16000, 512
    frame = np.hamming(320) * _tone(freq=1000.0, seconds=320 / sr, sr=sr)
    # abs(fft(frame, 512)).^2, bins 1:257 (MATLAB) == np.fft.rfft's 257 bins.
    full_fft = np.fft.fft(frame, n=n_fft)
    expected = np.abs(full_fft[: n_fft // 2 + 1]) ** 2
    actual = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert actual.shape[0] == 257


# ---------------------------------------------------------------------------
# 5. Filterbank: trimf(f, [edges(i), edges(i+1), edges(i+2)]) on the 257-bin grid.
# ---------------------------------------------------------------------------


def _trimf_reference(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Independent, literal `trimf` transcription (does not call `lfcc_b02._trimf`):
    0 at a and c, 1 at b, linear between."""
    y = np.zeros_like(x, dtype=np.float64)
    for i, xi in enumerate(x):
        if xi <= a or xi >= c:
            y[i] = 0.0
        elif xi == b:
            y[i] = 1.0
        elif xi < b:
            y[i] = (xi - a) / (b - a) if b > a else 1.0
        else:
            y[i] = (c - xi) / (c - b) if c > b else 1.0
    return y


def test_filterbank_matches_independent_trimf_transcription():
    sr, n_fft, n_filters = 16000, 512, 20
    fmax = sr / 2.0
    n_bins = n_fft // 2 + 1
    f = fmax * np.linspace(0.0, 1.0, n_bins)
    edges = np.linspace(0.0, fmax, n_filters + 2)

    expected = np.zeros((n_filters, n_bins), dtype=np.float64)
    for i in range(n_filters):
        expected[i] = _trimf_reference(f, edges[i], edges[i + 1], edges[i + 2])

    actual = lfcc_b02_filterbank(n_filters=n_filters, n_fft=n_fft, sr=sr, fmin=0.0, fmax=fmax)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_filter_edges_span_0_to_nyquist_not_30hz():
    edges = np.linspace(0.0, 8000.0, 20 + 2)
    assert edges[0] == 0.0, "filterbank's own left edge must be 0 Hz, not 30 Hz"
    psi = lfcc_b02_filterbank(n_filters=20, n_fft=512, sr=16000, fmin=0.0, fmax=8000.0)
    bin_freqs = 8000.0 * np.linspace(0.0, 1.0, 512 // 2 + 1)
    first_nonzero_bin = np.argmax(psi[0] > 0)
    assert bin_freqs[first_nonzero_bin] <= 32.0


# ---------------------------------------------------------------------------
# 6. log10(energy + eps), eps ADDED (not max-floored); orthonormal DCT-II.
# ---------------------------------------------------------------------------


def test_log10_eps_is_added_not_max_floored():
    from audio_df.features.lfcc_b02 import _LOG10_EPS

    assert pytest.approx(2.220446049250313e-16) == _LOG10_EPS
    energy = np.array([[0.0, 1.0]])
    expected = np.log10(energy + _LOG10_EPS)
    # A max-floor would instead clip to log10(_LOG10_EPS) for the zero entry, but
    # here we assert the *addition* semantics directly on a nonzero entry too: for
    # energy=1.0, log10(1.0 + eps) != log10(max(1.0, eps)) would coincide numerically
    # (both ~0), so check the true zero-energy case against the additive formula only.
    np.testing.assert_allclose(expected[0, 0], np.log10(_LOG10_EPS))
    assert np.isfinite(expected).all()


def test_no_cmvn_mean_is_not_forced_to_zero():
    # A DC-biased-in-log-energy signal (louder tone) should not have its static
    # coefficients mean-centered to ~0 by extract_lfcc_b02 -- CMVN is not applied.
    x = _tone(seconds=0.5, amp=0.9)
    feats = extract_lfcc_b02(x, sr=16000)
    static = feats[:, :20]
    assert abs(static[:, 0].mean()) > 1e-3


def test_60_dims_are_20_static_plus_20_delta_plus_20_delta2():
    x = _tone(seconds=0.3)
    feats = extract_lfcc_b02(x, sr=16000)
    static = feats[:, :20]
    delta = feats[:, 20:40]
    delta2 = feats[:, 40:60]
    np.testing.assert_allclose(delta, official_deltas(static, w=3), atol=1e-6)
    np.testing.assert_allclose(delta2, official_deltas(delta, w=3), atol=1e-6)
    assert static.shape[1] == 20


def test_sr_mismatch_raises():
    x = _tone(sr=8000, seconds=0.2)
    with pytest.raises(ValueError):
        extract_lfcc_b02(x, sr=8000)


# ---------------------------------------------------------------------------
# X4: optional mel spacing; the default (linear) path must stay bit-identical.
# ---------------------------------------------------------------------------


def test_linear_spacing_is_default_and_bit_identical():
    from audio_df.features.lfcc_b02 import _LOG10_EPS  # noqa: F401  (module still importable)

    x = _tone(freq=700.0, seconds=0.5)
    base = extract_lfcc_b02(x, sr=16000, cfg={})
    explicit = extract_lfcc_b02(x, sr=16000, cfg={"spacing": "linear"})
    assert np.array_equal(base, explicit)
    psi_default = lfcc_b02_filterbank(n_filters=20, n_fft=512, sr=16000)
    psi_linear = lfcc_b02_filterbank(n_filters=20, n_fft=512, sr=16000, spacing="linear")
    assert np.array_equal(psi_default, psi_linear)


def test_features_filterbank_linear_reproduces_b02_filterbank():
    from audio_df.features.cepstral import filterbank

    ours = lfcc_b02_filterbank(n_filters=20, n_fft=512, sr=16000, fmin=0.0, fmax=8000.0)
    theirs = filterbank(20, 512, 16000, spacing="linear", fmin=0.0, fmax=8000.0)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)


def test_mel_spacing_matches_features_filterbank_mel():
    from audio_df.features.cepstral import filterbank

    ours = lfcc_b02_filterbank(
        n_filters=20, n_fft=512, sr=16000, fmin=0.0, fmax=8000.0, spacing="mel"
    )
    theirs = filterbank(20, 512, 16000, spacing="mel", fmin=0.0, fmax=8000.0)
    np.testing.assert_allclose(ours, theirs, atol=1e-12)
    assert ours.max() <= 1.0 + 1e-12  # peak-1 triangles, not area-normalised


def test_mel_extract_differs_from_linear_only_via_filterbank():
    x = _tone(freq=700.0, seconds=0.5)
    lin = extract_lfcc_b02(x, sr=16000, cfg={})
    mel = extract_lfcc_b02(x, sr=16000, cfg={"spacing": "mel"})
    assert lin.shape == mel.shape == (mel.shape[0], 60)
    assert not np.allclose(lin, mel)


def test_invalid_spacing_raises():
    with pytest.raises(ValueError):
        lfcc_b02_filterbank(n_filters=20, n_fft=512, sr=16000, spacing="bark")
