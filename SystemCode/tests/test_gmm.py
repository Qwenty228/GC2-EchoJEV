"""Tests for the memory-bounded diagonal GMM (models/gmm.py) — synthetic data only."""

from __future__ import annotations

import numpy as np
import pytest

from audio_df.models.gmm import DiagGMM, score_llr


def _synthetic_mixture(rng, n_per_component, means, var):
    """Draw i.i.d. samples from a known diagonal-Gaussian mixture with equal weights."""
    parts = [rng.normal(loc=m, scale=np.sqrt(var), size=(n_per_component, len(m))) for m in means]
    return np.concatenate(parts, axis=0).astype(np.float32)


def test_recovers_known_low_dim_mixture():
    rng = np.random.default_rng(0)
    true_means = [np.array([-5.0, -5.0]), np.array([0.0, 0.0]), np.array([5.0, 5.0])]
    var = 0.25
    data = _synthetic_mixture(rng, 4000, true_means, var)

    gmm = DiagGMM(n_components=3, n_features=2, variance_floor=1e-3)
    gmm.fit_em(data, n_iter=15, chunk_frames=1000, seed=1, init_sample_size=4000)

    fitted_means = gmm.means
    assert fitted_means is not None
    # Match each true mean to its nearest fitted mean (component order from random
    # init is arbitrary) and check recovery within a small tolerance.
    for tm in true_means:
        dists = np.linalg.norm(fitted_means - tm[None, :], axis=1)
        assert dists.min() < 0.3, f"no fitted component close to true mean {tm}"


def test_chunked_em_matches_unchunked_on_small_data():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(500, 3)).astype(np.float32)

    gmm_chunked = DiagGMM(n_components=4, n_features=3)
    hist_chunked = gmm_chunked.fit_em(data, n_iter=5, chunk_frames=37, seed=7, init_sample_size=500)

    gmm_unchunked = DiagGMM(n_components=4, n_features=3)
    hist_unchunked = gmm_unchunked.fit_em(
        data, n_iter=5, chunk_frames=10_000, seed=7, init_sample_size=500
    )

    assert gmm_chunked.means is not None and gmm_unchunked.means is not None
    assert gmm_chunked.vars_ is not None and gmm_unchunked.vars_ is not None
    assert gmm_chunked.weights is not None and gmm_unchunked.weights is not None
    np.testing.assert_allclose(gmm_chunked.means, gmm_unchunked.means, atol=1e-6)
    np.testing.assert_allclose(gmm_chunked.vars_, gmm_unchunked.vars_, atol=1e-6)
    np.testing.assert_allclose(gmm_chunked.weights, gmm_unchunked.weights, atol=1e-6)
    np.testing.assert_allclose(hist_chunked, hist_unchunked, atol=1e-6)


def test_log_likelihood_is_non_decreasing():
    rng = np.random.default_rng(3)
    data = _synthetic_mixture(rng, 500, [np.array([0.0]), np.array([3.0])], 0.5)
    gmm = DiagGMM(n_components=2, n_features=1)
    history = gmm.fit_em(data, n_iter=10, chunk_frames=200, seed=2, init_sample_size=1000)
    diffs = np.diff(history)
    # EM guarantees non-decreasing likelihood up to floating point noise.
    assert (diffs > -1e-6).all(), f"log-likelihood decreased: {history}"


def test_chunking_bounds_responsibility_array_size(monkeypatch):
    rng = np.random.default_rng(5)
    data = rng.normal(size=(1000, 2)).astype(np.float32)
    gmm = DiagGMM(n_components=5, n_features=2)

    chunk_frames = 64
    seen_sizes = []
    original = DiagGMM._e_step_chunk

    def spy(self, x_chunk):
        seen_sizes.append(x_chunk.shape[0])
        assert x_chunk.shape[0] <= chunk_frames
        return original(self, x_chunk)

    monkeypatch.setattr(DiagGMM, "_e_step_chunk", spy)
    gmm.fit_em(data, n_iter=2, chunk_frames=chunk_frames, seed=1, init_sample_size=1000)

    assert all(s <= chunk_frames for s in seen_sizes)
    assert max(seen_sizes) > 5  # sanity: it actually ran on real chunks, not empties


def test_score_log_likelihood_chunked_matches_unchunked():
    rng = np.random.default_rng(9)
    data = rng.normal(size=(300, 2)).astype(np.float32)
    gmm = DiagGMM(n_components=3, n_features=2)
    gmm.fit_em(data, n_iter=3, chunk_frames=50, seed=1, init_sample_size=300)

    ll_chunked = gmm.score_log_likelihood(data, chunk_frames=17)
    ll_unchunked = gmm.score_log_likelihood(data, chunk_frames=10_000)
    np.testing.assert_allclose(ll_chunked, ll_unchunked, atol=1e-8)


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(size=(200, 2)).astype(np.float32)
    gmm = DiagGMM(n_components=3, n_features=2)
    gmm.fit_em(data, n_iter=3, chunk_frames=50, seed=1, init_sample_size=200)

    path = tmp_path / "gmm.npz"
    gmm.save(path)
    loaded = DiagGMM.load(path)

    assert loaded.means is not None and gmm.means is not None
    assert loaded.vars_ is not None and gmm.vars_ is not None
    assert loaded.weights is not None and gmm.weights is not None
    np.testing.assert_allclose(loaded.means, gmm.means)
    np.testing.assert_allclose(loaded.vars_, gmm.vars_)
    np.testing.assert_allclose(loaded.weights, gmm.weights)
    assert loaded.n_components == gmm.n_components
    assert loaded.n_features == gmm.n_features


def test_score_llr_higher_for_matching_class():
    rng = np.random.default_rng(11)
    bona_data = rng.normal(loc=0.0, scale=1.0, size=(2000, 2)).astype(np.float32)
    spoof_data = rng.normal(loc=6.0, scale=1.0, size=(2000, 2)).astype(np.float32)

    bona_gmm = DiagGMM(n_components=2, n_features=2)
    bona_gmm.fit_em(bona_data, n_iter=10, chunk_frames=500, seed=1, init_sample_size=2000)
    spoof_gmm = DiagGMM(n_components=2, n_features=2)
    spoof_gmm.fit_em(spoof_data, n_iter=10, chunk_frames=500, seed=2, init_sample_size=2000)

    # A held-out bona-fide-like utterance should score higher (more bona fide) than a
    # held-out spoof-like utterance.
    bona_utt = rng.normal(loc=0.0, scale=1.0, size=(100, 2)).astype(np.float32)
    spoof_utt = rng.normal(loc=6.0, scale=1.0, size=(100, 2)).astype(np.float32)

    score_bona = score_llr(bona_gmm, spoof_gmm, bona_utt)
    score_spoof = score_llr(bona_gmm, spoof_gmm, spoof_utt)
    assert score_bona > score_spoof


def test_init_raises_if_fewer_frames_than_components():
    gmm = DiagGMM(n_components=10, n_features=2)
    data = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        gmm.fit_em(data, n_iter=1, chunk_frames=10, seed=0, init_sample_size=3)
