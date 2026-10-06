"""Memory-bounded diagonal-covariance GMM, chunked EM, for A1 (LFCC-GMM).

`sklearn.mixture.GaussianMixture` is not used here: for the full ASVspoof 2019 LA train
partition (~7.5M spoof frames x 512 components), a `float64` full-`N x K` responsibility
matrix (as sklearn's E-step allocates) would need `7.5e6 * 512 * 8 bytes ~= 30.7 GB` — far
beyond this project's ~2 GB training-peak budget (see `CONVENTIONS.md`'s compute
budget and the A1 task brief). This module instead streams training frames from an
on-disk memmap in configurable chunks (default 50k frames): each chunk's `[chunk, K]`
responsibility matrix is `float32` and never holds more than one chunk at a time;
sufficient statistics (`N_k`, `sum_k x`, `sum_k x**2`) accumulate in `float64` across
chunks and are only combined into new means/variances once per EM iteration (the M-step).

This module does no file IO of its own beyond `save`/`load` (small `.npz` parameter
files) — reading LFCC frames into an on-disk memmap is `training/gmm.py`'s job
(per `CONVENTIONS.md`'s module boundaries: `models/` is architecture only).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["DiagGMM"]

# Variance floor: prevents a component collapsing onto a single frame (variance -> 0),
# which would make its log-likelihood diverge to +inf and destroy the whole mixture's
# log-likelihood-ratio score. 1e-3 is a project-local choice (not sourced from the
# official baseline, which uses VLFeat's `vl_gmm` with its own internal floor we did not
# independently verify — see `Miscellaneous/research/knowledge/methods/b02-lfcc-gmm.md`'s open
# question on `vl_gmm`'s exact internals) chosen to be small relative to the LFCC
# features' typical per-dimension variance (O(1-10) for log-filterbank/DCT features)
# while still being numerically safe in float32 chunked arithmetic.
DEFAULT_VARIANCE_FLOOR = 1e-3


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    # Guard against an all -inf row (should not happen with a variance floor + finite
    # inputs, but keeps this numerically total rather than producing nan on a
    # pathological chunk).
    m_safe = np.where(np.isfinite(m), m, 0.0)
    s = np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)
    return (m_safe + np.log(s)).squeeze(axis)


class DiagGMM:
    """Diagonal-covariance Gaussian mixture, fit by chunked EM.

    Parameters (`means`, `vars_`, `weights`) are `None` until `fit_em` or `load` is
    called. `n_components` (K) and `n_features` (D) are fixed at construction.
    """

    def __init__(
        self,
        n_components: int,
        n_features: int,
        variance_floor: float = DEFAULT_VARIANCE_FLOOR,
    ) -> None:
        self.n_components = n_components
        self.n_features = n_features
        self.variance_floor = variance_floor
        self.means: np.ndarray | None = None
        self.vars_: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.log_likelihood_history: list[float] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_params(self, init_sample: np.ndarray, rng: np.random.Generator) -> None:
        """Seeded random frame selection for means, global variance, equal weights.

        Matches the official baseline's documented "EM algorithm with random
        initialisation" (`Readme.txt`, quoted in `Miscellaneous/research/knowledge/methods/
        b02-lfcc-gmm.md`) as closely as a from-scratch reimplementation can, absent the
        exact internal init of VLFeat's compiled `vl_gmm` (never independently
        re-derived — flagged medium confidence in that research note). Concretely:
        means are K distinct frames drawn without replacement from `init_sample`
        (seeded); every component starts with the same global per-dimension variance of
        `init_sample` (floored); weights start uniform (`1/K`).
        """
        k = self.n_components
        n = init_sample.shape[0]
        if n < k:
            raise ValueError(
                f"init sample has only {n} frames, need >= n_components={k} to pick distinct means"
            )
        idx = rng.choice(n, size=k, replace=False)
        self.means = init_sample[idx].astype(np.float64).copy()
        global_var = np.var(init_sample.astype(np.float64), axis=0)
        global_var = np.maximum(global_var, self.variance_floor)
        self.vars_ = np.tile(global_var, (k, 1))
        self.weights = np.full(k, 1.0 / k, dtype=np.float64)

    # ------------------------------------------------------------------
    # E-step (chunked)
    # ------------------------------------------------------------------

    def _e_step_chunk(self, x_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Log-responsibilities `[B, K]` and per-frame total log-likelihood `[B]`.

        Uses matrix products (`x**2 @ (1/var).T`, etc.) rather than a per-component
        Python loop, and a chunk-local `logsumexp`, per the task brief. `x_chunk` is
        never larger than the caller's `chunk_frames`, so this never allocates a
        full-`N x K` array regardless of the total frame count.
        """
        assert self.means is not None and self.vars_ is not None and self.weights is not None
        means, vars_, weights = self.means, self.vars_, self.weights
        d = x_chunk.shape[1]

        inv_var = 1.0 / vars_  # [K, D]
        log_det = np.sum(np.log(vars_), axis=1)  # [K]
        x2 = x_chunk.astype(np.float64) ** 2  # [B, D]
        term_x2 = x2 @ inv_var.T  # [B, K]
        term_cross = x_chunk.astype(np.float64) @ (means * inv_var).T  # [B, K]
        term_mean = np.sum(means * means * inv_var, axis=1)  # [K]

        log_gauss = -0.5 * (
            d * np.log(2.0 * np.pi)
            + log_det[None, :]
            + term_x2
            - 2.0 * term_cross
            + term_mean[None, :]
        )
        log_weighted = log_gauss + np.log(weights)[None, :]
        log_norm = _logsumexp(log_weighted, axis=1)  # [B]
        log_resp = log_weighted - log_norm[:, None]  # [B, K]
        return log_resp, log_norm

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_em(
        self,
        data: np.ndarray,
        n_iter: int = 10,
        chunk_frames: int = 50_000,
        seed: int = 0,
        init_sample_size: int = 200_000,
    ) -> list[float]:
        """Fit by chunked EM over `data` (`[N, D]`, any array-like supporting slicing —
        an `np.memmap` is the intended input for the full train partition, and a plain
        `np.ndarray` works identically for tests/small data).

        Returns (and stores on `self.log_likelihood_history`) the average per-frame
        log-likelihood after each iteration, for the caller to log/verify monotonicity.
        `chunk_frames` bounds peak memory: at most one `[chunk_frames, n_components]`
        responsibility array (float64 here, float32 upstream when frames are read as
        float32 from disk) is ever materialised at once.
        """
        n = data.shape[0]
        rng = np.random.default_rng(seed)

        init_n = min(init_sample_size, n)
        init_idx = np.sort(rng.choice(n, size=init_n, replace=False))
        init_sample = np.asarray(data[init_idx], dtype=np.float64)
        self._init_params(init_sample, rng)

        k, d = self.n_components, self.n_features
        history: list[float] = []

        for _ in range(n_iter):
            n_k = np.zeros(k, dtype=np.float64)
            s_k = np.zeros((k, d), dtype=np.float64)
            s_k2 = np.zeros((k, d), dtype=np.float64)
            total_ll = 0.0

            for start in range(0, n, chunk_frames):
                chunk = np.asarray(data[start : start + chunk_frames])
                log_resp, log_norm = self._e_step_chunk(chunk)
                total_ll += float(np.sum(log_norm))
                resp = np.exp(log_resp)  # [B, K], float64
                chunk64 = chunk.astype(np.float64)
                n_k += resp.sum(axis=0)
                s_k += resp.T @ chunk64
                s_k2 += resp.T @ (chunk64 * chunk64)

            # M-step.
            n_k_safe = np.maximum(n_k, 1e-12)
            new_weights = n_k / n
            new_means = s_k / n_k_safe[:, None]
            new_vars = s_k2 / n_k_safe[:, None] - new_means * new_means
            new_vars = np.maximum(new_vars, self.variance_floor)

            self.weights = new_weights
            self.means = new_means
            self.vars_ = new_vars

            history.append(total_ll / n)

        self.log_likelihood_history = history
        return history

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_log_likelihood(self, data: np.ndarray, chunk_frames: int = 50_000) -> np.ndarray:
        """Per-frame total log-likelihood `log p(x)` for every row of `data`, chunked
        the same way as `fit_em` so scoring an arbitrarily long utterance never
        allocates a full `[N, K]` array either."""
        n = data.shape[0]
        out = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk_frames):
            chunk = np.asarray(data[start : start + chunk_frames])
            _, log_norm = self._e_step_chunk(chunk)
            out[start : start + len(chunk)] = log_norm
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        assert self.means is not None and self.vars_ is not None and self.weights is not None
        np.savez(
            path,
            means=self.means,
            vars=self.vars_,
            weights=self.weights,
            n_components=self.n_components,
            n_features=self.n_features,
            variance_floor=self.variance_floor,
            log_likelihood_history=np.asarray(self.log_likelihood_history, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path) -> DiagGMM:
        with np.load(path) as data:
            gmm = cls(
                n_components=int(data["n_components"]),
                n_features=int(data["n_features"]),
                variance_floor=float(data["variance_floor"]),
            )
            gmm.means = data["means"]
            gmm.vars_ = data["vars"]
            gmm.weights = data["weights"]
            gmm.log_likelihood_history = data["log_likelihood_history"].tolist()
        return gmm


def score_llr(
    bona_gmm: DiagGMM, spoof_gmm: DiagGMM, frames: np.ndarray, chunk_frames: int = 50_000
) -> float:
    """Mean per-frame log-likelihood-ratio (bona fide - spoof) over a whole utterance's
    frames — the official B02 CM score (`mean(llk_genuine) - mean(llk_spoof)`, per
    `Miscellaneous/research/knowledge/methods/b02-lfcc-gmm.md`). Higher = more bona fide."""
    ll_bona = bona_gmm.score_log_likelihood(frames, chunk_frames=chunk_frames)
    ll_spoof = spoof_gmm.score_log_likelihood(frames, chunk_frames=chunk_frames)
    return float(np.mean(ll_bona) - np.mean(ll_spoof))


__all__.append("score_llr")
