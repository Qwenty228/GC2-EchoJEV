"""Stage-2 feature extraction: STFT power -> filterbank -> log -> DCT, six ablation variants.

This module implements the controlled decomposition from the original project brief, "Stage 2":

    x[n] -> STFT X[m,k] -> power P[m,k] -> filterbank E[m,r] -> log L[m,r] -> DCT c[m,q]
    E[m,r] = sum_k P[m,k] * Psi_r[k]

`configs/features.yaml` names six variants (feat_a-feat_f) that each turn a *single* pipeline on or
off at a different stage, so that a downstream backend's performance difference can be
attributed to one representation choice at a time (frequency warping, filterbank smoothing,
log compression, DCT/decorrelation, coefficient truncation, temporal derivatives) rather than
lumped into a vague "LFCC vs MFCC" comparison.

This module only ever touches arrays (`x` is already a loaded, resampled waveform — that is
`audio.py`'s job, and this module must not open files). It caches through `cache.py`'s
`cache_key`/`FeatureCache` when a caller opts in via `extract(..., cache=..., utt_id=...)`;
it does not reimplement hashing or on-disk storage itself.

## DCT normalisation — a correction to the handoff's bare formula

The handoff states `c[m,q] = sum_r L[m,r] * cos(pi*q/J*(r+1/2))`. As verified in
`Miscellaneous/research/knowledge/methods/lfcc-mfcc-dct.md` (which cites scipy's own DCT-II
documentation), this bare formula has **no leading scale factor** and is therefore *not*
orthonormal — its implied transform matrix does not satisfy `O @ O.T = I`, so a naive
`idct(dct(L))` built from it does not reconstruct `L`. Making it orthonormal (invertible,
energy-preserving in the Parseval sense) requires exactly the standard DCT-II `norm="ortho"` row
scaling:

    c[m, 0] = sqrt(1/J) * sum_r L[m, r]
    c[m, q] = sqrt(2/J) * sum_r L[m, r] * cos(pi*q/J*(r+1/2))      for q = 1 .. J-1

`dct_coeffs(..., orthonormal=True)` (the default, and the only mode every variant
feat_a-feat_f below uses) applies exactly this scaling by delegating to
`scipy.fft.dct(type=2, norm="ortho")`
along the filterbank axis — the simplest correct implementation path, per the researcher's
note, and avoids hand-deriving the constants ourselves. `orthonormal=False` returns scipy's
un-normalised DCT-II instead (`norm=None`); that mode is *not* invertible and is provided only
so a caller can inspect the difference — it is never used by `extract`.

## Log-compression floor

`log(E)` is undefined at `E=0` (which happens for e.g. digital silence, or a filter channel
with genuinely zero energy in a synthetic test signal). We floor the filterbank energy at
`_LOG_FLOOR = 1e-10` before taking the log: `L = log(maximum(E, 1e-10))`. Consequences for
very quiet frames: any frame whose true filterbank energy falls below `1e-10` is clamped to
the same floor value (`log(1e-10) ~= -23.03`) regardless of how much quieter it actually is —
the floor makes the feature insensitive to *further* differences in near-silent energy below
that level, trading a small amount of dynamic-range resolution in the quietest frames for a
finite, comparable value instead of `-inf` or `nan`.
"""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from scipy.fft import dct

from audio_df.cache import FeatureCache, cache_key

__all__ = [
    "stft_power",
    "filterbank",
    "apply_filterbank",
    "dct_coeffs",
    "add_deltas",
    "extract",
    "hz_to_mel",
    "mel_to_hz",
]

# Log-compression floor; see module docstring "Log-compression floor".
_LOG_FLOOR = 1e-10


def hz_to_mel(f: np.ndarray | float) -> np.ndarray | float:
    """HTK-style mel scale: `mel(f) = 2595 * log10(1 + f/700)`.

    We use the HTK formula (rather than Slaney's piecewise linear/log variant, which
    `librosa.mel_frequencies` uses by default) because it is the simpler, single-branch
    definition and this module implements the filterbank construction directly rather than
    delegating to `librosa.filters.mel` — the choice only affects filter edge placement, not
    any of the pipeline stages the tests or the handoff care about.
    """
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray | float:
    """Inverse of `hz_to_mel`."""
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def stft_power(x: np.ndarray, cfg: dict) -> np.ndarray:
    """Power spectrogram `P[m, k]`, `m` = frame index, `k` = frequency bin.

    Reads `cfg["stft"]`: `n_fft`, `hop_length`, `win_length`, `window`, `center` — the same
    keys as `configs/features.yaml: stft`. Delegates the STFT itself to `librosa.stft` (a
    correct, well-tested implementation; nothing about the ablation grid depends on
    reimplementing the FFT/windowing/framing arithmetic by hand). `librosa.stft` returns
    `[K, M]` (frequency-major); we transpose to `[M, K]` to match this module's convention
    that time is always axis 0, matching `extract`'s documented `[T, D]` output shape.

    `K = n_fft // 2 + 1` (one-sided real spectrum). Power discards phase entirely — only
    `|X[m,k]|**2` survives past this point (handoff, explicit).
    """
    stft_cfg = cfg["stft"]
    x = np.asarray(x, dtype=np.float32)
    x_stft = librosa.stft(
        y=x,
        n_fft=stft_cfg["n_fft"],
        hop_length=stft_cfg["hop_length"],
        win_length=stft_cfg.get("win_length"),
        window=stft_cfg.get("window", "hann"),
        center=stft_cfg.get("center", True),
    )
    power = np.abs(x_stft) ** 2
    return power.T.astype(np.float64)


def filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    spacing: str = "linear",
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Triangular filterbank weights `Psi[r, k]`, `r` in `[0, n_filters)`, `k` in `[0, K)`.

    `spacing="linear"` places `n_filters + 2` triangle edges evenly spaced in Hz between
    `fmin` and `fmax`. `spacing="mel"` places them evenly spaced in mel (`hz_to_mel`), then
    converts back to Hz — the standard frequency-warping step that gives MFCC/mel-filterbank
    features their characteristic emphasis on lower frequencies (mel spacing is denser near
    `fmin` than linear spacing is). Each filter `r` is a triangle with left/centre/right edges
    `edges[r], edges[r+1], edges[r+2]`, rising linearly from 0 at the left edge to 1 at the
    centre and falling linearly back to 0 at the right edge — the standard MFCC/LFCC
    construction (e.g. `librosa.filters.mel`), reimplemented directly here so both spacings
    share one code path and the ablation between them is exactly "which edge sequence", with
    nothing else different.

    `K = n_fft // 2 + 1` bins are placed at `np.linspace(0, sr/2, K)`, matching the one-sided
    real spectrum `stft_power` returns for the same `n_fft`.
    """
    if spacing not in ("linear", "mel"):
        raise ValueError(f"spacing must be 'linear' or 'mel', got {spacing!r}")
    if fmax is None:
        fmax = sr / 2.0

    n_bins = n_fft // 2 + 1
    bin_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    edges: np.ndarray
    if spacing == "linear":
        edges = np.linspace(fmin, fmax, n_filters + 2)
    else:
        mel_edges = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)
        edges = np.asarray(mel_to_hz(mel_edges), dtype=np.float64)

    psi = np.zeros((n_filters, n_bins), dtype=np.float64)
    for r in range(n_filters):
        f_left, f_center, f_right = edges[r], edges[r + 1], edges[r + 2]

        left_denom = f_center - f_left
        right_denom = f_right - f_center
        # Degenerate filters (zero-width edge segment) contribute nothing rather than
        # dividing by zero; this only happens for pathological configs (n_filters far
        # exceeding the available bandwidth), not any variant in configs/features.yaml.
        rising = (bin_freqs - f_left) / left_denom if left_denom > 0 else np.zeros(n_bins)
        falling = (f_right - bin_freqs) / right_denom if right_denom > 0 else np.zeros(n_bins)

        psi[r] = np.clip(np.minimum(rising, falling), 0.0, None)

    return psi


def apply_filterbank(P: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    """`E[m, r] = sum_k P[m, k] * Psi[r, k]`, i.e. `P @ Psi.T`.

    `P` is `[M, K]` (from `stft_power`), `Psi` is `[n_filters, K]` (from `filterbank`);
    result is `[M, n_filters]`. This is the filterbank-pooling stage that discards fine
    FFT-bin detail, replacing `K` bins with `n_filters << K` smoothed channel energies
    (handoff, explicit) — any spectral feature narrower than a filter's bandwidth is averaged
    away here and cannot be recovered by any later stage.
    """
    return P @ Psi.T


def dct_coeffs(L: np.ndarray, n_coeff: int | None = None, orthonormal: bool = True) -> np.ndarray:
    """DCT-II along the last axis: `c[..., q]`, optionally truncated to `n_coeff` columns.

    `orthonormal=True` (the default, and the only mode `extract` uses) applies scipy's
    `norm="ortho"` scaling — `sqrt(1/J)` for `q=0`, `sqrt(2/J)` for `q>0`, `J` = input width —
    which is the correction the handoff's bare formula omits; see the module docstring "DCT
    normalisation" section. This is the mode for which `idct(dct(L)) == L` holds exactly
    (full, untruncated case) and for which Parseval's relation
    `sum_r L[r]**2 == sum_q c[q]**2` holds exactly.

    `orthonormal=False` returns scipy's un-normalised DCT-II (`norm=None`) instead — not
    invertible by a matching orthonormal IDCT, provided only so a caller can inspect the
    (non-orthonormal) alternative. It is never what `extract` uses for variants
    feat_b/feat_c/feat_e/feat_f.

    Truncation to `n_coeff < J` (variants feat_c and feat_e) is applied *after* the full DCT is
    computed, by slicing the first `n_coeff` (lowest-frequency, in DCT-coefficient sense)
    columns — the genuinely lossy step in the DCT stage (handoff, explicit): it discards the
    components of `L`'s row that vary fastest across filterbank channels, i.e. the
    finer-grained shape of the already-smoothed log-filterbank spectrum.
    """
    norm = "ortho" if orthonormal else None
    c_full = np.asarray(dct(L, type=2, axis=-1, norm=norm))
    if n_coeff is not None:
        c_full = c_full[..., :n_coeff]
    return c_full


def add_deltas(C: np.ndarray, order: int = 2) -> np.ndarray:
    """Append `order` temporal-derivative blocks to `C` along the feature axis.

    `C` is `[T, D]` (time-major, matching every other array in this module). Derivatives are
    computed along the time axis (axis 0) with `np.gradient`, which for interior frames is a
    central difference and at the two boundary frames falls back to a one-sided difference of
    the same order of accuracy — for a signal that actually is affine in time, both the
    central and one-sided finite differences reproduce the true derivative *exactly*
    (first derivative constant, second derivative exactly zero, including at the boundary
    frames), which is what this module's delta test checks. `order=2` (the default, matching
    variant feat_f's "static + delta + delta-delta") returns `concatenate([C, delta1, delta2],
    axis=1)`, i.e. `D -> 3*D`; `order=1` returns `D -> 2*D`; `order=0` returns `C` unchanged.
    """
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    blocks = [C]
    current = C
    for _ in range(order):
        current = np.gradient(current, axis=0)
        blocks.append(current)
    return np.concatenate(blocks, axis=1)


def extract(
    x: np.ndarray,
    variant: str,
    cfg: dict,
    cache: FeatureCache | None = None,
    utt_id: str | None = None,
) -> np.ndarray:
    """Run one of `configs/features.yaml`'s variants feat_a-feat_f end to end: `[T, D]`.

    `variant` is a key of `cfg["variants"]` (e.g. `"feat_a_linear_logfb"`), and `cfg` is the full
    feature config (`stft`, `filterbank`, `variants`, and optionally `sr` — defaults to
    16000 Hz, matching `audio.py.load_audio`'s default `target_sr`, if not given). Each
    variant entry supplies `spacing` (`"linear"`|`"mel"`), `log` (bool), `dct` (bool),
    `n_coeff` (`int | None`), `deltas` (bool) — see `configs/features.yaml`.

    Pipeline (handoff, exact order): STFT power -> filterbank -> (log) -> (DCT, optionally
    truncated) -> (deltas). Steps in parentheses are switched on/off per variant so that a
    downstream performance difference can be attributed to one representation choice at a
    time, per the handoff's "critical implementation experiments" list.

    Documented `D` per variant (given `configs/features.yaml`'s `n_filters=20`):
    feat_a (`linear`, no DCT): `D=20`. feat_b (`linear`, full DCT): `D=20`. feat_c (`linear`,
    DCT trunc 13): `D=13`. feat_d (`mel`, no DCT): `D=20`. feat_e (`mel`, DCT trunc 13,
    "mfcc"): `D=13`. feat_f (`linear`, full DCT + deltas, order=2): `D=20*3=60`.

    If `cache` is given, `utt_id` must be given too. The cache key is built from
    `cache_key(utt_id, {...})` over the *entire* extraction config relevant to this
    variant's output (the variant's own params plus the shared `stft`/`filterbank` blocks),
    per `cache.py`'s contract that any config change — including one buried in a nested
    block, such as a changed `n_fft` — must invalidate the cached entry. Caching is optional
    and unused by the test suite (`features.py` computes correctly with no cache at all);
    this module never reimplements hashing or on-disk storage itself (see `cache.py`).
    """
    if variant not in cfg["variants"]:
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(cfg['variants'])}")
    variant_cfg = cfg["variants"][variant]
    sr = cfg.get("sr", 16000)

    cache_cfg: dict[str, Any] = {
        "variant": variant,
        "variant_params": variant_cfg,
        "stft": cfg["stft"],
        "filterbank": cfg["filterbank"],
        "sr": sr,
    }
    key = None
    if cache is not None:
        if utt_id is None:
            raise ValueError("utt_id is required when cache is provided")
        key = cache_key(utt_id, cache_cfg)
        cached = cache.get(key)
        if cached is not None:
            return cached

    power = stft_power(x, cfg)
    psi = filterbank(
        n_filters=cfg["filterbank"]["n_filters"],
        n_fft=cfg["stft"]["n_fft"],
        sr=sr,
        spacing=variant_cfg["spacing"],
        fmin=cfg["filterbank"].get("fmin", 0.0),
        fmax=cfg["filterbank"].get("fmax"),
    )
    energy = apply_filterbank(power, psi)

    features = np.log(np.maximum(energy, _LOG_FLOOR)) if variant_cfg.get("log", True) else energy

    if variant_cfg.get("dct", False):
        features = dct_coeffs(features, n_coeff=variant_cfg.get("n_coeff"), orthonormal=True)

    if variant_cfg.get("deltas", False):
        features = add_deltas(features, order=2)

    if cache is not None:
        assert key is not None  # set above whenever cache is not None
        cache.put(key, features, meta={"variant": variant, "utt_id": utt_id})

    return features
