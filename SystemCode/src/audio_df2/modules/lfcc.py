"""LFCC front-end for A1, reproducing the official ASVspoof 2019 B02 baseline exactly.

Every parameter and formula below is taken directly from the official baseline code
(`ASVspoof_2019_baseline_CM_v1.zip: LFCC/extract_lfcc.m`, `LFCC/deltas.m`), read
directly (not `features.py`'s general STFT->filterbank->log->DCT grid), because
several official choices differ from that module's primitives:

  - `features.stft_power` defaults to a Hann window and librosa's *centered* framing
    (reflect-padded at both ends); the official code uses a **Hamming** window and
    MATLAB `buffer(..., 'nodelay')` framing, which starts the first frame at sample 0
    with no reflect padding and zero-pads only the final, otherwise-incomplete frame.
    Framing is therefore reimplemented here by hand (`_frame_signal`) rather than
    delegating to `stft_power`.
  - `features.py`'s log stage is natural log (`np.log`); the official code uses
    `log10`. `extract_lfcc_b02` uses `log10`, matching `t = dct(log10(filbanksum'+eps))`.
  - `features.add_deltas` uses `np.gradient` (central differences with one-sided,
    unhalved boundaries); the official `deltas.m` implements a different, specific FIR
    filter (see `official_deltas` below) — edge-replicated padding, and, for `w=3`, NO
    division by 2. See `official_deltas`'s docstring for the exact transcription.
  - `features.filterbank` is **not used here** even though its linear-spacing/0-Hz-edge
    construction is very close to the official one: this module reimplements the
    filterbank directly (`_trimf_filterbank`) as a literal transcription of
    `extract_lfcc.m`'s `trimf`/`linspace` calls, so this module never has to assume
    another module's primitive matches bit-for-bit (`tests/test_lfcc_b02.py` checks
    both against an independent third transcription at `atol=1e-12`).

All parameter values are sourced from a direct reading of the official baseline code
(`extract_lfcc.m`, `deltas.m`) and `Miscellaneous/research/knowledge/methods/b02-lfcc-gmm.md`, which
additionally corrects an earlier, unsourced belief that the filterbank's lower edge was
30 Hz — the code's own lower edge is 0 Hz (DC), which is what this module uses.

Pipeline, exactly as `extract_lfcc.m`:
  1. pre-emphasis, coefficient 0.97, unconditional, over the WHOLE signal before
     framing: `speech = filter([1 -0.97], 1, speech);` i.e.
     `y[n] = x[n] - 0.97*x[n-1]`, `y[0] = x[0]`. (No dithering: `extract_lfcc.m` has a
     dither line, but it is commented out in the released baseline.)
  2. framing: `w = hamming(frame_length_inSample)` — MATLAB's **symmetric** Hamming
     window (`np.hamming`, not `scipy.signal.get_window("hamming", N)`, which is
     periodic by default) — 320 samples (20 ms @ 16 kHz), 10 ms hop (50% overlap, 160
     samples), via `buffer(speech, frame_length_inSample,
     frame_length_inSample/2, 'nodelay')` semantics: first frame starts at sample 0;
     the final incomplete frame is zero-padded, never dropped.
  3. `abs(fft(frame, 512)).^2`, one-sided power spectrum, bins 1:257 (MATLAB
     1-indexed) == `np.fft.rfft(..., n=512)`'s 257 bins (0-indexed 0..256).
  4. `f = (Fs/2)*linspace(0,1,NFFT/2+1)`; `edges = linspace(min(f), max(f),
     No_Filter+2)`; filter `i` is `trimf(f, [edges(i), edges(i+1), edges(i+2)])` — 20
     linear-frequency triangular filters over 0-8000 Hz (`_trimf_filterbank`).
  5. `log10(filbanksum + eps)`, `eps = 2.220446049250313e-16` (float64 machine
     epsilon) **added**, not a max-floor.
  6. `t = dct(log10(...))`, MATLAB's default `dct()` == orthonormal DCT-II
     (`scipy.fft.dct(type=2, norm="ortho")`), taking all `No_Filter=20` outputs (the
     "0th cepstral coefficient" plus 19 more) as the 20 static coefficients.
  7. delta and delta-delta via `official_deltas(..., w=3)` twice
     (`deltas(stat', 3)` then `deltas(delta', 3)`).
  8. concatenate static + delta + delta-delta -> `[T, 60]`.

No CMVN anywhere (confirmed absent in the official code). Whole utterances only: this
module never crops or pads (that is `audio.py`'s job, and A1 explicitly does not use it
for LFCC-GMM, per the official baseline and `configs/a1_lfcc_gmm.yaml: preprocessing.
crop_seconds: null`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.fft import dct
from scipy.signal import lfilter

__all__ = [
    "extract_lfcc_b02",
    "official_deltas",
    "frame_count",
    "lfcc_b02_filterbank",
    "DEFAULT_LFCC_B02_CONFIG",
]

# Defaults, exactly as `ASVspoof2019_baseline_CM.m`'s call
# `extract_lfcc(x, fs, 20, 512, 20)` and `deltas(stat', 3)`.
DEFAULT_LFCC_B02_CONFIG: dict[str, Any] = {
    "sample_rate": 16000,
    "window_ms": 20,
    "hop_ms": 10,
    "n_fft": 512,
    "n_filters": 20,
    "preemph": 0.97,
    "delta_window": 3,
    "fmin": 0.0,
    "fmax": None,  # None -> sample_rate / 2, matching `Fs/2` in the official code
    # "linear" = official B02 (LFCC). "mel" = X4 MFCC arm: only the filter edge placement
    # changes (mel-spaced edges, same peak-1 trimf triangles, same everything else).
    "spacing": "linear",
}

_LOG10_EPS = np.finfo(np.float64).eps  # 2.220446049250313e-16, matches MATLAB's `eps`
# in `log10(filbanksum'+eps)` -- ADDED to the filterbank energy, not a max-floor.


def frame_count(n_samples: int, frame_length: int, hop: int) -> int:
    """Number of frames MATLAB `buffer(x, frame_length, frame_length-hop, 'nodelay')`
    produces: the first frame starts at sample 0, frames advance by `hop`, and the
    final, otherwise-incomplete frame is zero-padded rather than dropped -- but no
    *extra* all-zero frame is appended once a frame's end lands exactly on
    `n_samples` (verified against a direct loop-based transcription of `buffer`'s
    nodelay/zero-pad semantics in `tests/test_lfcc_b02.py`, including the boundary
    cases `n_samples <= frame_length` and `n_samples` an exact multiple of `hop`
    above `frame_length`).

    `num_frames = 1 + ceil(max(0, n_samples - frame_length) / hop)`.
    """
    if n_samples <= 0:
        return 1
    return 1 + int(np.ceil(max(0, n_samples - frame_length) / hop))


def _frame_signal(y: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    """MATLAB `buffer(y, frame_length, frame_length-hop, 'nodelay')`-equivalent framing.

    Returns `[num_frames, frame_length]`. Zero-pads `y` at the end just enough so every
    frame is full length; never reflect-pads and never centers (unlike
    `features.stft_power`'s librosa-backed framing).
    """
    n = len(y)
    num_frames = frame_count(n, frame_length, hop)
    needed = (num_frames - 1) * hop + frame_length
    if needed > n:
        y = np.pad(y, (0, needed - n), mode="constant", constant_values=0.0)
    frames = np.empty((num_frames, frame_length), dtype=np.float64)
    for i in range(num_frames):
        start = i * hop
        frames[i] = y[start : start + frame_length]
    return frames


def official_deltas(C: np.ndarray, w: int = 3) -> np.ndarray:
    """Exact transcription of the official `deltas.m` (dpwe 2003), called as
    `deltas(stat', 3)` and `deltas(delta', 3)`:

        hlen = floor(w/2);
        win = hlen:-1:-hlen;
        xx = [repmat(x(:,1), 1, hlen), x, repmat(x(:,end), 1, hlen)];   % edge-replicate
        d = filter(win, 1, xx, [], 2);
        d = d(:, 2*hlen + [1:nc]);

    For `w=3` this gives `hlen=1`, `win=[1, 0, -1]`, and (worked out below) reduces to

        d[t] = x_ext[t+1] - x_ext[t-1]

    with edge-replicated `x_ext` and, critically, **no division by 2 or by
    `sum(k**2)`** — earlier drafts of this module (and the un-quoted paraphrase in
    `Miscellaneous/research/knowledge/methods/b02-lfcc-gmm.md`, "window width 3") wrongly assumed a
    Furui-style *normalised* regression delta; the actual MATLAB FIR filter has no
    such normalisation.

    General odd `w` (`hlen = w // 2`), derived directly from the FIR-filter algebra
    above (`win(m+1) = hlen - m` for `m = 0..2*hlen`, substituting `k = hlen - m` and
    reading off the selected output columns `d(:, 2*hlen+1 : 2*hlen+nc)`):

        d[t] = sum_{k=1}^{hlen} k * (x_ext[t+k] - x_ext[t-k])

    where `x_ext` replicates `C`'s first/last row `hlen` times on each side
    (`np.pad(..., mode="edge")`). `w=1` (`hlen=0`) yields an all-zero delta (the FIR
    filter `win=[0]` is the zero filter), matching the MATLAB formula directly.

    `w=3` and `w=5` are both verified in `tests/test_lfcc_b02.py` against an
    independent, direct Python transcription of the `filter`/`repmat`/column-select
    steps quoted above (not just this closed-form reduction), at tight tolerance.
    """
    if w % 2 == 0 or w < 1:
        raise ValueError(f"w must be a positive odd integer, got {w}")
    hlen = w // 2
    if hlen == 0:
        return np.zeros_like(C, dtype=np.float64)

    T = C.shape[0]
    x_ext = np.pad(C, ((hlen, hlen), (0, 0)), mode="edge")
    out = np.zeros_like(C, dtype=np.float64)
    for k in range(1, hlen + 1):
        out += k * (x_ext[hlen + k : hlen + k + T] - x_ext[hlen - k : hlen - k + T])
    return out


def _trimf(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """MATLAB Fuzzy Logic Toolbox `trimf(x, [a, b, c])`: 0 at `a` and `c`, 1 at `b`,
    linear in between; 0 outside `[a, c]`. `a <= b <= c` is assumed (true for every
    call `lfcc_b02_filterbank` makes, since `edges` is a strictly increasing
    `linspace`)."""
    rising = (x - a) / (b - a) if b > a else np.ones_like(x)
    falling = (c - x) / (c - b) if c > b else np.ones_like(x)
    return np.clip(np.minimum(rising, falling), 0.0, None)


def lfcc_b02_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    spacing: str = "linear",
) -> np.ndarray:
    """Literal transcription of `extract_lfcc.m`'s filterbank construction:

        f = (Fs/2)*linspace(0,1,NFFT/2+1);
        filbandwidthsf = linspace(min(f), max(f), No_Filter+2);
        for i = 1:No_Filter
            filterbank(i,:) = trimf(f, [filbandwidthsf(i), ...
                filbandwidthsf(i+1), filbandwidthsf(i+2)]);
        end

    Returns `Psi[n_filters, n_fft//2+1]`. `fmax` defaults to `sr/2` (`Fs/2`), matching
    `min(f)=0`/`max(f)=Fs/2` exactly when `fmin=0.0` (the B02 default; NOT 30 Hz — see
    the module docstring).

    `spacing="mel"` (X4 only) replaces the `linspace` edges with edges equally spaced on
    the HTK mel scale (`features.hz_to_mel`) between `fmin` and `fmax`, mapped back to Hz;
    triangles are still peak-1 `trimf`. The default `"linear"` path is unchanged.
    """
    if spacing not in ("linear", "mel"):
        raise ValueError(f"spacing must be 'linear' or 'mel', got {spacing!r}")
    fmax = sr / 2.0 if fmax is None else fmax
    n_bins = n_fft // 2 + 1
    f = fmax * np.linspace(0.0, 1.0, n_bins)
    if spacing == "linear":
        edges = np.linspace(fmin, fmax, n_filters + 2)
    else:
        from audio_df2.helpers.signal import hz_to_mel, mel_to_hz

        edges = np.asarray(
            mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filters + 2)),
            dtype=np.float64,
        )
    psi = np.zeros((n_filters, n_bins), dtype=np.float64)
    for i in range(n_filters):
        psi[i] = _trimf(f, edges[i], edges[i + 1], edges[i + 2])
    return psi


def extract_lfcc_b02(
    x: np.ndarray, sr: int = 16000, cfg: dict[str, Any] | None = None
) -> np.ndarray:
    """Extract the official B02 LFCC feature: `[T, 60]` (20 static + 20 delta + 20 delta-delta).

    `x` is a 1-D waveform (already loaded/resampled by `audio_df.features.audio.load_audio`), `sr`
    must equal `cfg["sample_rate"]` (default 16000, matching the official baseline's
    training data). `cfg` overrides `DEFAULT_LFCC_B02_CONFIG` entries; unset keys fall back
    to the default. No cropping, no CMVN, no caching (A1's train-frame cache is a separate,
    on-disk mechanism built by `training/gmm.py`, not here — see that module's and
    `CONVENTIONS.md`'s module boundaries: this module only ever touches arrays).
    """
    full_cfg = {**DEFAULT_LFCC_B02_CONFIG, **(cfg or {})}
    if sr != full_cfg["sample_rate"]:
        raise ValueError(
            f"extract_lfcc_b02: sr={sr} does not match cfg['sample_rate']={full_cfg['sample_rate']}"
        )

    frame_length = int(round(sr / 1000 * full_cfg["window_ms"]))
    hop = int(round(sr / 1000 * full_cfg["hop_ms"]))
    n_fft = full_cfg["n_fft"]
    n_filters = full_cfg["n_filters"]
    preemph = full_cfg["preemph"]
    fmax = full_cfg["fmax"] if full_cfg["fmax"] is not None else sr / 2.0

    x = np.asarray(x, dtype=np.float64)

    # 1. Pre-emphasis over the whole signal: y[n] = x[n] - preemph*x[n-1], y[0] = x[0].
    #    No dithering (commented out in the released baseline).
    y = np.asarray(lfilter([1.0, -preemph], [1.0], x))

    # 2. Framing: symmetric Hamming window (np.hamming, NOT scipy's periodic default),
    #    buffer(...,'nodelay') semantics.
    frames = _frame_signal(y, frame_length, hop)
    window = np.hamming(frame_length)
    frames = frames * window[None, :]

    # 3. 512-point FFT, one-sided power spectrum: abs(fft(frame,512)).^2, bins 1:257.
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = np.abs(spectrum) ** 2  # [T, n_fft//2 + 1]

    # 4. 20 linear triangular filters, 0 Hz to sr/2 (research note: NOT 30 Hz), via a
    #    direct trimf/linspace transcription (lfcc_b02_filterbank), not features.py.
    psi = lfcc_b02_filterbank(
        n_filters=n_filters,
        n_fft=n_fft,
        sr=sr,
        fmin=full_cfg["fmin"],
        fmax=fmax,
        spacing=full_cfg["spacing"],
    )
    energy = power @ psi.T  # [T, n_filters]

    # 5. log10(filbanksum + eps): eps ADDED, not a max-floor.
    log_energy = np.log10(energy + _LOG10_EPS)

    # 6. Orthonormal DCT-II, all n_filters outputs (0th + (n_filters-1) more).
    static = np.asarray(dct(log_energy, type=2, axis=-1, norm="ortho"))

    # 7-8. Delta, delta-delta (official_deltas, no normalisation), concatenate.
    delta = official_deltas(static, w=full_cfg["delta_window"])
    delta2 = official_deltas(delta, w=full_cfg["delta_window"])
    return np.concatenate([static, delta, delta2], axis=1).astype(np.float32)


class LFCC:
    """Reusable B02 extractor; takes a 1-D waveform at sample_rate."""
    def __init__(self, sample_rate=16000, **config):
        self.sample_rate = sample_rate
        self.config = config

    def __call__(self, audio):
        return extract_lfcc_b02(audio, sr=self.sample_rate, cfg=self.config)
