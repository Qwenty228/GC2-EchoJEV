"""E4 robustness degradations: test-time-only signal transforms on 16 kHz mono float32.

Every public function here takes 16 kHz mono `float32` audio in and returns 16 kHz mono
`float32` audio out, **the same length as the input**. This module never loads files from
disk on its own initiative (callers pass arrays already loaded via `audio_df.audio.load_audio`)
and never touches manifests or configs — see `tasks/CONVENTIONS.md`'s module table; this is
new module scope carved out for E4, so it inherits `audio.py`'s "only ever touches waveforms"
contract rather than `audio.py` itself.

Alignment convention (documented once, used by every codec function below): ffmpeg's
mp3/opus encoders prepend encoder delay (priming samples) and the decoder can also emit a
few extra trailing samples from block padding, so a naive encode-decode round trip is NOT
sample-aligned with the input. Every codec path here corrects for this the same way: decode
to a buffer that is at least as long as the input plus a safety margin, then take a single
deterministic slice `decoded[:len(x)]` if the decode is at least as long (it always is with
our margin), otherwise zero-pad on the right up to `len(x)`. This is a pragmatic sample-count
alignment, not a phase-accurate delay-compensated alignment — it guarantees output length
equals input length and keeps the codec's audible artifact where it belongs (the start of
the signal keeps the encoder's actual transient response) but it does not attempt to undo the
codec's own algorithmic delay sample-for-sample. Silence padding is added at the tail (ffmpeg's
`-af apad`) before encoding so the encoder has enough real signal to prime + flush its window
without our trim eating into content.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

__all__ = [
    "mp3",
    "opus",
    "add_noise",
    "resample_roundtrip",
    "degrade",
    "clip_seed",
]

_FFMPEG = "ffmpeg"

# Extra tail padding (seconds) added before encoding so the codec's encoder delay /
# decoder flush never eats into real content once we slice back to len(x). 0.25 s is far
# more than mp3 (~1152 samples @ any sr, ~0.026 s at 44.1k) or Opus (a few ms) actually need;
# generous margin costs nothing since we discard it after decode.
_CODEC_PAD_S = 0.25


def _run_ffmpeg(args: list[str]) -> None:
    proc = subprocess.run(
        [_FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (args={args}): {proc.stderr}")


def _align_to_length(decoded: np.ndarray, n: int) -> np.ndarray:
    """Slice or zero-pad `decoded` to exactly `n` samples — see module docstring."""
    if len(decoded) >= n:
        return decoded[:n].astype(np.float32)
    return np.pad(decoded, (0, n - len(decoded))).astype(np.float32)


def _codec_roundtrip(
    x: np.ndarray,
    sr: int,
    *,
    encode_sr: int,
    encoder: str,
    bitrate_kbps: int,
    container_suffix: str,
) -> np.ndarray:
    """Shared encode-decode-through-ffmpeg path for mp3/opus.

    Writes the input (padded) as a WAV at `encode_sr`, encodes to `container_suffix` at
    `bitrate_kbps`, decodes back to a WAV at `sr`, then aligns to `len(x)` per the module
    docstring's convention. All I/O goes through a private `tempfile.TemporaryDirectory` so
    concurrent ffmpeg workers never collide on a shared path, and everything is cleaned up
    even on failure.
    """
    n = len(x)
    pad_samples = int(round(_CODEC_PAD_S * sr))
    padded = np.pad(x, (0, pad_samples)).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="audio_df_degrade_") as tmp:
        tmp_dir = Path(tmp)
        wav_in = tmp_dir / "in.wav"
        encoded = tmp_dir / f"encoded{container_suffix}"
        wav_out = tmp_dir / "out.wav"

        sf.write(wav_in, padded, sr, subtype="FLOAT")

        _run_ffmpeg(
            [
                "-i",
                str(wav_in),
                "-ar",
                str(encode_sr),
                "-ac",
                "1",
                "-c:a",
                encoder,
                "-b:a",
                f"{bitrate_kbps}k",
                str(encoded),
            ]
        )
        _run_ffmpeg(
            [
                "-i",
                str(encoded),
                "-ar",
                str(sr),
                "-ac",
                "1",
                str(wav_out),
            ]
        )

        decoded, decoded_sr = sf.read(wav_out, dtype="float32", always_2d=False)
        assert decoded_sr == sr, f"_codec_roundtrip: expected sr={sr}, decoder gave {decoded_sr}"

    return _align_to_length(decoded, n)


def mp3(x: np.ndarray, sr: int = 16000, bitrate_kbps: int = 32) -> np.ndarray:
    """Encode/decode through libmp3lame CBR at `bitrate_kbps`, then decode back to `sr`.

    mp3 has no fixed internal sample rate requirement the way Opus does, so encoding happens
    directly at `sr` (16 kHz) — libmp3lame supports 16 kHz CBR encoding natively. Uses `-b:a`
    for a constant bitrate target (CBR is what the E4 spec names: "MP3 32 kbps"), not VBR.
    Deterministic: ffmpeg's mp3 encoder has no random seed / non-determinism across identical
    runs on the same input.
    """
    return _codec_roundtrip(
        x,
        sr,
        encode_sr=sr,
        encoder="libmp3lame",
        bitrate_kbps=bitrate_kbps,
        container_suffix=".mp3",
    )


def opus(x: np.ndarray, sr: int = 16000, bitrate_kbps: int = 16) -> np.ndarray:
    """Encode/decode through libopus at `bitrate_kbps`.

    Opus's reference encoder runs its internal processing at one of a fixed set of rates
    (8/12/16/24/48 kHz); libopus itself resamples internally, but we resample explicitly to
    48 kHz before handing audio to ffmpeg's opus encoder and resample back to `sr` (16 kHz)
    after decode, rather than relying on ffmpeg's implicit `-ar` conversion around the codec,
    so the two resample steps this function performs are visible and match this module's own
    `resample_roundtrip` convention (scipy `resample_poly`) rather than a second, hidden
    resampler inside ffmpeg's `-ar` machinery.
    """
    opus_sr = 48000
    x_48k = resample_poly(x, opus_sr, sr).astype(np.float32) if sr != opus_sr else x

    with tempfile.TemporaryDirectory(prefix="audio_df_degrade_opus_") as tmp:
        tmp_dir = Path(tmp)
        pad_samples = int(round(_CODEC_PAD_S * opus_sr))
        padded = np.pad(x_48k, (0, pad_samples)).astype(np.float32)

        wav_in = tmp_dir / "in.wav"
        encoded = tmp_dir / "encoded.opus"
        wav_out = tmp_dir / "out.wav"

        sf.write(wav_in, padded, opus_sr, subtype="FLOAT")
        _run_ffmpeg(
            [
                "-i",
                str(wav_in),
                "-ar",
                str(opus_sr),
                "-ac",
                "1",
                "-c:a",
                "libopus",
                "-b:a",
                f"{bitrate_kbps}k",
                str(encoded),
            ]
        )
        _run_ffmpeg(
            [
                "-i",
                str(encoded),
                "-ar",
                str(opus_sr),
                "-ac",
                "1",
                str(wav_out),
            ]
        )
        decoded_48k, decoded_sr = sf.read(wav_out, dtype="float32", always_2d=False)
        assert decoded_sr == opus_sr

    decoded_16k = resample_poly(decoded_48k, sr, opus_sr).astype(np.float32)
    return _align_to_length(decoded_16k, len(x))


def add_noise(
    x: np.ndarray,
    noise: np.ndarray,
    snr_db: float = 10.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Mix `noise` into `x` at `snr_db`, using a random offset (and looping if shorter).

    Power definition: `P_signal = mean(x**2)`, `P_noise = mean(noise_segment**2)` over the
    WHOLE clip (not a VAD-restricted "active speech" region) — i.e. `snr_db = 10 *
    log10(P_signal / P_noise_scaled)` after scaling, so `noise` is rescaled by a factor `k`
    solved from `10*log10(P_signal / (k**2 * P_noise)) = snr_db`, i.e.
    `k = sqrt(P_signal / (P_noise * 10**(snr_db/10)))`.

    `noise` must already be at the same sample rate as `x` (callers resample the MUSAN noise
    file to 16 kHz once, not per-mix, since a single noise bank is reused across many clips).
    A random offset into `noise` is drawn from `rng` (required — this is the source of the
    "different clip -> different noise offset" determinism this module promises at the
    `degrade()` level); if `noise` is shorter than `x` it is tiled (looped) starting from that
    offset so every position in `x` gets covered.

    Guards: if `x` is silent (`P_signal == 0`) or `noise` is silent (`P_noise == 0`), returns
    `x` unchanged (mixing noise into true digital silence at a finite SNR is not well-defined,
    and scaling by an all-zero noise buffer cannot change `x` regardless) rather than raising
    or dividing by zero.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    x = x.astype(np.float32)
    p_signal = float(np.mean(np.square(x, dtype=np.float64)))
    if p_signal == 0.0 or len(noise) == 0:
        return x.copy()

    noise = noise.astype(np.float32)
    if len(noise) < n:
        n_tiles = int(np.ceil(n / len(noise)))
        noise_full = np.tile(noise, n_tiles)
    else:
        noise_full = noise

    max_start = len(noise_full) - n
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    segment = noise_full[start : start + n]

    p_noise = float(np.mean(np.square(segment, dtype=np.float64)))
    if p_noise == 0.0:
        return x.copy()

    k = np.sqrt(p_signal / (p_noise * (10.0 ** (snr_db / 10.0))))
    mixed = x + (k * segment).astype(np.float32)
    return mixed.astype(np.float32)


def resample_roundtrip(x: np.ndarray, sr: int = 16000, mid_sr: int = 8000) -> np.ndarray:
    """Downsample `sr` -> `mid_sr` -> `sr` with `scipy.signal.resample_poly` (polyphase FIR).

    `resample_poly` is chosen (over `librosa.resample`'s default soxr backend, which `audio.py`
    already uses for file-load resampling) so this module's one round-trip resampler is
    explicit and documented independently of `audio.py`'s choice — both are legitimate
    high-quality resamplers, but the spec asks for one documented choice, and
    `scipy.signal.resample_poly` has no optional native-code dependency beyond scipy itself.
    Uses `up`/`down` reduced by `sr`'s and `mid_sr`'s GCD via `resample_poly(x, up, down)`
    called twice (once each direction). The down-sampling step applies FIR anti-aliasing
    filtering with a cutoff below `mid_sr/2`, which is what removes content above 4 kHz when
    `mid_sr=8000` (Nyquist of 8 kHz) — the up-sampling step cannot recover that removed content,
    so the round trip is lossy by construction, which is the entire point of this condition.
    Output length is aligned to `len(x)` (slice/zero-pad), since `resample_poly`'s output
    length from a non-integer-ratio round trip does not always exactly equal the input length.
    """
    from math import gcd

    n = len(x)
    g1 = gcd(mid_sr, sr)
    down = resample_poly(x, mid_sr // g1, sr // g1).astype(np.float32)

    g2 = gcd(sr, mid_sr)
    up = resample_poly(down, sr // g2, mid_sr // g2).astype(np.float32)

    return _align_to_length(up, n)


def clip_seed(seed: int, clip_id: str, condition_name: str) -> int:
    """Deterministic per-clip RNG seed derived from `(seed, clip_id, condition_name)`.

    `sha256(f"{seed}|{clip_id}|{condition_name}")` truncated to 63 bits (fits any platform's
    `np.random.default_rng` / `np.random.Generator(PCG64(seed))` seed range, and avoids
    sign-bit ambiguity). This makes any single clip's degradation reproducible independently
    of processing order or worker count: the derived seed depends only on the three inputs,
    never on "which chunk / which worker / what ran before it".
    """
    payload = f"{seed}|{clip_id}|{condition_name}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def degrade(
    x: np.ndarray,
    sr: int,
    condition_name: str,
    cfg: dict[str, Any],
    seed: int,
    clip_id: str,
    noise_bank: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Apply the named condition's degradation from `cfg["conditions"][condition_name]`.

    `cfg` is `configs/e4_degradations.yaml`'s parsed contents (or an equivalent dict, as
    tests use). Dispatches on `cfg["conditions"][condition_name]["type"]`:
      - `"mp3"`: `mp3(x, sr, bitrate_kbps=...)`.
      - `"opus"`: `opus(x, sr, bitrate_kbps=...)`.
      - `"noise"`: `add_noise(x, noise, snr_db=..., rng=...)`, where `noise` is chosen from
        `noise_bank` (required for this condition; a list of already-16kHz-mono-float32
        arrays, one per MUSAN noise file) by `rng.integers(0, len(noise_bank))` using the
        SAME per-clip `rng` that then also picks the offset inside `add_noise` — i.e. one
        `np.random.Generator(np.random.PCG64(clip_seed(...)))` drives both the file choice
        and the offset choice for this clip, so the whole condition is reproducible from
        `(seed, clip_id, condition_name)` alone.
      - `"resample"`: `resample_roundtrip(x, sr, mid_sr=...)`.

    The per-clip `rng` is built from `clip_seed(seed, clip_id, condition_name)` regardless of
    condition (even mp3/opus, which do not consume randomness themselves) so every condition
    goes through the same seeding call for consistency and future-proofing.
    """
    condition = cfg["conditions"][condition_name]
    kind = condition["type"]
    rng = np.random.Generator(np.random.PCG64(clip_seed(seed, clip_id, condition_name)))

    if kind == "mp3":
        return mp3(x, sr, bitrate_kbps=condition["bitrate_kbps"])
    if kind == "opus":
        return opus(x, sr, bitrate_kbps=condition["bitrate_kbps"])
    if kind == "noise":
        if not noise_bank:
            raise ValueError(
                f"degrade: condition {condition_name!r} is type 'noise' but no noise_bank "
                f"was supplied (need MUSAN noise/ clips — see cfg['conditions'][{condition_name!r}]"
                f"['noise_dir'])"
            )
        idx = int(rng.integers(0, len(noise_bank)))
        return add_noise(x, noise_bank[idx], snr_db=condition["snr_db"], rng=rng)
    if kind == "resample":
        return resample_roundtrip(x, sr, mid_sr=condition["mid_sr"])

    raise ValueError(f"degrade: unknown condition type {kind!r} for {condition_name!r}")
