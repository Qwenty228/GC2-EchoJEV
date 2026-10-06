"""Frame-level features for the fusion model: an LFCC stream and an SSL stream per clip.

Both streams come from the **same** preprocessed clip (silence mode, then a fixed-length
crop), so they cover the same stretch of audio:

- **LFCC stream** `[T_lfcc, 60]`: B02 LFCC (static + delta + delta-delta), 10 ms hop, about
  100 frames per second (`features.lfcc_b02.extract_lfcc_b02`).
- **SSL stream** `[T_ssl, D]`: the frozen backbone's hidden states mixed across layers with
  a fixed weight vector, about 50 frames per second (20 ms) for WavLM and XLS-R.

`frame_times` gives each frame's centre in seconds, so a model can put both streams on one
clock even though their frame rates differ.

**Layer mix.** Caching every layer would cost `n_layers` times more disk; the mix is fixed
before caching. `layer_weights` accepts `"uniform"`, a list of layer indices (uniform over
those), or a path to an A2 run whose learned weights (`layer_weights.json`) should be used.

**Cache size.** For a 4 s crop at 16 kHz with WavLM Base+: SSL 199 x 768 and LFCC 399 x 60,
both fp16, measured at 345 KB per clip, so about 9 GB each for train (25,380 clips) and dev
(24,844). Extraction runs one backbone forward pass per clip (about 50 ms each on Apple M4
MPS).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from audio_df.features.audio import load_audio, preprocess
from audio_df.features.lfcc_b02 import extract_lfcc_b02
from audio_df.runtime.resources import ChunkedJob, MemoryGuard

__all__ = [
    "FrameCacheReader",
    "build_frame_cache",
    "clip_streams",
    "frame_times",
    "resolve_layer_weights",
    "ssl_frame_states",
]

LFCC_HOP_S = 0.010
SSL_HOP_S = 0.020


def resolve_layer_weights(spec: Any, n_layers: int) -> np.ndarray:
    """Turn a config `layer_weights` entry into a `[n_layers]` float32 vector summing to 1."""
    if spec == "uniform" or spec is None:
        w = np.ones(n_layers, dtype=np.float64)
    elif isinstance(spec, list | tuple):
        w = np.zeros(n_layers, dtype=np.float64)
        for layer in spec:
            if not 0 <= int(layer) < n_layers:
                raise ValueError(f"layer index {layer} out of range for {n_layers} layers")
            w[int(layer)] = 1.0
    else:
        path = Path(spec)
        weights_file = path / "layer_weights.json" if path.is_dir() else path
        w = np.asarray(json.loads(weights_file.read_text()), dtype=np.float64)
        if w.shape != (n_layers,):
            raise ValueError(f"{weights_file} has {w.shape[0]} weights, expected {n_layers}")
    if w.sum() <= 0:
        raise ValueError("layer weights must not all be zero")
    return (w / w.sum()).astype(np.float32)


def frame_times(n_frames: int, hop_s: float) -> np.ndarray:
    """Centre time in seconds of each of `n_frames` frames spaced `hop_s` apart."""
    return (np.arange(n_frames, dtype=np.float32) + 0.5) * np.float32(hop_s)


def ssl_frame_states(
    model: nn.Module,
    feature_extractor: Any,
    waveforms: np.ndarray,
    sample_rate: int,
    device: torch.device,
    layer_weights: np.ndarray,
) -> np.ndarray:
    """One forward pass -> layer-mixed hidden states, `[B, T, D]` fp16.

    `waveforms` is `[B, n_samples]` float32, already preprocessed to one fixed length.
    The mix is computed in fp32 on `device`; only the result is cast to fp16.
    """
    model.eval()
    inputs = feature_extractor(
        list(waveforms), sampling_rate=sample_rate, return_tensors="pt", padding=False
    )
    input_values = inputs["input_values"].to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        outputs = model(input_values, output_hidden_states=True)
    hidden = torch.stack(outputs.hidden_states, dim=1)  # [B, L, T, D]
    if hidden.shape[1] != len(layer_weights):
        raise ValueError(
            f"backbone returned {hidden.shape[1]} layers, layer_weights has {len(layer_weights)}"
        )
    w = torch.as_tensor(layer_weights, device=hidden.device, dtype=hidden.dtype)
    mixed = torch.einsum("l,bltd->btd", w, hidden)
    return mixed.to(torch.float16).cpu().numpy()


def clip_streams(
    x: np.ndarray,
    backbone: tuple[nn.Module, Any],
    cfg: dict[str, Any],
    layer_weights: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess one clip once and return `(lfcc [T_lfcc, 60], ssl [T_ssl, D])`, both fp16.

    `cfg` is a fusion config: `preprocessing` (sample rate, silence mode, crop) and
    `lfcc` (the B02 front-end block).
    """
    prep = cfg["preprocessing"]
    sr = int(prep["sample_rate"])
    x = preprocess(
        x,
        sr,
        silence_mode=prep["silence_mode"],
        crop_seconds=prep["crop_seconds"],
        crop_mode=prep.get("crop_mode", "center"),
    )
    lfcc = extract_lfcc_b02(x, sr=sr, cfg=cfg["lfcc"]).astype(np.float16)
    model, feature_extractor = backbone
    ssl = ssl_frame_states(model, feature_extractor, x[None, :], sr, device, layer_weights)[0]
    return lfcc, ssl


def build_frame_cache(
    ids: list[str],
    paths: list[str],
    backbone: tuple[nn.Module, Any],
    cfg: dict[str, Any],
    layer_weights: np.ndarray,
    out_dir: str | Path,
    device: torch.device,
    chunk_size: int = 256,
    memory_guard: MemoryGuard | None = None,
) -> ChunkedJob:
    """Extract both streams for every clip into a resumable chunked cache.

    Each chunk file holds `ids`, `lfcc` `[n, T_lfcc, 60]` and `ssl` `[n, T_ssl, D]` (fp16).
    Rerunning after an interruption skips completed chunks (`ChunkedJob`).
    """
    prep = cfg["preprocessing"]
    job_config = {
        "kind": "fusion_frames",
        "backbone": cfg["ssl"]["id"],
        "revision": cfg["ssl"].get("revision"),
        "layer_weights": [round(float(w), 6) for w in layer_weights],
        "preprocessing": prep,
        "lfcc": cfg["lfcc"],
    }
    id_to_path = dict(zip(ids, paths, strict=True))
    job = ChunkedJob(out_dir, ids, chunk_size, job_config, memory_guard=memory_guard)
    for chunk_idx, chunk_ids in job.pending():
        lfcc_parts, ssl_parts = [], []
        for utt_id in chunk_ids:
            x, _ = load_audio(id_to_path[utt_id], target_sr=int(prep["sample_rate"]))
            lfcc, ssl = clip_streams(x, backbone, cfg, layer_weights, device)
            lfcc_parts.append(lfcc)
            ssl_parts.append(ssl)
        job.save_chunk(
            chunk_idx, chunk_ids, {"lfcc": np.stack(lfcc_parts), "ssl": np.stack(ssl_parts)}
        )
    return job


class FrameCacheReader:
    """Stream a frame cache chunk by chunk, so training never holds it all in memory.

    `iter_batches` yields `(ids, lfcc, ssl)` batches as float32 tensors. With `shuffle`,
    chunk order and the order within each chunk are permuted by `rng` every pass.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.chunk_files = sorted(self.cache_dir.glob("chunk_*.npz"))
        if not self.chunk_files:
            raise FileNotFoundError(f"no frame-cache chunks under {self.cache_dir}")

    def ids(self) -> list[str]:
        out: list[str] = []
        for path in self.chunk_files:
            with np.load(path) as data:
                out.extend(str(i) for i in data["ids"])
        return out

    def iter_batches(
        self,
        batch_size: int,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterator[tuple[list[str], torch.Tensor, torch.Tensor]]:
        order = np.arange(len(self.chunk_files))
        if shuffle:
            if rng is None:
                raise ValueError("shuffle=True needs an rng, so the order is reproducible")
            order = rng.permutation(order)
        for chunk_idx in order:
            with np.load(self.chunk_files[chunk_idx]) as data:
                ids = [str(i) for i in data["ids"]]
                lfcc = data["lfcc"]
                ssl = data["ssl"]
            idx = rng.permutation(len(ids)) if shuffle and rng is not None else np.arange(len(ids))
            for start in range(0, len(idx), batch_size):
                sel = idx[start : start + batch_size]
                yield (
                    [ids[i] for i in sel],
                    torch.from_numpy(lfcc[sel].astype(np.float32)),
                    torch.from_numpy(ssl[sel].astype(np.float32)),
                )
