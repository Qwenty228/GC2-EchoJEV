"""Frozen-SSL per-layer, time-pooled hidden-state extraction and cache for System A2.

## What gets cached, and why

For every clip we store `[n_layers, hidden_dim]` fp16 — the **time-mean** of every hidden
state a frozen SSL backbone produces (`output_hidden_states=True`: WavLM Base+ gives 13
tensors of `[T, 768]`, embedding output + 12 transformer layers; XLS-R 300M gives 25 of
`[T, 1024]`, embedding output + 24 layers — both derived from a 199-frame, 4 s @ 16 kHz
input; see `Miscellaneous/research/knowledge/methods/ssl-hidden-states.md`). We never cache the raw
`[n_layers, T, hidden_dim]` frames for the whole corpus:

    frame-level cache: 25 layers x 199 frames x 1024 dims x 2 bytes (fp16)  ~= 10.2 MB/clip
                        -> 121,461 clips ~= 1.24 TB   (infeasible on this machine)
    pooled cache:       25 layers x 1024 dims x 2 bytes (fp16)              ~= 51.2 KB/clip
                        -> 121,461 clips ~= 6.2 GB     (fine)

(`CONVENTIONS.md`'s "one real blowup"; WavLM Base+ at 13 x 768 is smaller still,
~2.4 GB per silence mode for the full corpus.)

## Why the pooled cache is not a lossy shortcut for A2's head

A2's head (`audio_df.models.ssl_head.SSLHead`) computes a learned convex combination over
layers, then a mean over time (see that module's forward path via `weighted_sum`). Both
operations — the weighted sum over layers and the mean over time — are **linear**, and
linear operators commute:

    sum_l w_l * mean_t(H^(l))  ==  mean_t(sum_l w_l * H^(l))

So `SSLHead(pooled_cache)` is *exactly* equal (up to fp16 rounding) to "compute the
weighted sum over layers on the full frame-level hidden states, then mean-pool over time" —
never an approximation. This is why extraction here does the time-mean **before** any
LayerNorm: LayerNorm is not linear (it depends on the per-sample mean/variance of the
vector it normalizes), so normalizing frames before pooling would *not* commute with
pooling and would break this equivalence. Any normalization in `SSLHead` happens strictly
after the weighted sum over the already-pooled cache. `tests/test_ssl_cache.py` and
`tests/test_ssl_head.py` both check this numerically, from different ends.

## Per-backbone input normalization

WavLM Base+'s own feature extractor has `do_normalize=false`; XLS-R 300M's has
`do_normalize=true` (`Miscellaneous/research/knowledge/methods/ssl-hidden-states.md`). This module
never hardcodes either value: `load_backbone` always loads each model's own
`AutoFeatureExtractor`, and extraction always runs waveforms through that feature
extractor rather than feeding raw arrays directly to the model.

## Later (not implemented here)

If a future head needs frame sequences (e.g. a Transformer/CNN/ECAPA backend over the
learned weighted-sum frames), the artifact to cache is the *frame-level* weighted sum
`sum_l w_l * H^(l)` (`[T, hidden_dim]` per clip, ~37 GB fp16 for the whole corpus at
`hidden_dim=768`) for a bounded subset — not the full per-layer frame tensor above, and not
for the whole corpus. This module does not build that cache.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audio_df.features.audio import load_audio, preprocess
from audio_df.runtime.resources import ChunkedJob, MemoryGuard

__all__ = [
    "load_backbone",
    "extract_pooled_hidden_states",
    "cache_dir_for",
    "build_job_config",
    "WaveformDataset",
    "run_extraction",
    "load_cache",
]


def load_backbone(model_id: str, revision: str | None = None) -> tuple[nn.Module, Any]:
    """Download (first run only) and load a frozen SSL backbone plus its own feature extractor.

    Uses `transformers.AutoModel`/`AutoFeatureExtractor.from_pretrained(model_id)` so each
    backbone's own `preprocessor_config.json` (in particular `do_normalize`) is respected —
    never hardcode a shared preprocessing choice across backbones (see module docstring).
    The model is set to `.eval()`; the caller is responsible for `.to(device)` and freezing
    (no gradients are computed here, but `requires_grad_(False)` is applied defensively so a
    caller that forgets `torch.inference_mode()` still cannot accidentally update it).

    This function makes a real network call and is never exercised by this project's test
    suite (tests build a tiny `transformers` config locally instead, per
    `CONVENTIONS.md`); it is only invoked by `run_a2.py`'s runner, and only against a
    real, running machine.
    """
    from transformers import AutoFeatureExtractor, AutoModel

    kwargs: dict[str, Any] = {"revision": revision} if revision else {}
    model = AutoModel.from_pretrained(model_id, **kwargs)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, **kwargs)
    model.eval()
    model.requires_grad_(False)
    return model, feature_extractor


def extract_pooled_hidden_states(
    model: nn.Module,
    feature_extractor: Any,
    waveforms: np.ndarray,
    sample_rate: int,
    device: torch.device,
) -> np.ndarray:
    """One forward pass -> per-layer time-mean-pooled hidden states, fp16, `[B, L, D]`.

    `waveforms`: `[B, n_samples]` float32, already `preprocess()`-ed to a fixed length (this
    project's A2 crop is 4 s @ 16 kHz = 64,000 samples = 199 frames for both backbones — see
    module docstring); `sample_rate` is passed straight to `feature_extractor` so it applies
    whichever `do_normalize` its own `preprocessor_config.json` specifies. Runs in fp32 on
    `device` under `torch.inference_mode()` (frozen backbone, no gradients ever computed
    here); the pooled output is cast to fp16 only at the very end, once it is small, for the
    on-disk cache — this keeps the forward pass itself full precision.
    """
    model.eval()
    inputs = feature_extractor(
        list(waveforms),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs["input_values"].to(device=device, dtype=torch.float32)
    attention_mask = inputs.get("attention_mask")

    with torch.inference_mode():
        kwargs: dict[str, Any] = {"output_hidden_states": True}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(device)
        outputs = model(input_values, **kwargs)

    hidden_states = outputs.hidden_states  # tuple of L tensors, each [B, T, D]
    pooled = torch.stack([h.mean(dim=1) for h in hidden_states], dim=1)  # [B, L, D]
    return pooled.to(torch.float16).cpu().numpy()


def cache_dir_for(
    model_tag: str,
    silence_mode: str,
    testset: str,
    root: str | Path = "data/cache/ssl",
) -> Path:
    """`<root>/<model_tag>/<silence_mode>/<testset>/` — the cache layout every caller uses."""
    return Path(root) / model_tag / silence_mode / testset


def build_job_config(
    model_id: str,
    model_tag: str,
    n_layers: int,
    hidden_dim: int,
    silence_mode: str,
    crop_seconds: float,
    crop_mode: str,
    sample_rate: int,
    do_normalize: bool,
    revision: str | None = None,
    dtype: str = "float16",
) -> dict[str, Any]:
    """The `job_config` `ChunkedJob` hashes against `item_ids` to detect a changed recipe.

    Every field that would silently change what a cached clip's vector *means* if changed
    goes in here: model id/revision, silence mode, crop, normalization convention, dtype.
    """
    return {
        "model_id": model_id,
        "model_tag": model_tag,
        "revision": revision,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "silence_mode": silence_mode,
        "crop_seconds": crop_seconds,
        "crop_mode": crop_mode,
        "sample_rate": sample_rate,
        "do_normalize": do_normalize,
        "dtype": dtype,
    }


class WaveformDataset(Dataset):
    """Loads, silence-handles and crops one clip per `__getitem__`; returns `(utt_id, x)`.

    `x` is a fixed-length `float32` array (`preprocess(..., crop_seconds, crop_mode)`),
    ready to hand straight to a backbone's own feature extractor. Center crop is used for
    every split, including train — a fixed on-disk cache cannot support per-epoch random
    crop augmentation, so A2 deliberately gives up train-time random cropping in exchange
    for a cache that is built once (see `run_a2.py`'s module docstring for the tradeoff).
    """

    def __init__(
        self,
        ids: list[str],
        paths: list[str],
        sample_rate: int,
        silence_mode: Literal["raw", "trimmed"],
        crop_seconds: float,
        crop_mode: Literal["random", "center"] = "center",
    ) -> None:
        if len(ids) != len(paths):
            raise ValueError(f"ids and paths must have equal length ({len(ids)} != {len(paths)})")
        self.ids = list(ids)
        self.paths = list(paths)
        self.sample_rate = sample_rate
        self.silence_mode: Literal["raw", "trimmed"] = silence_mode
        self.crop_seconds = crop_seconds
        self.crop_mode: Literal["random", "center"] = crop_mode

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:
        utt_id = self.ids[idx]
        x, sr = load_audio(self.paths[idx], target_sr=self.sample_rate)
        x = preprocess(
            x,
            sr,
            silence_mode=self.silence_mode,
            crop_seconds=self.crop_seconds,
            crop_mode=self.crop_mode,
        )
        return utt_id, x.astype(np.float32)


def run_extraction(
    ids: list[str],
    paths: list[str],
    model: nn.Module,
    feature_extractor: Any,
    out_dir: str | Path,
    job_config: dict[str, Any],
    device: torch.device,
    sample_rate: int,
    silence_mode: Literal["raw", "trimmed"],
    crop_seconds: float,
    crop_mode: Literal["random", "center"],
    batch_size: int,
    num_workers: int = 0,
    chunk_size: int = 512,
    memory_guard: MemoryGuard | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
    generator: torch.Generator | None = None,
    blocking_pause: bool = True,
) -> ChunkedJob:
    """Extract pooled hidden states for `ids`/`paths` into a resumable chunked cache.

    Resumable by construction: `ChunkedJob` (see `audio_df.runtime.resources`) only re-hands out
    chunks whose `.npz` is missing or unreadable, so calling this again against the same
    `out_dir`/`job_config`/`ids` after an interruption (Ctrl-C, crash, `MemoryPressureExit`)
    picks up exactly where it left off, never re-extracting a completed chunk. Batching
    within a chunk goes through a plain `DataLoader` (`shuffle=False`, so a chunk's output
    order always matches `ids`' order) so `num_workers`/`worker_init_fn`/`generator` compose
    with this project's resource limits and seeding (`audio_df.runtime.resources.loader_workers`,
    `audio_df.runtime.seeding.worker_init_fn`/`make_generator`) exactly as every other loader does.
    """
    dataset = WaveformDataset(ids, paths, sample_rate, silence_mode, crop_seconds, crop_mode)
    id_to_index = {utt_id: i for i, utt_id in enumerate(ids)}

    job = ChunkedJob(
        out_dir,
        ids,
        chunk_size,
        job_config,
        memory_guard=memory_guard,
        blocking_pause=blocking_pause,
    )

    for chunk_idx, chunk_ids in job.pending():
        indices = [id_to_index[i] for i in chunk_ids]
        subset = torch.utils.data.Subset(dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            generator=generator,
        )

        pooled_parts: list[np.ndarray] = []
        for batch_ids, batch_x in loader:
            pooled = extract_pooled_hidden_states(
                model, feature_extractor, batch_x.numpy(), sample_rate, device
            )
            pooled_parts.append(pooled)
            del batch_ids  # order is guaranteed by shuffle=False; batch_ids unused beyond this

        pooled_all = np.concatenate(pooled_parts, axis=0)
        job.save_chunk(chunk_idx, chunk_ids, {"pooled": pooled_all})

    return job


def load_cache(out_dir: str | Path) -> dict[str, np.ndarray]:
    """Load and concatenate every completed `chunk_*.npz` file under `out_dir`.

    Loads in filename (i.e. chunk-index) order, which matches the extraction-time item
    order since `run_extraction` always processes chunks in index order with
    `shuffle=False`. Unlike reconstructing a `ChunkedJob` against the same `item_ids`, this
    does not need the caller to reproduce the original manifest/job_config; callers that do
    care about a config mismatch (e.g. `run_a2.py`'s `train`/`score` commands) should keep
    doing their own alignment against a manifest, not rely on this function to catch it.
    """
    out_dir = Path(out_dir)
    chunk_files = sorted(out_dir.glob("chunk_*.npz"))
    if not chunk_files:
        raise FileNotFoundError(f"no completed cache chunks found under {out_dir}")

    collected: dict[str, list[np.ndarray]] = {}
    for path in chunk_files:
        with np.load(path) as data:
            for key in data.files:
                collected.setdefault(key, []).append(data[key])

    return {key: np.concatenate(arrs, axis=0) for key, arrs in collected.items()}
