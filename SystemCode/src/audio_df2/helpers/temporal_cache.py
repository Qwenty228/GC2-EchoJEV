"""Resumable, bounded temporal features for D3; independent of A2 pooled cache."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from audio_df2.helpers.ssl_cache import AudioClips


def temporal_pool(frames: np.ndarray, tokens: int) -> np.ndarray:
    """Time-average to a fixed token count; [T,D] -> [tokens,D]."""
    x = torch.as_tensor(np.asarray(frames, dtype=np.float32).T.copy())[None]
    return F.adaptive_avg_pool1d(x, tokens)[0].T.numpy()


def wavlm_tokens(encoder, waveforms: np.ndarray, tokens: int) -> np.ndarray:
    """Use the frozen last transformer layer; do not alter WavLM.__call__."""
    model, processor = encoder._load()
    inputs = processor(list(waveforms), sampling_rate=encoder.config.sample_rate,
                       padding=False, return_tensors="pt")
    kwargs = {}
    if "attention_mask" in inputs:
        kwargs["attention_mask"] = inputs["attention_mask"].to(encoder.device)
    with torch.inference_mode():
        hidden = model(inputs["input_values"].to(encoder.device), **kwargs).last_hidden_state
        result = F.adaptive_avg_pool1d(hidden.transpose(1, 2), tokens).transpose(1, 2)
    return result.to(torch.float16).cpu().numpy()


def _identity(df, encoder, tokens: int) -> str:
    payload = {"manifest": df[["utt_id", "path"]].to_csv(index=False),
               "encoder": encoder.settings, "tokens": tokens,
               "representation": "last_hidden_state_temporal_avg_v1"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def temporal_cache(df, encoder, root: Path, split: str, tokens: int = 32) -> Path:
    """Return completed chunk directory; each chunk is written atomically and resumably."""
    cfg = encoder.config
    identity = _identity(df, encoder, tokens)
    directory = Path(root) / split / identity[:16]
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / "cache.json"
    if metadata.exists():
        if json.loads(metadata.read_text())["identity"] != identity:
            raise ValueError("Temporal cache identity mismatch")
    else:
        if list(directory.glob("chunk_*.npy")):
            raise ValueError("Temporal cache has chunks without metadata")
        temp = directory / "cache.json.tmp"
        temp.write_text(json.dumps({"identity": identity, "tokens": tokens,
                                    "count": len(df), "chunk_size": cfg.chunk_size}, indent=2))
        os.replace(temp, metadata)
    total = (len(df) + cfg.chunk_size - 1) // cfg.chunk_size
    for i in tqdm(range(total), desc=f"WavLM temporal {split}", unit="chunk"):
        path = directory / f"chunk_{i:05d}.npy"
        count = min(cfg.chunk_size, len(df) - i * cfg.chunk_size)
        if path.exists():
            try:
                existing = np.load(path, mmap_mode="r")
                if existing.shape == (count, tokens, cfg.hidden_dim) and existing.dtype == np.float16:
                    continue
            except (OSError, ValueError):
                pass
        subset = df.iloc[i * cfg.chunk_size:(i + 1) * cfg.chunk_size]
        loader = DataLoader(AudioClips(subset.path, cfg.sample_rate, cfg.crop_seconds),
                            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                            pin_memory=encoder.device.type == "cuda", shuffle=False)
        parts = [wavlm_tokens(encoder, batch.numpy(), tokens) for batch in tqdm(
            loader, desc=f"Temporal chunk {i+1}/{total}", leave=False, unit="batch")]
        array = np.concatenate(parts)
        if array.shape != (count, tokens, cfg.hidden_dim):
            raise ValueError(f"Invalid temporal feature shape: {array.shape}")
        temp = directory / f"chunk_{i:05d}.tmp.npy"
        np.save(temp, array)
        os.replace(temp, path)
    return directory


class TemporalPairs(torch.utils.data.Dataset):
    """Lazy chunk reader: no full SSL cache concatenation in RAM."""
    def __init__(self, df, encoder, acoustic, directory: Path, tokens: int):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder
        self.acoustic = acoustic
        self.directory = directory
        self.tokens = tokens
        self.chunk_size = encoder.config.chunk_size
        self._open_chunk = -1
        self._chunk = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        chunk_idx, local = divmod(index, self.chunk_size)

        if self._chunk is None or chunk_idx != self._open_chunk:
            self._chunk = np.load(
                self.directory / f"chunk_{chunk_idx:05d}.npy",
                mmap_mode="r",
            )
            self._open_chunk = chunk_idx

        chunk = self._chunk
        assert chunk is not None

        row = self.df.iloc[index]

        from audio_df2.helpers.audio import load_audio
        from audio_df2.helpers.ssl_cache import center_crop_repeat

        cfg = self.encoder.config
        audio = load_audio(row.path, cfg.sample_rate)
        audio = center_crop_repeat(
            audio,
            round(cfg.sample_rate * cfg.crop_seconds),
        )

        lfcc = temporal_pool(self.acoustic(audio), self.tokens)
        ssl = np.asarray(chunk[local], dtype=np.float32).copy()

        return (
            torch.from_numpy(lfcc),
            torch.from_numpy(ssl),
            torch.tensor(row.label == "bonafide", dtype=torch.float32),
        )
