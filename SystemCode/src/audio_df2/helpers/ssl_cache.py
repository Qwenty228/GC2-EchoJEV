"""Resumable pooled SSL feature cache, one atomic chunk at a time."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from audio_df2.helpers.audio import load_audio


def center_crop_repeat(x, n):
    """Match original A2's center crop / repeat padding, not zero padding."""
    if not len(x):
        return np.zeros(n, dtype=np.float32)
    if len(x) >= n:
        start = (len(x) - n) // 2
        return x[start:start + n]
    return np.tile(x, (n + len(x) - 1) // len(x))[:n]

class AudioClips(Dataset):
    def __init__(self, paths, sample_rate, seconds):
        self.paths = list(paths)
        self.sample_rate = sample_rate
        self.samples = round(sample_rate * seconds)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return center_crop_repeat(load_audio(self.paths[i], self.sample_rate), self.samples)


def iter_pooled_chunks(df, encoder, directory, name):
    """Yield cached [chunk,L,D] arrays in manifest order, without joining all in RAM."""
    cfg = encoder.config
    identity = hashlib.sha256((df[['utt_id', 'path']].to_csv(index=False) +
        json.dumps(encoder.settings, sort_keys=True)).encode()).hexdigest()
    # A changed manifest or preprocessing recipe gets its own cache.
    directory = Path(directory) / name / identity[:16]
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory / 'cache.json' 
    if metadata.exists():
        if json.loads(metadata.read_text())['identity'] != identity:
            raise ValueError(f'Incompatible SSL cache at {directory}; use a different cache path')
    else:
        if list(directory.glob('chunk_*.npy')):
            raise ValueError(f'Cache chunks exist without metadata: {directory}')
        temp_meta = directory / 'cache.json.tmp'
        temp_meta.write_text(json.dumps({'identity': identity, 'config': encoder.settings}, indent=2))
        os.replace(temp_meta, metadata)
    chunk_size = cfg.chunk_size
    total = (len(df) + chunk_size - 1) // chunk_size
    for i in tqdm(range(total), desc=f'WavLM {name}', unit='chunk'):
        out = directory / f'chunk_{i:05d}.npy'
        count = min(chunk_size, len(df) - i * chunk_size)
        if out.exists():
            try:
                existing = np.load(out, mmap_mode='r')
                if existing.shape == (count, cfg.n_layers, cfg.hidden_dim) and existing.dtype == np.float16:
                    yield existing
                    continue
            except (ValueError, OSError):
                pass
        subset = df.iloc[i * chunk_size:(i + 1) * chunk_size]
        dataset = AudioClips(subset['path'], cfg.sample_rate, cfg.crop_seconds)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=encoder.device.type == 'cuda')
        pooled = np.concatenate([encoder(batch.numpy()) for batch in loader])
        if pooled.shape != (count, cfg.n_layers, cfg.hidden_dim):
            raise ValueError(f'Unexpected WavLM shape {pooled.shape}')
        temp = directory / f'chunk_{i:05d}.tmp.npy'
        np.save(temp, pooled)
        os.replace(temp, out)
        yield np.load(out, mmap_mode='r')


def cached_pooled(df, encoder, directory, name):
    chunks = list(iter_pooled_chunks(df, encoder, directory, name))
    cfg = encoder.config
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0,cfg.n_layers,cfg.hidden_dim), dtype=np.float16)
