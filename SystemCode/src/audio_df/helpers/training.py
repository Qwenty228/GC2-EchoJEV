"""Reusable fitting utilities (not tied to A1)."""
from audio_df.modules.gmm import DiagGMM

def fit_gmm(frames, components=512, iterations=10, seed=1337, chunk_frames=50000,
            init_sample_size=200000, variance_floor=0.001):
    model = DiagGMM(components, frames.shape[1], variance_floor=variance_floor)
    model.fit_em(frames, n_iter=iterations, seed=seed, chunk_frames=chunk_frames,
                 init_sample_size=init_sample_size)
    return model


from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    lr: float
    batch_size: int
    max_epochs: int
    patience: int
