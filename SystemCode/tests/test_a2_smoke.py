"""Small, offline A2 smoke tests using a deterministic fake SSL encoder."""

import json
from dataclasses import replace
from typing import cast

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from audio_df.helpers.config import load_config
from audio_df.helpers.training import TrainingConfig
from audio_df.modules.neural import NeuralConfig, NeuralHead
from audio_df.modules.ssl_detector import SSLDetector
from audio_df.modules.wavlm import WavLM, WavLMConfig


SMOKE_HEAD = NeuralConfig(proj_dim=4, mlp_hidden=4, dropout=0.0)


def smoke_training_config() -> TrainingConfig:
    """Override only the training budget; never use full training defaults."""
    return replace(load_config(TrainingConfig), max_epochs=2, patience=1, batch_size=2, lr=1e-3)


class FakeEncoder:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = WavLMConfig("mock", None, 16000, 0.1, 3, 4, 2, 0, 2)

    @property
    def settings(self):
        from dataclasses import asdict
        return asdict(self.config)

    def __call__(self, waveforms):
        x = np.asarray(waveforms)
        m = x.mean(axis=1)
        s = x.std(axis=1)
        base = np.stack([m, s, m + s, m - s], axis=-1)
        return np.stack([base + i for i in range(3)], axis=1).astype(np.float16)


def test_end_to_end(tmp_path):
    rows = []
    for i in range(8):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.full(1600, (i + 1) * 0.02, dtype=np.float32), 16000)
        rows.append({
            "utt_id": f"clip{i}", "path": str(path),
            "label": "bonafide" if i % 2 else "spoof", "attack_id": "A01",
        })

    train = pd.DataFrame(rows[:6])
    dev = pd.DataFrame(rows[6:])
    head = NeuralHead(3, 4, SMOKE_HEAD)
    detector = SSLDetector(
        encoder=cast(WavLM, FakeEncoder()),
        head=head,
        train_config=smoke_training_config(),
    )
    detector.fit(train, tmp_path / "cache" / "a2", seed=1, dev=dev)
    assert len(detector.history["train_loss"]) <= 2 # type: ignore

    out = tmp_path / "artifact"
    detector.save(out)
    assert (out / "head.pt").exists()

    scored = detector.score_manifest(
        dev, tmp_path / "scores.csv", split="dev", cache_dir=tmp_path / "cache" / "a2"
    )
    assert len(scored) == 2 and np.isfinite(scored.score).all()
    assert len(json.loads((out / "layer_weights.json").read_text())) == 3


def test_checkpoint_load_without_download(tmp_path):
    encoder = WavLM(WavLMConfig("mock", None, 16000, 0.1, 3, 4, 2, 0, 2), device="cpu")
    model = SSLDetector(
        encoder=encoder,
        head=NeuralHead(3, 4, SMOKE_HEAD),
        train_config=smoke_training_config(),
    )
    model.save(tmp_path)
    loaded = SSLDetector.load(tmp_path, device="cpu")
    sample = torch.randn(2, 3, 4)
    with torch.no_grad():
        assert torch.allclose(model.head(sample), loaded.head(sample))
    assert loaded.encoder.model is None
