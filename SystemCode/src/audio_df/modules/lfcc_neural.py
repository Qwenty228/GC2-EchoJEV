"""A4: mean-pooled LFCC with the existing A2 neural projection/MLP head."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from audio_df.helpers.audio import load_audio
from audio_df.helpers.config import load_config
from audio_df.helpers.device import get_device
from audio_df.helpers.parallel import parallel_map
from audio_df.helpers.training import TrainingConfig
from audio_df.modules.detector import AudioFeatureExtractor
from audio_df.modules.lfcc import LFCC, LFCCConfig
from audio_df.modules.neural import NeuralConfig, NeuralHead


class LFCCDetector:
    """A4 binary classifier; NeuralHead has one LFCC representation, not SSL layers."""

    def __init__(
        self,
        features: LFCC,
        head: NeuralHead | None = None,
        head_config: NeuralConfig | None = None,
        train_config: TrainingConfig | None = None,
        device: str = "auto",
    ) -> None:
        if head is not None and head_config is not None:
            raise ValueError("Provide head or head_config, not both")
        self.features = features
        self.head = head if head is not None else NeuralHead(
            n_layers=1, hidden_dim=3 * features.settings.n_filters, config=head_config
        )
        self.train_config = train_config if train_config is not None else load_config(TrainingConfig)
        self.device = get_device(device)
        self.history: dict = {}

    @property
    def cache_namespace(self) -> tuple[str, ...]:
        return ("lfcc", "pooled")

    def _pooled(self, df: pd.DataFrame, cache_dir: Path, split: str) -> np.ndarray:
        """Cache one 60-D mean vector per clip; include paths, order and LFCC settings in key."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = hashlib.sha256(
            (df[["utt_id", "path"]].to_csv(index=False)
             + json.dumps(asdict(self.features.settings), sort_keys=True)).encode()
        ).hexdigest()[:16]
        target = cache_dir / f"{split}-{fingerprint}.npy"
        dims = 3 * self.features.settings.n_filters
        if target.exists():
            cached = np.load(target, mmap_mode="r")
            if cached.shape == (len(df), dims):
                tqdm.write(f"LFCC pooled cache hit: {target.name}")
                return np.asarray(cached)
        temp = cache_dir / f"{split}-{fingerprint}.building.npy"
        array = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32,
                                          shape=(len(df), dims))
        extractor = AudioFeatureExtractor(self.features)
        for i, frames in enumerate(parallel_map(extractor, df.path.tolist(),
                                                 workers=4, desc=f"LFCC {split}")):
            if frames.ndim != 2 or frames.shape[1] != dims:
                raise ValueError(f"Unexpected LFCC shape {frames.shape}")
            array[i] = np.asarray(frames, dtype=np.float32).mean(axis=0)
        array.flush()
        del array
        temp.replace(target)
        return np.load(target, mmap_mode="r")

    def fit(self, train: pd.DataFrame, cache_dir: Path, seed: int = 1337,
            dev: pd.DataFrame | None = None):
        if dev is None:
            raise ValueError("A4 requires dev for early stopping")
        torch.manual_seed(seed)
        x_train = torch.from_numpy(np.asarray(self._pooled(train, cache_dir, "train"), dtype=np.float32).copy()).unsqueeze(1)
        x_dev = torch.from_numpy(np.asarray(self._pooled(dev, cache_dir, "dev"), dtype=np.float32).copy()).unsqueeze(1)
        y_train = torch.tensor((train.label == "bonafide").to_numpy(), dtype=torch.float32)
        y_dev = torch.tensor((dev.label == "bonafide").to_numpy(), dtype=torch.float32)
        cfg = self.train_config
        weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
        head = self.head.to(self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=self.device))
        optimizer = torch.optim.Adam(head.parameters(), lr=cfg.lr)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(seed))
        best_loss, best_epoch, stale = float("inf"), 0, 0
        best_state = copy.deepcopy(head.state_dict())
        losses, dev_losses = [], []
        for epoch in tqdm(range(cfg.max_epochs), desc="Training A4 head", unit="epoch"):
            head.train()
            running = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(head(xb), yb)
                loss.backward()
                optimizer.step()
                running += float(loss.detach()) * len(xb)
            losses.append(running / len(x_train))
            head.eval()
            total = 0.0
            with torch.inference_mode():
                for start in range(0, len(x_dev), cfg.batch_size):
                    xb = x_dev[start:start + cfg.batch_size].to(self.device)
                    yb = y_dev[start:start + cfg.batch_size].to(self.device)
                    total += float(loss_fn(head(xb), yb)) * len(xb)
            dev_loss = total / len(x_dev)
            dev_losses.append(dev_loss)
            if dev_loss < best_loss:
                best_loss, best_epoch, stale = dev_loss, epoch + 1, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                stale += 1
            tqdm.write(f"A4 epoch {epoch + 1}: train={losses[-1]:.4f} dev={dev_loss:.4f}")
            if stale >= cfg.patience:
                break
        head.load_state_dict(best_state)
        head.eval()
        self.history = {"train_loss": losses, "dev_loss": dev_losses,
                        "best_epoch": best_epoch, "pos_weight": weight}
        return self

    def score_audio(self, waveform: np.ndarray) -> float:
        pooled = self.features(waveform).mean(axis=0).astype(np.float32)
        self.head.to(self.device).eval()
        with torch.inference_mode():
            return float(self.head(torch.from_numpy(pooled[None, None, :]).to(self.device))[0])

    def score_file(self, path: str | Path) -> float:
        return self.score_audio(load_audio(path, self.features.sample_rate))

    def score_manifest(self, df: pd.DataFrame, destination: Path, *, split: str,
                       cache_dir: Path) -> pd.DataFrame:
        pooled = self._pooled(df, cache_dir, split)
        self.head.to(self.device).eval()
        scores = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(pooled), self.train_config.batch_size),
                              desc=f"Scoring {split}", unit="batch"):
                xb = torch.from_numpy(np.asarray(pooled[start:start + self.train_config.batch_size],
                                                 dtype=np.float32).copy()).unsqueeze(1).to(self.device)
                scores.extend(self.head(xb).cpu().tolist())
        result = pd.DataFrame({"clip_id": df.utt_id.astype(str).to_numpy(),
                               "label": df.label.to_numpy(), "score": scores})
        if "attack_id" in df:
            result["attack_id"] = df.attack_id.to_numpy()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        return result

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.cpu().state_dict(), directory / "head.pt")
        (directory / "model.json").write_text(json.dumps({
            "features": asdict(self.features.settings),
            "head": asdict(self.head.config),
            "training": asdict(self.train_config),
        }, indent=2), encoding="utf-8")
        (directory / "training_curve.json").write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        self.head.to(self.device)

    @classmethod
    def load(cls, directory: str | Path, device: str = "auto") -> "LFCCDetector":
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        features = LFCC(config=LFCCConfig(**meta["features"]))
        head = NeuralHead(1, 3 * features.settings.n_filters, NeuralConfig(**meta["head"]))
        head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu", weights_only=True))
        head.eval()
        return cls(features, head=head, train_config=TrainingConfig(**meta["training"]), device=device)
