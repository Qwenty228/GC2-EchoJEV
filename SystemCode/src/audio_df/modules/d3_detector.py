"""D3 adapter for the existing Run.train/save/evaluate interface."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from audio_df.helpers.config import load_config
from audio_df.helpers.ssl_cache import center_crop_repeat
from audio_df.helpers.temporal_cache import TemporalPairs, temporal_cache, temporal_pool, wavlm_tokens
from audio_df.helpers.training import TrainingConfig
from audio_df.modules.bidirectional import BidirectionalConfig, BidirectionalHead
from audio_df.modules.lfcc import LFCC, LFCCConfig
from audio_df.modules.wavlm import WavLM, WavLMConfig

# LFCC is computed per clip on the CPU while batches load; workers keep the GPU fed.
LOADER_WORKERS = 6


class D3Detector:
    def __init__(self, acoustic: LFCC, ssl: WavLM,
                 head: BidirectionalHead | None = None,
                 config: BidirectionalConfig | None = None,
                 train_config: TrainingConfig | None = None):
        if head is not None and config is not None:
            raise ValueError("Specify head or config, not both")
        if acoustic.sample_rate != ssl.config.sample_rate:
            raise ValueError("LFCC and WavLM sample rates must match")
        self.acoustic, self.ssl = acoustic, ssl
        self.device = ssl.device
        self.head = head if head is not None else BidirectionalHead(
            3 * acoustic.settings.n_filters, ssl.config.hidden_dim, config)
        if abs(self.head.config.crop_seconds - ssl.config.crop_seconds) > 1e-6:
            raise ValueError("D3 attention timestamps must match WavLM crop_seconds")
        self.train_config = train_config or load_config(TrainingConfig)
        self.history = {}

    @property
    def cache_namespace(self) -> tuple[str, ...]:
        return ("wavlm-base-plus", "temporal")

    def _dataset(self, df: pd.DataFrame, split: str, cache_dir: Path) -> TemporalPairs:
        directory = temporal_cache(df, self.ssl, cache_dir, split, self.head.config.tokens,
                                   self.head.config.layer)
        return TemporalPairs(df, self.ssl, self.acoustic, directory, self.head.config.tokens)

    def fit(self, train: pd.DataFrame, cache_dir: Path, seed: int = 1337,
            dev: pd.DataFrame | None = None):
        if dev is None:
            raise ValueError("D3 needs dev data for early stopping")
        torch.manual_seed(seed)
        training = self._dataset(train, "train", cache_dir)
        validation = self._dataset(dev, "dev", cache_dir)
        cfg = self.train_config
        loader = DataLoader(training, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=LOADER_WORKERS, persistent_workers=True,
                            generator=torch.Generator().manual_seed(seed))
        dev_loader = DataLoader(validation, batch_size=cfg.batch_size, shuffle=False,
                                num_workers=LOADER_WORKERS, persistent_workers=True)
        weight = float((train.label == "spoof").sum()) / max(float((train.label == "bonafide").sum()), 1.)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=self.device))
        head = self.head.to(self.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=cfg.lr)
        best, stale, best_epoch = float("inf"), 0, 0
        best_state = copy.deepcopy(head.state_dict())
        train_losses, dev_losses = [], []
        for epoch in tqdm(range(cfg.max_epochs), desc=f"{type(self).__name__} training", unit="epoch"):
            head.train()
            running = 0.
            for a, s, y in loader:
                a, s, y = a.to(self.device), s.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(head(a, s), y)
                loss.backward()
                optimizer.step()
                running += float(loss.detach()) * len(y)
            train_loss = running / len(training)
            head.eval()
            running = 0.
            with torch.inference_mode():
                for a, s, y in dev_loader:
                    a, s, y = a.to(self.device), s.to(self.device), y.to(self.device)
                    running += float(criterion(head(a, s), y)) * len(y)
            dev_loss = running / len(validation)
            train_losses.append(train_loss)
            dev_losses.append(dev_loss)
            tqdm.write(f"{type(self).__name__} epoch {epoch+1}: train={train_loss:.4f} dev={dev_loss:.4f}")
            if dev_loss < best:
                best, best_epoch, stale = dev_loss, epoch+1, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                stale += 1
            if stale >= cfg.patience:
                break
        head.load_state_dict(best_state)
        head.eval()
        self.history = {"train_loss": train_losses, "dev_loss": dev_losses,
                        "best_epoch": best_epoch, "pos_weight": weight}
        return self

    def score_manifest(self, df: pd.DataFrame, destination: Path, *, split: str,
                       cache_dir: Path) -> pd.DataFrame:
        dataset = self._dataset(df, split, cache_dir)
        loader = DataLoader(dataset, batch_size=self.train_config.batch_size, shuffle=False,
                            num_workers=LOADER_WORKERS)
        self.head.to(self.device).eval()
        scores = []
        with torch.inference_mode():
            for a, s, _ in tqdm(loader, desc=f"{type(self).__name__} scoring {split}", unit="batch"):
                scores.extend(self.head(a.to(self.device), s.to(self.device)).cpu().tolist())
        result = pd.DataFrame({"clip_id": df.utt_id.astype(str).to_numpy(),
                               "label": df.label.to_numpy(), "score": scores})
        if "attack_id" in df:
            result["attack_id"] = df.attack_id.to_numpy()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        return result

    def score_audio(self, waveform: np.ndarray) -> float:
        cfg = self.ssl.config
        x = center_crop_repeat(np.asarray(waveform, dtype=np.float32),
                               round(cfg.sample_rate * cfg.crop_seconds))
        a = temporal_pool(self.acoustic(x), self.head.config.tokens)[None]
        s = wavlm_tokens(self.ssl, x[None], self.head.config.tokens,
                         self.head.config.layer).astype(np.float32)
        self.head.to(self.device).eval()
        with torch.inference_mode():
            return float(self.head(torch.from_numpy(a).to(self.device),
                                   torch.from_numpy(s).to(self.device))[0])

    def score_file(self, path: str | Path) -> float:
        from audio_df.helpers.audio import load_audio
        return self.score_audio(load_audio(path, self.ssl.config.sample_rate))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.cpu().state_dict(), directory / "head.pt")
        (directory / "model.json").write_text(json.dumps({
            "acoustic": asdict(self.acoustic.settings), "ssl": asdict(self.ssl.config),
            "fusion": asdict(self.head.config), "training": asdict(self.train_config),
        }, indent=2))
        (directory / "training_curve.json").write_text(json.dumps(self.history, indent=2))
        self.head.to(self.device)

    @classmethod
    def load(cls, directory: Path, device: str = "auto") -> "D3Detector":
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text())
        acoustic = LFCC(config=LFCCConfig(**meta["acoustic"]))
        ssl = WavLM(WavLMConfig(**meta["ssl"]), device=device)
        config = BidirectionalConfig(**meta["fusion"])
        head = BidirectionalHead(3 * acoustic.settings.n_filters, ssl.config.hidden_dim, config)
        head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu", weights_only=True))
        head.eval()
        return cls(acoustic, ssl, head=head, train_config=TrainingConfig(**meta["training"]))
