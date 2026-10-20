"""D1: independent LFCC + frozen WavLM feature-level concatenation detector."""
from __future__ import annotations

import copy
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
from audio_df.helpers.ssl_cache import cached_pooled, center_crop_repeat, iter_pooled_chunks
from audio_df.helpers.training import TrainingConfig
from audio_df.modules.fusion import ConcatHead
from audio_df.modules.lfcc import LFCC, LFCCConfig
from audio_df.modules.lfcc_neural import LFCCDetector
from audio_df.modules.neural import NeuralConfig
from audio_df.modules.wavlm import WavLM, WavLMConfig


class FusionDetector:
    """Trains only ConcatHead; LFCC and WavLM are fixed feature extractors."""

    def __init__(self, acoustic: LFCC, ssl: WavLM,
                 head: ConcatHead | None = None,
                 head_config: NeuralConfig | None = None,
                 train_config: TrainingConfig | None = None):
        if head is not None and head_config is not None:
            raise ValueError("Specify head or head_config, not both")
        self.acoustic = acoustic
        self.ssl = ssl
        self.device = ssl.device
        self.head = head if head is not None else ConcatHead(
            ssl.config.n_layers, ssl.config.hidden_dim,
            3 * acoustic.settings.n_filters, head_config,
        )
        self.train_config = train_config if train_config is not None else load_config(TrainingConfig)
        self.history: dict = {}
        # Reuse A4's exact LFCC pooled feature-cache implementation, without training A4.
        self._lfcc_cache = LFCCDetector(features=acoustic, device=str(self.device))

    @property
    def cache_namespace(self) -> tuple[str, ...]:
        # Runtime passes data/cache/fusion; this detector resolves shared feature caches.
        return ("fusion",)

    def _dirs(self, cache_dir: Path) -> tuple[Path, Path]:
        root = Path(cache_dir).parent
        return root / "lfcc" / "pooled", root / self.ssl.config.model_id.rsplit("/", 1)[-1] / "pooled"

    def _features(self, df: pd.DataFrame, split: str, cache_dir: Path):
        lfcc_dir, ssl_dir = self._dirs(cache_dir)
        lfcc = self._lfcc_cache._pooled(df, lfcc_dir, split)
        ssl = cached_pooled(df, self.ssl, ssl_dir, split)
        if len(lfcc) != len(ssl) or len(lfcc) != len(df):
            raise ValueError("Feature-cache row count mismatch")
        return lfcc, ssl

    def fit(self, train: pd.DataFrame, cache_dir: Path, seed: int = 1337,
            dev: pd.DataFrame | None = None):
        if dev is None:
            raise ValueError("D1 requires a development manifest for early stopping")
        torch.manual_seed(seed)
        lfcc_train, ssl_train = self._features(train, "train", cache_dir)
        lfcc_dev, ssl_dev = self._features(dev, "dev", cache_dir)
        x_l = torch.from_numpy(np.asarray(lfcc_train, dtype=np.float32).copy())
        x_s = torch.from_numpy(np.asarray(ssl_train, dtype=np.float32).copy())
        d_l = torch.from_numpy(np.asarray(lfcc_dev, dtype=np.float32).copy())
        d_s = torch.from_numpy(np.asarray(ssl_dev, dtype=np.float32).copy())
        y = torch.tensor((train.label == "bonafide").to_numpy(), dtype=torch.float32)
        yd = torch.tensor((dev.label == "bonafide").to_numpy(), dtype=torch.float32)
        cfg = self.train_config
        head = self.head.to(self.device)
        weight = float((y == 0).sum()) / max(float((y == 1).sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=self.device))
        optimizer = torch.optim.Adam(head.parameters(), lr=cfg.lr)
        loader = DataLoader(TensorDataset(x_l, x_s, y), batch_size=cfg.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(seed))
        best_loss, best_epoch, stale = float("inf"), 0, 0
        best_state = copy.deepcopy(head.state_dict())
        losses, dev_losses = [], []
        for epoch in tqdm(range(cfg.max_epochs), desc="Training D1 fusion", unit="epoch"):
            head.train()
            running = 0.0
            for lb, sb, yb in loader:
                lb, sb, yb = lb.to(self.device), sb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(head(lb, sb), yb)
                loss.backward()
                optimizer.step()
                running += float(loss.detach()) * len(yb)
            losses.append(running / len(y))
            head.eval()
            total = 0.0
            with torch.inference_mode():
                for start in range(0, len(yd), cfg.batch_size):
                    lb = d_l[start:start + cfg.batch_size].to(self.device)
                    sb = d_s[start:start + cfg.batch_size].to(self.device)
                    yb = yd[start:start + cfg.batch_size].to(self.device)
                    total += float(loss_fn(head(lb, sb), yb)) * len(yb)
            dev_loss = total / len(yd)
            dev_losses.append(dev_loss)
            if dev_loss < best_loss:
                best_loss, best_epoch, stale = dev_loss, epoch + 1, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                stale += 1
            tqdm.write(f"D1 epoch {epoch + 1}: train={losses[-1]:.4f} dev={dev_loss:.4f}")
            if stale >= cfg.patience:
                break
        head.load_state_dict(best_state)
        head.eval()
        self.history = {"train_loss": losses, "dev_loss": dev_losses,
                        "best_epoch": best_epoch, "pos_weight": weight}
        return self

    def score_manifest(self, df: pd.DataFrame, destination: Path, *, split: str,
                       cache_dir: Path) -> pd.DataFrame:
        lfcc_dir, ssl_dir = self._dirs(cache_dir)
        lfcc = self._lfcc_cache._pooled(df, lfcc_dir, split)
        scores: list[float] = []
        offset = 0
        self.head.to(self.device).eval()
        with torch.inference_mode():
            for pooled in iter_pooled_chunks(df, self.ssl, ssl_dir, split):
                for start in range(0, len(pooled), self.train_config.batch_size):
                    end = start + self.train_config.batch_size
                    sl = slice(offset + start, offset + min(end, len(pooled)))
                    lb = torch.from_numpy(np.asarray(lfcc[sl], dtype=np.float32).copy()).to(self.device)
                    sb = torch.from_numpy(np.asarray(pooled[start:end], dtype=np.float32).copy()).to(self.device)
                    scores.extend(self.head(lb, sb).cpu().tolist())
                offset += len(pooled)
        if len(scores) != len(df):
            raise ValueError("D1 scores are not aligned with the manifest")
        result = pd.DataFrame({"clip_id": df.utt_id.astype(str).to_numpy(),
                               "label": df.label.to_numpy(), "score": scores})
        if "attack_id" in df:
            result["attack_id"] = df.attack_id.to_numpy()
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(destination, index=False)
        return result

    def score_audio(self, waveform: np.ndarray) -> float:
        x = np.asarray(waveform, dtype=np.float32)
        lfcc = self.acoustic(x).mean(axis=0).astype(np.float32)
        ssl_cfg = self.ssl.config
        cropped = center_crop_repeat(x, round(ssl_cfg.sample_rate * ssl_cfg.crop_seconds))
        ssl = self.ssl(cropped[None, :]).astype(np.float32)
        self.head.to(self.device).eval()
        with torch.inference_mode():
            return float(self.head(torch.from_numpy(lfcc[None, :]).to(self.device),
                                   torch.from_numpy(ssl).to(self.device))[0])

    def score_file(self, path: str | Path) -> float:
        if self.acoustic.sample_rate != self.ssl.config.sample_rate:
            raise ValueError("LFCC and WavLM sample rates must match")
        return self.score_audio(load_audio(path, self.ssl.config.sample_rate))

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.cpu().state_dict(), directory / "head.pt")
        (directory / "model.json").write_text(json.dumps({
            "acoustic": asdict(self.acoustic.settings),
            "ssl": asdict(self.ssl.config),
            "head": asdict(self.head.config),
            "training": asdict(self.train_config),
        }, indent=2), encoding="utf-8")
        (directory / "training_curve.json").write_text(json.dumps(self.history, indent=2))
        (directory / "layer_weights.json").write_text(json.dumps(
            self.head.layer_weights().detach().cpu().tolist(), indent=2))
        self.head.to(self.device)

    @classmethod
    def load(cls, directory: Path, device: str = "auto") -> "FusionDetector":
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        acoustic = LFCC(config=LFCCConfig(**meta["acoustic"]))
        ssl = WavLM(WavLMConfig(**meta["ssl"]), device=device)
        head = ConcatHead(ssl.config.n_layers, ssl.config.hidden_dim,
                          3 * acoustic.settings.n_filters, NeuralConfig(**meta["head"]))
        head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu", weights_only=True))
        head.eval()
        return cls(acoustic, ssl, head=head, train_config=TrainingConfig(**meta["training"]))
