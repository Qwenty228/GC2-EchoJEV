"""A2 as a detector: frozen SSL backbone -> per-layer time-mean pooling -> `SSLHead` logit.

Two scoring paths share one head:

- `score_pooled(pooled)` scores per-layer pooled hidden states (`[B, L, D]`, the layout of
  `data/cache/ssl/...`). It is the exact computation `pipelines/run_a2.py` used to write its
  score files, and the one experiments use.
- `score_waveform(x)` runs the backbone first (`features.ssl.extract_pooled_hidden_states`),
  for single clips in the application. The cache stores pooled states in fp16, so the two
  paths agree to within fp16 rounding, not bit for bit.

The backbone is loaded lazily on first use (about 0.4 GB for WavLM Base+), so a detector
that only scores cached features never downloads it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from audio_df.features.audio import preprocess
from audio_df.features.ssl import extract_pooled_hidden_states, load_backbone
from audio_df.models.base import (
    Detector,
    dev_threshold_from_run,
    read_bundle_meta,
    write_bundle_meta,
)
from audio_df.models.ssl_head import SSLHead
from audio_df.runtime.device import resolve_device

__all__ = ["SslHeadDetector"]

_HEAD_FILE = "head_state_dict.pt"


def head_from_config(model_cfg: dict[str, Any], head_cfg: dict[str, Any]) -> SSLHead:
    """An untrained `SSLHead` shaped by a run's `model` and `head` config blocks."""
    return SSLHead(
        n_layers=model_cfg["n_layers"],
        hidden_dim=model_cfg["hidden_dim"],
        proj_dim=head_cfg["proj_dim"],
        mlp_hidden=head_cfg["mlp_hidden"],
        dropout=head_cfg["dropout"],
    )


class SslHeadDetector(Detector):
    """System A2. `model` and `head` are the run's config blocks (`configs/a2_*.yaml`);
    `preprocessing` sets the crop and silence mode applied before the backbone."""

    system = "a2_ssl_head"

    def __init__(
        self,
        head: SSLHead,
        model: dict[str, Any],
        head_config: dict[str, Any],
        preprocessing: dict[str, Any],
        threshold: float,
        model_version: str,
        device: str | torch.device = "auto",
        backbone: tuple[nn.Module, Any] | None = None,
    ) -> None:
        super().__init__(threshold, model_version)
        self.model = dict(model)
        self.head_config = dict(head_config)
        self.preprocessing = dict(preprocessing)
        self.sample_rate = int(self.preprocessing["sample_rate"])
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.head = head.to(self.device).eval()
        self._backbone = backbone

    # -- scoring --------------------------------------------------------------

    def score_pooled(self, pooled: np.ndarray) -> np.ndarray:
        """Head logits for pooled hidden states `[B, L, D]` (any float dtype) -> `[B]`."""
        x = torch.tensor(np.asarray(pooled).astype(np.float32), device=self.device)
        with torch.no_grad():
            logit = self.head(x).cpu().numpy()
        return logit.astype(np.float64)

    def backbone(self) -> tuple[nn.Module, Any]:
        """`(model, feature_extractor)`, loaded on first call and moved to `self.device`."""
        if self._backbone is None:
            model, feature_extractor = load_backbone(self.model["id"], self.model.get("revision"))
            self._backbone = (model.to(self.device), feature_extractor)
        return self._backbone

    def pooled_from_waveform(self, x: np.ndarray) -> np.ndarray:
        """Preprocess one clip and run the backbone: `[1, L, D]` fp16, as cached."""
        x = preprocess(
            x,
            self.sample_rate,
            silence_mode=self.preprocessing["silence_mode"],
            crop_seconds=self.preprocessing["crop_seconds"],
            crop_mode=self.preprocessing.get("crop_mode", "center"),
        )
        model, feature_extractor = self.backbone()
        return extract_pooled_hidden_states(
            model, feature_extractor, x[None, :], self.sample_rate, self.device
        )

    def score_waveform(self, x: np.ndarray) -> tuple[float, dict[str, Any]]:
        score = float(self.score_pooled(self.pooled_from_waveform(x))[0])
        return score, {"backbone": self.model["id"]}

    def layer_weights(self) -> list[float]:
        return [float(w) for w in self.head.layer_weights().detach().cpu()]

    # -- bundle ---------------------------------------------------------------

    def save(self, bundle_dir: str | Path) -> Path:
        bundle_dir = write_bundle_meta(
            bundle_dir,
            {
                "system": self.system,
                "model_version": self.model_version,
                "threshold": self.threshold,
                "model": self.model,
                "head": self.head_config,
                "preprocessing": self.preprocessing,
            },
        )
        torch.save(self.head.state_dict(), bundle_dir / _HEAD_FILE)
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: str | Path, device: str | torch.device = "auto") -> SslHeadDetector:
        bundle_dir = Path(bundle_dir)
        meta = read_bundle_meta(bundle_dir)
        head = head_from_config(meta["model"], meta["head"])
        head.load_state_dict(torch.load(bundle_dir / _HEAD_FILE, map_location="cpu"))
        return cls(
            head,
            model=meta["model"],
            head_config=meta["head"],
            preprocessing=meta["preprocessing"],
            threshold=meta["threshold"],
            model_version=meta["model_version"],
            device=device,
        )

    @classmethod
    def from_run(
        cls,
        run_dir: str | Path,
        silence_mode: str | None = None,
        device: str | torch.device = "auto",
    ) -> SslHeadDetector:
        """Build from a trained, scored A2 run (`runs/<id>/`: head weights, config, metrics)."""
        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        head = head_from_config(cfg["model"], cfg["head"])
        head.load_state_dict(torch.load(run_dir / _HEAD_FILE, map_location="cpu"))
        preprocessing = dict(cfg["preprocessing"])
        if silence_mode is not None:
            preprocessing["silence_mode"] = silence_mode
        return cls(
            head,
            model=cfg["model"],
            head_config=cfg["head"],
            preprocessing=preprocessing,
            threshold=dev_threshold_from_run(run_dir),
            model_version=run_dir.name,
            device=device,
        )
