"""The fusion model behind the Detector interface (system `fusion_xattn`).

Scores either cached frame streams (`score_streams`, used by the pipeline) or a raw clip
(`score_waveform`, used by applications): the clip is preprocessed once, both streams are
computed (`features.frames.clip_streams`) and the cross-attention model returns one logit.
The backbone loads lazily on first waveform scored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from audio_df.features.frames import LFCC_HOP_S, SSL_HOP_S, clip_streams
from audio_df.features.ssl import load_backbone
from audio_df.models.base import Detector, read_bundle_meta, run_threshold, write_bundle_meta
from audio_df.models.fusion import CrossAttentionFusion
from audio_df.runtime.device import resolve_device

__all__ = ["FusionDetector", "build_fusion_model"]

_MODEL_FILE = "model_state.pt"
_MIX_FILE = "layer_mix.json"


def build_fusion_model(cfg: dict[str, Any]) -> CrossAttentionFusion:
    """An untrained `CrossAttentionFusion` shaped by a fusion config (`configs/fusion_*.yaml`)."""
    return CrossAttentionFusion(
        lfcc_dim=60,
        ssl_dim=int(cfg["ssl"]["hidden_dim"]),
        lfcc_hop_s=LFCC_HOP_S,
        ssl_hop_s=SSL_HOP_S,
        **cfg["model"],
    )


class FusionDetector(Detector):
    system = "fusion_xattn"

    def __init__(
        self,
        model: CrossAttentionFusion,
        config: dict[str, Any],
        layer_weights: np.ndarray,
        threshold: float,
        model_version: str,
        device: str | torch.device = "auto",
        backbone: tuple[nn.Module, Any] | None = None,
    ) -> None:
        super().__init__(threshold, model_version)
        self.config = dict(config)
        self.preprocessing = dict(config["preprocessing"])
        self.sample_rate = int(self.preprocessing["sample_rate"])
        self.layer_weights = np.asarray(layer_weights, dtype=np.float32)
        self.device = resolve_device(device) if isinstance(device, str) else device
        self.model = model.to(self.device).eval()
        self._backbone = backbone

    def score_streams(self, lfcc: Any, ssl: Any) -> np.ndarray:
        """Logits for batched streams `lfcc [B, T1, 60]`, `ssl [B, T2, D]` -> `[B]`."""
        lfcc_t = torch.as_tensor(np.asarray(lfcc, dtype=np.float32), device=self.device)
        ssl_t = torch.as_tensor(np.asarray(ssl, dtype=np.float32), device=self.device)
        with torch.no_grad():
            logit = self.model(lfcc_t, ssl_t).cpu().numpy()
        return logit.astype(np.float64)

    def backbone(self) -> tuple[nn.Module, Any]:
        if self._backbone is None:
            ssl_cfg = self.config["ssl"]
            model, feature_extractor = load_backbone(ssl_cfg["id"], ssl_cfg.get("revision"))
            self._backbone = (model.to(self.device), feature_extractor)
        return self._backbone

    def score_waveform(self, x: np.ndarray) -> tuple[float, dict[str, Any]]:
        lfcc, ssl = clip_streams(x, self.backbone(), self.config, self.layer_weights, self.device)
        score = float(self.score_streams(lfcc[None], ssl[None])[0])
        return score, {"n_lfcc_frames": int(len(lfcc)), "n_ssl_frames": int(len(ssl))}

    # -- bundle ---------------------------------------------------------------

    def save(self, bundle_dir: str | Path) -> Path:
        bundle_dir = write_bundle_meta(
            bundle_dir,
            {
                "system": self.system,
                "model_version": self.model_version,
                "threshold": self.threshold,
                "config": self.config,
                "layer_weights": [float(w) for w in self.layer_weights],
            },
        )
        torch.save(self.model.state_dict(), bundle_dir / _MODEL_FILE)
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: str | Path, device: str | torch.device = "auto") -> FusionDetector:
        bundle_dir = Path(bundle_dir)
        meta = read_bundle_meta(bundle_dir)
        model = build_fusion_model(meta["config"])
        model.load_state_dict(torch.load(bundle_dir / _MODEL_FILE, map_location="cpu"))
        return cls(
            model,
            meta["config"],
            np.asarray(meta["layer_weights"], dtype=np.float32),
            threshold=meta["threshold"],
            model_version=meta["model_version"],
            device=device,
        )

    @classmethod
    def from_run(cls, run_dir: str | Path, device: str | torch.device = "auto") -> FusionDetector:
        """Build from a trained fusion run (`runs/<id>/`: weights, config, layer mix)."""
        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        model = build_fusion_model(cfg)
        model.load_state_dict(torch.load(run_dir / _MODEL_FILE, map_location="cpu"))
        mix = np.asarray(json.loads((run_dir / _MIX_FILE).read_text()), dtype=np.float32)
        return cls(
            model,
            cfg,
            mix,
            threshold=run_threshold(run_dir),
            model_version=run_dir.name,
            device=device,
        )
