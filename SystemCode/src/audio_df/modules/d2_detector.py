"""D2 detector: reuse D3's temporal cache, dataset, fitting and scoring."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from audio_df.helpers.training import TrainingConfig
from audio_df.modules.bidirectional import BidirectionalConfig
from audio_df.modules.d3_detector import D3Detector
from audio_df.modules.lfcc import LFCC, LFCCConfig
from audio_df.modules.unidirectional import CrossAttentionHead
from audio_df.modules.wavlm import WavLM, WavLMConfig


class D2Detector(D3Detector):
    def __init__(
        self, acoustic: LFCC, ssl: WavLM,
        head: CrossAttentionHead | None = None,
        config: BidirectionalConfig | None = None,
        train_config: TrainingConfig | None = None,
    ):
        if head is not None and config is not None:
            raise ValueError("Specify head or config, not both")
        if head is None:
            head = CrossAttentionHead(
                3 * acoustic.settings.n_filters, ssl.config.hidden_dim, config
            )
        super().__init__(acoustic, ssl, head=head, train_config=train_config)

    @classmethod
    def load(cls, directory: Path, device: str = "auto") -> "D2Detector":
        directory = Path(directory)
        meta = json.loads((directory / "model.json").read_text())
        acoustic = LFCC(config=LFCCConfig(**meta["acoustic"]))
        ssl = WavLM(WavLMConfig(**meta["ssl"]), device=device)
        config = BidirectionalConfig(**meta["fusion"])
        head = CrossAttentionHead(
            3 * acoustic.settings.n_filters, ssl.config.hidden_dim, config
        )
        head.load_state_dict(torch.load(
            directory / "head.pt", map_location="cpu", weights_only=True
        ))
        head.eval()
        return cls(acoustic, ssl, head=head,
                   train_config=TrainingConfig(**meta["training"]))
