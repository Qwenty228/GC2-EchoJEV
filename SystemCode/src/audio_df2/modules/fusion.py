"""D1 fusion head: learned WavLM layer mixture + pooled LFCC concatenation."""

import torch
from torch import nn

from audio_df2.helpers.config import load_config
from audio_df2.modules.neural import NeuralConfig


class ConcatHead(nn.Module):
    """Inputs: LFCC [B,F], WavLM pooled states [B,L,D]; output: bona fide logit."""

    def __init__(self, n_layers: int, hidden_dim: int, lfcc_dim: int,
                 config: NeuralConfig | None = None):
        super().__init__()
        self.config = config if config is not None else load_config(NeuralConfig)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(hidden_dim + lfcc_dim)
        self.proj = nn.Linear(hidden_dim + lfcc_dim, self.config.proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.config.proj_dim, self.config.mlp_hidden),
            nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(self.config.mlp_hidden, 1),
        )

    def layer_weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(dim=0)

    def forward(self, lfcc: torch.Tensor, ssl: torch.Tensor) -> torch.Tensor:
        if ssl.shape[0] != lfcc.shape[0] or ssl.shape[1] != self.layer_logits.numel():
            raise ValueError("Incompatible LFCC/WavLM batch or layer count")
        pooled_ssl = torch.einsum("l,bld->bd", self.layer_weights(), ssl)
        combined = torch.cat((lfcc, pooled_ssl), dim=-1)
        return self.mlp(self.proj(self.norm(combined))).squeeze(-1)
