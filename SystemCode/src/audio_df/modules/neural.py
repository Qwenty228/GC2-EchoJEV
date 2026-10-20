"""Trainable WavLM layer-weighted binary classification head."""
from dataclasses import dataclass
import torch
from torch import nn
from audio_df.helpers.config import load_config

@dataclass(frozen=True)
class NeuralConfig:
    proj_dim: int
    mlp_hidden: int
    dropout: float

class NeuralHead(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int, config: NeuralConfig | None = None):
        super().__init__()
        self.config = config if config is not None else load_config(NeuralConfig)
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, self.config.proj_dim)
        self.mlp = nn.Sequential(nn.Linear(self.config.proj_dim, self.config.mlp_hidden),
                                 nn.GELU(), nn.Dropout(self.config.dropout),
                                 nn.Linear(self.config.mlp_hidden, 1))

    def layer_weights(self):
        return torch.softmax(self.layer_logits, dim=0)

    def forward(self, pooled):
        z = torch.einsum('l,bld->bd', self.layer_weights(), pooled)
        return self.mlp(self.proj(self.norm(z))).squeeze(-1)
