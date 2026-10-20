"""D2 one-way LFCC-query/WavLM-key-value temporal fusion.

Matches D3's token count, projections, time encoding, depth and classifier;
only the SSL->LFCC update is removed. WavLM features remain frozen.
"""
from __future__ import annotations

import torch
from torch import nn

from audio_df.modules.bidirectional import BidirectionalConfig, BidirectionalHead, _ffn


class CrossAttentionBlock(nn.Module):
    """Update LFCC tokens using SSL tokens; leave SSL tokens unchanged."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.lfcc_q = nn.LayerNorm(width)
        self.ssl_kv = nn.LayerNorm(width)
        self.lfcc_from_ssl = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.lfcc_ffn = _ffn(width, dropout)

    def forward(
        self,
        lfcc: torch.Tensor,
        ssl: torch.Tensor,
        lfcc_pad: torch.Tensor | None = None,
        ssl_pad: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        kv = self.ssl_kv(ssl)
        delta, weights = self.lfcc_from_ssl(
            self.lfcc_q(lfcc), kv, kv,
            key_padding_mask=ssl_pad,
            need_weights=return_attention,
        )
        lfcc = lfcc + delta
        lfcc = lfcc + self.lfcc_ffn(lfcc)
        return lfcc, ssl, {"lfcc_to_ssl": weights} if return_attention else None


class CrossAttentionHead(BidirectionalHead):
    """D3-compatible forward interface, with only LFCC attending to SSL."""

    def __init__(
        self, lfcc_dim: int, ssl_dim: int,
        config: BidirectionalConfig | None = None,
    ):
        super().__init__(lfcc_dim, ssl_dim, config)
        c = self.config
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(c.width, c.heads, c.dropout)
            for _ in range(c.blocks)
        )
