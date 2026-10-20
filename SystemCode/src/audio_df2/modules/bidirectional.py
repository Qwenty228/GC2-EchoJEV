"""Reusable compressed-temporal bidirectional LFCC/WavLM fusion head.

Both streams represent equally spaced bins spanning the same center-cropped audio.
Attention weights are diagnostics, not causal explanations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BidirectionalConfig:
    tokens: int = 32
    width: int = 128
    heads: int = 4
    dropout: float = 0.1
    blocks: int = 2
    crop_seconds: float = 4.0


def time_encoding(times_s: torch.Tensor, width: int, max_period_s: float = 10.0) -> torch.Tensor:
    """Encode actual bin-center timestamps in seconds, not raw frame indices."""
    if width % 2:
        raise ValueError("Attention width must be even")
    half = width // 2
    periods = 0.02 * (max_period_s / 0.02) ** (
        torch.arange(half, device=times_s.device, dtype=torch.float32) / max(half - 1, 1)
    )
    phase = 2 * math.pi * times_s.float().unsqueeze(-1) / periods
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


def _ffn(width: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                         nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(4 * width, width), nn.Dropout(dropout))


class FusionBlock(nn.Module):
    """Sequential SSL<-LFCC then LFCC<-updated SSL, as in the original fusion."""

    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.ssl_q = nn.LayerNorm(width)
        self.lfcc_kv = nn.LayerNorm(width)
        self.ssl_from_lfcc = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.lfcc_q = nn.LayerNorm(width)
        self.ssl_kv = nn.LayerNorm(width)
        self.lfcc_from_ssl = nn.MultiheadAttention(width, heads, dropout, batch_first=True)
        self.ssl_ffn = _ffn(width, dropout)
        self.lfcc_ffn = _ffn(width, dropout)

    def forward(self, lfcc: torch.Tensor, ssl: torch.Tensor,
                lfcc_pad: torch.Tensor | None = None,
                ssl_pad: torch.Tensor | None = None,
                return_attention: bool = False):
        kv = self.lfcc_kv(lfcc)
        delta, s2l = self.ssl_from_lfcc(self.ssl_q(ssl), kv, kv,
                                        key_padding_mask=lfcc_pad,
                                        need_weights=return_attention)
        ssl = ssl + delta
        ssl = ssl + self.ssl_ffn(ssl)
        kv = self.ssl_kv(ssl)
        delta, l2s = self.lfcc_from_ssl(self.lfcc_q(lfcc), kv, kv,
                                        key_padding_mask=ssl_pad,
                                        need_weights=return_attention)
        lfcc = lfcc + delta
        lfcc = lfcc + self.lfcc_ffn(lfcc)
        return lfcc, ssl, {"ssl_to_lfcc": s2l, "lfcc_to_ssl": l2s} if return_attention else None


class BidirectionalHead(nn.Module):
    """Two temporally aligned token streams -> bona fide logit (higher = genuine)."""

    def __init__(self, lfcc_dim: int, ssl_dim: int, config: BidirectionalConfig | None = None):
        super().__init__()
        self.config = config or BidirectionalConfig()
        c = self.config
        if c.width % c.heads or c.width % 2 or c.blocks < 1 or c.tokens < 1:
            raise ValueError("Invalid attention width, heads, blocks or tokens")
        self.lfcc_proj = nn.Sequential(nn.LayerNorm(lfcc_dim), nn.Linear(lfcc_dim, c.width))
        self.ssl_proj = nn.Sequential(nn.LayerNorm(ssl_dim), nn.Linear(ssl_dim, c.width))
        self.blocks = nn.ModuleList(FusionBlock(c.width, c.heads, c.dropout)
                                    for _ in range(c.blocks))
        self.classifier = nn.Sequential(nn.LayerNorm(2 * c.width),
                                        nn.Linear(2 * c.width, c.width), nn.GELU(),
                                        nn.Dropout(c.dropout), nn.Linear(c.width, 1))

    @staticmethod
    def _pool(x: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        if pad is None:
            return x.mean(dim=1)
        keep = (~pad).unsqueeze(-1).to(x.dtype)
        return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1)

    def forward(self, lfcc: torch.Tensor, ssl: torch.Tensor,
                lfcc_pad: torch.Tensor | None = None,
                ssl_pad: torch.Tensor | None = None,
                return_attention: bool = False):
        if lfcc.ndim != 3 or ssl.ndim != 3:
            raise ValueError("Expected [batch, time, features] for both streams")
        # Adaptive temporal pooling in temporal_cache.py makes uniform bins over
        # the same 4-second crop. Use the center of each bin in seconds.
        def centers(length: int, x: torch.Tensor) -> torch.Tensor:
            return (torch.arange(length, device=x.device, dtype=torch.float32) + 0.5) * (
                self.config.crop_seconds / length)

        a = self.lfcc_proj(lfcc) + time_encoding(centers(lfcc.shape[1], lfcc), self.config.width)
        s = self.ssl_proj(ssl) + time_encoding(centers(ssl.shape[1], ssl), self.config.width)
        maps = []
        for block in self.blocks:
            a, s, attn = block(a, s, lfcc_pad, ssl_pad, return_attention)
            if return_attention:
                maps.append(attn)
        z = torch.cat((self._pool(a, lfcc_pad), self._pool(s, ssl_pad)), dim=-1)
        score = self.classifier(z).squeeze(-1)
        return (score, maps) if return_attention else score
