"""The main model (scaffold): bidirectional cross-attention between the A1 and A2 streams.

    lfcc [B, T_lfcc, 60]  --LayerNorm, Linear-->  [B, T_lfcc, d] + time(t_lfcc)
    ssl  [B, T_ssl, 768]  --LayerNorm, Linear-->  [B, T_ssl, d]  + time(t_ssl)
    N x FusionBlock:
        ssl  <- ssl  + CrossAttn(query=ssl,  key/value=lfcc) ; + FFN
        lfcc <- lfcc + CrossAttn(query=lfcc, key/value=ssl)  ; + FFN
    masked mean over time of each stream -> concat [2d] -> MLP -> one logit

The logit is oriented like every score in the project: **higher = more bona fide**.

**One clock.** The streams run at different frame rates (LFCC every 10 ms, WavLM every
20 ms). Both get a sinusoidal encoding of each frame's centre time **in seconds**
(`time_encoding`), so attention can match frames by when they happen, not by index.

**Input scale.** LFCC c0 reaches about -70 on digital silence while other coefficients are
O(1); a LayerNorm over the 60 input dimensions comes before the projection.

This is a starting point, not a tuned design. Extension points for the team: the block
(`FusionBlock`), the pooling (`_pool`), and `forward(..., return_attention=True)`, which
returns every block's attention maps for explainability.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

__all__ = ["CrossAttentionFusion", "FusionBlock", "time_encoding"]


def time_encoding(times_s: torch.Tensor, d_model: int, max_period_s: float = 10.0) -> torch.Tensor:
    """Sinusoidal encoding of frame times in seconds: `[..., T]` -> `[..., T, d_model]`.

    Wavelengths run geometrically from 20 ms to `max_period_s`, so the encoding resolves
    single SSL frames and still distinguishes positions across a whole clip.
    """
    if d_model % 2:
        raise ValueError("d_model must be even for the sin/cos time encoding")
    half = d_model // 2
    min_period_s = 0.02
    periods = min_period_s * (max_period_s / min_period_s) ** (
        torch.arange(half, device=times_s.device, dtype=torch.float32) / max(half - 1, 1)
    )
    angles = 2 * math.pi * times_s.unsqueeze(-1).float() / periods
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class FusionBlock(nn.Module):
    """One round of bidirectional cross-attention, pre-LayerNorm, with residuals and FFNs."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm_ssl_q = nn.LayerNorm(d_model)
        self.norm_lfcc_kv = nn.LayerNorm(d_model)
        self.ssl_from_lfcc = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.norm_lfcc_q = nn.LayerNorm(d_model)
        self.norm_ssl_kv = nn.LayerNorm(d_model)
        self.lfcc_from_ssl = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.ffn_ssl = _ffn(d_model, dropout)
        self.ffn_lfcc = _ffn(d_model, dropout)

    def forward(
        self,
        lfcc: torch.Tensor,
        ssl: torch.Tensor,
        lfcc_pad: torch.Tensor | None,
        ssl_pad: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        kv = self.norm_lfcc_kv(lfcc)
        upd, attn_s2l = self.ssl_from_lfcc(
            self.norm_ssl_q(ssl), kv, kv, key_padding_mask=lfcc_pad, need_weights=True
        )
        ssl = ssl + upd
        ssl = ssl + self.ffn_ssl(ssl)

        kv = self.norm_ssl_kv(ssl)
        upd, attn_l2s = self.lfcc_from_ssl(
            self.norm_lfcc_q(lfcc), kv, kv, key_padding_mask=ssl_pad, need_weights=True
        )
        lfcc = lfcc + upd
        lfcc = lfcc + self.ffn_lfcc(lfcc)
        return lfcc, ssl, {"ssl_to_lfcc": attn_s2l, "lfcc_to_ssl": attn_l2s}


def _ffn(d_model: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, 4 * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(4 * d_model, d_model),
        nn.Dropout(dropout),
    )


class CrossAttentionFusion(nn.Module):
    """LFCC x SSL cross-attention detector head. See the module docstring for the shape."""

    def __init__(
        self,
        lfcc_dim: int = 60,
        ssl_dim: int = 768,
        d_model: int = 128,
        n_heads: int = 4,
        n_blocks: int = 2,
        mlp_hidden: int = 64,
        dropout: float = 0.1,
        lfcc_hop_s: float = 0.010,
        ssl_hop_s: float = 0.020,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.lfcc_hop_s = lfcc_hop_s
        self.ssl_hop_s = ssl_hop_s
        self.lfcc_in = nn.Sequential(nn.LayerNorm(lfcc_dim), nn.Linear(lfcc_dim, d_model))
        self.ssl_in = nn.Sequential(nn.LayerNorm(ssl_dim), nn.Linear(ssl_dim, d_model))
        self.blocks = nn.ModuleList(FusionBlock(d_model, n_heads, dropout) for _ in range(n_blocks))
        self.out_norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def _times(self, n: int, hop_s: float, device: torch.device) -> torch.Tensor:
        return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) * hop_s

    @staticmethod
    def _pool(x: torch.Tensor, pad: torch.Tensor | None) -> torch.Tensor:
        """Mean over valid (non-padded) frames: `[B, T, d]` -> `[B, d]`."""
        if pad is None:
            return x.mean(dim=1)
        keep = (~pad).unsqueeze(-1).to(x.dtype)
        return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        lfcc: torch.Tensor,
        ssl: torch.Tensor,
        lfcc_pad: torch.Tensor | None = None,
        ssl_pad: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> Any:
        """`lfcc [B, T1, 60]`, `ssl [B, T2, D]`; `*_pad` are `[B, T]` bool masks, True on
        padding. Returns logits `[B]`, plus a list of per-block attention maps when
        `return_attention`."""
        h_l = self.lfcc_in(lfcc) + time_encoding(
            self._times(lfcc.shape[1], self.lfcc_hop_s, lfcc.device), self.d_model
        )
        h_s = self.ssl_in(ssl) + time_encoding(
            self._times(ssl.shape[1], self.ssl_hop_s, ssl.device), self.d_model
        )
        maps = []
        for block in self.blocks:
            h_l, h_s, attn = block(h_l, h_s, lfcc_pad, ssl_pad)
            maps.append(attn)
        z = self.out_norm(torch.cat([self._pool(h_l, lfcc_pad), self._pool(h_s, ssl_pad)], -1))
        logit = self.head(z).squeeze(-1)
        return (logit, maps) if return_attention else logit
