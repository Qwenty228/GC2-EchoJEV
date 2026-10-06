"""A2's head: learnable softmax layer weights -> LayerNorm -> projection -> MLP -> logit.

This is the only trainable part of System A2. The frozen SSL backbone (WavLM Base+ or
XLS-R 300M, see `audio_df.features.ssl`) is never touched here; this module only knows about
a `[batch, n_layers, hidden_dim]` tensor of per-layer, time-mean-pooled hidden states (the
pooled cache `ssl_cache.py` writes to disk) and turns it into one scalar logit per clip.

## The mean-pool / weighted-sum equivalence this head relies on

Let `H^(l) in R^{T x D}` be layer `l`'s full frame-level hidden states for one clip (`T`
frames, `D` dims), `w_l = softmax(a)_l` the learned layer weights, and `mean_t(.)` the
mean over the time axis. Both the weighted sum over layers and the mean over time are
linear operators, and they commute:

    sum_l w_l * mean_t(H^(l))  ==  mean_t(sum_l w_l * H^(l))

So caching only `mean_t(H^(l))` per layer (`[n_layers, hidden_dim]` per clip, `ssl_cache.py`'s
"pooled cache") and computing `z = sum_l w_l * pooled_l` here gives **exactly** the same `z`
as the conceptually simpler "combine all frames first, then pool" pipeline would — without
ever storing a frame-level tensor for the full corpus (`ssl-hidden-states.md`; and see
`ssl_cache.py`'s module docstring for the disk-size argument that makes this equivalence
load-bearing, not just a nice property). `tests/test_ssl_head.py` checks this numerically.

Everything *after* the weighted sum (`LayerNorm`, the linear projection, the MLP) is applied
to the already-pooled `[batch, D]` vector `z`, so it plays no part in the equivalence above
and is free to be nonlinear.

## Output orientation

`forward` returns one raw logit per clip, oriented so that a **higher logit means more
bona fide** (`sigmoid(logit) ~ P(bonafide)`), matching this project's ASVspoof/`metrics.py`
convention (`higher_is_spoof=False`) rather than `training.py`'s local
`higher_is_spoof=True` convention for its own generic `train_model` loop. `run_a2.py` does
not use `training.py`'s loop for this reason — see its own training-loop docstring for how
it maps this head's orientation onto `nn.BCEWithLogitsLoss`.

## Layer count is backbone-specific, never shared

WavLM Base+ has 13 hidden-state tensors (embedding output + 12 transformer layers, hidden
size 768); XLS-R 300M has 25 (embedding output + 24 layers, hidden size 1024) — see
`Miscellaneous/research/knowledge/methods/ssl-hidden-states.md`. `SSLHead(n_layers=...,
hidden_dim=...)` must be constructed with the values that match whichever cache it will be trained
on; there is deliberately no shared default and no cross-backbone weight sharing.

## Later (not implemented here)

A frame-level (per-time-step) alternative head — e.g. a small Transformer/CNN/ECAPA encoder
over the learned weighted-sum frame sequence — would need the *frame-level* weighted sum
(`sum_l w_l * H^(l)`, shape `[T, D]` per clip, ~37 GB fp16 for the whole corpus at D=768) to
be cached separately. That is explicitly out of scope for this module and is not built here;
`ssl_cache.py`'s module docstring repeats this note from the caller's side.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["SSLHead"]


class SSLHead(nn.Module):
    """Softmax-weighted layer sum -> LayerNorm(D) -> Linear(D, proj_dim) -> MLP -> logit.

    `n_layers` and `hidden_dim` must match the pooled cache this head is trained on exactly
    (13/768 for WavLM Base+, 25/1024 for XLS-R 300M — never shared between backbones).
    """

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        proj_dim: int = 256,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # One trainable scalar per layer; softmax turns it into a convex combination.
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))

        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, proj_dim)
        self.mlp = nn.Sequential(
            nn.Linear(proj_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def layer_weights(self) -> torch.Tensor:
        """`softmax(layer_logits)`, shape `[n_layers]` — exposed for plotting/inspection."""
        return torch.softmax(self.layer_logits, dim=0)

    def weighted_sum(self, pooled: torch.Tensor) -> torch.Tensor:
        """`sum_l w_l * pooled[:, l, :]` -> `[batch, hidden_dim]`, before LayerNorm/proj/MLP.

        Exposed separately from `forward` so the equivalence property (this module's
        docstring) can be tested directly against a frame-level reference without also
        depending on the (nonlinear) LayerNorm/MLP stages.
        """
        if pooled.shape[-2:] != (self.n_layers, self.hidden_dim):
            raise ValueError(
                f"expected pooled hidden states of shape [..., {self.n_layers}, "
                f"{self.hidden_dim}], got {tuple(pooled.shape)}"
            )
        w = self.layer_weights()
        return torch.einsum("l,...ld->...d", w, pooled)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """`pooled`: `[batch, n_layers, hidden_dim]` -> one logit per clip, `[batch]`.

        Higher output means more bona fide (see module docstring).
        """
        z = self.weighted_sum(pooled)
        z = self.norm(z)
        h = self.proj(z)
        logit = self.mlp(h).squeeze(-1)
        return logit
