"""SSLHead: shapes, layer weights, the pooled-cache equivalence, and output orientation."""

from __future__ import annotations

import numpy as np
import torch

from audio_df.models.ssl_head import SSLHead


def test_forward_shape():
    head = SSLHead(n_layers=5, hidden_dim=8, proj_dim=16, mlp_hidden=12)
    pooled = torch.randn(3, 5, 8)
    out = head(pooled)
    assert out.shape == (3,)


def test_layer_weights_are_a_softmax_and_sum_to_one():
    head = SSLHead(n_layers=4, hidden_dim=6)
    with torch.no_grad():
        head.layer_logits.copy_(torch.tensor([0.0, 1.0, 2.0, -1.0]))
    w = head.layer_weights()
    expected = torch.softmax(torch.tensor([0.0, 1.0, 2.0, -1.0]), dim=0)
    torch.testing.assert_close(w, expected)
    assert abs(float(w.sum()) - 1.0) < 1e-6


def test_weighted_sum_matches_frame_level_weighted_sum_then_mean_pool():
    """The equivalence the whole pooled-cache design rests on, from the head's own side:
    head.weighted_sum(mean_t(H)) == mean_t(sum_l w_l H^(l)) for the SAME learned weights."""
    torch.manual_seed(0)
    n_layers, hidden_dim, batch, t = 7, 10, 4, 23
    head = SSLHead(n_layers=n_layers, hidden_dim=hidden_dim)
    with torch.no_grad():
        head.layer_logits.copy_(torch.randn(n_layers))

    frames = torch.randn(batch, n_layers, t, hidden_dim)  # [B, L, T, D]
    pooled = frames.mean(dim=2)  # [B, L, D] -- what ssl_cache.py actually caches

    candidate = head.weighted_sum(pooled)  # [B, D]

    w = head.layer_weights()
    reference = torch.einsum("l,bltd->btd", w, frames).mean(dim=1)  # [B, D]

    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-5)


def test_wrong_shape_raises():
    head = SSLHead(n_layers=5, hidden_dim=8)
    bad = torch.randn(3, 4, 8)  # wrong n_layers
    try:
        head(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for mismatched pooled shape")


def test_training_reduces_dev_loss_and_orients_higher_is_bonafide():
    """Synthetic, linearly separable pooled cache: bonafide clips have a larger mean in one
    layer's features than spoof clips. A few epochs of gradient descent on BCE (with the
    sign convention training/head.py uses: target=is_spoof, input=-logit) should push mean logits
    for bonafide clips above mean logits for spoof clips, and reduce loss."""
    torch.manual_seed(0)
    n_layers, hidden_dim = 3, 6
    n_per_class = 64

    rng = np.random.default_rng(1)
    bonafide = rng.normal(loc=1.0, scale=0.3, size=(n_per_class, n_layers, hidden_dim))
    spoof = rng.normal(loc=-1.0, scale=0.3, size=(n_per_class, n_layers, hidden_dim))
    pooled = torch.tensor(np.concatenate([bonafide, spoof], axis=0), dtype=torch.float32)
    is_spoof = torch.tensor([0.0] * n_per_class + [1.0] * n_per_class)

    head = SSLHead(
        n_layers=n_layers, hidden_dim=hidden_dim, proj_dim=16, mlp_hidden=12, dropout=0.0
    )
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def loss_at_current_params() -> float:
        head.eval()
        with torch.no_grad():
            logit = head(pooled)
            return float(loss_fn(-logit, is_spoof))

    initial_loss = loss_at_current_params()

    head.train()
    for _ in range(200):
        opt.zero_grad()
        logit = head(pooled)
        loss = loss_fn(-logit, is_spoof)
        loss.backward()
        opt.step()

    final_loss = loss_at_current_params()
    assert final_loss < initial_loss

    head.eval()
    with torch.no_grad():
        logit = head(pooled)
    mean_logit_bonafide = float(logit[:n_per_class].mean())
    mean_logit_spoof = float(logit[n_per_class:].mean())
    assert mean_logit_bonafide > mean_logit_spoof  # higher logit -> more bona fide
