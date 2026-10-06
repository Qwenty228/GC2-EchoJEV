"""audio_df.evaluation.xai_layers: single-layer heads, leave-one-out, LR probe, table builders.

Tiny synthetic arrays only; no real data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from audio_df.evaluation import xai_layers as xl
from audio_df.evaluation.metrics import compute_eer
from audio_df.models.ssl_head import SSLHead

HEAD_CFG = {"proj_dim": 8, "mlp_hidden": 4, "dropout": 0.0, "lr": 1e-2, "batch_size": 32,
            "max_epochs": 30, "patience": 5, "pos_weight": "auto"}
L, D = 4, 6


def _data(n=200, informative=2, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.float32)  # 1 = bonafide
    x = rng.normal(size=(n, L, D)).astype(np.float32)
    direction = np.linspace(-1.0, 1.0, D, dtype=np.float32)  # LayerNorm removes constant shifts
    x[:, informative, :] += (y[:, None] * 3.0 - 1.0) * direction
    return x.astype(np.float16), y


def test_loo_head_zeroes_layer_and_renormalises():
    head = SSLHead(L, D, 8, 4, 0.0)
    with torch.no_grad():
        head.layer_logits.copy_(torch.tensor([0.5, -0.2, 1.0, 0.1]))
    loo = xl.leave_one_layer_out(head, 2)
    w = loo.layer_weights().detach().numpy()
    assert w[2] == 0.0 and w.sum() == pytest.approx(1.0)
    rest = torch.softmax(torch.tensor([0.5, -0.2, 0.1]), 0).numpy()
    np.testing.assert_allclose(w[[0, 1, 3]], rest, rtol=1e-5)
    # original untouched
    assert head.layer_weights()[2] > 0.3


def test_loo_output_ignores_removed_layer():
    head = SSLHead(L, D, 8, 4, 0.0).eval()
    loo = xl.leave_one_layer_out(head, 1)
    x, _ = _data()
    a = xl.score_head(loo, x)
    x2 = x.copy()
    x2[:, 1, :] = 7.0
    b = xl.score_head(loo, x2)
    np.testing.assert_allclose(a, b, atol=1e-6)


def test_score_head_is_batch_invariant_and_layer_slices():
    head = SSLHead(1, D, 8, 4, 0.0).eval()
    x, _ = _data()
    a = xl.score_head(head, x, layer=3, batch_size=17)
    b = xl.score_head(head, x, layer=3, batch_size=500)
    np.testing.assert_allclose(a, b, atol=1e-6)
    direct = head(torch.tensor(x[:, 3:4].astype(np.float32))).detach().numpy()
    np.testing.assert_allclose(a, direct, atol=1e-6)


def test_train_single_layer_head_prefers_informative_layer():
    x, y = _data(seed=1)
    xd, yd = _data(seed=2)
    xt, yt = _data(n=400, seed=3)
    eers = {}
    for layer in (0, 2):
        head = xl.train_layer_head(x, y, xd, yd, layer=layer, seed=5, head_cfg=HEAD_CFG,
                                   n_layers=L, hidden_dim=D)
        s = xl.score_head(head, xt, layer=layer)
        lab = np.where(yt == 1, "bonafide", "spoof")
        eers[layer], _ = compute_eer(s, lab, higher_is_spoof=False)
    assert eers[2] < 0.1 < eers[0]


def test_train_is_deterministic_for_seed():
    x, y = _data(seed=1)
    xd, yd = _data(seed=2)
    kw = dict(layer=None, seed=9, head_cfg=HEAD_CFG, n_layers=L, hidden_dim=D)
    h1 = xl.train_layer_head(x, y, xd, yd, **kw)
    h2 = xl.train_layer_head(x, y, xd, yd, **kw)
    np.testing.assert_array_equal(xl.score_head(h1, xd), xl.score_head(h2, xd))


def test_lr_probe_scores_orientation():
    x, y = _data(seed=1)
    xt, yt = _data(n=300, seed=3)
    probe = xl.fit_lr_probe(x, y, layer=2)
    s = probe(xt)
    assert s[yt == 1].mean() > s[yt == 0].mean()


def test_per_attack_table_and_summary_row():
    metrics = {"testsets": {
        "dev": {"eer": 0.01, "roc_auc": 0.99, "per_attack_eer": {"per_attack": {"A01": 0.0}}},
        "eval": {"eer": 0.1, "per_attack_eer": {"per_attack": {"A07": 0.2, "A08": 0.1}}},
        "eval_trimmed": {"eer": 0.2, "per_attack_eer": {"per_attack": {"A07": 0.3, "A08": 0.2}}},
        "itw": {"eer": 0.15, "roc_auc": 0.9, "per_attack_eer": None},
    }}
    row = xl.summary_row("v", metrics, bonafide_rejected=0.05)
    assert row["itw_eer"] == 0.15 and row["itw_bonafide_rejected"] == 0.05
    tab = xl.per_attack_table({"v1": metrics, "v2": metrics}, "eval")
    assert list(tab.index) == ["A07", "A08"] and list(tab.columns) == ["v1", "v2"]
    assert tab.loc["A07", "v1"] == 0.2
    md = xl.to_markdown(tab, pct=True)
    assert "20.00" in md
