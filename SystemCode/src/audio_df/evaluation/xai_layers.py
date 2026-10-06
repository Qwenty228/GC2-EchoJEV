"""X6: which WavLM layers carry the evidence? Single-layer heads, leave-one-layer-out, LR probe.

Single-layer head, the least invasive way: `SSLHead(n_layers=1, ...)` fed `x[:, l:l+1, :]`.
Its softmax over one logit is exactly 1.0, so the head is the standard A2 head with the
layer mixture fixed to a one-hot; nothing in `ssl_head.py` or `train_head` changes.

Leave-one-layer-out acts on an already-trained 13-layer head: that layer's logit is set to
-inf, so softmax gives it weight 0 and renormalises the rest. No retraining.

All training goes through `run_a2.train_head`. Caches stay fp16 in memory and are converted
to float32 per batch (`score_head`) or per training set (`train_layer_head`).
Metrics are NOT computed here; callers use `audio_df.evaluation.scoring` (this module only builds
tables from its output).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch

from audio_df.models.ssl_head import SSLHead
from audio_df.pipelines.run_a2 import train_head
from audio_df.runtime.seeding import make_generator

__all__ = [
    "leave_one_layer_out",
    "score_head",
    "train_layer_head",
    "fit_lr_probe",
    "summary_row",
    "per_attack_table",
    "to_markdown",
]


def leave_one_layer_out(head: SSLHead, layer: int) -> SSLHead:
    """Copy of `head` whose layer `layer` has weight exactly 0 (others renormalised)."""
    out = copy.deepcopy(head)
    with torch.no_grad():
        out.layer_logits[layer] = float("-inf")
    return out.eval()


def score_head(
    head: SSLHead, x: np.ndarray, *, layer: int | None = None, batch_size: int = 4096
) -> np.ndarray:
    """Logits (higher = more bona fide) on CPU. `layer` slices `x[:, layer:layer+1]` first."""
    head = head.to("cpu").eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size]
            if layer is not None:
                xb = xb[:, layer : layer + 1]
            out.append(head(torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32))).numpy())
    return np.concatenate(out).astype(np.float64)


def train_layer_head(
    x_train: np.ndarray,
    y_bonafide_train: np.ndarray,
    x_dev: np.ndarray,
    y_bonafide_dev: np.ndarray,
    *,
    layer: int | None,
    seed: int,
    head_cfg: dict[str, Any],
    n_layers: int,
    hidden_dim: int,
) -> SSLHead:
    """Train the A2 head on all layers (`layer=None`) or on one layer, same recipe as A2.

    Seeding order mirrors `run_a2.cmd_train` (global seed, then head init, then loader
    generator from the same seed), so `seed=1337, layer=None` reproduces the E1 recipe.
    """
    torch.manual_seed(seed)
    head = SSLHead(
        n_layers=n_layers if layer is None else 1,
        hidden_dim=hidden_dim,
        proj_dim=head_cfg["proj_dim"],
        mlp_hidden=head_cfg["mlp_hidden"],
        dropout=head_cfg["dropout"],
    )
    pw = head_cfg["pos_weight"]

    def sl(a: np.ndarray) -> torch.Tensor:
        a = a if layer is None else a[:, layer : layer + 1]
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))

    result = train_head(
        head,
        sl(x_train),
        torch.from_numpy(np.asarray(y_bonafide_train, dtype=np.float32)),
        sl(x_dev),
        torch.from_numpy(np.asarray(y_bonafide_dev, dtype=np.float32)),
        lr=head_cfg["lr"],
        batch_size=head_cfg["batch_size"],
        max_epochs=head_cfg["max_epochs"],
        patience=head_cfg["patience"],
        pos_weight=None if pw == "auto" else float(pw),
        device=torch.device("cpu"),
        generator=make_generator(seed),
    )
    return result.head.eval()


def fit_lr_probe(
    x_train: np.ndarray, y_bonafide_train: np.ndarray, *, layer: int, max_iter: int = 1000
) -> Callable[[np.ndarray], np.ndarray]:
    """Standardise + sklearn LogisticRegression (default C) on one layer; returns a scorer.

    Scorer returns the decision function (higher = more bona fide, target = bona fide = 1).
    `max_iter` is raised from sklearn's 100 only so the default lbfgs converges.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    xl = np.asarray(x_train[:, layer, :], dtype=np.float32)
    scaler = StandardScaler().fit(xl)
    clf = LogisticRegression(max_iter=max_iter).fit(scaler.transform(xl), y_bonafide_train)

    def scorer(x: np.ndarray) -> np.ndarray:
        z = scaler.transform(np.asarray(x[:, layer, :], dtype=np.float32))
        return clf.decision_function(z).astype(np.float64)

    return scorer


def summary_row(name: str, metrics: dict[str, Any], bonafide_rejected: float) -> dict[str, Any]:
    ts = metrics["testsets"]
    return {
        "variant": name,
        "dev_eer": ts["dev"]["eer"],
        "eval_eer": ts["eval"]["eer"],
        "eval_trimmed_eer": ts["eval_trimmed"]["eer"],
        "itw_eer": ts["itw"]["eer"],
        "itw_auc": ts["itw"]["roc_auc"],
        "itw_bonafide_rejected": bonafide_rejected,
    }


def per_attack_table(metrics_by_variant: dict[str, dict], testset: str) -> pd.DataFrame:
    """Rows = attacks, columns = variants, values = per-attack EER (fraction)."""
    cols = {
        v: m["testsets"][testset]["per_attack_eer"]["per_attack"]
        for v, m in metrics_by_variant.items()
    }
    return pd.DataFrame(cols).sort_index()


def to_markdown(df: pd.DataFrame, *, pct: bool = False, index_name: str = "") -> str:
    """Plain markdown table; `pct` renders fractions as percent with 2 decimals."""
    cols = list(df.columns)
    label = index_name or str(df.index.name or "")
    head = "| " + label + " | " + " | ".join(map(str, cols)) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{v * 100:.2f}" if pct else f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
