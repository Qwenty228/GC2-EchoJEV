"""A2 head training: binary cross-entropy on pooled SSL states, checkpoint chosen by dev loss.

- **Checkpoint selection is by dev loss, not dev EER.** Dev only contains the training
  attacks A01-A06, so dev EER saturates near-perfect and stops being a useful early-stopping
  signal long before dev loss does.
- **Loss orientation.** `SSLHead.forward` returns a logit oriented so that higher = more
  bona fide. `train_head` trains against that orientation directly: target = `is_bonafide`
  (1 for bona fide, 0 for spoof), with `pos_weight = n_spoof / n_bonafide` in
  `nn.BCEWithLogitsLoss`. `pos_weight` only reweights the `target == 1` term, so putting
  bona fide (the minority class, about 1:8.8 in train) there is what corrects the imbalance;
  a `target = is_spoof` framing would reweight the majority class instead.
- **No metrics here.** Only losses; EER and friends come from `scripts/score_experiment.py`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = ["TrainResult", "align_cache_with_manifest", "train_head"]


def align_cache_with_manifest(
    cache: dict[str, np.ndarray], manifest_df: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Reorder `manifest_df` to match `cache["ids"]`'s order; raise on any missing id."""
    ids = [str(i) for i in cache["ids"]]
    indexed = manifest_df.set_index("utt_id")
    missing = sorted(set(ids) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"{len(missing)} cached id(s) not found in manifest, e.g. {missing[:5]}")
    indexed.index = indexed.index.astype(str)
    ordered = indexed.loc[ids].reset_index()
    return cache["pooled"], ordered


HeadT = TypeVar("HeadT", bound=nn.Module)


@dataclass
class TrainResult(Generic[HeadT]):
    head: HeadT
    best_epoch: int
    train_losses: list[float] = field(default_factory=list)
    dev_losses: list[float] = field(default_factory=list)
    pos_weight: float = 1.0


def train_head(
    head: HeadT,
    x_train: torch.Tensor,
    y_bonafide_train: torch.Tensor,
    x_dev: torch.Tensor,
    y_bonafide_dev: torch.Tensor,
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    pos_weight: float | None = None,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> TrainResult[HeadT]:
    """Train `head` with BCE, checkpoint-select by DEV LOSS (not dev EER — see module doc).

    `y_bonafide_*` must be 1.0 for bonafide, 0.0 for spoof (matching `SSLHead`'s
    higher-logit-means-bonafide orientation; see the module docstring for why this, and not
    a `target=is_spoof` framing, is what makes `pos_weight` correct class-imbalance
    reweighting rather than backwards reweighting).

    `pos_weight=None` (the config's `"auto"`) computes `n_spoof / n_bonafide` from
    `y_bonafide_train` itself. No metric (EER/AUC/etc.) is computed here — only losses.
    """
    if device is None:
        device = torch.device("cpu")
    if pos_weight is None:
        n_bonafide = float((y_bonafide_train == 1).sum())
        n_spoof = float((y_bonafide_train == 0).sum())
        pos_weight = n_spoof / max(n_bonafide, 1.0)

    head = head.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
    opt = torch.optim.Adam(head.parameters(), lr=lr)

    dataset = TensorDataset(x_train, y_bonafide_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    x_dev = x_dev.to(device)
    y_bonafide_dev = y_bonafide_dev.to(device)

    train_losses: list[float] = []
    dev_losses: list[float] = []
    best_dev_loss = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    epochs_since_improvement = 0

    for epoch in range(max_epochs):
        head.train()
        running, n_batches = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = head(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n_batches += 1
        train_losses.append(running / max(n_batches, 1))

        head.eval()
        with torch.no_grad():
            dev_loss = float(loss_fn(head(x_dev), y_bonafide_dev))
        dev_losses.append(dev_loss)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_state = copy.deepcopy(head.state_dict())
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                break

    head.load_state_dict(best_state)
    return TrainResult(
        head=head,
        best_epoch=best_epoch,
        train_losses=train_losses,
        dev_losses=dev_losses,
        pos_weight=float(pos_weight),
    )
