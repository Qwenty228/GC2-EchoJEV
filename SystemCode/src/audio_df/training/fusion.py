"""Train the fusion model by streaming a frame cache: BCE, checkpoint chosen by dev loss.

Same choices as A2's head (`training.head`): target 1 for bona fide, `pos_weight =
n_spoof / n_bonafide` on that minority class, checkpoint by dev loss because dev EER
saturates on the training attacks. The difference is scale: the frame cache does not fit
in memory, so batches stream chunk by chunk (`features.frames.FrameCacheReader`), with chunk
and within-chunk order reshuffled every epoch from a seeded generator.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from audio_df.features.frames import FrameCacheReader

__all__ = ["FusionTrainResult", "predict_cache", "train_fusion"]


@dataclass
class FusionTrainResult:
    model: nn.Module
    best_epoch: int
    train_losses: list[float] = field(default_factory=list)
    dev_losses: list[float] = field(default_factory=list)
    pos_weight: float = 1.0


def _targets(ids: list[str], is_bonafide: dict[str, float], device: torch.device) -> torch.Tensor:
    return torch.tensor([is_bonafide[i] for i in ids], dtype=torch.float32, device=device)


def _mean_loss(
    model: nn.Module,
    reader: FrameCacheReader,
    is_bonafide: dict[str, float],
    loss_fn: nn.Module,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for ids, lfcc, ssl in reader.iter_batches(batch_size):
            logit = model(lfcc.to(device), ssl.to(device))
            total += float(loss_fn(logit, _targets(ids, is_bonafide, device))) * len(ids)
            n += len(ids)
    return total / max(n, 1)


def train_fusion(
    model: nn.Module,
    train_reader: FrameCacheReader,
    train_is_bonafide: dict[str, float],
    dev_reader: FrameCacheReader,
    dev_is_bonafide: dict[str, float],
    *,
    lr: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    pos_weight: float | None = None,
    device: torch.device | None = None,
    seed: int = 1337,
) -> FusionTrainResult:
    """Train `model` on the train cache; keep the weights with the lowest dev loss.

    `*_is_bonafide` map clip id -> 1.0 (bona fide) or 0.0 (spoof). `pos_weight=None`
    computes `n_spoof / n_bonafide` from the training labels.
    """
    device = device or torch.device("cpu")
    if pos_weight is None:
        labels = np.asarray([train_is_bonafide[i] for i in train_reader.ids()])
        pos_weight = float((labels == 0).sum()) / max(float((labels == 1).sum()), 1.0)

    torch.manual_seed(seed)
    model = model.to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    train_losses: list[float] = []
    dev_losses: list[float] = []
    best_dev, best_epoch, since_best = float("inf"), 0, 0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(max_epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        running, n = 0.0, 0
        for ids, lfcc, ssl in train_reader.iter_batches(batch_size, shuffle=True, rng=rng):
            opt.zero_grad()
            loss = loss_fn(
                model(lfcc.to(device), ssl.to(device)), _targets(ids, train_is_bonafide, device)
            )
            loss.backward()
            opt.step()
            running += float(loss.detach()) * len(ids)
            n += len(ids)
        train_losses.append(running / max(n, 1))

        dev_loss = _mean_loss(model, dev_reader, dev_is_bonafide, loss_fn, batch_size, device)
        dev_losses.append(dev_loss)
        if dev_loss < best_dev:
            best_dev, best_epoch, since_best = dev_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            since_best += 1
            if since_best >= patience:
                break

    model.load_state_dict(best_state)
    return FusionTrainResult(model, best_epoch, train_losses, dev_losses, float(pos_weight))


def predict_cache(
    model: nn.Module, reader: FrameCacheReader, batch_size: int, device: torch.device
) -> tuple[list[str], np.ndarray]:
    """Logits for every cached clip, in cache order."""
    model = model.to(device).eval()
    ids_out: list[str] = []
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for ids, lfcc, ssl in reader.iter_batches(batch_size):
            parts.append(model(lfcc.to(device), ssl.to(device)).cpu().numpy())
            ids_out.extend(ids)
    return ids_out, np.concatenate(parts).astype(np.float64)
