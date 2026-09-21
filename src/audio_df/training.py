"""Training loop, seeding and early stopping.

This module owns *how* a model is trained — seeding, the epoch loop, early stopping on
dev EER, parameter counting, timing and (best-effort, honestly-labelled) memory tracking.
It never defines an architecture (see the module boundary table in `tasks/CONVENTIONS.md`):
callers pass an already-constructed `nn.Module` and get a trained one back.

## Model output convention

`train_model` and `measure_rtf` assume the model returns a single raw logit per utterance
(shape `[B]` or `[B, 1]`), oriented so that a **higher logit means more spoof-like** — i.e.
`sigmoid(logit)` is treated as `P(spoof)`, trained against labels where `1 = spoof`,
`0 = bonafide`. This is a `training.py`-local convention (chosen because
`nn.BCEWithLogitsLoss` needs a fixed positive class, and "spoof = positive" is the more
common framing for a detector), not the ASVspoof scoring convention used elsewhere in this
project (`metrics.py`'s default orientation is `higher_is_spoof=False`, higher = bonafide).
`cfg.higher_is_spoof` is still a required, explicit field — exactly like every function in
`metrics.py` — precisely so a caller whose model happens to emit the opposite orientation
does not get a silently flipped EER; it is not defaulted to match either convention.

## Peak memory: measured, not assumed

CUDA has `torch.cuda.max_memory_allocated()`, a true peak tracker. MPS has no equivalent as
of torch 2.14 (`torch.mps.current_allocated_memory()` is a point-in-time reading only). This
module does not pretend otherwise:

- On `mps`, it *samples* `torch.mps.current_allocated_memory()` after every training step and
  keeps a running max. That running max is a lower bound on the true peak (memory could spike
  and free between samples) — the returned dict names the metric
  `"mps_sampled_peak_allocated_mb"`, not `"peak_mb"`, so nobody downstream mistakes it for a
  CUDA-grade peak.
- On `cpu` (and any other device), it reads `resource.getrusage(RUSAGE_SELF).ru_maxrss`, the
  OS-reported peak resident set size for the *whole process since it started* (not scoped to
  this training call). Named `"process_peak_rss_mb_since_start"` for the same reason.
"""

from __future__ import annotations

import copy
import random
import resource
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from .device import resolve_device
from .metrics import compute_eer

__all__ = [
    "TrainConfig",
    "TrainResult",
    "set_seed",
    "train_model",
    "count_params",
    "measure_rtf",
]


def set_seed(seed: int) -> None:
    """Seed python `random`, numpy and torch (CPU, CUDA-all, and MPS) with one call.

    Covering all three is the whole point: a model trained twice with the same seed must
    produce bit-identical losses (see `tests/test_training.py`), and any one of these RNGs
    left unseeded would break that on the corresponding backend. `torch.mps.manual_seed`
    exists as of torch 2.14 and is called whenever MPS is built, regardless of whether it is
    the resolved device for this process, so state is deterministic either way.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@dataclass
class TrainConfig:
    """Training hyperparameters and provenance-relevant knobs.

    `higher_is_spoof` has no default, deliberately (see module docstring): it is the
    orientation of `model`'s raw output, used to compute dev EER for early stopping.
    """

    epochs: int
    lr: float
    patience: int
    seed: int
    higher_is_spoof: bool
    device: str = "auto"
    weight_decay: float = 0.0
    grad_clip: float | None = None
    min_epochs: int = 1
    keep_checkpoint_history: bool = False


@dataclass
class TrainResult:
    """Everything `training.py` knows about one completed training run.

    `checkpoints` (only populated when `cfg.keep_checkpoint_history=True`, default off to
    avoid an O(epochs) memory cost on real models) holds the raw `state_dict()` after each
    epoch's training+eval step, in epoch order (index 0 is epoch 1), *before* the final
    best-epoch restore. Exists so callers/tests can verify that the returned `model` holds
    the best epoch's weights and not the last epoch's: compare `model.state_dict()` against
    `checkpoints[best_epoch - 1]` (should match) and `checkpoints[-1]` (should differ
    whenever `best_epoch != epochs_run`).
    """

    model: nn.Module
    best_epoch: int
    best_dev_eer: float
    train_losses: list[float]
    dev_eers: list[float]
    epochs_run: int
    stopped_early: bool
    wall_time_s: float
    peak_mem: dict[str, Any]
    trainable_params: int
    total_params: int
    device: str
    seed: int
    checkpoints: list[dict[str, torch.Tensor]] = field(default_factory=list, repr=False)


def count_params(model: nn.Module) -> tuple[int, int]:
    """`(trainable, total)` parameter counts, distinguishing frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _peak_memory(device: torch.device, mps_peak_bytes: float | None) -> dict[str, Any]:
    """Best-effort peak memory, labelled honestly per-device. See module docstring."""
    if device.type == "mps":
        if mps_peak_bytes is None:
            mps_peak_bytes = torch.mps.current_allocated_memory()
        return {"metric": "mps_sampled_peak_allocated_mb", "value_mb": mps_peak_bytes / 1e6}
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux reports it in KiB. Both are converted to MB.
    divisor = 1e6 if sys.platform == "darwin" else 1e3
    return {"metric": "process_peak_rss_mb_since_start", "value_mb": ru_maxrss / divisor}


def train_model(
    model: nn.Module,
    loaders: dict[str, Iterable[tuple[torch.Tensor, torch.Tensor]]],
    cfg: TrainConfig,
) -> TrainResult:
    """Train `model` with Adam + BCE-with-logits, early stopping on dev EER (mode=min).

    `loaders["train"]` and `loaders["dev"]` are iterables of `(x, y)` batches, `y` being
    0 (bonafide) / 1 (spoof). Calls `set_seed(cfg.seed)` itself at the top, so two calls
    with identical `model` initial state, `loaders` and `cfg` are bit-identical (see
    `tests/test_training.py::test_seeding_twice_gives_identical_losses`).

    Early stopping: after every epoch's dev EER, if it is the best seen so far the model's
    weights are checkpointed (deep-copied) and the patience counter resets; otherwise the
    counter increments. Training stops once the counter reaches `cfg.patience` (and at least
    `cfg.min_epochs` epochs have run). The **returned model always has the best epoch's
    weights loaded**, never the last epoch's — that is the entire point of monitoring dev EER
    instead of train loss.
    """
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = loaders["train"]
    dev_loader = loaders["dev"]

    best_state = copy.deepcopy(model.state_dict())
    best_dev_eer = float("inf")
    best_epoch = 0
    epochs_since_improve = 0
    train_losses: list[float] = []
    dev_eers: list[float] = []
    stopped_early = False
    mps_peak_bytes = 0.0
    epoch_checkpoints: list[dict[str, torch.Tensor]] = []

    start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float().reshape(-1)
            optimizer.zero_grad()
            logits = model(x).reshape(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            if cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            bsz = x.shape[0]
            running_loss += loss.item() * bsz
            n_seen += bsz
            if device.type == "mps":
                mps_peak_bytes = max(mps_peak_bytes, torch.mps.current_allocated_memory())
        train_losses.append(running_loss / max(n_seen, 1))

        model.eval()
        dev_scores: list[float] = []
        dev_labels: list[float] = []
        with torch.no_grad():
            for x, y in dev_loader:
                x = x.to(device)
                logits = model(x).reshape(-1)
                probs = torch.sigmoid(logits)
                dev_scores.extend(probs.cpu().numpy().tolist())
                dev_labels.extend(y.reshape(-1).cpu().numpy().tolist())
        eer, _ = compute_eer(dev_scores, dev_labels, higher_is_spoof=cfg.higher_is_spoof)
        dev_eers.append(eer)

        if cfg.keep_checkpoint_history:
            epoch_checkpoints.append(copy.deepcopy(model.state_dict()))

        if eer < best_dev_eer:
            best_dev_eer = eer
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.patience and epoch >= cfg.min_epochs:
                stopped_early = True
                break

    wall_time_s = time.perf_counter() - start

    model.load_state_dict(best_state)
    trainable, total = count_params(model)
    peak_mem = _peak_memory(device, mps_peak_bytes if device.type == "mps" else None)

    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        best_dev_eer=best_dev_eer,
        train_losses=train_losses,
        dev_eers=dev_eers,
        epochs_run=len(train_losses),
        stopped_early=stopped_early,
        wall_time_s=wall_time_s,
        peak_mem=peak_mem,
        trainable_params=trainable,
        total_params=total,
        device=str(device),
        seed=cfg.seed,
        checkpoints=epoch_checkpoints,
    )


def measure_rtf(
    model: nn.Module,
    loader: Iterable[Any],
    device: str | torch.device,
    sample_rate: int = 16000,
    audio_seconds_per_utt: float | None = None,
) -> float:
    """Real-time factor: wall-clock inference time / audio duration processed.

    `RTF < 1` means faster than real time. `loader` yields batches whose first element is
    the model input tensor `x` (a 2-tuple `(x, y)` or a bare tensor both work). By default,
    audio duration per utterance is inferred as `x.shape[-1] / sample_rate` — i.e. this
    assumes the last dimension of `x` is raw waveform samples at `sample_rate` (16 kHz
    project-wide). For inputs where that assumption does not hold (frame-based features,
    pooled embeddings), pass `audio_seconds_per_utt` explicitly instead of relying on the
    inferred value.
    """
    resolved = resolve_device(device) if isinstance(device, str) else device
    model = model.to(resolved)
    model.eval()

    total_wall_s = 0.0
    total_audio_s = 0.0

    with torch.no_grad():
        for batch in loader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            x = x.to(resolved)

            if resolved.type == "mps":
                torch.mps.synchronize()
            elif resolved.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if resolved.type == "mps":
                torch.mps.synchronize()
            elif resolved.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            bsz = x.shape[0]
            total_wall_s += t1 - t0
            if audio_seconds_per_utt is not None:
                total_audio_s += audio_seconds_per_utt * bsz
            else:
                total_audio_s += (x.shape[-1] / sample_rate) * bsz

    if total_audio_s <= 0:
        raise ValueError("no audio duration accumulated; cannot compute RTF")
    return total_wall_s / total_audio_s
