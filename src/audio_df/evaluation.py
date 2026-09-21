"""Scoring runs and the dev-only threshold rule.

Per `tasks/CONVENTIONS.md`: "Thresholds are chosen on dev. Never on eval." This module makes
that structurally hard to get wrong rather than relying on a comment:

1. `evaluate()` never computes any threshold, and never even looks at an aggregate statistic
   over the scores it returns — it maps `model` over `loader` and hands back per-utterance
   scores. There is no code path inside it that *could* select an operating point.
2. Every `ScoreSet` is tagged with the `split` it came from (`"train" | "dev" | "eval"`),
   supplied as a required, explicit argument to `evaluate()` at the call site — not inferred,
   not defaulted. The tag travels with the data for its entire lifetime, because `ScoreSet`
   is a frozen dataclass (the tag cannot be mutated after creation, e.g. to "launder" an eval
   ScoreSet into a dev one).
3. `pick_threshold()` is the *only* function in this codebase that selects a threshold, and it
   runtime-asserts `dev_scores.split == "dev"` before doing anything else. Passing it an
   eval-tagged `ScoreSet` raises `ValueError` immediately — it does not silently compute *a*
   threshold from whatever it was given.

Combined, this is stronger than a docstring warning: misuse fails loudly (a raised exception),
not quietly (a plausible-looking but eval-contaminated threshold).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from .device import resolve_device
from .metrics import compute_eer

__all__ = ["ScoreSet", "EvalConfig", "evaluate", "pick_threshold"]

_VALID_SPLITS = ("train", "dev", "eval")
Split = Literal["train", "dev", "eval"]


@dataclass(frozen=True)
class ScoreSet:
    """Per-utterance scores from one `evaluate()` call, immutably tagged with their split.

    `scores` are raw model outputs after `sigmoid` (see `evaluation.evaluate`'s docstring for
    the polarity convention), oriented per `higher_is_spoof`. `attack_ids` is `None` unless
    the loader supplied per-utterance attack/generator ids.
    """

    split: Split
    utt_ids: tuple[str, ...]
    scores: np.ndarray
    labels: np.ndarray
    higher_is_spoof: bool
    attack_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.split not in _VALID_SPLITS:
            raise ValueError(f"split must be one of {_VALID_SPLITS}, got {self.split!r}")
        n = len(self.utt_ids)
        if len(self.scores) != n or len(self.labels) != n:
            raise ValueError(
                f"utt_ids ({n}), scores ({len(self.scores)}) and labels "
                f"({len(self.labels)}) must have equal length"
            )
        if self.attack_ids is not None and len(self.attack_ids) != n:
            raise ValueError(f"attack_ids ({len(self.attack_ids)}) must match utt_ids ({n})")


@dataclass
class EvalConfig:
    """Config for `evaluate()`. `higher_is_spoof` has no default — see `metrics.py`."""

    higher_is_spoof: bool
    device: str = "auto"


def evaluate(
    model: nn.Module, loader: Iterable[Any], cfg: EvalConfig, split: Split
) -> ScoreSet:
    """Score every utterance in `loader`, tagged with `split`. Never selects a threshold.

    `split` is required and positional/keyword — there is no default, so a caller cannot
    forget to state which split they are scoring. `loader` yields batches of
    `(x, y, utt_ids)` or `(x, y, utt_ids, attack_ids)`; `y` is 0 (bonafide) / 1 (spoof).
    `model(x)` must return one raw logit per utterance, oriented per the `training.py`
    module docstring convention (`sigmoid(logit) = P(spoof)`); this function converts that
    logit to a probability score via `sigmoid` before returning it, so a raw `higher_is_spoof`
    of `True` in `cfg` reflects that the stored scores are `P(spoof)`.
    """
    device = resolve_device(cfg.device)
    model = model.to(device)
    model.eval()

    utt_ids: list[str] = []
    scores: list[float] = []
    labels: list[float] = []
    attack_ids: list[str] = []
    have_attack_ids = False

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                x, y, ids, attacks = batch
                have_attack_ids = True
            elif len(batch) == 3:
                x, y, ids = batch
                attacks = None
            else:
                raise ValueError(
                    "eval loader batches must be (x, y, utt_ids) or "
                    "(x, y, utt_ids, attack_ids), got a batch of length "
                    f"{len(batch)}"
                )
            x = x.to(device)
            logits = model(x).reshape(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probs.tolist())
            y_arr = y.cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            labels.extend(y_arr.reshape(-1).tolist())
            utt_ids.extend(list(ids))
            if attacks is not None:
                attack_ids.extend(list(attacks))

    return ScoreSet(
        split=split,
        utt_ids=tuple(utt_ids),
        scores=np.asarray(scores, dtype=float),
        labels=np.asarray(labels),
        higher_is_spoof=cfg.higher_is_spoof,
        attack_ids=tuple(attack_ids) if have_attack_ids else None,
    )


def pick_threshold(dev_scores: ScoreSet) -> float:
    """Select an operating threshold from DEV scores ONLY.

    Raises `TypeError`/`ValueError` rather than silently proceeding if handed anything but a
    `split="dev"` `ScoreSet` — see the module docstring for why this is a hard guard, not a
    convention. The threshold returned is `compute_eer`'s own operating threshold (the
    dev-set EER point), matching the project's stated protocol of choosing thresholds on dev.
    """
    if not isinstance(dev_scores, ScoreSet):
        raise TypeError(f"pick_threshold expects a ScoreSet, got {type(dev_scores).__name__}")
    if dev_scores.split != "dev":
        raise ValueError(
            f"pick_threshold requires a ScoreSet with split='dev', got split={dev_scores.split!r}; "
            "thresholds must never be selected from eval (or train) scores"
        )
    _, thr = compute_eer(dev_scores.scores, dev_scores.labels, dev_scores.higher_is_spoof)
    return float(thr)
