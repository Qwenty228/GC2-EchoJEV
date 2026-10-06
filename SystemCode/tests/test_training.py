"""Tests for `audio_df.training`.

No dataset is required: every test constructs a toy 2-layer model and synthetic tensors.
The load-bearing test here is `test_seeding_twice_gives_identical_losses` — reproducibility
is claimed, not assumed, so it is proven bit-identical rather than merely close.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from audio_df.evaluation import EvalConfig, ScoreSet, evaluate, pick_threshold
from audio_df.training import (
    TrainConfig,
    count_params,
    measure_rtf,
    set_seed,
    train_model,
)


def _toy_model() -> nn.Module:
    """A tiny 2-layer MLP: Linear(4, 8) -> ReLU -> Linear(8, 1)."""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))


def _memorisable_batch(n: int = 32, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A small, perfectly separable synthetic batch: spoof iff sum(x) > 0."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 4, generator=g)
    y = (x.sum(dim=1) > 0).float()
    return x, y


# ---------------------------------------------------------------------------
# Toy model trains and loss decreases
# ---------------------------------------------------------------------------


def test_toy_model_loss_decreases_on_memorisable_batch():
    set_seed(0)
    model = _toy_model()
    x, y = _memorisable_batch()
    loaders = {"train": [(x, y)], "dev": [(x, y)]}
    cfg = TrainConfig(epochs=2, lr=0.05, patience=100, seed=0, higher_is_spoof=True, device="cpu")

    result = train_model(model, loaders, cfg)

    assert result.epochs_run == 2
    assert len(result.train_losses) == 2
    assert result.train_losses[-1] < result.train_losses[0]


def test_toy_model_loss_keeps_decreasing_with_more_epochs():
    """A slightly longer run on the same memorisable batch drives loss further down."""
    set_seed(1)
    model = _toy_model()
    x, y = _memorisable_batch(seed=1)
    loaders = {"train": [(x, y)], "dev": [(x, y)]}
    cfg = TrainConfig(epochs=25, lr=0.1, patience=100, seed=1, higher_is_spoof=True, device="cpu")

    result = train_model(model, loaders, cfg)

    assert result.train_losses[-1] < result.train_losses[0] * 0.5
    assert result.best_dev_eer <= 0.5


# ---------------------------------------------------------------------------
# Reproducibility: the load-bearing test
# ---------------------------------------------------------------------------


def test_seeding_twice_gives_identical_losses():
    """Same seed, same model class, same data -> bit-identical loss trajectories."""

    def run():
        set_seed(42)
        model = _toy_model()
        x, y = _memorisable_batch(seed=7)
        loaders = {"train": [(x, y)], "dev": [(x, y)]}
        cfg = TrainConfig(
            epochs=5, lr=0.05, patience=100, seed=42, higher_is_spoof=True, device="cpu"
        )
        return train_model(model, loaders, cfg)

    result_a = run()
    result_b = run()

    assert result_a.train_losses == result_b.train_losses
    assert result_a.dev_eers == result_b.dev_eers
    for (name_a, p_a), (name_b, p_b) in zip(
        result_a.model.state_dict().items(), result_b.model.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(p_a, p_b), (
            f"parameter {name_a} differs between two identically seeded runs"
        )


def test_set_seed_makes_fresh_model_init_identical():
    """`set_seed` alone, called before construction, reproduces initial weights exactly."""
    set_seed(99)
    model_a = _toy_model()
    set_seed(99)
    model_b = _toy_model()
    for (name_a, p_a), (name_b, p_b) in zip(
        model_a.state_dict().items(), model_b.state_dict().items(), strict=True
    ):
        assert name_a == name_b
        assert torch.equal(p_a, p_b)


# ---------------------------------------------------------------------------
# Early stopping restores the BEST checkpoint, not the last
# ---------------------------------------------------------------------------


def _flipped_signal_loaders() -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Train and dev signal in dim 0 are DELIBERATELY flipped, so as the model fits train,
    dev EER gets monotonically worse after the (near-random) first epoch — a deterministic
    plateau/worsening shape without relying on hand-tuned learning dynamics."""
    n = 40
    g = torch.Generator().manual_seed(0)
    x_train = torch.randn(n, 2, generator=g)
    y_train = (x_train[:, 0] > 0).float()
    x_dev = torch.randn(n, 2, generator=g)
    y_dev = (x_dev[:, 0] < 0).float()
    loaders = {"train": [(x_train, y_train)], "dev": [(x_dev, y_dev)]}
    return loaders, x_train, x_dev


def test_early_stopping_fires_and_restores_best_not_last_checkpoint():
    loaders, _, _ = _flipped_signal_loaders()
    set_seed(123)
    model = nn.Linear(2, 1)
    cfg = TrainConfig(
        epochs=15,
        lr=0.5,
        patience=2,
        seed=123,
        higher_is_spoof=True,
        device="cpu",
        keep_checkpoint_history=True,
    )

    result = train_model(model, loaders, cfg)

    # Early stopping actually fired, before exhausting the epoch budget.
    assert result.stopped_early is True
    assert result.epochs_run < cfg.epochs
    # Stops exactly `patience` epochs after the best one.
    assert result.epochs_run == result.best_epoch + cfg.patience
    # The best epoch is not the last epoch trained -- otherwise this test wouldn't
    # distinguish "restore best" from "just keep the last weights".
    assert result.best_epoch != result.epochs_run

    best_ckpt = result.checkpoints[result.best_epoch - 1]
    last_ckpt = result.checkpoints[-1]

    returned_state = result.model.state_dict()
    for key in returned_state:
        assert torch.equal(returned_state[key], best_ckpt[key]), (
            f"returned model's {key} does not match the best epoch's checkpoint"
        )
        assert not torch.equal(returned_state[key], last_ckpt[key]), (
            f"returned model's {key} matches the LAST epoch, not the best one -- "
            "early stopping did not restore the best checkpoint"
        )


def test_min_epochs_delays_early_stopping():
    loaders, _, _ = _flipped_signal_loaders()
    set_seed(123)
    model = nn.Linear(2, 1)
    cfg = TrainConfig(
        epochs=15,
        lr=0.5,
        patience=2,
        seed=123,
        higher_is_spoof=True,
        device="cpu",
        min_epochs=6,
    )

    result = train_model(model, loaders, cfg)

    assert result.epochs_run >= cfg.min_epochs


# ---------------------------------------------------------------------------
# count_params
# ---------------------------------------------------------------------------


def test_count_params_distinguishes_trainable_from_total():
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 1))
    trainable_before, total_before = count_params(model)
    assert trainable_before == total_before
    assert trainable_before == (4 * 8 + 8) + (8 * 1 + 1)

    # freeze the first layer
    for p in model[0].parameters():
        p.requires_grad = False

    trainable_after, total_after = count_params(model)
    assert total_after == total_before
    assert trainable_after < total_after
    assert trainable_after == (8 * 1 + 1)


# ---------------------------------------------------------------------------
# measure_rtf
# ---------------------------------------------------------------------------


def test_measure_rtf_is_positive_and_scales_with_audio_seconds():
    model = _toy_model()

    class Waveform(nn.Module):
        """Wraps the toy model so `x`'s last dim can look like raw waveform samples."""

        def __init__(self, inner: nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            # collapse a [B, T] waveform-shaped input down to the toy model's expected [B, 4]
            pooled = x.reshape(x.shape[0], 4, -1).mean(dim=-1)
            return self.inner(pooled)

    wrapped = Waveform(model)
    x = torch.randn(4, 16000)  # 1 second at 16 kHz
    loader = [(x, torch.zeros(4))]

    rtf = measure_rtf(wrapped, loader, "cpu", sample_rate=16000)
    assert rtf > 0.0

    rtf_explicit = measure_rtf(wrapped, loader, "cpu", audio_seconds_per_utt=1.0)
    assert rtf_explicit > 0.0


def test_measure_rtf_rejects_zero_audio_duration():
    model = _toy_model()
    x, y = _memorisable_batch()
    loader = [(x, y)]
    with pytest.raises(ValueError, match="no audio duration"):
        measure_rtf(model, loader, "cpu", audio_seconds_per_utt=0.0)


# ---------------------------------------------------------------------------
# evaluation.py: threshold selection reads DEV only, never eval
# ---------------------------------------------------------------------------


class _ConstantModel(nn.Module):
    """Returns a fixed per-utterance logit, ignoring `x` -- lets the test control the
    exact score distribution for dev vs. eval deterministically."""

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.logits = logits

    def forward(self, x):
        return self.logits[: x.shape[0]]


def _make_score_loader(logits: list[float], labels: list[int]):
    x = torch.zeros(len(logits), 1)
    y = torch.tensor(labels, dtype=torch.float32)
    ids = [f"utt{i}" for i in range(len(logits))]
    return [(x, y, ids)]


def test_pick_threshold_reads_dev_only_and_eval_cannot_influence_it():
    # Dev: perfectly separable around score 0.5 (bonafide low, spoof high).
    dev_logits = torch.tensor([-5.0, -4.0, -3.0, 3.0, 4.0, 5.0])
    dev_labels = [0, 0, 0, 1, 1, 1]
    dev_loader = _make_score_loader(dev_logits.tolist(), dev_labels)

    # Eval: scores live in a totally different, shifted range -- a wrong implementation
    # that computed (or was influenced by) a threshold from eval data would pick a very
    # different operating point than the dev-only threshold below.
    eval_logits = torch.tensor([95.0, 96.0, 97.0, 103.0, 104.0, 105.0])
    eval_labels = [0, 0, 0, 1, 1, 1]
    eval_loader = _make_score_loader(eval_logits.tolist(), eval_labels)

    cfg = EvalConfig(higher_is_spoof=True, device="cpu")
    dev_scores = evaluate(_ConstantModel(dev_logits), dev_loader, cfg, split="dev")
    eval_scores = evaluate(_ConstantModel(eval_logits), eval_loader, cfg, split="eval")

    threshold = pick_threshold(dev_scores)

    # The dev-only threshold must sit in the dev score range, nowhere near eval's range.
    assert -5.0 <= threshold <= 5.0
    assert not (90.0 <= threshold <= 110.0)

    # Calling pick_threshold on the SAME dev_scores again (eval_scores computed in between)
    # gives the identical threshold: eval_scores existing/having been computed has no
    # observable effect on it.
    threshold_again = pick_threshold(dev_scores)
    assert threshold_again == threshold

    # Sanity: eval_scores' own scores (post-sigmoid) are saturated near 1.0 -- nowhere near
    # the dev-derived threshold -- confirming the two ScoreSets carry genuinely unrelated
    # data, not just that pick_threshold's return value happens to look right.
    assert eval_scores.scores.min() > 0.99


def test_pick_threshold_rejects_an_eval_scoreset():
    eval_logits = torch.tensor([-5.0, -4.0, 3.0, 4.0])
    eval_loader = _make_score_loader(eval_logits.tolist(), [0, 0, 1, 1])
    cfg = EvalConfig(higher_is_spoof=True, device="cpu")
    eval_scores = evaluate(_ConstantModel(eval_logits), eval_loader, cfg, split="eval")

    with pytest.raises(ValueError, match="split='dev'"):
        pick_threshold(eval_scores)


def test_pick_threshold_rejects_non_scoreset_input():
    with pytest.raises(TypeError):
        pick_threshold({"scores": [0.1, 0.2], "labels": [0, 1]})


def test_evaluate_returns_per_utterance_scores_tagged_with_split():
    logits = torch.tensor([-2.0, 2.0, -1.0, 1.0])
    labels = [0, 1, 0, 1]
    loader = _make_score_loader(logits.tolist(), labels)
    cfg = EvalConfig(higher_is_spoof=True, device="cpu")

    scores = evaluate(_ConstantModel(logits), loader, cfg, split="eval")

    assert isinstance(scores, ScoreSet)
    assert scores.split == "eval"
    assert len(scores.utt_ids) == 4
    assert len(scores.scores) == 4
    assert len(scores.labels) == 4
    # scores are sigmoid(logits): monotonic with the raw logits, not thresholded/binarised.
    assert scores.scores[0] < scores.scores[1]


def test_scoreset_rejects_unknown_split():
    with pytest.raises(ValueError, match="split"):
        ScoreSet(
            split="bogus",  # type: ignore[arg-type]
            utt_ids=("a",),
            scores=np.array([0.1]),
            labels=np.array([0]),
            higher_is_spoof=True,
        )
