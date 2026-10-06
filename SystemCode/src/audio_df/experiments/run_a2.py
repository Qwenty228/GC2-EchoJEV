"""System A2 runner: frozen SSL -> per-layer hidden states -> learned layer-weighted head.

    timing  [--n 200]                                          builds no cache/experiment
    extract --testsets train,dev,eval [--silence-mode] [--resume]   builds the pooled cache
    train   --exp e1 [--resume]                                     trains the head, scores
                                                                     dev+eval (raw)
    score   --model-from <raw exp dir> --exp e1|e3
            --silence-mode raw|trimmed --testsets eval[,itw,...]    scores extra test sets

Backbone is chosen by `--config` (`configs/a2_wavlm_base_plus.yaml` default,
`configs/a2_xlsr_300m.yaml` for XLS-R; run XLS-R only after Base+ works end to end, and
raw-only — see that config's own header comment).

## Why extraction is a cache-building job, not an `experiment_run`

`extract` only ever produces `data/cache/ssl/...` — no `runs/<id>/` directory, no
registry row. It still goes through this project's full resource discipline
(`apply_resource_limits`, `check_can_start`, `heavy_job_lock`, `MemoryGuard`, `ChunkedJob`)
directly via `audio_df.resources`, because it is exactly the kind of long-running,
resumable, memory-sensitive job that discipline exists for; it just isn't an "experiment"
in this project's sense (a run that produces scored predictions).

## Why train/score DO go through `experiment_run`

`train` (and `score`, against an already-trained head) produce `scores/<testset>.csv`
files under `runs/<exp_id>/`, so they go through the shared lifecycle
(`audio_df.experiments.lifecycle.experiment_run`): resource limits, seeding, disk/pressure
checks, the heavy-job lock, a registry row, and `run_summary.json` on exit.

## Fixed A2 design decisions (see the handoff spec this runner implements)

- **Center crop for every split, including train.** A2's cache is built once, ahead of
  training; a fixed on-disk cache cannot support per-epoch random-crop augmentation the way
  an on-the-fly audio pipeline could. This is a deliberate loss of one kind of augmentation
  in exchange for not having to re-run a ~0.4-1.8 TB-scale extraction job every epoch.
- **No trimmed training, ever.** `train` always trains on the raw cache. `trimmed` only
  ever appears as a diagnostic: the raw-trained head is scored against a *trimmed* eval (or
  In-the-Wild) cache via `score --silence-mode trimmed`. `score_experiment.py`'s threshold
  always comes from the *raw* dev scores (`--threshold-from <raw exp dir>`), never from a
  trimmed dev (there is no trimmed dev cache to threshold on in the first place).
- **Checkpoint selection is by dev LOSS, not dev EER.** Dev only contains the training
  attacks A01-A06, so dev EER saturates near-perfect and stops being a useful early-stopping
  signal long before dev loss does (project decision, recorded here since it deviates from
  `training.py`'s usual dev-EER-based early stopping).
- **Training's loss-sign convention.** `SSLHead.forward` returns a logit oriented so that
  higher = more bona fide (`audio_df.models.ssl_head`'s own convention, matching
  `metrics.py`'s `higher_is_spoof=False`). `train_head` below trains directly against that
  orientation: target = `is_bonafide` (1 for bonafide, 0 for spoof), input = the head's raw
  logit, with `pos_weight = n_spoof / n_bonafide` in `nn.BCEWithLogitsLoss` — `pos_weight`
  only reweights the loss term where `target == 1`, so putting bona fide (the minority
  class, per the train split's ~1:8.8 bonafide:spoof ratio) in that slot is what actually
  corrects the imbalance; putting `is_spoof` there instead (the more common BCE framing,
  used by `training.py`'s *own* local convention for its unrelated generic loop) would
  reweight the already-majority class and get the correction backwards. No sign-flip of the
  logit is needed under this framing, unlike a target=`is_spoof` framing would need.
- **No metrics computed here.** Only `scores/<testset>.csv` files are written (via
  `audio_df.exp_io.write_scores`); `scripts/score_experiment.py` is the only place EER/AUC/
  minDCF are computed, run by the user afterward — this file prints the exact command.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from audio_df.audio import load_audio, preprocess
from audio_df.device import resolve_device
from audio_df.exp_io import make_exp_id, write_scores
from audio_df.experiments.lifecycle import experiment_run
from audio_df.models.ssl_head import SSLHead
from audio_df.resources import (
    MemoryGuard,
    MemoryPressureExit,
    apply_resource_limits,
    check_can_start,
    check_free_disk,
    format_timing,
    heavy_job_lock,
    load_resources,
    loader_workers,
    memory_status,
    run_resource_summary,
    timing_run,
)
from audio_df.seeding import make_generator
from audio_df.seeding import worker_init_fn as seed_worker_init_fn
from audio_df.ssl_cache import (
    build_job_config,
    cache_dir_for,
    extract_pooled_hidden_states,
    load_backbone,
    load_cache,
    run_extraction,
)

__all__ = [
    "load_config",
    "resolve_manifest",
    "assert_backbone_matches",
    "align_cache_with_manifest",
    "TrainResult",
    "train_head",
    "score_testset",
    "main",
]

DEFAULT_CONFIG = "configs/a2_wavlm_base_plus.yaml"

# Extrapolation targets for `timing`. `in_the_wild` is an approximate, UNVERIFIED clip
# count (Miscellaneous/research/fact-checks/FC-plan-2026-09-30.md item 3: only aggregate hours --
# 37.9h total / 17.2h fake / 20.7-20.8h bona fide -- were confirmed against a primary source; no
# clip count was). It is printed as an explicit approximation, never as a fact.
_EXTRAPOLATION_TARGETS: dict[str, int] = {
    "all_raw_splits (train+dev+eval)": 121_461,
    "trimmed_eval": 71_237,
    "in_the_wild (~unverified count)": 31_800,
    "e4_subset (4 x 10k)": 4 * 10_000,
}

# Approximate first-download sizes (HF model card listings), printed once by `timing`.
_MODEL_DOWNLOAD_GB = {
    "microsoft/wavlm-base-plus": 0.38,
    "facebook/wav2vec2-xls-r-300m": 1.27,
}


# ---------------------------------------------------------------------------
# Config / manifest helpers
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_manifest(cfg: dict[str, Any], testset: str) -> str:
    """`"train"` resolves to `cfg["data"]["train_manifest"]`; anything else looks up
    `cfg["data"]["testsets"][testset]` (`dev`, `eval`, `itw`, `eval_mp3_32k`, ...)."""
    if testset == "train":
        return cfg["data"]["train_manifest"]
    testsets = cfg["data"]["testsets"]
    if testset not in testsets:
        raise ValueError(f"unknown testset {testset!r}; known: {['train', *sorted(testsets)]}")
    return testsets[testset]


def assert_backbone_matches(model: nn.Module, cfg: dict[str, Any]) -> None:
    """Fail loudly if the loaded model's shape does not match what the config claims.

    A silent mismatch here would poison the entire cache with the wrong `n_layers`/
    `hidden_dim`, discovered only much later when a head refuses to load.
    """
    # `nn.Module` has no `config`; the Hugging Face backbone does, with untyped fields.
    config: Any = model.config
    n_layers = config.num_hidden_layers + 1
    hidden_dim = config.hidden_size
    expected_layers = cfg["model"]["n_layers"]
    expected_hidden = cfg["model"]["hidden_dim"]
    if n_layers != expected_layers or hidden_dim != expected_hidden:
        raise ValueError(
            f"loaded backbone {cfg['model']['id']!r} has {n_layers} hidden-state tensors x "
            f"{hidden_dim} dims, but config says n_layers={expected_layers}, "
            f"hidden_dim={expected_hidden}"
        )


def align_cache_with_manifest(
    cache: dict[str, np.ndarray], manifest_df: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Reorder `manifest_df` to match `cache["ids"]`'s order; raise on any missing id."""
    ids = [str(i) for i in cache["ids"]]
    indexed = manifest_df.set_index("utt_id")
    missing = sorted(set(ids) - set(indexed.index.astype(str)))
    if missing:
        raise ValueError(
            f"{len(missing)} cached id(s) not found in manifest, e.g. {missing[:5]}"
        )
    indexed.index = indexed.index.astype(str)
    ordered = indexed.loc[ids].reset_index()
    return cache["pooled"], ordered


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


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


def score_testset(
    head: nn.Module, cache_dir: str | Path, manifest_df: pd.DataFrame, device: torch.device
) -> pd.DataFrame:
    """Score every cached clip in `cache_dir`, aligned to `manifest_df`'s labels.

    `score` = the head's raw logit (higher = more bona fide, matching this project's score
    file convention, `exp_io`'s module docstring, `higher_is_spoof=False`).
    """
    cache = load_cache(cache_dir)
    pooled, ordered = align_cache_with_manifest(cache, manifest_df)

    head = head.to(device)
    head.eval()
    x = torch.tensor(pooled.astype(np.float32), device=device)
    with torch.no_grad():
        logit = head(x).cpu().numpy()

    df = pd.DataFrame(
        {
            "clip_id": ordered["utt_id"].astype(str),
            "label": ordered["label"],
            "score": logit.astype(np.float64),
        }
    )
    if "attack_id" in ordered.columns:
        df["attack_id"] = ordered["attack_id"]
    return df


def _load_head_from_config(cfg: dict[str, Any]) -> SSLHead:
    return SSLHead(
        n_layers=cfg["model"]["n_layers"],
        hidden_dim=cfg["model"]["hidden_dim"],
        proj_dim=cfg["head"]["proj_dim"],
        mlp_hidden=cfg["head"]["mlp_hidden"],
        dropout=cfg["head"]["dropout"],
    )


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def cmd_timing(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    res = load_resources()
    apply_resource_limits(res)

    download_gb = _MODEL_DOWNLOAD_GB.get(cfg["model"]["id"])
    if download_gb is not None:
        print(
            f"[timing] first run downloads {cfg['model']['id']} weights from Hugging Face "
            f"(approx. {download_gb} GB)."
        )

    model, feature_extractor = load_backbone(cfg["model"]["id"], cfg["model"]["revision"])
    assert_backbone_matches(model, cfg)
    device = resolve_device("auto")
    model = model.to(device)

    train_manifest = pd.read_csv(cfg["data"]["train_manifest"])
    items = list(zip(train_manifest["utt_id"], train_manifest["path"], strict=True))

    prep = cfg["preprocessing"]

    def _one(item: tuple[str, str]) -> np.ndarray:
        _, path = item
        x, sr = load_audio(path, target_sr=prep["sample_rate"])
        x = preprocess(
            x,
            sr,
            silence_mode=prep["silence_mode"],
            crop_seconds=prep["crop_seconds"],
            crop_mode=prep["crop_mode"],
        )
        return extract_pooled_hidden_states(
            model, feature_extractor, x[None, :], prep["sample_rate"], device
        )

    n = args.n if args.n is not None else res["resources"]["timing_run_clips"]
    result = timing_run(_one, items, n=n, cfg=res)
    print(format_timing(result))

    extrapolations: dict[str, float | None] = {}
    for name, total in _EXTRAPOLATION_TARGETS.items():
        mean_s = result["mean_s_per_item"]
        extrapolations[name] = (total * mean_s) / 3600.0 if mean_s > 0 else None
        print(f"  extrapolated for {name} ({total} clips): {extrapolations[name]:.2f} h")

    out_path = Path("data/cache/timing") / f"a2_{cfg['model']['tag']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"timing": result, "extrapolations_hours": extrapolations}, indent=2)
    )
    print(f"wrote {out_path}")
    print(
        f"next: uv run python -m audio_df.experiments.run_a2 extract "
        f"--config {args.config} --testsets train,dev,eval"
    )


def cmd_extract(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    res = load_resources()
    apply_resource_limits(res)
    status_fn = getattr(args, "status_fn", memory_status)
    check_can_start(res, status_fn=status_fn)

    model, feature_extractor = load_backbone(cfg["model"]["id"], cfg["model"]["revision"])
    assert_backbone_matches(model, cfg)
    device = resolve_device("auto")
    model = model.to(device)
    guard = MemoryGuard(res, status_fn=status_fn)

    testsets = [t.strip() for t in args.testsets.split(",") if t.strip()]
    prep = cfg["preprocessing"]
    log_path = Path(cfg["cache_dir"]) / cfg["model"]["tag"] / "extraction_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with heavy_job_lock(
            "a2-extract", lock_path=getattr(args, "lock_path", "data/cache/.heavy_job.lock")
        ):
            for testset in testsets:
                manifest_path = resolve_manifest(cfg, testset)
                manifest_df = pd.read_csv(manifest_path)
                ids = manifest_df["utt_id"].astype(str).tolist()
                paths = manifest_df["path"].astype(str).tolist()

                out_dir = cache_dir_for(
                    cfg["model"]["tag"], args.silence_mode, testset, root=cfg["cache_dir"]
                )
                need_gb = len(ids) * cfg["model"]["n_layers"] * cfg["model"]["hidden_dim"] * 2 / 1e9
                out_dir.mkdir(parents=True, exist_ok=True)
                check_free_disk(out_dir, need_gb, res)

                job_config = build_job_config(
                    model_id=cfg["model"]["id"],
                    model_tag=cfg["model"]["tag"],
                    n_layers=cfg["model"]["n_layers"],
                    hidden_dim=cfg["model"]["hidden_dim"],
                    silence_mode=args.silence_mode,
                    crop_seconds=prep["crop_seconds"],
                    crop_mode=prep["crop_mode"],
                    sample_rate=prep["sample_rate"],
                    do_normalize=cfg["model"]["do_normalize"],
                    revision=cfg["model"]["revision"],
                )
                num_workers = loader_workers(1, res, kind="ssl")

                run_extraction(
                    ids,
                    paths,
                    model,
                    feature_extractor,
                    out_dir,
                    job_config,
                    device,
                    sample_rate=prep["sample_rate"],
                    silence_mode=args.silence_mode,
                    crop_seconds=prep["crop_seconds"],
                    crop_mode=prep["crop_mode"],
                    batch_size=res["resources"]["ssl_extract_batch_size"],
                    num_workers=num_workers,
                    chunk_size=res["resources"]["chunk_size"],
                    memory_guard=guard,
                    worker_init_fn=seed_worker_init_fn,
                    generator=make_generator(cfg["seed"]),
                )

                summary = run_resource_summary(guard)
                print(f"[extract] {testset} ({args.silence_mode}): {summary}")
                with open(log_path, "a") as f:
                    f.write(
                        json.dumps(
                            {"testset": testset, "silence_mode": args.silence_mode, **summary}
                        )
                        + "\n"
                    )
    except MemoryPressureExit as exc:
        print(f"[extract] memory pressure critical: {exc.args[0]}; rerun with --resume")
        sys.exit(3)

    print(
        f"next: uv run python -m audio_df.experiments.run_a2 train --config {args.config} --exp e1"
    )


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    exp_id = make_exp_id(date.today(), args.exp, cfg["system"], "raw")
    full_config = {**cfg, "experiment": args.exp}

    with experiment_run(
        exp_id,
        full_config,
        job_name="a2-train",
        resume=args.resume,
        root=getattr(args, "root", "runs"),
        lock_path=getattr(args, "lock_path", "data/cache/.heavy_job.lock"),
        status_fn=getattr(args, "status_fn", memory_status),
    ) as run:
        device = resolve_device("auto")
        tag = cfg["model"]["tag"]
        cache_root = cfg["cache_dir"]

        train_manifest = pd.read_csv(cfg["data"]["train_manifest"])
        dev_manifest = pd.read_csv(cfg["data"]["testsets"]["dev"])
        eval_manifest = pd.read_csv(cfg["data"]["testsets"]["eval"])

        train_cache_dir = cache_dir_for(tag, "raw", "train", root=cache_root)
        dev_cache_dir = cache_dir_for(tag, "raw", "dev", root=cache_root)
        eval_cache_dir = cache_dir_for(tag, "raw", "eval", root=cache_root)

        pooled_train, train_aligned = align_cache_with_manifest(
            load_cache(train_cache_dir), train_manifest
        )
        pooled_dev, dev_aligned = align_cache_with_manifest(load_cache(dev_cache_dir), dev_manifest)

        x_train = torch.tensor(pooled_train.astype(np.float32))
        y_train = torch.tensor(
            (train_aligned["label"] == "bonafide").astype(np.float32).to_numpy()
        )
        x_dev = torch.tensor(pooled_dev.astype(np.float32))
        y_dev = torch.tensor((dev_aligned["label"] == "bonafide").astype(np.float32).to_numpy())

        head = _load_head_from_config(cfg)
        generator = make_generator(cfg["seed"])
        pw_cfg = cfg["head"]["pos_weight"]
        pos_weight = None if pw_cfg == "auto" else float(pw_cfg)

        result = train_head(
            head,
            x_train,
            y_train,
            x_dev,
            y_dev,
            lr=cfg["head"]["lr"],
            batch_size=cfg["head"]["batch_size"],
            max_epochs=cfg["head"]["max_epochs"],
            patience=cfg["head"]["patience"],
            pos_weight=pos_weight,
            device=device,
            generator=generator,
        )

        torch.save(result.head.state_dict(), run.exp_dir / "head_state_dict.pt")
        (run.exp_dir / "training_curve.json").write_text(
            json.dumps(
                {
                    "train_loss": result.train_losses,
                    "dev_loss": result.dev_losses,
                    "best_epoch": result.best_epoch,
                    "pos_weight": result.pos_weight,
                },
                indent=2,
            )
        )
        (run.exp_dir / "layer_weights.json").write_text(
            json.dumps(result.head.layer_weights().detach().cpu().numpy().tolist(), indent=2)
        )

        dev_scores = score_testset(result.head, dev_cache_dir, dev_manifest, device)
        write_scores(run.exp_dir, "dev", dev_scores)
        eval_scores = score_testset(result.head, eval_cache_dir, eval_manifest, device)
        write_scores(run.exp_dir, "eval", eval_scores)

        run.note(f"best dev loss at epoch {result.best_epoch} (dev loss checkpoint selection)")

    print(
        f"next: uv run python scripts/score_experiment.py {run.exp_dir} "
        f"&& uv run python -m audio_df.experiments.run_a2 score --model-from {run.exp_dir} "
        f"--exp {args.exp} --silence-mode raw --testsets itw"
    )


def cmd_score(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model_from = Path(args.model_from)
    head = _load_head_from_config(cfg)
    head.load_state_dict(torch.load(model_from / "head_state_dict.pt", map_location="cpu"))
    device = resolve_device("auto")
    head = head.to(device)

    exp_id = make_exp_id(date.today(), args.exp, cfg["system"], args.silence_mode)
    full_config = {
        **cfg,
        "experiment": args.exp,
        "model_from": str(model_from),
        # Record the silence mode actually scored (2026-10-03: trimmed runs said "raw").
        "preprocessing": {**cfg["preprocessing"], "silence_mode": args.silence_mode},
    }
    testsets = [t.strip() for t in args.testsets.split(",") if t.strip()]

    # Through experiment_run like every other run, so a score run gets its own registry row,
    # resource summary and failure record (2026-10-01: without a row, score_experiment.py
    # crashed on these runs).
    with experiment_run(
        exp_id,
        full_config,
        job_name="a2-score",
        config_summary=(
            f"a2 score, silence_mode={args.silence_mode}, testsets={','.join(testsets)}, "
            f"model_from={model_from}"
        ),
        resume=args.resume,
        root=getattr(args, "root", "runs"),
        lock_path=getattr(args, "lock_path", "data/cache/.heavy_job.lock"),
        status_fn=getattr(args, "status_fn", memory_status),
    ) as run:
        if args.silence_mode == "trimmed":
            run.note(
                f"Trimmed-silence diagnostic: the raw-trained head from {model_from} scoring "
                "trimmed test audio. The threshold must come from the RAW experiment's dev scores."
            )
        for testset in testsets:
            manifest_df = pd.read_csv(resolve_manifest(cfg, testset))
            cache_dir = cache_dir_for(
                cfg["model"]["tag"], args.silence_mode, testset, root=cfg["cache_dir"]
            )
            df = score_testset(head, cache_dir, manifest_df, device)
            write_scores(run.exp_dir, testset, df, force=args.resume)
        exp_dir = run.exp_dir

    testset_list = ",".join(testsets)
    print(
        f"next: uv run python scripts/score_experiment.py {exp_dir} "
        f"--threshold-from {model_from} "
        f"# scores written for {testset_list} ({args.silence_mode})"
    )


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="System A2: frozen SSL + layer-weighted head")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    p_timing = sub.add_parser("timing")
    p_timing.add_argument("--n", type=int, default=None)
    p_timing.set_defaults(func=cmd_timing)

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--testsets", required=True)
    p_extract.add_argument("--silence-mode", default="raw", choices=["raw", "trimmed"])
    p_extract.add_argument("--resume", action="store_true")
    p_extract.set_defaults(func=cmd_extract)

    p_train = sub.add_parser("train")
    p_train.add_argument("--exp", required=True)
    p_train.add_argument("--resume", action="store_true")
    p_train.set_defaults(func=cmd_train)

    p_score = sub.add_parser("score")
    p_score.add_argument("--model-from", required=True)
    p_score.add_argument("--exp", required=True)
    p_score.add_argument("--silence-mode", required=True, choices=["raw", "trimmed"])
    p_score.add_argument("--testsets", required=True)
    p_score.add_argument("--resume", action="store_true")
    p_score.set_defaults(func=cmd_score)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
