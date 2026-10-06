"""End-to-end test of the A1 runner (train + score) on a tiny synthetic dataset.

No real dataset, no network: audio is synthesised with numpy/soundfile in `tmp_path`,
memory pressure is mocked to "normal" throughout via `run_a1._STATUS_FN`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from audio_df.experiments import run_a1

NORMAL_STATUS = {
    "pressure_level": 1,
    "pressure_name": "normal",
    "swap_used_gb": 0.1,
    "swap_total_gb": 4.0,
    "rss_mb": 50.0,
}

SR = 16000


def _tone(freq: float, seconds: float, sr: int = SR, amp: float = 0.6) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_resources_yaml(path: Path) -> None:
    cfg = {
        "resources": {
            "mps_high_watermark_ratio": 0.3,
            "cpu_threads": 1,
            "max_loader_workers": 0,
            "max_loader_workers_ssl": 0,
            "ssl_extract_batch_size": 1,
            "max_ffmpeg_workers": 0,
            "min_free_disk_gb": 0,
            "timing_run_clips": 5,
            "chunk_size": 4,
            "memory_guard": {
                "poll": "between_chunks",
                "swap_growth_limit_gb": 10.0,
                "pause_on": "warn",
                "exit_on": "critical",
            },
            "close_before_heavy_jobs": [],
        }
    }
    path.write_text(yaml.safe_dump(cfg))


def _build_dataset(tmp_path: Path, n_per_class: dict[str, int]) -> dict[str, Path]:
    """Bona fide = low-frequency tone, spoof = high-frequency tone: cheap, separable
    synthetic classes for a tiny GMM to actually distinguish (a genuine, if trivial,
    signal difference, not a random label)."""
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()

    manifest_paths: dict[str, Path] = {}
    rng = np.random.default_rng(0)
    for split, n in n_per_class.items():
        rows = []
        for cls, freq, attack in (("bonafide", 220.0, "-"), ("spoof", 3200.0, "A99")):
            for i in range(n):
                jitter = rng.uniform(-5, 5)
                x = _tone(freq + jitter, seconds=0.3)
                utt_id = f"{split}_{cls}_{i}"
                path = audio_dir / f"{utt_id}.wav"
                sf.write(str(path), x, SR)
                rows.append(
                    {"utt_id": utt_id, "path": str(path), "label": cls, "attack_id": attack}
                )
        df = pd.DataFrame(rows)
        out_path = manifests_dir / f"{split}.csv"
        df.to_csv(out_path, index=False)
        manifest_paths[split] = out_path

    return manifest_paths


def _build_config(tmp_path: Path, manifest_paths: dict[str, Path]) -> Path:
    cfg = {
        "system": "a1",
        "experiment": "smoke",
        "features": {
            "name": "lfcc_b02",
            "window_ms": 20,
            "hop_ms": 10,
            "n_fft": 512,
            "n_filters": 20,
            "fmin": 0.0,
            "fmax": None,
            "preemph": 0.97,
            "delta_window": 3,
            "total_dims": 60,
        },
        "gmm": {
            "n_components": 2,
            "covariance": "diag",
            "em_iters": 2,
            "init": "random_frames",
            "chunk_frames": 1000,
            "variance_floor": 0.01,
            "init_sample_size": 5000,
        },
        "data": {
            "train": str(manifest_paths["train"]),
            "testsets": {"dev": str(manifest_paths["dev"]), "eval": str(manifest_paths["eval"])},
        },
        "preprocessing": {"sample_rate": SR, "crop_seconds": None, "silence_mode": "raw"},
        "seed": 1337,
    }
    path = tmp_path / "a1_config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_train_then_score_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_a1, "_STATUS_FN", lambda: NORMAL_STATUS)

    manifest_paths = _build_dataset(tmp_path, {"train": 8, "dev": 3, "eval": 3})
    config_path = _build_config(tmp_path, manifest_paths)
    resources_path = tmp_path / "resources.yaml"
    _write_resources_yaml(resources_path)

    parser = run_a1.build_arg_parser()
    train_args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--resources",
            str(resources_path),
            "train",
            "--exp",
            "smoke",
        ]
    )
    train_args.fn(train_args)

    exp_dirs = sorted((tmp_path / "runs").glob("E-*-smoke-a1-raw"))
    assert len(exp_dirs) == 1
    raw_exp_dir = exp_dirs[0]

    # Score files exist for both configured test sets.
    dev_scores = pd.read_csv(raw_exp_dir / "scores" / "dev.csv")
    eval_scores = pd.read_csv(raw_exp_dir / "scores" / "eval.csv")
    assert {"clip_id", "label", "score", "attack_id"} <= set(dev_scores.columns)
    assert len(dev_scores) == 6  # 3 bonafide + 3 spoof
    assert len(eval_scores) == 6

    # Higher score = more bona fide (project convention): bona fide clips should score
    # higher on average than spoof clips, on both test sets.
    for scores_df in (dev_scores, eval_scores):
        bona_mean = scores_df.loc[scores_df["label"] == "bonafide", "score"].mean()
        spoof_mean = scores_df.loc[scores_df["label"] == "spoof", "score"].mean()
        assert bona_mean > spoof_mean

    # run_summary.json and a REGISTRY.md row exist.
    assert (raw_exp_dir / "run_summary.json").exists()
    registry_text = (tmp_path / "runs" / "REGISTRY.md").read_text()
    assert raw_exp_dir.name in registry_text
    assert "| done |" in registry_text

    # Model files were saved and are loadable.
    assert (raw_exp_dir / "bona_gmm.npz").exists()
    assert (raw_exp_dir / "spoof_gmm.npz").exists()

    # --- score subcommand: trimmed diagnostic against the raw-trained models ---
    score_args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--resources",
            str(resources_path),
            "score",
            "--model-from",
            str(raw_exp_dir),
            "--exp",
            "e3",  # differs from the config's "smoke", so the override is actually tested
            "--silence-mode",
            "trimmed",
            "--testsets",
            "eval",
        ]
    )
    score_args.fn(score_args)

    trimmed_dirs = sorted((tmp_path / "runs").glob("E-*-e3-a1-trimmed"))
    assert len(trimmed_dirs) == 1
    trimmed_dir = trimmed_dirs[0]
    trimmed_eval = pd.read_csv(trimmed_dir / "scores" / "eval.csv")
    assert len(trimmed_eval) == 6
    assert (
        "trimmed diagnostic"
        in "\n".join(line for line in (trimmed_dir / "notes.md").read_text().splitlines()).lower()
    )
    # 2026-10-03: the score run's config kept the train config's experiment, so the
    # results tables filed In-the-Wild runs (--exp e3) under E1.
    trimmed_cfg = yaml.safe_load((trimmed_dir / "config.yaml").read_text())
    assert trimmed_cfg["experiment"] == "e3"
    assert trimmed_cfg["preprocessing"]["silence_mode"] == "trimmed"


def test_resume_replays_no_completed_chunk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_a1, "_STATUS_FN", lambda: NORMAL_STATUS)

    manifest_paths = _build_dataset(tmp_path, {"train": 4, "dev": 2, "eval": 2})
    config_path = _build_config(tmp_path, manifest_paths)
    resources_path = tmp_path / "resources.yaml"
    _write_resources_yaml(resources_path)

    parser = run_a1.build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--resources",
            str(resources_path),
            "train",
            "--exp",
            "smoke",
        ]
    )
    args.fn(args)

    exp_dir = next((tmp_path / "runs").glob("E-*-smoke-a1-raw"))
    n_frames_before = json.loads(
        (Path("data/cache/a1").glob("*/bonafide_frames.f32.json").__next__()).read_text()
    )["n_frames"]

    # Re-running with --resume must not error (config matches, chunks already done).
    args_resume = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--resources",
            str(resources_path),
            "train",
            "--exp",
            "smoke",
            "--resume",
        ]
    )
    args_resume.fn(args_resume)

    n_frames_after = json.loads(
        (Path("data/cache/a1").glob("*/bonafide_frames.f32.json").__next__()).read_text()
    )["n_frames"]
    assert n_frames_after == n_frames_before
    assert exp_dir.exists()


def test_front_end_hash_unchanged_without_spacing_and_differs_for_mel():
    base = {
        "features": {"name": "lfcc_b02", "n_filters": 20, "fmin": 0.0, "fmax": None},
        "preprocessing": {"sample_rate": SR, "crop_seconds": None, "silence_mode": "raw"},
        "seed": 1337,
    }
    h0 = run_a1._front_end_config_hash(base)
    # Hash must not depend on the seed (frames are seed-independent; seeds share a cache).
    assert run_a1._front_end_config_hash({**base, "seed": 2337}) == h0
    mel = {**base, "features": {**base["features"], "spacing": "mel"}}
    assert run_a1._front_end_config_hash(mel) != h0


def test_existing_lfcc_config_hash_is_the_cached_one():
    cfg = run_a1.load_config("configs/a1_lfcc_gmm.yaml")
    assert "spacing" not in cfg["features"]
    assert run_a1._front_end_config_hash(cfg) == "06c95e9be4b94e0f"


def test_mfcc_config_differs_from_lfcc_only_in_spacing_and_name():
    lfcc = run_a1.load_config("configs/a1_lfcc_gmm.yaml")
    mfcc = run_a1.load_config("configs/a1_mfcc_b02.yaml")
    assert mfcc["features"]["spacing"] == "mel"
    mfcc_feat = {k: v for k, v in mfcc["features"].items() if k != "spacing"}
    assert mfcc_feat == lfcc["features"]
    for key in ("gmm", "data", "preprocessing", "seed"):
        assert mfcc[key] == lfcc[key]


def test_tag_and_seed_flags(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_a1, "_STATUS_FN", lambda: NORMAL_STATUS)

    manifest_paths = _build_dataset(tmp_path, {"train": 4, "dev": 2, "eval": 2})
    config_path = _build_config(tmp_path, manifest_paths)
    resources_path = tmp_path / "resources.yaml"
    _write_resources_yaml(resources_path)
    common = ["--config", str(config_path), "--resources", str(resources_path)]

    parser = run_a1.build_arg_parser()
    args = parser.parse_args(
        [*common, "train", "--exp", "smoke", "--tag", "mfcc-s2337", "--seed", "2337"]
    )
    args.fn(args)
    exp_dir = next((tmp_path / "runs").glob("E-*-smoke-a1-raw-mfcc-s2337"))
    saved = yaml.safe_load((exp_dir / "config.yaml").read_text())
    assert saved["seed"] == 2337

    args = parser.parse_args(
        [
            *common,
            "score",
            "--model-from",
            str(exp_dir),
            "--exp",
            "smoke",
            "--silence-mode",
            "trimmed",
            "--testsets",
            "eval",
            "--tag",
            "mfcc-s2337",
            "--seed",
            "2337",
        ]
    )
    args.fn(args)
    score_dir = next((tmp_path / "runs").glob("E-*-smoke-a1-trimmed-mfcc-s2337"))
    assert yaml.safe_load((score_dir / "config.yaml").read_text())["seed"] == 2337

    # No flags -> default id, config seed untouched.
    args = parser.parse_args([*common, "train", "--exp", "e1"])
    assert args.tag is None and args.seed is None
