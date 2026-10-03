"""run_a2.py: config/manifest helpers, train_head, score_testset, and an end-to-end CLI run
against a tiny locally-built WavLM model and synthetic audio (no network, no real data)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import yaml

import audio_df.experiments.run_a2 as run_a2
from audio_df.models.ssl_head import SSLHead

NORMAL_STATUS = {
    "pressure_level": 1,
    "pressure_name": "normal",
    "swap_used_gb": 0.1,
    "swap_total_gb": 1.0,
    "rss_mb": 1.0,
}


def _status_fn():
    return dict(NORMAL_STATUS)


class _FakeFeatureExtractor:
    def __init__(self, do_normalize: bool = False):
        self.do_normalize = do_normalize

    def __call__(self, waveforms, sampling_rate, return_tensors="pt", padding=False):
        arr = np.stack([np.asarray(w, dtype=np.float32) for w in waveforms])
        if self.do_normalize:
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True) + 1e-7
            arr = (arr - mean) / std
        return {"input_values": torch.from_numpy(arr)}


def _tiny_wavlm():
    from transformers import WavLMConfig, WavLMModel

    config = WavLMConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
        do_stable_layer_norm=False,
        feat_extract_norm="group",
    )
    model = WavLMModel(config)
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Small unit tests
# ---------------------------------------------------------------------------


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"a": 1, "b": {"c": 2}}))
    cfg = run_a2.load_config(path)
    assert cfg == {"a": 1, "b": {"c": 2}}


def test_resolve_manifest():
    cfg = {"data": {"train_manifest": "t.csv", "testsets": {"dev": "d.csv", "eval": "e.csv"}}}
    assert run_a2.resolve_manifest(cfg, "train") == "t.csv"
    assert run_a2.resolve_manifest(cfg, "dev") == "d.csv"
    with pytest.raises(ValueError, match="unknown testset"):
        run_a2.resolve_manifest(cfg, "nope")


def test_assert_backbone_matches_raises_on_mismatch():
    model, config = _tiny_wavlm()
    cfg_ok = {"model": {"id": "x", "n_layers": 3, "hidden_dim": 16}}
    run_a2.assert_backbone_matches(model, cfg_ok)  # no raise

    cfg_bad = {"model": {"id": "x", "n_layers": 13, "hidden_dim": 768}}
    with pytest.raises(ValueError, match="hidden-state tensors"):
        run_a2.assert_backbone_matches(model, cfg_bad)


def test_align_cache_with_manifest_reorders_and_detects_missing():
    cache = {"ids": np.array(["b", "a", "c"]), "pooled": np.zeros((3, 2, 4))}
    manifest = pd.DataFrame({"utt_id": ["a", "b", "c"], "label": ["bonafide", "spoof", "spoof"]})
    pooled, ordered = run_a2.align_cache_with_manifest(cache, manifest)
    assert list(ordered["utt_id"]) == ["b", "a", "c"]
    assert pooled.shape == (3, 2, 4)

    cache_missing = {"ids": np.array(["z"]), "pooled": np.zeros((1, 2, 4))}
    with pytest.raises(ValueError, match="not found in manifest"):
        run_a2.align_cache_with_manifest(cache_missing, manifest)


def test_train_head_reduces_dev_loss_and_selects_best_dev_epoch():
    torch.manual_seed(0)
    n_layers, hidden_dim = 3, 6
    rng = np.random.default_rng(2)

    def _make(n_per_class: int) -> tuple[torch.Tensor, torch.Tensor]:
        bonafide = rng.normal(loc=1.0, scale=0.2, size=(n_per_class, n_layers, hidden_dim))
        spoof = rng.normal(loc=-1.0, scale=0.2, size=(n_per_class, n_layers, hidden_dim))
        x = torch.tensor(np.concatenate([bonafide, spoof]), dtype=torch.float32)
        y = torch.tensor([1.0] * n_per_class + [0.0] * n_per_class)
        return x, y

    x_train, y_train = _make(30)
    x_dev, y_dev = _make(15)

    head = SSLHead(n_layers=n_layers, hidden_dim=hidden_dim, proj_dim=12, mlp_hidden=8, dropout=0.0)
    result = run_a2.train_head(
        head, x_train, y_train, x_dev, y_dev,
        lr=1e-2, batch_size=16, max_epochs=40, patience=10,
    )
    assert result.dev_losses[result.best_epoch] == min(result.dev_losses)
    assert result.dev_losses[result.best_epoch] < result.dev_losses[0]
    assert result.pos_weight == pytest.approx(1.0)  # balanced synthetic data


def test_score_testset_writes_expected_columns_and_orientation(tmp_path):
    head = SSLHead(n_layers=2, hidden_dim=3, proj_dim=4, mlp_hidden=4, dropout=0.0)
    with torch.no_grad():
        # Force a large positive bias so every clip scores strongly bona fide, then rely on
        # a hand-picked weight to separate label A from label B deterministically.
        for p in head.parameters():
            p.zero_()
        head.proj.bias.copy_(torch.tensor([5.0, -5.0, 0.0, 0.0]))

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    ids = np.array(["u1", "u2", "u3"])
    pooled = np.zeros((3, 2, 3), dtype=np.float16)
    pooled[0] = 1.0   # -> bonafide-ish input
    pooled[1] = -1.0
    pooled[2] = 0.5
    np.savez(cache_dir / "chunk_00000.npz", ids=ids, pooled=pooled)

    manifest = pd.DataFrame(
        {
            "utt_id": ["u1", "u2", "u3"],
            "label": ["bonafide", "spoof", "bonafide"],
            "attack_id": ["-", "A01", "-"],
        }
    )
    df = run_a2.score_testset(head, cache_dir, manifest, torch.device("cpu"))
    assert list(df.columns) == ["clip_id", "label", "score", "attack_id"]
    assert list(df["clip_id"]) == ["u1", "u2", "u3"]
    assert np.isfinite(df["score"]).all()


# ---------------------------------------------------------------------------
# End-to-end CLI run (tiny model, synthetic audio, no network, mocked memory status)
# ---------------------------------------------------------------------------


def _sine(n_samples: int, freq: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(n_samples) / sr
    return (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_split(audio_dir: Path, n_bonafide: int, n_spoof: int, prefix: str) -> pd.DataFrame:
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for label, count in (("bonafide", n_bonafide), ("spoof", n_spoof)):
        for _ in range(count):
            utt_id = f"{prefix}_{idx:03d}"
            path = audio_dir / f"{utt_id}.wav"
            sf.write(str(path), _sine(8000, freq=150.0 + idx), 16000)
            rows.append(
                {
                    "utt_id": utt_id,
                    "path": str(path),
                    "label": label,
                    "attack_id": "-" if label == "bonafide" else "A01",
                }
            )
            idx += 1
    return pd.DataFrame(rows)


@pytest.fixture
def a2_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    audio_root = tmp_path / "audio"
    train_df = _write_split(audio_root / "train", 4, 4, "train")
    dev_df = _write_split(audio_root / "dev", 3, 3, "dev")
    eval_df = _write_split(audio_root / "eval", 3, 3, "eval")

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    train_path = manifests_dir / "train.csv"
    dev_path = manifests_dir / "dev.csv"
    eval_path = manifests_dir / "eval.csv"
    train_df.to_csv(train_path, index=False)
    dev_df.to_csv(dev_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    cache_dir = tmp_path / "cache"

    cfg = {
        "system": "a2",
        "model": {
            "id": "tiny/wavlm",
            "tag": "tiny-wavlm",
            "revision": None,
            "n_layers": 3,
            "hidden_dim": 16,
            "do_normalize": False,
        },
        "preprocessing": {
            "sample_rate": 16000,
            "crop_seconds": 0.5,
            "crop_mode": "center",
            "silence_mode": "raw",
        },
        "head": {
            "proj_dim": 12,
            "mlp_hidden": 8,
            "dropout": 0.0,
            "lr": 1.0e-2,
            "batch_size": 8,
            "max_epochs": 3,
            "patience": 2,
            "pos_weight": "auto",
        },
        "data": {
            "train_manifest": str(train_path),
            "testsets": {"dev": str(dev_path), "eval": str(eval_path)},
        },
        "features": {"name": "ssl_tiny_wavlm_layer_weighted"},
        "cache_dir": str(cache_dir),
        "seed": 1337,
    }
    config_path = tmp_path / "a2_tiny.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    # Copy configs/resources.yaml's real content so load_resources() (default path
    # "configs/resources.yaml") finds a valid file relative to the chdir'd tmp_path.
    real_resources = Path(__file__).resolve().parents[1] / "configs" / "resources.yaml"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "resources.yaml").write_text(real_resources.read_text())

    return {
        "config_path": config_path,
        "cache_dir": cache_dir,
        "tmp_path": tmp_path,
    }


def test_run_a2_end_to_end(a2_project, monkeypatch):
    model, config = _tiny_wavlm()
    fe = _FakeFeatureExtractor()
    monkeypatch.setattr(run_a2, "load_backbone", lambda model_id, revision=None: (model, fe))

    cfg_path = a2_project["config_path"]
    tmp_path = a2_project["tmp_path"]

    extract_args = argparse.Namespace(
        config=str(cfg_path),
        testsets="train,dev,eval",
        silence_mode="raw",
        resume=False,
        status_fn=_status_fn,
        lock_path=str(tmp_path / "extract.lock"),
    )
    run_a2.cmd_extract(extract_args)

    cache_dir = a2_project["cache_dir"]
    assert (cache_dir / "tiny-wavlm" / "raw" / "train").exists()
    assert (cache_dir / "tiny-wavlm" / "raw" / "dev").exists()
    assert (cache_dir / "tiny-wavlm" / "raw" / "eval").exists()
    log_path = cache_dir / "tiny-wavlm" / "extraction_log.jsonl"
    assert log_path.exists()
    log_lines = log_path.read_text().strip().splitlines()
    assert len(log_lines) == 3
    json.loads(log_lines[0])  # valid JSON

    train_args = argparse.Namespace(
        config=str(cfg_path),
        exp="e1",
        resume=False,
        root=str(tmp_path / "experiments"),
        lock_path=str(tmp_path / "train.lock"),
        status_fn=_status_fn,
    )
    run_a2.cmd_train(train_args)

    today = date.today().strftime("%Y%m%d")
    exp_dir = tmp_path / "experiments" / f"E-{today}-e1-a2-raw"
    assert exp_dir.is_dir()
    assert (exp_dir / "head_state_dict.pt").exists()
    curve = json.loads((exp_dir / "training_curve.json").read_text())
    assert "train_loss" in curve and "dev_loss" in curve and "best_epoch" in curve
    weights = json.loads((exp_dir / "layer_weights.json").read_text())
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-4

    dev_scores = pd.read_csv(exp_dir / "scores" / "dev.csv")
    eval_scores = pd.read_csv(exp_dir / "scores" / "eval.csv")
    assert set(dev_scores["label"]) <= {"bonafide", "spoof"}
    assert len(dev_scores) == 6
    assert len(eval_scores) == 6

    registry_text = (tmp_path / "experiments" / "REGISTRY.md").read_text()
    assert "| done |" in registry_text

    score_args = argparse.Namespace(
        config=str(cfg_path),
        model_from=str(exp_dir),
        exp="e3",
        silence_mode="raw",
        testsets="eval",
        resume=False,
        root=str(tmp_path / "experiments"),
        lock_path=str(tmp_path / "score.lock"),
        status_fn=_status_fn,
    )
    run_a2.cmd_score(score_args)

    score_exp_dir = tmp_path / "experiments" / f"E-{today}-e3-a2-raw"
    assert (score_exp_dir / "scores" / "eval.csv").exists()
    rescored = pd.read_csv(score_exp_dir / "scores" / "eval.csv")
    assert len(rescored) == 6
    # 2026-10-01: score runs had no registry row, so score_experiment.py crashed on them.
    registry_text = (tmp_path / "experiments" / "REGISTRY.md").read_text()
    assert f"| E-{today}-e3-a2-raw |" in registry_text
    # The config must record what this score run did, since the results tables read it.
    score_cfg = yaml.safe_load((score_exp_dir / "config.yaml").read_text())
    assert score_cfg["experiment"] == "e3"
    assert score_cfg["preprocessing"]["silence_mode"] == "raw"
    assert score_cfg["model_from"] == str(exp_dir)

    # Re-running with --resume (e.g. to repair a run) overwrites its scores instead of failing.
    run_a2.cmd_score(argparse.Namespace(**{**vars(score_args), "resume": True}))
    assert len(pd.read_csv(score_exp_dir / "scores" / "eval.csv")) == 6
