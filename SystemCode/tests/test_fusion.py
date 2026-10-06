"""The fusion scaffold: model, frame features, cache, trainer, detector and pipeline.

Synthetic throughout: a tiny WavLM built from a config, tones as audio. No dataset, no
download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import yaml

from audio_df.features.frames import (
    FrameCacheReader,
    build_frame_cache,
    clip_streams,
    resolve_layer_weights,
    ssl_frame_states,
)
from audio_df.inference import load_detector
from audio_df.models.fusion import CrossAttentionFusion, time_encoding
from audio_df.models.fusion_detector import FusionDetector, build_fusion_model
from audio_df.pipelines import fusion as fusion_pipeline
from audio_df.training.fusion import predict_cache, train_fusion

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SR = 16000
NORMAL_STATUS = {
    "pressure_level": 1,
    "pressure_name": "normal",
    "swap_used_gb": 0.1,
    "swap_total_gb": 1.0,
    "rss_mb": 1.0,
}


def _tiny_backbone():
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
    torch.manual_seed(0)
    model = WavLMModel(config).eval()

    class _FeatureExtractor:
        def __call__(self, waveforms, sampling_rate, return_tensors="pt", padding=False):
            arr = np.stack([np.asarray(w, dtype=np.float32) for w in waveforms])
            return {"input_values": torch.from_numpy(arr)}

    return model, _FeatureExtractor()


def _cfg(tmp_path: Path) -> dict:
    lfcc = yaml.safe_load((CONFIGS / "fusion_xattn.yaml").read_text())["lfcc"]
    return {
        "system": "fusion",
        "experiment": "smoke",
        "features": {"name": "lfcc_b02_x_ssl_frames"},
        "ssl": {
            "id": "tiny",
            "revision": None,
            "n_layers": 3,
            "hidden_dim": 16,
            "layer_weights": "uniform",
        },
        "lfcc": lfcc,
        "preprocessing": {
            "sample_rate": SR,
            "crop_seconds": 0.5,
            "crop_mode": "center",
            "silence_mode": "raw",
        },
        "model": {"d_model": 16, "n_heads": 2, "n_blocks": 1, "mlp_hidden": 8, "dropout": 0.0},
        "train": {
            "lr": 3e-3,
            "batch_size": 4,
            "max_epochs": 6,
            "patience": 6,
            "pos_weight": "auto",
        },
        "data": {
            "train_manifest": str(tmp_path / "m" / "train.csv"),
            "testsets": {
                "dev": str(tmp_path / "m" / "dev.csv"),
                "eval": str(tmp_path / "m" / "eval.csv"),
            },
        },
        "cache_dir": str(tmp_path / "cache"),
        "cache_tag": "tiny-uniform",
        "cache_chunk_size": 3,
        "seed": 1337,
    }


def _tone(freq: float, seconds: float = 0.6) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# -- model --------------------------------------------------------------------


def test_time_encoding_shape_and_even_width() -> None:
    enc = time_encoding(torch.tensor([0.005, 0.015, 0.025]), 8)
    assert enc.shape == (3, 8)
    with pytest.raises(ValueError):
        time_encoding(torch.tensor([0.0]), 7)


def _model() -> CrossAttentionFusion:
    torch.manual_seed(0)
    return CrossAttentionFusion(ssl_dim=16, d_model=16, n_heads=2, n_blocks=2, dropout=0.0).eval()


def test_forward_returns_one_logit_per_clip() -> None:
    out = _model()(torch.randn(3, 40, 60), torch.randn(3, 20, 16))
    assert out.shape == (3,)


def test_padding_does_not_change_the_output() -> None:
    model = _model()
    lfcc, ssl = torch.randn(1, 40, 60), torch.randn(1, 20, 16)
    base = model(lfcc, ssl)
    lfcc_pad = torch.cat([lfcc, torch.randn(1, 10, 60)], dim=1)
    ssl_pad = torch.cat([ssl, torch.randn(1, 5, 16)], dim=1)
    mask_l = torch.zeros(1, 50, dtype=torch.bool)
    mask_l[:, 40:] = True
    mask_s = torch.zeros(1, 25, dtype=torch.bool)
    mask_s[:, 20:] = True
    padded = model(lfcc_pad, ssl_pad, lfcc_pad=mask_l, ssl_pad=mask_s)
    torch.testing.assert_close(padded, base, rtol=1e-5, atol=1e-5)


def test_attention_maps_are_returned_per_block() -> None:
    logit, maps = _model()(torch.randn(2, 40, 60), torch.randn(2, 20, 16), return_attention=True)
    assert logit.shape == (2,)
    assert len(maps) == 2
    assert maps[0]["ssl_to_lfcc"].shape == (2, 20, 40)
    assert maps[0]["lfcc_to_ssl"].shape == (2, 40, 20)


def test_gradients_reach_both_streams() -> None:
    model = _model().train()
    model(torch.randn(2, 40, 60), torch.randn(2, 20, 16)).sum().backward()
    for proj in (model.lfcc_in[1], model.ssl_in[1]):
        assert isinstance(proj, torch.nn.Linear)
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum() > 0


# -- frame features -------------------------------------------------------------


def test_resolve_layer_weights(tmp_path: Path) -> None:
    np.testing.assert_allclose(resolve_layer_weights("uniform", 4), [0.25] * 4)
    np.testing.assert_allclose(resolve_layer_weights([1, 2], 4), [0, 0.5, 0.5, 0])
    (tmp_path / "layer_weights.json").write_text(json.dumps([1.0, 3.0]))
    np.testing.assert_allclose(resolve_layer_weights(str(tmp_path), 2), [0.25, 0.75])
    with pytest.raises(ValueError):
        resolve_layer_weights([5], 4)


def test_ssl_frame_states_is_the_weighted_layer_sum() -> None:
    model, fe = _tiny_backbone()
    x = _tone(220.0, 0.5)[None, :]
    weights = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    got = ssl_frame_states(model, fe, x, SR, torch.device("cpu"), weights)
    with torch.no_grad():
        hidden = model(torch.from_numpy(x), output_hidden_states=True).hidden_states
    expected = torch.einsum("l,lbtd->btd", torch.from_numpy(weights), torch.stack(hidden)).numpy()
    np.testing.assert_allclose(got.astype(np.float32), expected, atol=2e-3, rtol=2e-3)


def test_clip_streams_cover_the_same_crop(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    lfcc, ssl = clip_streams(
        _tone(220.0, 1.2),
        _tiny_backbone(),
        cfg,
        resolve_layer_weights("uniform", 3),
        torch.device("cpu"),
    )
    assert lfcc.shape[1] == 60 and ssl.shape[1] == 16
    # 0.5 s crop: about 100 LFCC frames per second and about 50 SSL frames per second.
    assert abs(lfcc.shape[0] - 50) <= 2 and abs(ssl.shape[0] - 25) <= 2


def _write_dataset(tmp_path: Path, n: int = 4) -> None:
    (tmp_path / "audio").mkdir(exist_ok=True)
    (tmp_path / "m").mkdir(exist_ok=True)
    for split in ("train", "dev", "eval"):
        rows = []
        for label, freq, attack in (("bonafide", 220.0, "-"), ("spoof", 3200.0, "A99")):
            for i in range(n):
                path = tmp_path / "audio" / f"{split}_{label}_{i}.wav"
                sf.write(path, _tone(freq + 7 * i), SR)
                rows.append(
                    {
                        "utt_id": f"{split}_{label}_{i}",
                        "path": str(path),
                        "label": label,
                        "attack_id": attack,
                    }
                )
        pd.DataFrame(rows).to_csv(tmp_path / "m" / f"{split}.csv", index=False)


def _cache(tmp_path: Path, split: str) -> Path:
    cfg = _cfg(tmp_path)
    df = pd.read_csv(tmp_path / "m" / f"{split}.csv")
    out = tmp_path / "cache" / split
    build_frame_cache(
        df["utt_id"].tolist(),
        df["path"].tolist(),
        _tiny_backbone(),
        cfg,
        resolve_layer_weights("uniform", 3),
        out,
        torch.device("cpu"),
        chunk_size=3,
    )
    return out


def test_frame_cache_reader_streams_every_clip(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    out = _cache(tmp_path, "train")
    reader = FrameCacheReader(out)
    assert sorted(reader.ids()) == sorted(pd.read_csv(tmp_path / "m" / "train.csv")["utt_id"])
    seen = [i for ids, _, _ in reader.iter_batches(2) for i in ids]
    assert seen == reader.ids()
    a = [i for ids, _, _ in reader.iter_batches(2, True, np.random.default_rng(0)) for i in ids]
    b = [i for ids, _, _ in reader.iter_batches(2, True, np.random.default_rng(0)) for i in ids]
    assert a == b and sorted(a) == sorted(seen)
    _cache(tmp_path, "train")  # resuming a complete cache is a no-op


def test_train_fusion_learns_and_predicts(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    train, dev = (
        FrameCacheReader(_cache(tmp_path, "train")),
        FrameCacheReader(_cache(tmp_path, "dev")),
    )
    labels = {i: float("bonafide" in i) for i in train.ids() + dev.ids()}
    cfg = _cfg(tmp_path)
    result = train_fusion(
        build_fusion_model(cfg),
        train,
        labels,
        dev,
        labels,
        lr=3e-3,
        batch_size=4,
        max_epochs=8,
        patience=8,
    )
    assert result.train_losses[-1] < result.train_losses[0]
    assert result.dev_losses[result.best_epoch] == min(result.dev_losses)
    ids, logits = predict_cache(result.model, dev, 4, torch.device("cpu"))
    assert sorted(ids) == sorted(dev.ids()) and np.isfinite(logits).all()


# -- detector ---------------------------------------------------------------------


def test_fusion_detector_predicts_and_round_trips(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    detector = FusionDetector(
        build_fusion_model(cfg),
        cfg,
        resolve_layer_weights("uniform", 3),
        threshold=0.0,
        model_version="test",
        device="cpu",
        backbone=_tiny_backbone(),
    )
    result = detector.predict(_tone(220.0, 1.0), SR)
    assert result.system == "fusion_xattn" and np.isfinite(result.score)
    detector.save(tmp_path / "bundle")
    loaded = load_detector(tmp_path / "bundle")
    assert isinstance(loaded, FusionDetector)
    loaded = FusionDetector.load(tmp_path / "bundle", device="cpu")
    lfcc = np.random.default_rng(0).standard_normal((2, 50, 60)).astype(np.float32)
    ssl = np.random.default_rng(1).standard_normal((2, 25, 16)).astype(np.float32)
    np.testing.assert_array_equal(
        loaded.score_streams(lfcc, ssl), detector.score_streams(lfcc, ssl)
    )


# -- pipeline -----------------------------------------------------------------------


def test_fusion_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write_dataset(tmp_path)
    cfg_path = tmp_path / "fusion.yaml"
    cfg_path.write_text(yaml.safe_dump(_cfg(tmp_path)))
    resources = yaml.safe_load((CONFIGS / "resources.yaml").read_text())
    resources["resources"]["min_free_disk_gb"] = 0
    (tmp_path / "resources.yaml").write_text(yaml.safe_dump(resources))
    common = {
        "config": str(cfg_path),
        "resources": str(tmp_path / "resources.yaml"),
        "status_fn": lambda: NORMAL_STATUS,
        "lock_path": str(tmp_path / "job.lock"),
        "root": str(tmp_path / "runs"),
    }
    fusion_pipeline.cmd_prepare(
        argparse.Namespace(
            **common,
            testsets="train,dev,eval",
            silence_mode="raw",
            resume=False,
            backbone=_tiny_backbone(),
        )
    )
    fusion_pipeline.cmd_train(argparse.Namespace(**common, exp="smoke", resume=False))
    run = next((tmp_path / "runs").glob("E-*-smoke-fusion-raw"))
    for name in ("model_state.pt", "training_curve.json", "layer_mix.json"):
        assert (run / name).exists()
    dev = pd.read_csv(run / "scores" / "dev.csv")
    assert len(dev) == 8 and set(dev.columns) >= {"clip_id", "label", "score", "attack_id"}

    fusion_pipeline.cmd_score(
        argparse.Namespace(
            **common,
            model_from=str(run),
            exp="e3",
            silence_mode="raw",
            testsets="eval",
            resume=False,
        )
    )
    scored = next((tmp_path / "runs").glob("E-*-e3-fusion-raw"))
    assert len(pd.read_csv(scored / "scores" / "eval.csv")) == 8
