"""The Detector interface: A1, A2 and the JEV wrapper score, save, load and dispatch.

Synthetic throughout: tiny GMMs fitted on random frames, a tiny WavLM built from a config,
the dry-run JEV client. No dataset and no download.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from audio_df.features.audio import preprocess
from audio_df.features.lfcc_b02 import extract_lfcc_b02
from audio_df.inference import load_detector
from audio_df.models.a1_lfcc_gmm import LfccGmmDetector
from audio_df.models.a2_ssl_head import SslHeadDetector, head_from_config
from audio_df.models.base import BUNDLE_FILE, DetectionResult, decide
from audio_df.models.gmm import DiagGMM, score_llr
from audio_df.models.jev.wrapper import JEV_THRESHOLD, JevEscalation

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SR = 16000


def _tone(seconds: float = 1.5, freq: float = 220.0, sr: int = SR, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(len(t))).astype(
        np.float32
    )


def _gmm(seed: int) -> DiagGMM:
    rng = np.random.default_rng(seed)
    gmm = DiagGMM(n_components=2, n_features=60, variance_floor=1e-3)
    gmm.fit_em(rng.normal(seed, 1.0, size=(400, 60)), n_iter=2, chunk_frames=100, seed=seed)
    return gmm


@pytest.fixture
def a1() -> LfccGmmDetector:
    cfg = yaml.safe_load((CONFIGS / "a1_lfcc_gmm.yaml").read_text())
    return LfccGmmDetector(
        _gmm(0),
        _gmm(1),
        features=cfg["features"],
        preprocessing={"sample_rate": SR, "crop_seconds": None, "silence_mode": "raw"},
        threshold=0.5,
        model_version="test-a1",
    )


def test_decide_is_strictly_above_threshold() -> None:
    assert decide(1.0, 0.5) == "bonafide"
    assert decide(0.5, 0.5) == "spoof"


def test_a1_score_matches_the_runner_formula(a1: LfccGmmDetector) -> None:
    x = _tone()
    frames = extract_lfcc_b02(preprocess(x, SR, silence_mode="raw"), sr=SR, cfg=a1.features)
    expected = score_llr(a1.bona_gmm, a1.spoof_gmm, frames)
    score, evidence = a1.score_waveform(x)
    assert score == expected
    assert evidence["n_frames"] == len(frames)


def test_a1_predict_returns_structured_result(a1: LfccGmmDetector) -> None:
    result = a1.predict(_tone(), SR)
    assert isinstance(result, DetectionResult)
    assert result.label == decide(result.score, 0.5)
    assert result.system == "a1_lfcc_gmm"
    json.dumps(result.to_dict())  # JSON-ready


def test_a1_predict_resamples_and_downmixes(a1: LfccGmmDetector) -> None:
    stereo_8k = np.stack([_tone(sr=8000), _tone(sr=8000)], axis=1)
    result = a1.predict(stereo_8k, 8000)
    assert np.isfinite(result.score)


def test_a1_bundle_round_trip(a1: LfccGmmDetector, tmp_path: Path) -> None:
    a1.save(tmp_path / "a1")
    assert (tmp_path / "a1" / BUNDLE_FILE).exists()
    loaded = load_detector(tmp_path / "a1")
    assert isinstance(loaded, LfccGmmDetector)
    x = _tone(seed=3)
    assert loaded.score_waveform(x)[0] == a1.score_waveform(x)[0]
    assert loaded.threshold == a1.threshold


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


@pytest.fixture
def a2() -> SslHeadDetector:
    model_cfg = {"id": "tiny", "revision": None, "n_layers": 3, "hidden_dim": 16}
    head_cfg = {"proj_dim": 8, "mlp_hidden": 4, "dropout": 0.0}
    torch.manual_seed(0)
    return SslHeadDetector(
        head_from_config(model_cfg, head_cfg),
        model=model_cfg,
        head_config=head_cfg,
        preprocessing={
            "sample_rate": SR,
            "crop_seconds": 1.0,
            "crop_mode": "center",
            "silence_mode": "raw",
        },
        threshold=0.0,
        model_version="test-a2",
        device="cpu",
        backbone=_tiny_backbone(),
    )


def test_a2_waveform_path_equals_pooled_path(a2: SslHeadDetector) -> None:
    x = _tone()
    pooled = a2.pooled_from_waveform(x)
    assert pooled.shape == (1, 3, 16)
    assert a2.score_waveform(x)[0] == float(a2.score_pooled(pooled)[0])


def test_a2_score_pooled_matches_head_forward(a2: SslHeadDetector) -> None:
    pooled = np.random.default_rng(0).standard_normal((5, 3, 16)).astype(np.float16)
    with torch.no_grad():
        expected = a2.head(torch.tensor(pooled.astype(np.float32))).numpy()
    np.testing.assert_array_equal(a2.score_pooled(pooled), expected.astype(np.float64))


def test_a2_bundle_round_trip(a2: SslHeadDetector, tmp_path: Path) -> None:
    a2.save(tmp_path / "a2")
    loaded = load_detector(tmp_path / "a2")
    assert isinstance(loaded, SslHeadDetector)
    loaded = SslHeadDetector.load(tmp_path / "a2", device="cpu")
    pooled = np.random.default_rng(1).standard_normal((4, 3, 16)).astype(np.float32)
    np.testing.assert_array_equal(loaded.score_pooled(pooled), a2.score_pooled(pooled))


@pytest.fixture
def jev_cfg() -> dict:
    return yaml.safe_load((CONFIGS / "a3_jev.yaml").read_text())


def test_jev_always_mode_reports_the_jev_score(a1: LfccGmmDetector, jev_cfg: dict) -> None:
    wrapped = JevEscalation(a1, jev_cfg)
    result = wrapped.predict(_tone(), SR)
    assert result.threshold == JEV_THRESHOLD
    assert result.evidence["escalated"] is True
    assert result.score == result.evidence["jev_score"]
    assert result.evidence["route"] == "dry-run"
    assert "label" not in json.dumps(result.evidence["state"])


def test_jev_state_uses_the_base_threshold(a1: LfccGmmDetector, jev_cfg: dict) -> None:
    wrapped = JevEscalation(a1, jev_cfg)
    assert wrapped.jev_config["thresholds"]["detector"]["eer_threshold"] == a1.threshold


def test_jev_escalation_band(a1: LfccGmmDetector, jev_cfg: dict) -> None:
    x = _tone()
    base_score = a1.score_waveform(x)[0]
    never = JevEscalation(a1, jev_cfg, escalate_margin=0.0)
    accepted = never.predict(x, SR)
    assert accepted.evidence["escalated"] is False
    assert accepted.score == base_score
    assert accepted.threshold == a1.threshold

    always = JevEscalation(a1, jev_cfg, escalate_margin=float("inf"))
    escalated = always.predict(x, SR)
    assert escalated.evidence["escalated"] is True
    expected_label = decide(escalated.evidence["jev_score"], JEV_THRESHOLD)
    assert escalated.label == expected_label


def test_jev_bundle_round_trip(a1: LfccGmmDetector, jev_cfg: dict, tmp_path: Path) -> None:
    wrapped = JevEscalation(a1, jev_cfg, state_variant="words", question_set="all3")
    wrapped.save(tmp_path / "a1-jev")
    loaded = load_detector(tmp_path / "a1-jev")
    assert isinstance(loaded, JevEscalation)
    assert isinstance(loaded.base, LfccGmmDetector)
    x = _tone()
    assert loaded.predict(x, SR).score == wrapped.predict(x, SR).score
    assert set(loaded.predict(x, SR).evidence["answers"]) == {"noul", "choice", "score"}
