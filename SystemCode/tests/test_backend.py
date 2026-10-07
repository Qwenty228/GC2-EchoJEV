"""The FastAPI backend: health, model info, detection, and its error paths.

A tiny A1 bundle (two 2-component GMMs on random frames) stands in for a trained model.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml
from fastapi.testclient import TestClient

from audio_df.models.a1_lfcc_gmm import LfccGmmDetector
from audio_df.models.gmm import DiagGMM
from backend.app import create_app

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SR = 16000


def _gmm(seed: int) -> DiagGMM:
    gmm = DiagGMM(n_components=2, n_features=60, variance_floor=1e-3)
    data = np.random.default_rng(seed).normal(seed, 1.0, size=(300, 60))
    gmm.fit_em(data, n_iter=2, chunk_frames=100, seed=seed)
    return gmm


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    features = yaml.safe_load((CONFIGS / "a1_lfcc_gmm.yaml").read_text())["features"]
    LfccGmmDetector(
        _gmm(0),
        _gmm(1),
        features=features,
        preprocessing={"sample_rate": SR, "crop_seconds": None, "silence_mode": "raw"},
        threshold=0.5,
        model_version="test-a1",
    ).save(tmp_path / "bundle")
    return TestClient(create_app(tmp_path / "bundle"))


def _wav(seconds: float = 1.0, sr: int = SR, fmt: str = "WAV") -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format=fmt)
    return buf.getvalue()


def test_health_and_model(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}
    model = client.get("/api/model").json()
    assert model["system"] == "a1_lfcc_gmm"
    assert model["threshold"] == 0.5


def test_detect_returns_a_verdict(client: TestClient) -> None:
    r = client.post("/api/detect", files={"file": ("clip.wav", _wav(), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"bonafide", "spoof"}
    assert body["label"] == ("bonafide" if body["score"] > body["threshold"] else "spoof")
    assert body["filename"] == "clip.wav"
    assert body["duration_s"] == pytest.approx(1.0)


def test_detect_accepts_other_rates_and_formats(client: TestClient) -> None:
    flac_8k = _wav(sr=8000, fmt="FLAC")
    r = client.post("/api/detect", files={"file": ("clip.flac", flac_8k, "audio/flac")})
    assert r.status_code == 200


def test_detect_rejects_non_audio(client: TestClient) -> None:
    r = client.post("/api/detect", files={"file": ("notes.txt", b"not audio", "text/plain")})
    assert r.status_code == 400


def test_detect_rejects_long_clips(client: TestClient) -> None:
    r = client.post("/api/detect", files={"file": ("long.wav", _wav(61.0), "audio/wav")})
    assert r.status_code == 413


def test_missing_bundle_is_a_clear_503(tmp_path: Path) -> None:
    r = TestClient(create_app(tmp_path / "nothing")).get("/api/model")
    assert r.status_code == 503
    assert "export_bundle.py" in r.json()["detail"]


def test_frontend_is_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Voice spoof check" in r.text
