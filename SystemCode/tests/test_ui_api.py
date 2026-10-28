"""Test web UI API endpoints, model comparison, and rankings."""
from __future__ import annotations

import io
import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Use a1 for fast headless fixture testing
    app = create_app(system="a1")
    return TestClient(app)


def test_health_and_model(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}

    r_model = client.get("/api/model")
    assert r_model.status_code == 200
    data = r_model.json()
    assert data["system"] == "a1"
    assert "threshold" in data
    assert data["score_direction"] == "higher_is_bonafide"


def test_models_list(client: TestClient) -> None:
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "models" in data
    assert len(data["models"]) >= 2
    # a1 should be marked available because artifacts/a1 exists
    a1_meta = next(m for m in data["models"] if m["id"] == "a1")
    assert a1_meta["available"] is True


def test_rankings(client: TestClient) -> None:
    for dataset in ["asvspoof2019_eval", "in_the_wild"]:
        r = client.get(f"/api/rankings?dataset={dataset}")
        assert r.status_code == 200
        data = r.json()
        assert data["dataset"] == dataset
        assert len(data["data"]["items"]) >= 5
        # Top rank should be present
        top = data["data"]["items"][0]
        assert top["rank"] == 1
        assert "accuracy" in top
        assert "coverage" in top


def test_detect_audio(client: TestClient) -> None:
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    wav_bytes = buf.getvalue()

    r = client.post("/api/detect", files={"file": ("test.wav", wav_bytes, "audio/wav")})
    assert r.status_code == 200
    res = r.json()
    assert "score" in res
    assert "verdict" in res
    assert res["system"] == "a1"


def test_compare_models(client: TestClient) -> None:
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    wav_bytes = buf.getvalue()

    # Compare a1 with a1 (valid available model)
    r = client.post(
        "/api/compare",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        data={"model_a": "a1", "model_b": "a1", "abstain_margin": 0.25},
    )
    assert r.status_code == 200
    res = r.json()
    assert "arbitration" in res
    assert res["arbitration"]["agreement"] is True
    assert res["arbitration"]["status"] == "unanimous"
    assert "model_a" in res
    assert "model_b" in res


def test_compare_invalid_model(client: TestClient) -> None:
    sr = 16000
    waveform = np.zeros(sr * 2, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    wav_bytes = buf.getvalue()

    r = client.post(
        "/api/compare",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
        data={"model_a": "invalid_model", "model_b": "a1"},
    )
    assert r.status_code == 400

