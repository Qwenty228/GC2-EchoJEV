"""Test web UI API endpoints, model comparison, and rankings."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf
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


def test_rankings(client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("ADF_ARTIFACTS", str(tmp_path))
    assert client.get("/api/rankings").status_code == 404  # no rankings.json yet
    data = {"final": "d2", "testsets": [], "detectors": [{"id": "d2"}], "arbitration": []}
    (tmp_path / "rankings.json").write_text(json.dumps(data))
    r = client.get("/api/rankings")
    assert r.status_code == 200
    assert r.json() == data


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


def _tone_wav() -> bytes:
    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def _calibrated_pair() -> list[str]:
    """Two detectors on this machine whose artifact matches its stored calibration."""
    from audio_df.helpers import artifacts
    from backend.app import ROOT

    present = artifacts.available(ROOT).items()
    ok = [s for s, a in present if (artifacts.threshold(a) or {}).get("platt")]
    return ok[:2]


def test_compare_models(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    pair = _calibrated_pair()
    if len(pair) < 2:
        pytest.skip("needs two calibrated detectors on this machine")
    monkeypatch.setenv("LAYA_DRY_RUN", "1")  # deterministic offline judge
    r = client.post(
        "/api/compare",
        files={"file": ("test.wav", _tone_wav(), "audio/wav")},
        data={"model_a": pair[0], "model_b": pair[1], "judge": "laya"},
    )
    assert r.status_code == 200
    res = r.json()
    arb = res["arbitration"]
    assert arb["judge"] == "laya"
    assert arb["verdict"] in {"bonafide", "spoof", "uncertain"}
    assert set(arb["probabilities"]) == {"bonafide", "spoof", "uncertain"}
    assert len(arb["request"]["context"]) == 2
    assert 0.0 <= res["model_a"]["p_bonafide"] <= 1.0


def test_compare_same_detector(client: TestClient) -> None:
    r = client.post(
        "/api/compare",
        files={"file": ("test.wav", _tone_wav(), "audio/wav")},
        data={"model_a": "a2", "model_b": "a2"},
    )
    assert r.status_code == 400


def test_compare_jev_needs_key(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    r = client.post(
        "/api/compare",
        files={"file": ("test.wav", _tone_wav(), "audio/wav")},
        data={"model_a": "a2", "model_b": "a2", "judge": "jev"},
    )
    assert r.status_code == 400


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

