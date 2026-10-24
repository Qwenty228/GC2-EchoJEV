"""Serve a trained EchoJEV artifact and the static frontend.

From SystemCode/:
    ADF_SYSTEM=a1 uv run uvicorn backend.app:app --reload
    ADF_SYSTEM=a2 ADF_THRESHOLD=0.5 uv run uvicorn backend.app:app

ADF_SYSTEM selects a1, a2, a4, d1, d2, or d3. ADF_MODEL_FROM optionally
pins an artifact directory; otherwise the latest COMPLETE artifact is used.
ADF_THRESHOLD must be calibrated for that exact artifact before a verdict
is returned. Without it, /api/detect returns a continuous score only.

Uploads are silence-trimmed with the same recipe as the trimmed dev set the
thresholds are calibrated on; silent or too-short clips are rejected.
"""
from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from audio_df.helpers.audio import load_audio  # noqa: F401 - shared resampling convention
from audio_df.helpers.eval_audio import trim_silence
from audio_df.modules.detector import GMMDetector
from audio_df.modules.ssl_detector import SSLDetector
from audio_df.modules.lfcc_neural import LFCCDetector
from audio_df.modules.fusion_detector import FusionDetector
from audio_df.modules.d3_detector import D3Detector
from audio_df.modules.d2_detector import D2Detector

__all__ = ["app", "create_app"]

DETECTORS = {
    "a1": GMMDetector,
    "a2": SSLDetector,
    "a4": LFCCDetector,
    "d1": FusionDetector,
    "d2": D2Detector,
    "d3": D3Detector,
}
ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 60.0
MIN_SPEECH_SECONDS = 1.0
SILENT_PEAK = 1e-3  # about -60 dBFS


def resolve_artifact(root: Path, system: str) -> Path:
    parent = root / "artifacts" / system
    valid = sorted(p for p in parent.iterdir() if p.is_dir() and (p / "COMPLETE").is_file()) if parent.is_dir() else []
    if not valid:
        raise FileNotFoundError(f"No completed {system} artifact under {parent}")
    return valid[-1]


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError, TypeError, ValueError) as exc:
        raise HTTPException(400, f"Could not decode audio: {exc}") from exc
    if sr <= 0 or x.size == 0:
        raise HTTPException(400, "Audio is empty or invalid")
    if len(x) / sr > MAX_SECONDS:
        raise HTTPException(413, f"Clip exceeds {MAX_SECONDS:g} seconds")
    if not np.isfinite(x).all():
        raise HTTPException(400, "Audio contains non-finite samples")
    return np.asarray(x, dtype=np.float32), int(sr)


def _mono_resample(x: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != target_sr:
        from librosa import resample
        x = resample(np.asarray(x, dtype=np.float32), orig_sr=sr, target_sr=target_sr)
    return np.asarray(x, dtype=np.float32)


def _sample_rate(model: Any, system: str) -> int:
    if system in {"a1", "a4"}:
        return int(model.features.sample_rate)
    if system == "a2":
        return int(model.encoder.config.sample_rate)
    return int(model.ssl.config.sample_rate)


def create_app(
    artifact_dir: str | Path | None = None,
    *,
    system: str | None = None,
    threshold: float | None = None,
) -> FastAPI:
    selected = (system or os.getenv("ADF_SYSTEM", "a2")).lower()
    if selected not in DETECTORS:
        raise ValueError(f"Unsupported ADF_SYSTEM={selected!r}; choose {', '.join(DETECTORS)}")
    source = artifact_dir or os.getenv("ADF_MODEL_FROM")
    configured_threshold = threshold
    if configured_threshold is None and os.getenv("ADF_THRESHOLD") is not None:
        configured_threshold = float(os.environ["ADF_THRESHOLD"])
    if configured_threshold is not None and not np.isfinite(configured_threshold):
        raise ValueError("ADF_THRESHOLD must be finite")

    api = FastAPI(title="EchoJEV audio deepfake detection", version="0.2.0")
    state: dict[str, Any] = {}
    lock = threading.Lock()

    def detector() -> Any:
        with lock:
            if "model" not in state:
                chosen = Path(source) if source is not None else resolve_artifact(ROOT, selected)
                if not chosen.is_absolute():
                    chosen = ROOT / chosen
                chosen = chosen.resolve()
                if not (chosen / "COMPLETE").is_file():
                    raise HTTPException(503, f"No completed model artifact at {chosen}")
                state["model"] = DETECTORS[selected].load(chosen)
                state["artifact"] = chosen
            return state["model"]

    @api.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/api/model")
    def model_info() -> dict[str, Any]:
        model = detector()
        artifact = state["artifact"]
        return {
            "system": selected,
            "artifact": str(artifact),
            "model_version": artifact.name,
            "sample_rate": _sample_rate(model, selected),
            "threshold": configured_threshold,
            "score_direction": "higher_is_bonafide",
        }

    @api.post("/api/detect")
    def detect(file: UploadFile = File(...)) -> dict[str, Any]:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        x, sr = decode_audio(data)
        model = detector()
        target_sr = _sample_rate(model, selected)
        waveform = _mono_resample(x, sr, target_sr)
        if np.abs(waveform).max() < SILENT_PEAK:
            raise HTTPException(422, "No sound detected; check the microphone")
        waveform = trim_silence(waveform)
        speech_s = len(waveform) / target_sr
        if speech_s < MIN_SPEECH_SECONDS:
            raise HTTPException(422, f"Only {speech_s:.1f} s of sound after trimming silence; "
                                     f"need at least {MIN_SPEECH_SECONDS:g} s of speech")
        try:
            score = float(model.score_audio(waveform))
        except Exception as exc:
            raise HTTPException(500, "Detector inference failed") from exc
        if not np.isfinite(score):
            raise HTTPException(500, "Detector returned a non-finite score")
        verdict = None if configured_threshold is None else ("bonafide" if score >= configured_threshold else "spoof")
        return {
            "filename": file.filename,
            "duration_s": round(len(x) / sr, 3),
            "speech_s": round(speech_s, 3),
            "system": selected,
            "model_version": state["artifact"].name,
            "score": score,
            "score_direction": "higher_is_bonafide",
            "threshold": configured_threshold,
            "verdict": verdict,
        }

    if FRONTEND_DIR.is_dir():
        api.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return api


app = create_app()
