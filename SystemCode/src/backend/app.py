"""FastAPI app: upload a voice sample, get a bona fide / spoof verdict as JSON.

    uv run uvicorn backend.app:app --reload          # from SystemCode/
    ADF_BUNDLE=artifacts/a1-lfcc-gmm uv run uvicorn backend.app:app

Serves the single-page frontend (`SystemCode/frontend/`) at `/` and the API under `/api`:

- `GET  /api/health`  -> `{"status": "ok"}`
- `GET  /api/model`   -> the loaded detector: system, version, threshold, sample rate
- `POST /api/detect`  -> multipart field `file` (WAV, FLAC, OGG or MP3) ->
  `audio_df.models.base.DetectionResult` as JSON

The detector is any model bundle (`scripts/export_bundle.py`), chosen by the `ADF_BUNDLE`
environment variable (default `artifacts/a2-wavlm-base-plus`) and loaded once, on the first
request. Swapping A1, A2, the fusion model or a JEV-wrapped model is a different bundle,
not different code.
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

from audio_df.inference import load_detector
from audio_df.models.base import Detector

__all__ = ["app", "create_app"]

DEFAULT_BUNDLE = "artifacts/a2-wavlm-base-plus"
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 60.0


def decode_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode an uploaded file to `float32` samples (mono or `[n, channels]`) and its rate."""
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    except (sf.LibsndfileError, RuntimeError, TypeError) as exc:
        raise HTTPException(400, f"could not decode audio: {exc}") from exc
    if len(x) == 0:
        raise HTTPException(400, "the audio file is empty")
    if len(x) / sr > MAX_SECONDS:
        raise HTTPException(413, f"clip longer than {MAX_SECONDS:.0f} s")
    return np.asarray(x, dtype=np.float32), int(sr)


def create_app(bundle_dir: str | Path | None = None) -> FastAPI:
    """Build the app around one bundle (`ADF_BUNDLE`, else the default, if not given)."""
    bundle = Path(bundle_dir or os.environ.get("ADF_BUNDLE", DEFAULT_BUNDLE))
    api = FastAPI(title="Audio deepfake detection", version="0.1.0")
    state: dict[str, Detector] = {}
    lock = threading.Lock()

    def detector() -> Detector:
        with lock:
            if "detector" not in state:
                if not bundle.exists():
                    raise HTTPException(
                        503,
                        f"no model bundle at {bundle}; export one with "
                        "scripts/export_bundle.py and set ADF_BUNDLE",
                    )
                state["detector"] = load_detector(bundle)
            return state["detector"]

    @api.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.get("/api/model")
    def model() -> dict[str, Any]:
        d = detector()
        return {
            "bundle": str(bundle),
            "system": d.system,
            "model_version": d.model_version,
            "threshold": d.threshold,
            "sample_rate": d.sample_rate,
        }

    # A plain `def`: inference is CPU/GPU-bound, so FastAPI runs it in a worker thread
    # instead of blocking the event loop.
    @api.post("/api/detect")
    def detect(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"file larger than {MAX_UPLOAD_BYTES // 2**20} MB")
        x, sr = decode_audio(data)
        result = detector().predict(x, sr).to_dict()
        result["filename"] = file.filename
        result["duration_s"] = round(len(x) / sr, 3)
        return result

    if FRONTEND_DIR.exists():
        api.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return api


app = create_app()
