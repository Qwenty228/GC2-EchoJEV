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
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from audio_df.helpers.audio import load_audio  # noqa: F401 - shared resampling convention
from audio_df.helpers.eval_audio import trim_silence
from audio_df.modules.detector import GMMDetector
from audio_df.modules.ssl_detector import SSLDetector
from audio_df.modules.lfcc_neural import LFCCDetector
from audio_df.modules.fusion_detector import FusionDetector
from audio_df.modules.d3_detector import D3Detector
from audio_df.modules.d2_detector import D2Detector
from audio_df.helpers.research.runs import artifact_key
from audio_df.modules import jev, laya

__all__ = ["app", "create_app"]

DETECTORS = {
    "a1": GMMDetector,
    "a2": SSLDetector,
    "a4": LFCCDetector,
    "d1": FusionDetector,
    "d2": D2Detector,
    "d3": D3Detector,
}
POLICIES = {  # optional decision layer applied to the detector score
    "jev": ("JEV (TypeSafe API, paid per request)", jev.decide),
    "laya": ("LAYA (local model, free)", laya.decide),
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


def dev_threshold(root: Path, system: str, artifact: Path) -> tuple[float, str] | None:
    """Dev EER threshold from the newest evaluation run of this artifact (what FP1/FP2 use)."""
    key = f"{system}/{artifact.name}"
    for run in sorted((root / "runs").glob("E-*"), reverse=True):
        cfg_path, cal_path = run / "config.yaml", run / "calibration.json"
        if cfg_path.is_file() and cal_path.is_file():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            if cfg.get("architecture") == system and artifact_key(cfg.get("model_artifact", "")) == key:
                return float(json.loads(cal_path.read_text(encoding="utf-8"))["eer_threshold"]), f"dev EER ({run.name})"
    return None


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
                found = None if configured_threshold is not None else dev_threshold(ROOT, selected, chosen)
                state["threshold"], state["threshold_source"] = (
                    (configured_threshold, "ADF_THRESHOLD") if configured_threshold is not None
                    else found if found is not None else (None, "not set"))
            return state["model"]

    def apply_policy(name: str, score: float) -> dict[str, Any]:
        """Run JEV or LAYA on the same case FP1/FP2 use; return its answers and the full request."""
        if state["threshold"] is None:
            raise HTTPException(409, "A decision policy needs a threshold; set ADF_THRESHOLD")
        label, decide = POLICIES[name]
        case = jev.detector_case(score, state["threshold"])
        started = time.monotonic()
        try:
            raw = decide(case)
        except RuntimeError as exc:  # missing API key, package or local model
            raise HTTPException(503, f"{label} unavailable: {exc}") from exc
        response = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
        answers = response.get("answers", {}) if isinstance(response, dict) else {}
        return {
            "name": name, "label": label, "latency_s": round(time.monotonic() - started, 3),
            "model": response.get("model") if isinstance(response, dict) else None,
            "decision": answers.get("decision", {}).get("choice"),
            "answers": answers,
            "request": {"system_prompt": case.prompt, "state": {"context": dict(case.context)},
                        "questions": case.questions},
        }

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
            "threshold": state["threshold"],
            "threshold_source": state["threshold_source"],
            "score_direction": "higher_is_bonafide",
            "policies": {"none": "Detector only", **{k: v[0] for k, v in POLICIES.items()}},
        }

    @api.post("/api/detect")
    def detect(file: UploadFile = File(...), policy: str = Form("none")) -> dict[str, Any]:
        if policy != "none" and policy not in POLICIES:
            raise HTTPException(400, f"Unknown policy {policy!r}; choose none, {', '.join(POLICIES)}")
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
        thr = state["threshold"]
        verdict = None if thr is None else ("bonafide" if score >= thr else "spoof")
        return {
            "filename": file.filename,
            "duration_s": round(len(x) / sr, 3),
            "speech_s": round(speech_s, 3),
            "system": selected,
            "model_version": state["artifact"].name,
            "score": score,
            "score_direction": "higher_is_bonafide",
            "threshold": thr,
            "threshold_source": state["threshold_source"],
            "verdict": verdict,
            "policy": None if policy == "none" else apply_policy(policy, score),
        }

    if FRONTEND_DIR.is_dir():
        api.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return api


app = create_app()
