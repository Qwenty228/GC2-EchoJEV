"""Serve a trained EchoJEV artifact and the static frontend.

From SystemCode/:
    ADF_SYSTEM=a1 uv run uvicorn backend.app:app --reload
    ADF_SYSTEM=a2 ADF_THRESHOLD=0.5 uv run uvicorn backend.app:app

ADF_SYSTEM selects a1, a2, a4, d1, d2, or d3. ADF_MODEL_FROM optionally
pins an artifact directory; otherwise the latest COMPLETE artifact is used, from
ADF_ARTIFACTS when set (e.g. weights mounted into a container), else artifacts/.
Only systems with an artifact on this machine can be loaded or compared.
The threshold is ADF_THRESHOLD when set, else the artifact's threshold.json
(its dev EER threshold, written by the runtime after evaluation).

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
from typing import Annotated, Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from audio_df.helpers import artifacts
from audio_df.helpers.audio import load_audio  # noqa: F401 - shared resampling convention
from audio_df.helpers.config import load_config
from audio_df.helpers.eval_audio import trim_silence
from audio_df.modules import arbitrator, jev, laya
from audio_df.modules.arbitrator import ArbitratorConfig
from audio_df.modules.d2_detector import D2Detector
from audio_df.modules.d3_detector import D3Detector
from audio_df.modules.detector import GMMDetector
from audio_df.modules.fusion_detector import FusionDetector
from audio_df.modules.lfcc_neural import LFCCDetector
from audio_df.modules.ssl_detector import SSLDetector

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

SYSTEM_METADATA: dict[str, dict[str, Any]] = {
    "a1": {
        "name": "LFCC-GMM (B02)",
        "description": "512-component diagonal Gaussian Mixture Models over 60-dim LFCCs",
        "domain": "Spectral Envelope & Acoustic Ensembles",
        "sample_rate": 16000,
    },
    "a2": {
        "name": "WavLM Base+ (SSL)",
        "description": "Pre-trained WavLM Base+ transformer with temporal average pooling",
        "domain": "Deep Contextual Semantics & Representations",
        "sample_rate": 16000,
    },
    "a4": {
        "name": "LFCC-MLP",
        "description": "Small MLP over time-averaged 60-dim LFCCs",
        "domain": "Spectral Envelope (Time-Averaged)",
        "sample_rate": 16000,
    },
    "d1": {
        "name": "Feature Fusion (Early)",
        "description": "Early concatenation of LFCC features and WavLM SSL representations",
        "domain": "Concatenated Multi-Modal Features",
        "sample_rate": 16000,
    },
    "d2": {
        "name": "Cross-Attention Fusion",
        "description": "Cross-attention fusion between LFCC filterbanks and WavLM Layer-6 tokens",
        "domain": "Cross-Attended Spectral & Contextual Representations",
        "sample_rate": 16000,
    },
    "d3": {
        "name": "Two-Way Cross-Attention",
        "description": "LFCC and WavLM layer-6 tokens attend to each other",
        "domain": "Two-Way Spectral & SSL Attention",
        "sample_rate": 16000,
    },
}

def artifact_threshold(artifact: Path | None) -> tuple[float | None, str]:
    """The artifact's stored dev EER threshold (threshold.json) and where it came from."""
    stored = artifacts.threshold(artifact) if artifact is not None else None
    if stored is None:
        return None, "not set"
    return float(stored["eer_threshold"]), f"{stored['source']} EER ({stored['run']})"


def detector_name(system: str) -> str:
    return f"{system.upper()} {SYSTEM_METADATA[system]['name']}"


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

    api = FastAPI(title="EchoJEV audio deepfake detection", version="0.3.0")
    model_pool: dict[str, tuple[Any, Path]] = {}
    lock = threading.Lock()

    def get_loaded_detector(sys_name: str) -> tuple[Any, Path]:
        with lock:
            if sys_name not in model_pool:
                source_path = source if (sys_name == selected and source is not None) else None
                chosen = (Path(source_path) if source_path is not None
                          else artifacts.latest(sys_name, ROOT))
                if chosen is None:
                    present = ", ".join(artifacts.available(ROOT)) or "none"
                    raise HTTPException(503, f"{sys_name} is not available on this machine; "
                                             f"available: {present}")
                if not chosen.is_absolute():
                    chosen = ROOT / chosen
                chosen = chosen.resolve()
                if not (chosen / "COMPLETE").is_file():
                    raise HTTPException(503, f"No completed model artifact at {chosen}")
                model_pool[sys_name] = (DETECTORS[sys_name].load(chosen), chosen)
            return model_pool[sys_name]

    def require_available(field: str, value: str) -> str:
        """A detector id this machine can run (the startup system may be pinned elsewhere)."""
        sys_name = (value or selected).lower().strip()
        present = artifacts.available(ROOT)
        if sys_name != selected and sys_name not in present:
            raise HTTPException(400, f"{field}={sys_name!r} is not available on this machine; "
                                     f"choose from {list(present)}")
        return sys_name

    def threshold_for(sys_name: str, artifact: Path) -> tuple[float | None, str]:
        """ADF_THRESHOLD for the startup system, else the artifact's stored dev threshold."""
        if sys_name == selected and configured_threshold is not None:
            return configured_threshold, "ADF_THRESHOLD"
        return artifact_threshold(artifact)

    def read_upload(file: UploadFile) -> tuple[np.ndarray, int]:
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        return decode_audio(data)

    def score_single(model: Any, sys_name: str, x: np.ndarray, orig_sr: int) -> float:
        target_sr = _sample_rate(model, sys_name)
        waveform = _mono_resample(x, orig_sr, target_sr)
        if np.abs(waveform).max() < SILENT_PEAK:
            raise HTTPException(422, "No sound detected; check the microphone")
        waveform = trim_silence(waveform)
        speech_s = len(waveform) / target_sr
        if speech_s < MIN_SPEECH_SECONDS:
            raise HTTPException(
                422,
                f"Only {speech_s:.1f} s of sound after trimming silence; "
                f"need at least {MIN_SPEECH_SECONDS:g} s of speech",
            )
        try:
            score = float(model.score_audio(waveform))
        except Exception as exc:
            raise HTTPException(500, f"Detector {sys_name} inference failed") from exc
        if not np.isfinite(score):
            raise HTTPException(500, f"Detector {sys_name} returned non-finite score")
        return score

    def apply_policy(name: str, score: float, threshold: float | None) -> dict[str, Any]:
        """Run JEV or LAYA on the same case FP1/FP2 use; return its answers and the full request."""
        if threshold is None:
            raise HTTPException(409, "A decision policy needs a threshold; set ADF_THRESHOLD")
        label, decide = POLICIES[name]
        case = jev.detector_case(score, threshold)
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
        model, artifact = get_loaded_detector(selected)
        threshold_value, threshold_source = threshold_for(selected, artifact)
        return {
            "system": selected,
            "artifact": str(artifact),
            "model_version": artifact.name,
            "sample_rate": _sample_rate(model, selected),
            "threshold": threshold_value,
            "threshold_source": threshold_source,
            "score_direction": "higher_is_bonafide",
            "policies": {"none": "Detector only", **{k: v[0] for k, v in POLICIES.items()}},
        }

    @api.get("/api/models")
    def list_models() -> dict[str, Any]:
        """Supported detectors (which are on this machine, their thresholds) and usable judges."""
        present = artifacts.available(ROOT)
        models_list = [
            {
                "id": sys_id,
                "name": meta["name"],
                "description": meta["description"],
                "domain": meta["domain"],
                "threshold": artifact_threshold(present[sys_id])[0] if sys_id in present else None,
                "sample_rate": meta["sample_rate"],
                "available": sys_id in present,
                "artifact": present[sys_id].name if sys_id in present else None,
                "is_current": sys_id == selected,
            }
            for sys_id, meta in SYSTEM_METADATA.items()
        ]
        usable = arbitrator.judges()
        judges = {k: {"label": v, "available": k in usable} for k, v in arbitrator.LABELS.items()}
        return {"current_system": selected, "models": models_list, "judges": judges,
                "default_judge": load_config(ArbitratorConfig).judge}

    @api.get("/api/rankings")
    def get_rankings() -> dict[str, Any]:
        """Leaderboard from <artifacts>/rankings.json, built by `uv run make-rankings`."""
        path = artifacts.artifacts_dir(ROOT) / "rankings.json"
        if not path.is_file():
            raise HTTPException(404, "No rankings yet: run `uv run make-rankings` and copy "
                                     "rankings.json into the artifacts folder")
        return json.loads(path.read_text(encoding="utf-8"))

    @api.post("/api/detect")
    def detect(
        file: Annotated[UploadFile, File()],
        policy: Annotated[str, Form()] = "none",
        system: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        if policy != "none" and policy not in POLICIES:
            choices = ", ".join(POLICIES)
            raise HTTPException(400, f"Unknown policy {policy!r}; choose none, {choices}")
        sys_name = require_available("system", system)
        x, sr = read_upload(file)
        model, artifact = get_loaded_detector(sys_name)
        score = score_single(model, sys_name, x, sr)
        thr, thr_source = threshold_for(sys_name, artifact)
        return {
            "filename": file.filename,
            "duration_s": round(len(x) / sr, 3),
            "system": sys_name,
            "model_version": artifact.name,
            "score": score,
            "score_direction": "higher_is_bonafide",
            "threshold": thr,
            "threshold_source": thr_source,
            "verdict": None if thr is None else ("bonafide" if score >= thr else "spoof"),
            "policy": None if policy == "none" else apply_policy(policy, score, thr),
        }

    @api.post("/api/compare")
    def compare(
        file: Annotated[UploadFile, File()],
        model_a: Annotated[str, Form()] = "a2",
        model_b: Annotated[str, Form()] = "a1",
        judge: Annotated[str, Form()] = "",
    ) -> dict[str, Any]:
        """Score one clip with two detectors; a LAYA or JEV judge arbitrates."""
        config = load_config(ArbitratorConfig)
        judge = judge or config.judge
        usable = arbitrator.judges()
        if judge not in usable:
            raise HTTPException(400, f"Judge {judge!r} is not usable here; "
                                     f"choose from {list(usable)}")
        if (model_a or selected).lower().strip() == (model_b or selected).lower().strip():
            raise HTTPException(400, "Pick two different detectors to arbitrate between")
        x, sr = read_upload(file)
        sides = []
        for field, value in (("model_a", model_a), ("model_b", model_b)):
            sys_name = require_available(field, value)
            model, artifact = get_loaded_detector(sys_name)
            score = score_single(model, sys_name, x, sr)
            thr = threshold_for(sys_name, artifact)[0]
            stored = artifacts.threshold(artifact)
            p = artifacts.p_bonafide(stored, score)
            if thr is None or p is None:
                raise HTTPException(409, f"No calibration for {sys_name} artifact {artifact.name}: "
                                         f"artifacts/{sys_name}/threshold.json is missing or for "
                                         "another artifact. Evaluate this artifact, or run "
                                         "`python -m audio_df.helpers.artifacts` if runs/ has it")
            sides.append({
                "system": sys_name, "name": detector_name(sys_name), "artifact": artifact.name,
                "score": score, "threshold": thr, "margin": score - thr, "p_bonafide": p,
                "error_rate_pct": stored.get("error_rate_pct"),
                "prediction": "bonafide" if score >= thr else "spoof",
                "features": SYSTEM_METADATA[sys_name]["domain"],
            })

        evidence = [{"name": s["name"], "verdict": s["prediction"], "p_bonafide": s["p_bonafide"],
                     "error_rate_pct": s["error_rate_pct"]} for s in sides]
        case = arbitrator.request(evidence, config.variant)
        try:
            answer = arbitrator.decide(case, config, judge=judge)
        except RuntimeError as exc:  # missing API key, package or local model
            raise HTTPException(503, f"{usable[judge]} unavailable: {exc}") from exc

        verdict = answer["choice"]
        agreement = sides[0]["prediction"] == sides[1]["prediction"]
        backers = [s["name"] for s in sides if s["prediction"] == verdict]
        if verdict == "uncertain":
            status = "escalated"
            rationale = f"{usable[judge]} abstained: the evidence is too mixed to decide."
        else:
            status = "unanimous" if agreement and backers else "arbitrated"
            sided = f"siding with {' and '.join(backers)}" if backers else "against both detectors"
            rationale = f"{usable[judge]} answered {verdict}, {sided}."

        def card(side: dict[str, Any]) -> dict[str, Any]:
            rounded = {k: round(side[k], 4) for k in ("score", "threshold", "margin", "p_bonafide")}
            keep = ("system", "name", "artifact", "error_rate_pct", "prediction", "features")
            return {**{k: side[k] for k in keep}, **rounded}

        return {
            "filename": file.filename,
            "duration_s": round(len(x) / sr, 3),
            "model_a": card(sides[0]),
            "model_b": card(sides[1]),
            "arbitration": {
                "judge": judge, "judge_label": usable[judge], "variant": config.variant,
                "verdict": verdict, "status": status, "agreement": agreement,
                "probabilities": answer["probabilities"], "rationale": rationale,
                "request": {"prompt": case.prompt, "context": dict(case.context),
                            "questions": case.questions},
            },
        }

    if FRONTEND_DIR.is_dir():
        api.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return api


app = create_app()
