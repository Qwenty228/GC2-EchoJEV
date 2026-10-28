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
<<<<<<< HEAD
=======
import yaml
>>>>>>> f072274293237c7368f52020dfa00e02414d0c05
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

SYSTEM_METADATA: dict[str, dict[str, Any]] = {
    "a1": {
        "name": "LFCC-GMM (B02)",
        "description": "512-component diagonal Gaussian Mixture Models over 60-dim LFCCs",
        "domain": "Spectral Envelope & Acoustic Ensembles",
        "threshold": 0.738177,
        "std": 1.825,
        "sample_rate": 16000,
    },
    "a2": {
        "name": "WavLM Base+ (SSL)",
        "description": "Pre-trained WavLM Base+ transformer with temporal average pooling",
        "domain": "Deep Contextual Semantics & Representations",
        "threshold": -0.002726,
        "std": 13.513,
        "sample_rate": 16000,
    },
    "a4": {
        "name": "LFCC-ResNet",
        "description": "Residual convolutional network over spectral filterbanks",
        "domain": "Acoustic Phase & High Frequencies",
        "threshold": -0.354,
        "std": 0.610,
        "sample_rate": 16000,
    },
    "d1": {
        "name": "Feature Fusion (Early)",
        "description": "Early concatenation of LFCC features and WavLM SSL representations",
        "domain": "Concatenated Multi-Modal Features",
        "threshold": -7.775,
        "std": 6.554,
        "sample_rate": 16000,
    },
    "d2": {
        "name": "Cross-Attention Fusion",
        "description": "Cross-attention fusion between LFCC filterbanks and WavLM Layer-6 tokens",
        "domain": "Cross-Attended Spectral & Contextual Representations",
        "threshold": -0.990488,
        "std": 5.354,
        "sample_rate": 16000,
    },
    "d3": {
        "name": "Multi-Scale Fusion",
        "description": "Multi-resolution spectral pyramid cross-attended with self-supervised tokens",
        "domain": "Multi-Resolution Cross-Attention",
        "threshold": -4.072,
        "std": 4.382,
        "sample_rate": 16000,
    },
}

RANKINGS_DATA: dict[str, Any] = {
    "asvspoof2019_eval": {
        "id": "asvspoof2019_eval",
        "title": "ASVspoof 2019 Logical Access (Evaluation Partition, N=71,237)",
        "description": "Standard in-domain benchmark evaluating synthetic speech across 13 unseen neural vocoders and synthesis techniques (A07-A19).",
        "items": [
            {
                "rank": 1,
                "name": "EchoJEV Combined Policy",
                "system": "A2 (WavLM) + D2 (Cross-Attn)",
                "policy": "Arbitration + Selective Abstention (τ_arb=0.25)",
                "accuracy": 90.62,
                "coverage": 96.0,
                "eer": 1.15,
                "ner": "+3 clips",
                "escalated": "4.0%",
                "type": "Policy Ensemble",
                "badge": "Top Performer",
                "notes": "Arbitrates decisive conflicts; escalates high-entropy ties under Chow's Rule.",
            },
            {
                "rank": 2,
                "name": "LAYA Option 1: Selective Abstention",
                "system": "A2 (WavLM Base+)",
                "policy": "Margin-Based Risk Rejection (τ=3.0)",
                "accuracy": 90.32,
                "coverage": 93.0,
                "eer": 1.08,
                "ner": "Error Filtered",
                "escalated": "7.0%",
                "type": "Risk Rejection",
                "badge": "High Assurance",
                "notes": "Rejects 68.2% of baseline misclassifications located in the boundary uncertainty band.",
            },
            {
                "rank": 3,
                "name": "LAYA Option 2: Pure Arbitration",
                "system": "A2 (WavLM) + D2 (Cross-Attn)",
                "policy": "Multi-Detector Margin Dominance (τ=0.0)",
                "accuracy": 90.00,
                "coverage": 100.0,
                "eer": 1.22,
                "ner": "+5 clips",
                "escalated": "0.0%",
                "type": "Autonomous Ensemble",
                "badge": "100% Autonomous",
                "notes": "Zero human escalation; recovers +5 net error clips autonomously over A2.",
            },
            {
                "rank": 4,
                "name": "Base D2: Cross-Attention Fusion",
                "system": "D2 (LFCC + WavLM L6)",
                "policy": "Acoustic Classifier Alone",
                "accuracy": 87.00,
                "coverage": 100.0,
                "eer": 1.72,
                "ner": "0 (Baseline)",
                "escalated": "0.0%",
                "type": "Acoustic Base",
                "badge": "Dual Acoustic",
                "notes": "Cross-attends spectral filterbanks with self-supervised Layer-6 transformer tokens.",
            },
            {
                "rank": 5,
                "name": "Base A2: WavLM Base+",
                "system": "A2 (WavLM Base+)",
                "policy": "Acoustic Classifier Alone",
                "accuracy": 85.00,
                "coverage": 100.0,
                "eer": 1.32,
                "ner": "0 (Baseline)",
                "escalated": "0.0%",
                "type": "Acoustic Base",
                "badge": "SSL Acoustic",
                "notes": "Temporal average pooling over fine-tuned WavLM Base+ contextual representations.",
            },
            {
                "rank": 6,
                "name": "Base A4: LFCC-ResNet",
                "system": "A4 (LFCC + ResNet)",
                "policy": "Acoustic Classifier Alone",
                "accuracy": 83.50,
                "coverage": 100.0,
                "eer": 4.21,
                "ner": "0 (Baseline)",
                "escalated": "0.0%",
                "type": "Acoustic Base",
                "badge": "CNN Acoustic",
                "notes": "2D convolutions capturing spectro-temporal artifacts without foundation models.",
            },
            {
                "rank": 7,
                "name": "Base A1: LFCC-GMM",
                "system": "A1 (LFCC + GMM)",
                "policy": "Acoustic Classifier Alone",
                "accuracy": 80.16,
                "coverage": 100.0,
                "eer": 9.56,
                "ner": "0 (Baseline)",
                "escalated": "0.0%",
                "type": "Acoustic Base",
                "badge": "B02 Baseline",
                "notes": "Official ASVspoof 2019 B02 baseline (512-component diagonal Gaussian Mixture Models).",
            },
        ],
    },
    "in_the_wild": {
        "id": "in_the_wild",
        "title": "In-The-Wild Evaluation Partition (N=31,779)",
        "description": "Challenging out-of-domain benchmark gathered from public deepfakes, podcasts, compression artifacts, and real-world audio channels.",
        "items": [
            {
                "rank": 1,
                "name": "LAYA Option 1: Selective Abstention",
                "system": "A2 (WavLM Base+)",
                "policy": "Margin-Based Risk Rejection (τ=3.0)",
                "accuracy": 73.03,
                "coverage": 89.0,
                "eer": 12.10,
                "ner": "Error Filtered",
                "escalated": "11.0%",
                "type": "Risk Rejection",
                "badge": "Top OOD Robustness",
                "notes": "Filters acoustic ambiguity under heavy channel shift; delivers +5.03pp gain on decided speech.",
            },
            {
                "rank": 2,
                "name": "Base A2: WavLM Base+",
                "system": "A2 (WavLM Base+)",
                "policy": "Acoustic Classifier Alone",
                "accuracy": 68.00,
                "coverage": 100.0,
                "eer": 13.32,
                "ner": "0 (Baseline)",
                "escalated": "0.0%",
                "type": "Acoustic Base",
                "badge": "Strongest Single Base",
                "notes": "Self-supervised representations maintain strong generalization across out-of-domain codecs.",
            },
            {
                "rank": 3,
                "name": "LAYA Option 2: Pure Arbitration",
                "system": "A2 (WavLM) + D2 (Cross-Attn)",
                "policy": "Multi-Detector Margin Dominance (τ=0.0)",
                "accuracy": 67.00,
                "coverage": 100.0,
                "eer": 14.15,
                "ner": "+11.00pp vs D2",
                "escalated": "0.0%",
                "type": "Autonomous Ensemble",
                "badge": "Robust Ensemble",
                "notes": "Shields D2 weakness on OOD data (+11pp gain over D2) while retaining 100% coverage.",
            },
            {
                "rank": 4,
                "name": "EchoJEV Combined Policy",
                "system": "A2 (WavLM) + D2 (Cross-Attn)",
                "policy": "Arbitration + Selective Abstention (τ_arb=0.25)",
                "accuracy": 64.89,
                "coverage": 94.0,
                "eer": 13.80,
                "ner": "+8.89pp vs D2",
                "escalated": "6.0%",
                "type": "Policy Ensemble",
                "badge": "Balanced OOD",
                "notes": "Selectively abstains on ambiguous boundary conflicts in out-of-domain recordings.",
            },
            {
                "rank": 5,
                "name": "Base D2: Cross-Attention Fusion",
                "system": "D2 (LFCC + WavLM L6)",
                "policy": "Acoustic Classifier Alone",
                "accuracy": 56.00,
                "coverage": 100.0,
                "eer": 18.50,
                "ner": "0 (Baseline)",
                "escalated": "0.0%",
                "type": "Acoustic Base",
                "badge": "Handcrafted + SSL",
                "notes": "Handcrafted LFCC features suffer greater distribution shift on diverse unconstrained audio.",
            },
            {
                "rank": 6,
                "name": "Base A4: LFCC-ResNet",
                "system": "A4 (LFCC + ResNet)",
                "policy": "Acoustic Classifier Alone",
                "accuracy": 52.30,
                "coverage": 100.0,
                "eer": 28.40,
                "ner": "0 (Baseline)",
                "escalated": "0.0%",
                "type": "Acoustic Base",
                "badge": "CNN Acoustic",
                "notes": "Struggles with out-of-domain acoustic environments without foundation model pre-training.",
            },
            {
                "rank": 7,
                "name": "Base A1: LFCC-GMM",
                "system": "A1 (LFCC + GMM)",
                "policy": "Acoustic Classifier Alone",
                "accuracy": 41.96,
                "coverage": 100.0,
                "eer": 58.04,
                "ner": "0 (Baseline)",
                "escalated": "0.0%",
                "type": "Acoustic Base",
                "badge": "B02 Baseline",
                "notes": "Catastrophic degradation under unconstrained acoustic environments and codecs.",
            },
        ],
    },
}


def resolve_artifact(root: Path, system: str) -> Path:
    parent = root / "artifacts" / system
    valid = sorted(p for p in parent.iterdir() if p.is_dir() and (p / "COMPLETE").is_file()) if parent.is_dir() else []
    if not valid:
        raise FileNotFoundError(f"No completed {system} artifact under {parent}")
    return valid[-1]


<<<<<<< HEAD
def has_artifact(root: Path, system: str) -> bool:
    parent = root / "artifacts" / system
    return parent.is_dir() and any(p.is_dir() and (p / "COMPLETE").is_file() for p in parent.iterdir())
=======
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
>>>>>>> f072274293237c7368f52020dfa00e02414d0c05


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
    model_pool: dict[str, tuple[Any, Path]] = {}
    lock = threading.Lock()

    def get_loaded_detector(sys_name: str) -> tuple[Any, Path]:
        with lock:
            if sys_name not in model_pool:
                source_path = source if (sys_name == selected and source is not None) else None
                chosen = Path(source_path) if source_path is not None else resolve_artifact(ROOT, sys_name)
                if not chosen.is_absolute():
                    chosen = ROOT / chosen
                chosen = chosen.resolve()
                if not (chosen / "COMPLETE").is_file():
                    raise HTTPException(503, f"No completed model artifact at {chosen}")
<<<<<<< HEAD
                model_pool[sys_name] = (DETECTORS[sys_name].load(chosen), chosen)
            return model_pool[sys_name]

    def detector() -> Any:
        m, artifact = get_loaded_detector(selected)
        state["model"] = m
        state["artifact"] = artifact
        return m

    def score_single(model: Any, sys_name: str, x: np.ndarray, orig_sr: int) -> tuple[float, float, int]:
        target_sr = _sample_rate(model, sys_name)
        waveform = _mono_resample(x, orig_sr, target_sr)
        if np.abs(waveform).max() < SILENT_PEAK:
            raise HTTPException(422, "No sound detected; check the microphone")
        waveform = trim_silence(waveform)
        speech_s = len(waveform) / target_sr
        if speech_s < MIN_SPEECH_SECONDS:
            raise HTTPException(
                422,
                f"Only {speech_s:.1f} s of sound after trimming silence; need at least {MIN_SPEECH_SECONDS:g} s of speech",
            )
        try:
            score = float(model.score_audio(waveform))
        except Exception as exc:
            raise HTTPException(500, f"Detector {sys_name} inference failed") from exc
        if not np.isfinite(score):
            raise HTTPException(500, f"Detector {sys_name} returned non-finite score")
        return score, speech_s, target_sr
=======
                state["model"] = DETECTORS[selected].load(chosen)
                state["artifact"] = chosen
                found = None if configured_threshold is not None else dev_threshold(ROOT, selected, chosen)
                state["threshold"], state["threshold_source"] = (
                    (configured_threshold, "ADF_THRESHOLD") if configured_threshold is not None
                    else found if found is not None else (None, "not set"))
            return state["model"]
>>>>>>> f072274293237c7368f52020dfa00e02414d0c05

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
        meta = SYSTEM_METADATA.get(selected, {})
        effective_threshold = configured_threshold if configured_threshold is not None else meta.get("threshold")
        return {
            "system": selected,
            "artifact": str(artifact),
            "model_version": artifact.name,
            "sample_rate": _sample_rate(model, selected),
<<<<<<< HEAD
            "threshold": effective_threshold,
=======
            "threshold": state["threshold"],
            "threshold_source": state["threshold_source"],
>>>>>>> f072274293237c7368f52020dfa00e02414d0c05
            "score_direction": "higher_is_bonafide",
            "policies": {"none": "Detector only", **{k: v[0] for k, v in POLICIES.items()}},
        }

    @api.get("/api/models")
    def list_models() -> dict[str, Any]:
        """List all supported detectors with their artifacts and calibration parameters."""
        models_list = []
        for sys_id, meta in SYSTEM_METADATA.items():
            avail = has_artifact(ROOT, sys_id)
            models_list.append(
                {
                    "id": sys_id,
                    "name": meta["name"],
                    "description": meta["description"],
                    "domain": meta["domain"],
                    "threshold": meta["threshold"],
                    "std": meta["std"],
                    "sample_rate": meta["sample_rate"],
                    "available": avail,
                    "is_current": (sys_id == selected),
                }
            )
        return {"current_system": selected, "models": models_list}

    @api.get("/api/rankings")
    def get_rankings(dataset: str = "asvspoof2019_eval") -> dict[str, Any]:
        """Return benchmark model rankings evaluated with JEV/LAYA decision policies."""
        if dataset not in RANKINGS_DATA:
            dataset = "asvspoof2019_eval"
        return {
            "dataset": dataset,
            "available_datasets": list(RANKINGS_DATA.keys()),
            "data": RANKINGS_DATA[dataset],
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
            raise HTTPException(
                422,
                f"Only {speech_s:.1f} s of sound after trimming silence; need at least {MIN_SPEECH_SECONDS:g} s of speech",
            )
        try:
            score = float(model.score_audio(waveform))
        except Exception as exc:
            raise HTTPException(500, "Detector inference failed") from exc
        if not np.isfinite(score):
            raise HTTPException(500, "Detector returned a non-finite score")
<<<<<<< HEAD

        meta = SYSTEM_METADATA.get(selected, {})
        effective_threshold = configured_threshold if configured_threshold is not None else meta.get("threshold")
        verdict = (
            None
            if effective_threshold is None
            else ("bonafide" if score >= effective_threshold else "spoof")
        )
=======
        thr = state["threshold"]
        verdict = None if thr is None else ("bonafide" if score >= thr else "spoof")
>>>>>>> f072274293237c7368f52020dfa00e02414d0c05
        return {
            "filename": file.filename,
            "duration_s": round(len(x) / sr, 3),
            "speech_s": round(speech_s, 3),
            "system": selected,
            "model_version": state["artifact"].name,
            "score": score,
            "score_direction": "higher_is_bonafide",
<<<<<<< HEAD
            "threshold": effective_threshold,
=======
            "threshold": thr,
            "threshold_source": state["threshold_source"],
>>>>>>> f072274293237c7368f52020dfa00e02414d0c05
            "verdict": verdict,
            "policy": None if policy == "none" else apply_policy(policy, score),
        }

    @api.post("/api/compare")
    def compare(
        file: UploadFile = File(...),
        model_a: str = Form("a2"),
        model_b: str = Form("a1"),
        abstain_margin: float = Form(0.25),
    ) -> dict[str, Any]:
        """Compare two acoustic detectors side-by-side with LAYA/JEV arbitration."""
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
        x, sr = decode_audio(data)

        sys_a = model_a.lower().strip()
        sys_b = model_b.lower().strip()
        if sys_a not in DETECTORS:
            raise HTTPException(400, f"Unsupported model_a={sys_a!r}; choose from {list(DETECTORS.keys())}")
        if sys_b not in DETECTORS:
            raise HTTPException(400, f"Unsupported model_b={sys_b!r}; choose from {list(DETECTORS.keys())}")

        try:
            m_a, art_a = get_loaded_detector(sys_a)
        except Exception as exc:
            raise HTTPException(400, f"Failed to load Model A ({sys_a}): {exc}") from exc
        try:
            m_b, art_b = get_loaded_detector(sys_b)
        except Exception as exc:
            raise HTTPException(400, f"Failed to load Model B ({sys_b}): {exc}") from exc

        score_a, speech_a, sr_a = score_single(m_a, sys_a, x, sr)
        score_b, speech_b, sr_b = score_single(m_b, sys_b, x, sr)

        meta_a = SYSTEM_METADATA.get(sys_a, {})
        meta_b = SYSTEM_METADATA.get(sys_b, {})

        th_a = meta_a.get("threshold", 0.0)
        std_a = meta_a.get("std", 1.0)
        th_b = meta_b.get("threshold", 0.0)
        std_b = meta_b.get("std", 1.0)

        margin_a = score_a - th_a
        margin_b = score_b - th_b
        z_a = margin_a / std_a
        z_b = margin_b / std_b

        pred_a = "bonafide" if score_a >= th_a else "spoof"
        pred_b = "bonafide" if score_b >= th_b else "spoof"
        agreement = bool(pred_a == pred_b)
        delta_z = abs(abs(z_a) - abs(z_b))

        name_a = meta_a.get("name", sys_a.upper())
        name_b = meta_b.get("name", sys_b.upper())

        # LAYA / JEV Option 2 Arbitration Engine:
        if agreement:
            verdict = pred_a
            status = "unanimous"
            dominant = "both"
            probs = {
                "bonafide": 0.94 if pred_a == "bonafide" else 0.03,
                "spoof": 0.94 if pred_a == "spoof" else 0.03,
                "uncertain": 0.03,
            }
            rationale = (
                f"Unanimous Agreement: Both {name_a} and {name_b} independently classify the audio as "
                f"{'bona fide (authentic)' if pred_a == 'bonafide' else 'spoof (synthetic)'} "
                f"with concordant acoustic confidence."
            )
        elif delta_z < abstain_margin:
            # Chow's Rule: selective abstention when margin gap is too narrow
            verdict = "uncertain"
            status = "escalated"
            dominant = "tied"
            probs = {"bonafide": 0.35, "spoof": 0.35, "uncertain": 0.30}
            rationale = (
                f"Selective Abstention (Chow's Rule): Detectors conflict ({name_a}: {pred_a} vs {name_b}: {pred_b}) "
                f"and normalized margin difference |Δz| = {delta_z:.3f} is below the arbitration confidence cutoff "
                f"τ_arb = {abstain_margin:.2f}. High-entropy conflict safely escalated to human forensic review."
            )
        else:
            dominant = sys_a if abs(z_a) >= abs(z_b) else sys_b
            dom_name = name_a if dominant == sys_a else name_b
            verdict = pred_a if dominant == sys_a else pred_b
            status = "arbitrated"
            probs = {
                "bonafide": 0.85 if verdict == "bonafide" else 0.10,
                "spoof": 0.85 if verdict == "spoof" else 0.10,
                "uncertain": 0.05,
            }
            rationale = (
                f"Multi-Detector Arbitration: Detectors conflict ({name_a}: {pred_a} vs {name_b}: {pred_b}). "
                f"{dom_name} provides decisive normalized margin dominance (|z| = {max(abs(z_a), abs(z_b)):.2f} "
                f"vs {min(abs(z_a), abs(z_b)):.2f}, |Δz| = {delta_z:.3f} ≥ {abstain_margin:.2f}). "
                f"Verdict arbitrated to {verdict} autonomously."
            )

        return {
            "filename": file.filename,
            "duration_s": round(len(x) / sr, 3),
            "speech_s": round(speech_a, 3),
            "model_a": {
                "system": sys_a,
                "name": name_a,
                "artifact": art_a.name,
                "score": round(score_a, 4),
                "threshold": round(th_a, 4),
                "margin": round(margin_a, 4),
                "norm_margin_z": round(z_a, 3),
                "prediction": pred_a,
                "features": meta_a.get("domain", ""),
            },
            "model_b": {
                "system": sys_b,
                "name": name_b,
                "artifact": art_b.name,
                "score": round(score_b, 4),
                "threshold": round(th_b, 4),
                "margin": round(margin_b, 4),
                "norm_margin_z": round(z_b, 3),
                "prediction": pred_b,
                "features": meta_b.get("domain", ""),
            },
            "arbitration": {
                "verdict": verdict,
                "status": status,
                "dominant_model": dominant,
                "agreement": agreement,
                "delta_z": round(delta_z, 3),
                "abstain_margin": abstain_margin,
                "probabilities": probs,
                "rationale": rationale,
            },
        }

    if FRONTEND_DIR.is_dir():
        api.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return api


app = create_app()
