"""The one entry point an application needs: load a model bundle, score a voice sample.

    from audio_df.inference import load_detector

    detector = load_detector("artifacts/a2-wavlm-base-plus")
    result = detector.predict(samples, sample_rate)   # audio_df.models.base.DetectionResult
    payload = result.to_dict()                         # JSON-ready

Any registered system (A1, A2, the fusion model, any of them wrapped with JEV) loads the
same way; the bundle's `bundle.json` names the system.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from audio_df.features.audio import load_audio
from audio_df.models.base import DetectionResult, Detector, read_bundle_meta

__all__ = ["load_detector", "predict_file"]


def load_detector(bundle_dir: str | Path) -> Detector:
    from audio_df.models.registry import detector_class

    meta = read_bundle_meta(bundle_dir)
    return detector_class(meta["system"]).load(bundle_dir)


def predict_file(detector: Detector, path: str | Path) -> DetectionResult:
    """Load an audio file at the detector's sample rate and score it."""
    x, sr = load_audio(str(path), target_sr=detector.sample_rate)
    return detector.predict(np.asarray(x), sr)
