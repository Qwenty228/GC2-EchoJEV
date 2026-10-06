"""The detector interface every system implements, and the on-disk model bundle.

A **detector** turns one voice sample into one structured verdict:

    detector = load_detector("artifacts/a2-wavlm")      # audio_df.inference
    result = detector.predict(samples, sample_rate)      # -> DetectionResult
    result.to_dict()  # {"label": "spoof", "score": -12.8, "threshold": -2.43, ...}

A1, A2, the fusion model and any of them wrapped with JEV are interchangeable behind this
interface, so experiment runners and the web application never branch on the system.

**Score convention.** `score` is the system's raw score, **higher = more bona fide**, the
same convention as every `runs/<id>/scores/*.csv` file. `label` is `"bonafide"` when
`score > threshold`, else `"spoof"`; `threshold` is fixed on the development set (the dev
EER threshold), never on evaluation data.

**Bundle.** A detector saves itself to a directory holding `bundle.json` (the system name,
version, threshold and everything needed to rebuild it) plus its weight files.
`audio_df.inference.load_detector` reads `bundle.json`, looks the system up in
`audio_df.models.registry`, and calls that class's `load`.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from audio_df.features.audio import to_mono_at

__all__ = [
    "BUNDLE_FILE",
    "DetectionResult",
    "dev_threshold_from_run",
    "decide",
    "Detector",
    "read_bundle_meta",
    "write_bundle_meta",
]

BUNDLE_FILE = "bundle.json"

Label = Literal["bonafide", "spoof"]


@dataclass(frozen=True)
class DetectionResult:
    """One verdict on one voice sample.

    `evidence` holds system-specific, JSON-serialisable detail: component scores, frame
    counts, JEV answers. It never holds the label of the clip or anything about its source.
    """

    label: Label
    score: float
    threshold: float
    system: str
    model_version: str
    evidence: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide(score: float, threshold: float) -> Label:
    """`"bonafide"` above the threshold, `"spoof"` at or below it (the EER convention:
    a miss is a bona fide clip with `score <= threshold`)."""
    return "bonafide" if score > threshold else "spoof"


class Detector(ABC):
    """Base class for every system. Subclasses set `system` and implement four methods."""

    system: str = ""
    sample_rate: int = 16000

    def __init__(self, threshold: float, model_version: str) -> None:
        self.threshold = float(threshold)
        self.model_version = model_version

    @abstractmethod
    def score_waveform(self, x: np.ndarray) -> tuple[float, dict[str, Any]]:
        """Score mono `float32` samples already at `self.sample_rate`.

        Returns `(score, evidence)`; the subclass applies its own preprocessing (silence
        handling, cropping) so callers never need to know it.
        """

    @abstractmethod
    def save(self, bundle_dir: str | Path) -> Path:
        """Write `bundle.json` and the weight files into `bundle_dir`; return it."""

    @classmethod
    @abstractmethod
    def load(cls, bundle_dir: str | Path) -> Detector:
        """Rebuild the detector from a directory written by `save`."""

    def predict(self, x: np.ndarray, sr: int) -> DetectionResult:
        """Score one voice sample at any sample rate, mono or multi-channel."""
        t0 = time.perf_counter()
        x = to_mono_at(x, sr, self.sample_rate)
        score, evidence = self.score_waveform(x)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return DetectionResult(
            label=decide(score, self.threshold),
            score=float(score),
            threshold=self.threshold,
            system=self.system,
            model_version=self.model_version,
            evidence=evidence,
            latency_ms=latency_ms,
        )


def write_bundle_meta(bundle_dir: str | Path, meta: dict[str, Any]) -> Path:
    """Create `bundle_dir` and write `meta` (must include `system`) as `bundle.json`."""
    if "system" not in meta:
        raise ValueError("bundle metadata must name its 'system'")
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / BUNDLE_FILE).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    return bundle_dir


def read_bundle_meta(bundle_dir: str | Path) -> dict[str, Any]:
    path = Path(bundle_dir) / BUNDLE_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: not a model bundle directory")
    meta: dict[str, Any] = json.loads(path.read_text())
    return meta


def dev_threshold_from_run(run_dir: str | Path) -> float:
    """The dev EER threshold recorded in a scored run's `metrics.json`.

    Prefers `testsets.dev.eer_threshold`; falls back to `threshold_source.value` for runs
    that borrowed their threshold from another run. Raises if the run is unscored.
    """
    metrics_path = Path(run_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found; score the run first: "
            f"uv run python scripts/score_experiment.py {run_dir}"
        )
    metrics = json.loads(metrics_path.read_text())
    dev = ((metrics.get("testsets") or {}).get("dev") or {}).get("eer_threshold")
    if dev is not None:
        return float(dev)
    source = (metrics.get("threshold_source") or {}).get("value")
    if source is not None:
        return float(source)
    raise ValueError(f"{metrics_path} records no dev EER threshold")
