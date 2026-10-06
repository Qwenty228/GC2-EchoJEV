"""A1 as a detector: LFCC-B02 front end + two diagonal GMMs, scored by mean frame LLR.

The scoring path is exactly the one `pipelines/run_a1.py` used to write its score files:
`preprocess` (silence mode, no crop) -> `extract_lfcc_b02` -> mean per-frame
log-likelihood ratio, bona fide minus spoof (`models.gmm.score_llr`). Training stays in
the pipeline (it streams millions of frames through chunked EM); this class only holds the
two trained models and turns a waveform into a verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from audio_df.features.audio import preprocess
from audio_df.features.lfcc_b02 import extract_lfcc_b02
from audio_df.models.base import (
    Detector,
    dev_threshold_from_run,
    read_bundle_meta,
    write_bundle_meta,
)
from audio_df.models.gmm import DiagGMM

__all__ = ["LfccGmmDetector"]


class LfccGmmDetector(Detector):
    """System A1. `features` and `preprocessing` are the config blocks of the run that
    trained the GMMs (`configs/a1_lfcc_gmm.yaml`); they travel with the bundle so a loaded
    detector always extracts the features its models were trained on."""

    system = "a1_lfcc_gmm"

    def __init__(
        self,
        bona_gmm: DiagGMM,
        spoof_gmm: DiagGMM,
        features: dict[str, Any],
        preprocessing: dict[str, Any],
        threshold: float,
        model_version: str,
    ) -> None:
        super().__init__(threshold, model_version)
        self.bona_gmm = bona_gmm
        self.spoof_gmm = spoof_gmm
        self.features = dict(features)
        self.preprocessing = dict(preprocessing)
        self.sample_rate = int(self.preprocessing["sample_rate"])

    def frames(self, x: np.ndarray) -> np.ndarray:
        """Preprocess and extract the `[T, 60]` LFCC frames the GMMs score."""
        x = preprocess(
            x,
            self.sample_rate,
            silence_mode=self.preprocessing["silence_mode"],
            crop_seconds=self.preprocessing.get("crop_seconds"),
        )
        return extract_lfcc_b02(x, sr=self.sample_rate, cfg=self.features)

    def score_waveform(self, x: np.ndarray) -> tuple[float, dict[str, Any]]:
        frames = self.frames(x)
        ll_bona = self.bona_gmm.score_log_likelihood(frames)
        ll_spoof = self.spoof_gmm.score_log_likelihood(frames)
        # Same expression as models.gmm.score_llr, so scores match the run files exactly.
        score = float(np.mean(ll_bona) - np.mean(ll_spoof))
        frame_llr = ll_bona - ll_spoof
        evidence = {
            "n_frames": int(len(frames)),
            "share_frames_leaning_bonafide": float(np.mean(frame_llr > 0)),
        }
        return score, evidence

    # -- bundle ---------------------------------------------------------------

    def save(self, bundle_dir: str | Path) -> Path:
        bundle_dir = write_bundle_meta(
            bundle_dir,
            {
                "system": self.system,
                "model_version": self.model_version,
                "threshold": self.threshold,
                "features": self.features,
                "preprocessing": self.preprocessing,
            },
        )
        self.bona_gmm.save(bundle_dir / "bona_gmm.npz")
        self.spoof_gmm.save(bundle_dir / "spoof_gmm.npz")
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: str | Path) -> LfccGmmDetector:
        bundle_dir = Path(bundle_dir)
        meta = read_bundle_meta(bundle_dir)
        return cls(
            DiagGMM.load(bundle_dir / "bona_gmm.npz"),
            DiagGMM.load(bundle_dir / "spoof_gmm.npz"),
            features=meta["features"],
            preprocessing=meta["preprocessing"],
            threshold=meta["threshold"],
            model_version=meta["model_version"],
        )

    @classmethod
    def from_run(cls, run_dir: str | Path, silence_mode: str | None = None) -> LfccGmmDetector:
        """Build from a trained, scored A1 run (`runs/<id>/`: GMMs, config, metrics).

        The threshold is that run's dev EER threshold. `silence_mode` overrides the run's
        preprocessing (the trimmed diagnostic scores raw-trained GMMs on trimmed audio).
        """
        run_dir = Path(run_dir)
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        preprocessing = dict(cfg["preprocessing"])
        if silence_mode is not None:
            preprocessing["silence_mode"] = silence_mode
        return cls(
            DiagGMM.load(run_dir / "bona_gmm.npz"),
            DiagGMM.load(run_dir / "spoof_gmm.npz"),
            features=cfg["features"],
            preprocessing=preprocessing,
            threshold=dev_threshold_from_run(run_dir),
            model_version=run_dir.name,
        )
