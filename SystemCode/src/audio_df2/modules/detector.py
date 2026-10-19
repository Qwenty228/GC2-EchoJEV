"""Binary LFCC/GMM detector: two class distributions, one likelihood ratio."""
import hashlib
import json
from pathlib import Path

from audio_df2.helpers.audio import load_audio
from audio_df2.helpers.cache import cache_frames
from audio_df2.helpers.training import fit_gmm
from audio_df2.modules.gmm import DiagGMM, score_llr
from audio_df2.modules.lfcc import LFCC


class GMMDetector:
    def __init__(self, features: LFCC, components: int = 512, iterations: int = 10):
        self.features = features
        self.components = components
        self.iterations = iterations
        self.bonafide: DiagGMM | None = None
        self.spoof: DiagGMM | None = None

    def fit(self, train, cache_dir: Path, seed: int = 1337):
        fingerprint = hashlib.sha256(
            (train.to_csv(index=False) + repr(self.features.config)).encode()
        ).hexdigest()[:16]
        for label in ("bonafide", "spoof"):
            paths = train.loc[train.label == label, "path"]
            frames = cache_frames(
                paths,
                lambda path: self.features(load_audio(path, self.features.sample_rate)),
                cache_dir,
                f"{label}-{fingerprint}",
                3 * (self.features.config.get("n_filters", 20)),
            )
            fitted = fit_gmm(
                frames, components=self.components, iterations=self.iterations, seed=seed
            )
            setattr(self, label, fitted)
        return self

    def score_audio(self, waveform):
        if self.bonafide is None or self.spoof is None:
            raise RuntimeError("Detector is not fitted")
        return score_llr(self.bonafide, self.spoof, self.features(waveform))

    def score_file(self, path):
        return self.score_audio(load_audio(path, self.features.sample_rate))

    def save(self, directory):
        if self.bonafide is None or self.spoof is None:
            raise RuntimeError("Detector is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.bonafide.save(directory / "bonafide.npz")
        self.spoof.save(directory / "spoof.npz")
        (directory / "model.json").write_text(json.dumps({
            "sample_rate": self.features.sample_rate,
            "features": self.features.config,
            "components": self.components,
            "iterations": self.iterations,
        }, indent=2))

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        cfg = json.loads((directory / "model.json").read_text())
        model = cls(LFCC(cfg["sample_rate"], **cfg["features"]),
                    cfg["components"], cfg["iterations"])
        model.bonafide = DiagGMM.load(directory / "bonafide.npz")
        model.spoof = DiagGMM.load(directory / "spoof.npz")
        return model
