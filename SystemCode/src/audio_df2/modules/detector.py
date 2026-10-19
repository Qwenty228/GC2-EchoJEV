"""A1 detector: two class-specific diagonal GMMs, one likelihood ratio."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from audio_df2.helpers.audio import load_audio
from audio_df2.helpers.cache import cache_frames
from audio_df2.helpers.config import load_config
from audio_df2.helpers.training import fit_gmm
from audio_df2.modules.gmm import DiagGMM, GMMConfig, score_llr
from audio_df2.modules.lfcc import LFCC, LFCCConfig


class GMMDetector:
    def __init__(
        self,
        features: LFCC | None = None,
        config: GMMConfig | None = None,
    ) -> None:
        self.features = features if features is not None else LFCC()
        self.config = config if config is not None else load_config(GMMConfig)
        self.bonafide: DiagGMM | None = None
        self.spoof: DiagGMM | None = None

    def fit(self, train, cache_dir: Path, seed: int | None = None):
        cfg = self.config
        actual_seed = cfg.seed if seed is None else seed
        fingerprint = hashlib.sha256(
            (train.to_csv(index=False) + json.dumps(self.features.config, sort_keys=True)).encode()
        ).hexdigest()[:16]
        for label in ("bonafide", "spoof"):
            paths = train.loc[train.label == label, "path"]
            frames = cache_frames(
                paths,
                lambda path: self.features(load_audio(path, self.features.sample_rate)),
                cache_dir,
                f"{label}-{fingerprint}",
                3 * self.features.settings.n_filters,
            )
            fitted = fit_gmm(
                frames,
                components=cfg.components,
                iterations=cfg.iterations,
                seed=actual_seed,
                chunk_frames=cfg.chunk_frames,
                init_sample_size=cfg.init_sample_size,
                variance_floor=cfg.variance_floor,
            )
            setattr(self, label, fitted)
        return self

    def score_audio(self, waveform):
        if self.bonafide is None or self.spoof is None:
            raise RuntimeError("Detector is not fitted")
        return score_llr(self.bonafide, self.spoof, self.features(waveform),
                         chunk_frames=self.config.chunk_frames)

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
            "features": asdict(self.features.settings),
            "gmm": asdict(self.config),
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        cfg = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        model = cls(
            features=LFCC(config=LFCCConfig(**cfg["features"])),
            config=GMMConfig(**cfg["gmm"]),
        )
        model.bonafide = DiagGMM.load(directory / "bonafide.npz")
        model.spoof = DiagGMM.load(directory / "spoof.npz")
        return model
