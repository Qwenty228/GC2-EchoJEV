"""Fusion + P1: apply JEV to a completed D1, D2 or D3 detector.

Evaluation-only composition; no detector weights are updated.
Pick the fusion architecture with --base (default d2), e.g. `uv run fp1 evaluate --base d3`.
"""
from audio_df.modules.d2_detector import D2Detector
from audio_df.modules.d3_detector import D3Detector
from audio_df.modules.fusion_detector import FusionDetector
from audio_df.modules.jev import decide
from audio_df.runtime import Run

DETECTORS = {
    "d1": FusionDetector,
    "d2": D2Detector,
    "d3": D3Detector,
}


@Run.cli(dataset="2019LA")
def main(run: Run, limit: int = 10, jev_model: str = "jev-latest", base: str = "d2"):
    """Fusion + P1: evaluate a frozen fusion detector with JEV (API costs apply)."""
    if run.command == "train":
        raise ValueError("Fusion + P1 is evaluation-only; train the base detector separately")
    if base not in DETECTORS:
        raise ValueError(f"Unknown fusion base {base!r}; choose from {sorted(DETECTORS)}")
    print("WARNING: Limited Fusion + P1 runs are API demonstrations, NOT full benchmarks.", flush=True)
    model = DETECTORS[base].load(run.model_from_for(base))
    return run.evaluate_policy(model, decide, base=base, limit=limit, jev_model=jev_model)


if __name__ == "__main__":
    main()
