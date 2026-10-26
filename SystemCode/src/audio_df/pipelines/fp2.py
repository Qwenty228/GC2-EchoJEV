"""Fusion + P2: apply local LAYA (P2 policy) to a completed D1, D2 or D3 detector.

Open-source, free, local evaluation composition; no detector weights are updated.
Change BASE to select a different fusion architecture for ablations.
"""
from audio_df.modules.d2_detector import D2Detector
from audio_df.modules.d3_detector import D3Detector
from audio_df.modules.fusion_detector import FusionDetector
from audio_df.modules.laya import decide
from audio_df.runtime import Run

DETECTORS = {
    "d1": FusionDetector,
    "d2": D2Detector,
    "d3": D3Detector,
}

BASE = "d2"  # round 3 D2 (WavLM layer 6), best dual model; "d1" or "d3" for an ablation.


@Run.cli(dataset="2019LA")
def main(run: Run, limit: int = 50, laya_model: str = "convaiinnovations/laya"):
    """Fusion + P2: evaluate a frozen fusion detector with local LAYA."""
    if run.command == "train":
        raise ValueError("Fusion + P2 is evaluation-only; train the base detector separately")
    if BASE not in DETECTORS:
        raise ValueError(f"Unknown fusion base {BASE!r}; choose from {sorted(DETECTORS)}")

    model = DETECTORS[BASE].load(run.model_from_for(BASE))
    return run.evaluate_policy(model, decide, base=BASE, limit=limit, jev_model=laya_model)


if __name__ == "__main__":
    main()
