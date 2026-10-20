"""A3 = frozen A2 + JEV P1; evaluation-only composition, no new detector."""
from audio_df.runtime import Run
from audio_df.modules.ssl_detector import SSLDetector
from audio_df.modules.jev import decide
from audio_df.modules.detector import GMMDetector


@Run.cli(dataset="2019LA")
def main(run: Run, limit: int = 1, jev_model: str = "jev-latest"):
    """A3: evaluate frozen A2 predictions with JEV (API costs apply)."""
    if run.command == "train":
        raise ValueError("A3 is evaluation-only; train A2 separately")
    print("WARNING: Limited A3 runs are API demonstrations, NOT full benchmarks.", flush=True)
    
    base = "a1"
    # model = SSLDetector.load(run.model_from_for("a2"))
    model = GMMDetector.load(run.model_from_for(base))
    return run.evaluate_policy(model, decide, base=base, limit=limit, jev_model=jev_model)


if __name__ == "__main__":
    main()
