"""A3-Laya: evaluate frozen A2 predictions with local LAYA (P2 decision policy).

Open-source, free, local execution. No API costs or rate limits apply.
"""
from audio_df.modules.laya import decide
from audio_df.modules.ssl_detector import SSLDetector
from audio_df.runtime import Run


@Run.cli(dataset="2019LA")
def main(run: Run, limit: int = 50, laya_model: str = "convaiinnovations/laya"):
    """A3-Laya: evaluate frozen A2 predictions with local LAYA."""
    if run.command == "train":
        raise ValueError("A3-Laya is evaluation-only; train A2 separately")

    base = "a2"
    model = SSLDetector.load(run.model_from_for(base))
    return run.evaluate_policy(model, decide, base=base, limit=limit, jev_model=laya_model)


if __name__ == "__main__":
    main()
