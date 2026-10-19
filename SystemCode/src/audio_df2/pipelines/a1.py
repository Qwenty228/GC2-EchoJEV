"""A1: LFCC -> diagonal GMM detector."""
from audio_df2.modules.detector import GMMDetector, GMMConfig
from audio_df2.modules.lfcc import LFCC, LFCCConfig
from audio_df2.runtime import Run



# lfcc = LFCC(config=LFCCConfig())
# gmm_config = GMMConfig()

@Run.cli(dataset="2019LA")
def main(run: Run):
    """Train, evaluate, or run the LFCC-GMM baseline."""
    if run.command != "evaluate":
        model = GMMDetector()
        run.train(model)
        run.save(model)
    else:
        model = GMMDetector.load(run.model_from)

    if run.command != "train":
        run.evaluate(model)


if __name__ == "__main__":
    main()
