from audio_df2.modules import LFCC, NeuralHead, LFCCDetector
from audio_df2.runtime import Run


@Run.cli(dataset="2019LA")
def main(run: Run):
    """A4: LFCC → Neural Head."""

    if run.command != "evaluate":
        model = LFCCDetector(
            features=LFCC(),
            head=NeuralHead(),
        )
        run.train(model)
        run.save(model)
    else:
        model = LFCCDetector.load(run.model_from, device=run.device)

    if run.command != "train":
        run.evaluate(model)


if __name__ == "__main__":
    main()