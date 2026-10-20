"""A2: frozen WavLM Base+ -> layer-weighted neural head."""
from audio_df2.modules.ssl_detector import SSLDetector
from audio_df2.runtime import Run


@Run.cli(dataset="2019LA")
def main(run: Run):
    """A2: frozen WavLM Base+ -> neural head (train/evaluate/run)."""
    if run.command != "evaluate":
        model = SSLDetector()
        run.train(model)
        run.save(model)
    else:
        model = SSLDetector.load(run.model_from)
    if run.command != "train":
        run.evaluate(model)


if __name__ == "__main__":
    main()
