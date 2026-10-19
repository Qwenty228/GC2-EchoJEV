"""A1: LFCC -> two class-conditional diagonal GMMs."""
from audio_df2.modules.detector import GMMDetector
from audio_df2.modules.lfcc import LFCC
from audio_df2.runtime import Run


@Run.cli(dataset="2019LA")
def main(run: Run, components: int = 512, iterations: int = 10):
    """A1: official B02 LFCC + diagonal GMM baseline.
        commands:
            train: trains a GMM model on the training set and saves it to disk.
            evaluate: scores the test set using a pre-trained model.
            run: runs the full pipeline (train + evaluate) and saves the results to disk.
    
    """
    if run.command != "evaluate":
        # train and run create and save model
        model = GMMDetector(features=LFCC(), components=components, iterations=iterations)
        run.train(model)
        run.save(model)
    else:
        # eval loads
        model = GMMDetector.load(run.model_from)    
        
    if run.command != "train":
        # run and evaluate score the test set
        run.evaluate(model)
    

if __name__ == "__main__":
    main()
