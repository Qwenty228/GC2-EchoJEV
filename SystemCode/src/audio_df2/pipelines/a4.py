"""A4 — LFCC → neural classifier (acoustic-only discriminative control).

Architecture: 16 kHz audio → B02 LFCC [T,60] → mean over time [60]
→ existing NeuralHead with one input representation: LayerNorm(60)
→ Linear(60,256) → Linear(256,128) → GELU → Dropout → Linear(128,1).
The 1-layer softmax weight is identically 1, so no SSL layer weighting occurs.
The exact hidden widths/dropout come from configs/neural.yaml.

Purpose: A1 vs A4 tests GMM versus neural classification on LFCC.
A2 vs A4 vs D1 tests single representations versus fusion. Same train/dev
protocol and shared evaluation metrics; higher logit means bona fide.

Training: ASVspoof 2019 LA train, weighted BCE, early stopping by dev loss.
Caveat: time-mean LFCC loses local/temporal cues; not a strong temporal model.
This is a controlled baseline, not an explainability claim.
"""
from audio_df2.modules.lfcc import LFCC
from audio_df2.modules.lfcc_neural import LFCCDetector
from audio_df2.runtime import Run


@Run.cli(dataset="2019LA")
def main(run: Run):
    """A4: LFCC → neural head (train/evaluate/run)."""
    if run.command != "evaluate":
        model = LFCCDetector(features=LFCC())
        run.train(model)
        run.save(model)
    else:
        model = LFCCDetector.load(run.model_from)
    if run.command != "train":
        run.evaluate(model)


if __name__ == "__main__":
    main()
