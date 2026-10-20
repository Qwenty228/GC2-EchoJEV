"""D1 — pooled LFCC + frozen WavLM Base+ → concatenation → neural classifier.

Architecture
------------
Audio (16 kHz) → LFCC-B02 [T,60] → time mean [60] ────────────┐
Audio (16 kHz) → frozen WavLM → 13 pooled layers [13,768]      │
                     → learned layer-weighted sum [768] ──────┤
                                      concat [828] → LayerNorm
                                      → projection → MLP → bona fide logit

Scientific purpose
------------------
Compare A2 vs D1 to test whether adding explicit LFCC information improves
SSL-based detection. Compare A4 vs D1 to test whether adding SSL information
helps the LFCC neural baseline. D1 is trained independently; it does NOT load
A1's GMM or A2's trained head. Only its fusion head is trainable.

Training / evaluation
---------------------
2019 LA train; dev-loss early stopping; weighted BCE; WavLM frozen.
Reuse the identical pooled LFCC cache from A4 and WavLM cache from A2.
Use the shared run evaluator (EER, minDCF, ROC AUC, F1).
Note: LFCC pools the whole utterance (as in A4), while WavLM pools a 4 s
center crop (as in A2). This mismatch must be disclosed in comparisons.

Limitations
-----------
Time pooling removes local timing information. This is a simple concatenation
control, not temporal cross-attention or a faithful explanation mechanism.
"""
from audio_df2.modules.fusion_detector import FusionDetector
from audio_df2.modules.lfcc import LFCC
from audio_df2.modules.wavlm import WavLM
from audio_df2.runtime import Run


@Run.cli(dataset="2019LA")
def main(run: Run):
    """D1: pooled LFCC + frozen WavLM → concatenation classifier."""
    if run.command != "evaluate":
        model = FusionDetector(acoustic=LFCC(), ssl=WavLM())
        run.train(model)
        run.save(model)
    else:
        model = FusionDetector.load(run.model_from)
    if run.command != "train":
        run.evaluate(model)


if __name__ == "__main__":
    main()
