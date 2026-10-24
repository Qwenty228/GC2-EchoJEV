"""D3 — Bidirectional temporal LFCC/WavLM cross-attention.

Audio -> shared 4 s center crop -> LFCC frames / frozen WavLM final-layer frames
      -> adaptive average pooling to 32 temporal bins per branch
      -> per-branch LayerNorm + Linear(128) + sinusoidal bin-center time (seconds)
      -> 2 sequential bidirectional attention blocks:
           SSL queries LFCC -> residual -> FFN;
           LFCC queries updated SSL -> residual -> FFN
      -> masked mean pooling -> concat -> MLP -> bona fide logit.

Only projection, attention, FFN and classification weights train. WavLM stays
frozen. Training selects checkpoint by dev BCE loss; scoring uses shared metrics.

The 32-bin cache is reusable by other temporal fusion heads when encoder,
preprocessing, tokenization and manifest match. A2's all-layer time-pooled
cache is NOT interchangeable. Adaptive pooling loses fine-grained timing;
attention maps are diagnostic, not causal explanations. The original ADF1
fusion uses frame-level features, so this is not exact mathematical parity.
"""
from audio_df.modules.bidirectional import BidirectionalConfig
from audio_df.modules.d3_detector import D3Detector
from audio_df.modules.lfcc import LFCC
from audio_df.modules.wavlm import WavLM
from audio_df.runtime import Run


# Dual-model settings: hard-coded BidirectionalConfig defaults (not YAML), overridden here.
# layer = WavLM hidden state used for the tokens (0 = CNN output, 12 = last). Round 2 used 12;
# round 3 uses 6: notebook 04's per-layer probe averaged over every test set (2019 eval, its
# degraded copies, In-the-Wild, 2021 LA) is best at layer 6 by both EER and minDCF.
CONFIG = BidirectionalConfig(layer=6)


@Run.cli(dataset="2019LA")
def main(run: Run):
    """D3: frozen WavLM + LFCC -> bidirectional temporal attention."""
    if run.command != "evaluate":
        model = D3Detector(acoustic=LFCC(), ssl=WavLM(), config=CONFIG)
        run.train(model)
        run.save(model)
    else:
        model = D3Detector.load(run.model_from)
    if run.command != "train":
        run.evaluate(model)


if __name__ == "__main__":
    main()
