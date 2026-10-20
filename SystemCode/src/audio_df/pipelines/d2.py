"""D2 — One-way temporal LFCC/WavLM cross-attention.

Audio -> shared 4 s center crop -> LFCC / frozen WavLM last-layer frames
      -> 32 adaptive temporal bins per stream (reuse D3 temporal SSL cache)
      -> LayerNorm + projection + sinusoidal bin-center time encoding
      -> 2 LFCC-query / WavLM-key-value cross-attention blocks with residual + FFN
      -> pool LFCC and unchanged SSL streams -> concatenate -> MLP -> bona fide logit.

Only the LFCC/SSL projections, LFCC cross-attention, FFN and classifier train.
The pretrained WavLM encoder is always frozen. Unlike D3, the SSL stream is
not updated by LFCC. Both use the same compressed tokens and classifier layout,
but parameter counts differ, so differences are not a pure direction-only test.
Attention maps are diagnostics, not causal explanations. D2 uses the final
WavLM layer, not A2's pooled multi-layer representation.
"""
from audio_df.modules.d2_detector import D2Detector
from audio_df.modules.lfcc import LFCC
from audio_df.modules.wavlm import WavLM
from audio_df.runtime import Run


@Run.cli(dataset="2019LA")
def main(run: Run):
    """D2: frozen WavLM + LFCC -> one-way temporal attention."""
    if run.command != "evaluate":
        model = D2Detector(acoustic=LFCC(), ssl=WavLM())
        run.train(model)
        run.save(model)
    else:
        model = D2Detector.load(run.model_from)
    if run.command != "train":
        run.evaluate(model)


if __name__ == "__main__":
    main()
