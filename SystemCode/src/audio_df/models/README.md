# `audio_df.models`: how a detector is built

Every system, from the baselines to the main model, is a **detector**: one voice sample in,
one structured verdict out. Experiment pipelines, the web backend and notebooks only ever talk
to that interface, so swapping models never touches them.

```python
from audio_df.inference import load_detector

detector = load_detector("artifacts/<bundle>")  # any system
result = detector.predict(samples, sample_rate)  # -> DetectionResult
```

## What is here

| file | what it is |
|---|---|
| `base.py` | `Detector` (the interface), `DetectionResult`, bundle helpers |
| `registry.py` | system name -> detector class; `load_detector` dispatches through it |
| `a1_lfcc_gmm.py`, `gmm.py` | baseline A1: B02 LFCC + two diagonal GMMs |
| `a2_ssl_head.py`, `ssl_head.py` | baseline A2: frozen WavLM + layer-weighted head |
| `fusion.py`, `fusion_detector.py` | **main model (scaffold)**: A1 x A2 cross-attention |
| `jev/` | JEV client, evidence state, questions, cache, and `wrapper.JevEscalation` |

## The three layers of a model

Keep these apart; it is what makes a model reusable.

1. **Architecture**: a plain `torch.nn.Module` (or, for A1, a density model). Tensors in,
   logit out. No file I/O, no configs, no thresholds. Example: `fusion.CrossAttentionFusion`.
2. **Detector**: a `base.Detector` subclass that owns the architecture plus everything needed
   to go from raw samples to a verdict: preprocessing, feature extraction, the dev threshold,
   saving and loading. Example: `fusion_detector.FusionDetector`.
3. **Training and runs** live outside `models/`: fitting in `audio_df/training/`, the
   command-line pipeline in `audio_df/pipelines/`, settings in `configs/`.

## Rules every detector follows

- **Score orientation:** higher = more bona fide, for every system. `label` is `"bonafide"`
  when `score > threshold`.
- **Threshold:** the dev-set EER threshold, read from the run's `metrics.json`
  (`base.run_threshold`). Never chosen on eval or In-the-Wild.
- **Preprocessing belongs to the detector.** `score_waveform` receives mono samples at
  `self.sample_rate` and applies its own silence handling and cropping, so callers cannot get
  it wrong. `predict` already handles other sample rates and stereo.
- **Evidence is label-free and JSON-ready:** component scores, frame counts, attention
  summaries. Never the clip's label, attack or speaker.
- **A bundle rebuilds the detector completely:** `save()` writes `bundle.json` (with
  `"system"`) plus weights; `load()` needs nothing else.

## Adding or changing a model

To try a variant of the main model, change `fusion.py` (the block, the pooling, the layer mix
in `configs/fusion_xattn.yaml`) and keep `FusionDetector`; nothing else needs to move.

To add a new system:

1. **Architecture**: `models/<name>.py`, an `nn.Module` returning one logit per clip.
2. **Detector**: subclass `Detector`, set a unique `system = "<name>"`, implement
   `score_waveform`, `save`, `load`, and a `from_run(run_dir)` that reads `config.yaml`, the
   weights and `run_threshold(run_dir)`. Copy `fusion_detector.py` as the starting point.
3. **Register** it in `registry.py` (`SYSTEMS`).
4. **Train**: put the fitting loop in `training/<name>.py` (`training/fusion.py` streams a
   frame cache with dev-loss checkpointing; `training/head.py` trains from in-memory tensors).
5. **Pipeline**: `pipelines/<name>.py` with a `COMMANDS` dict (`prepare`, `train`, `score`),
   added to `PIPELINES` in `pipelines/run.py`, and a `configs/<name>.yaml` with `system: <name>`.
6. **Export** with `scripts/export_bundle.py` (add the system to its `_FROM_RUN`), then serve it:
   `ADF_BUNDLE=artifacts/<bundle> uv run uvicorn backend.app:app`.
7. **Test** with synthetic data only: shapes, a save/load round trip through `load_detector`,
   and a tiny end-to-end pipeline run (see `tests/test_fusion.py`).

## JEV: an add-on, not a system

`jev.wrapper.JevEscalation(base, jev_config, escalate_margin=...)` wraps any detector. With
`escalate_margin=None` it asks JEV about every clip (the ablation "model + JEV"); with a margin
it asks only when the base score is close to its threshold. Export it with
`scripts/export_bundle.py ... --jev configs/a3_jev.yaml`. JEV is a typed classifier service,
not a large language model, and it only ever sees the label-free evidence state.
