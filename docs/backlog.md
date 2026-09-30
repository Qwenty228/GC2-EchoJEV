# Backlog

Parked on 2026-09-30 when the plan changed; code is kept, nothing deleted.

- **A2-alt** — needs a GPU; this machine is a fanless MacBook Air M4, 16 GB.
- **RawNet2** — out of scope for the three-system benchmark.
- **XLS-R layer-wise probes** — old Stage 6; superseded by A2's learned layer weights.
- **MLAAD held-out-architecture set** — ADR-001 option B; parked, not needed for E1–E6 now.
- **VoxENES 2026** — modern-generator benchmark; parked until A1–A3 run end to end.
- **Old two-branch fusion + multi-task model** — `configs/fusion_multitask.yaml`; replaced by A1/A2/A3.
- **Old Wave 4 T12/T13 baselines over the feature grid `feat_a`…`feat_f`** — superseded; feature code kept.
- **A2 alternative heads Transformer/CNN/ECAPA** — need frame sequences; only after the MLP head works, cache the learned weighted-sum frames, 1×768 per frame, ~37 GB fp16 for 121,461 clips.
- **T23 hand-written attention example** — parked (old plan): gated on T21 (layer-wise probing), which is itself superseded.
- **T26 fusion + shared trunk + detection head** — parked (old plan): superseded by A1/A2/A3.
- **T27 multi-task head** — parked (old plan): superseded by A1/A2/A3; no fusion model to attach it to.
- **T28 required ablation grid over `configs/fusion_multitask.yaml:ablations`** — parked (old plan): the fusion/multi-task model it ablates is replaced by A1/A2/A3.
- **T29 AASIST comparison** — parked (old plan): outside the three-system benchmark scope.
- **T30 WavLM backend comparison** — parked (old plan): WavLM Base+ is now A2's primary backend directly, not a comparison arm.
- **T36 controlled MLAAD subset** — parked (old plan): same rationale as the MLAAD held-out-architecture set above; not needed for E1–E6 now.
- **T37 self-generated voice-cloning family (OpenVoice)** — parked (old plan): reversal condition (T31 vs T36 disagreement) cannot be evaluated while T36 itself is parked.
