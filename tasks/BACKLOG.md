# Backlog — Stages 5–10, deliberately unscheduled

These are not "later tasks we ran out of time for". The handoff is explicit: *do not optimize
the final architecture before the diagnostic stages*. Each item below is gated on evidence
that Stages 0–4 must produce first.

| id | stage | work | gate |
|---|---|---|---|
| T20 | 5 | RawNet2 baseline (prefer official implementation; record commit + config) | Stage 4 ResNet baseline reproducible across 3 seeds |
| T21 | 6 | wav2vec 2.0 frozen layer-wise linear probing; EER vs layer index | T11 registry stable; embedding cache proven on disk budget |
| T22 | 6 | Layer aggregation comparison: final / best-dev / mean / learned softmax weights | T21 shows layer-dependent EER |
| T23 | 7 | One tiny hand-written attention example, then library implementations only | T21 done — attention is explained where it is used |
| T24 | 8 | XLS-R branch: frozen, learned layer weighting → pooling → 256-d `z_X` | T22 justifies the weighting empirically |
| T25 | 9 | Handcrafted branch: best Stage-2 variant → compact ResNet → 256-d `z_L` | ADR-001 records the chosen representation |
| T26 | 10 | Fusion + shared trunk + detection head | T24 and T25 both benchmarked alone |
| T27 | 10 | Multi-task head — **only** if a defensible dataset-provided auxiliary label exists | researcher note establishing the label is real, not invented |
| T28 | 10 | Required ablation grid (7 rows, see `configs/fusion_multitask.yaml:ablations`) | T26 complete |
| T29 | — | AASIST comparison | compute budget check |
| T30 | — | WavLM backend comparison | T21 infrastructure reusable |
| T31 | — | Unseen-generator + cross-dataset evaluation | whole-family holdout implemented in T05 |
| T32 | — | Robustness: codec, additive noise, resampling, short slices | T31 done |
| T33 | — | Efficiency sweep: params, checkpoint size, peak memory, wall time, RTF | all models built |
| T34 | — | Final report sections 7–13 + benchmark table | everything above |

**Complementarity is a claim, not an assumption.** T26–T28 may not assert that handcrafted and
SSL features are complementary until the ablations show it, and the headline judgement is
whether any gain survives unseen-generator and cross-domain evaluation — not in-domain EER.
