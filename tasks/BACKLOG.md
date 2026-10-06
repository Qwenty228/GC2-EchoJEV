> Superseded by the A1/A2/A3 plan. Parked work is listed in docs/technical-reference.md §13.

# Backlog — Stages 5–10, deliberately unscheduled

These are not "later tasks we ran out of time for". The handoff is explicit: *do not optimize
the final architecture before the diagnostic stages*. Each item below is gated on evidence
that Stages 0–4 must produce first.

| id | stage | work | gate |
|---|---|---|---|
| T20 | 5 | RawNet2 baseline (prefer official implementation; record commit + config) | Stage 4 ResNet baseline reproducible across 3 seeds |
| T21 | 6 | wav2vec 2.0 frozen layer-wise linear probing; EER vs layer index. **Pool over time before caching** - un-pooled 25-layer states cost 1.24 TB for the corpus vs 6.22 GB pooled, measured | T11 registry stable |
| T22 | 6 | Layer aggregation comparison: final / best-dev / mean / learned softmax weights | T21 shows layer-dependent EER |
| T23 | 7 | One tiny hand-written attention example, then library implementations only | T21 done — attention is explained where it is used |
| T24 | 8 | XLS-R branch: frozen, learned layer weighting → pooling → 256-d `z_X` | T22 justifies the weighting empirically |
| T25 | 9 | Handcrafted branch: best Stage-2 variant → compact ResNet → 256-d `z_L` | ADR-001 records the chosen representation |
| T26 | 10 | Fusion + shared trunk + detection head | T24 and T25 both benchmarked alone |
| T27 | 10 | Multi-task head — **only** if a defensible dataset-provided auxiliary label exists | researcher note establishing the label is real, not invented |
| T28 | 10 | Required ablation grid (7 rows, see `configs/fusion_multitask.yaml:ablations`) | T26 complete |
| T29 | — | AASIST comparison | compute budget check |
| T30 | — | WavLM backend comparison | T21 infrastructure reusable |
| T31 | — | **Level 2, free:** unseen-generator EER on ASVspoof 2019 LA's official eval (A07–A19 unseen vs A01–A06 trained). No new data needed | whole-family holdout implemented in T05 |
| T35 | — | **Level 3:** In-the-Wild as a frozen external eval set (8.16 GB, CC-BY-SA-4.0). **No generator labels, no official split** — per-generator EER is impossible; any internal split must be speaker-disjoint across its 58 speakers and justified | T31 done; set must pass `scripts/audit_leakage.py` first |
| T36 | — | Controlled MLAAD subset: pin the **version** (v1 = 63.0 h/21 models, v11 = 1153.6 h/205 models — "MLAAD" alone is not reproducible), select 8–12 architectures from `meta.csv`, hold out ≤2 whole families, save the manifest. CC-BY-NC-4.0 | ADR-001; per-architecture duration is wildly unbalanced, so naive sampling hides family imbalance |
| T37 | — | *Optional:* one self-generated voice-cloning family (OpenVoice, MIT — **not** XTTS-v2, which is CPML non-commercial with no licence path since Coqui shut down). Driven from an already-manifested bonafide corpus's speakers/transcripts so recording chain and transcript distribution are inherited, with identical post-processing and actively equalised silence and duration distributions for both classes | ADR-001 reversal condition: only if T31 and T36 disagree sharply. Must pass the leakage audit before reporting any number |
| T32 | — | Robustness: codec, additive noise, resampling, short slices | T31 done |
| T33 | — | Efficiency sweep: params, checkpoint size, peak memory, wall time, RTF | all models built |
| T34 | — | Final report sections 7–13 + benchmark table | everything above |

**Complementarity is a claim, not an assumption.** T26–T28 may not assert that handcrafted and
SSL features are complementary until the ablations show it, and the headline judgement is
whether any gain survives unseen-generator and cross-domain evaluation — not in-domain EER.


## Data strategy

See [`research/decisions/ADR-001-generalization-data-strategy.md`](../research/decisions/ADR-001-generalization-data-strategy.md).

The short version: unseen-generator evidence comes from ASVspoof's own A07–A19 eval split
(free) and from holding out MLAAD architectures (whose bonafide pairing its authors already
validated) **before** any self-generated audio is considered. Pairing self-sourced fakes
against a differently-sourced bonafide corpus is the highest-probability way to produce an
indefensible result in this entire plan — the detector learns corpus, not synthesis, and the
EER looks excellent. Any constructed or subsetted set must pass `scripts/audit_leakage.py`
before it may produce a reported number.
