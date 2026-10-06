# Conventions — binding on every agent

## Environment: uv, always
```bash
uv sync                      # create/refresh .venv from pyproject + uv.lock
uv run pytest                # tests
uv run ruff check src tests  # lint
uv run ruff format src tests
uv run mypy                  # types (src only)
uv run python scripts/prepare_data.py --config configs/data.yaml
```
Never `pip install`, never bare `python`, never activate the venv by hand.
Adding a dependency is `uv add <pkg>` (or `uv add --dev <pkg>`) and is **only** allowed to the
agent that owns `pyproject.toml` for that task — everyone else requests it in their report.

## Code boundaries
- `src/audio_df/` holds all reusable logic. Notebooks import from it and hold questions,
  plots and interpretation only. A function used twice moves into `src/`.
- Package purposes are fixed:
  | package | does | must not |
  |---|---|---|
  | `data/` | manifests, protocol parsing, splits, subsets, degradations, synthetic fixtures | train anything |
  | `features/` | audio load and preprocessing, LFCC-B02, cepstral grid, SSL extraction, caches | know labels or thresholds |
  | `models/` | detectors behind one interface (`base.Detector`): A1, A2, the JEV wrapper; their architectures | do run bookkeeping |
  | `training/` | fitting: A1 frame cache and chunked EM, A2 head training | write score files |
  | `evaluation/` | metrics, scoring from score files, statistics, leakage audit, figures, XAI | choose thresholds from eval |
  | `runtime/` | device, seeding, resource limits, run directories, registry | know what a model is |
  | `pipelines/` | the `adf` command: one module per system wiring the above into runs | hold reusable logic |
  | `inference.py` | `load_detector(bundle)`, the entry point for applications | know about runs |
  | `config.py` | load YAML configs | interpret them |

## Reproducibility (every run, no exceptions)
Seed numpy/torch/python; record git commit, config path + hash, dataset split hash, seed,
device, package versions. Every run writes `runs/<run id>/` (config, environment, scores) and
adds one row to `runs/REGISTRY.md`; metrics come only from `scripts/score_experiment.py`.

## Testing
- Tests run with **no dataset present**. Synthesise audio with numpy (tones, noise, chirps).
- Real-data tests are `@pytest.mark.needs_data`; model-download tests are `@pytest.mark.slow`.
- Numerical code gets a property or an analytic reference, not just a shape assertion.
  (e.g. EER of a perfectly separable score set is 0.0; an orthonormal DCT round-trips.)

## Device
Apple Silicon MPS or CPU. `device: auto` resolves cuda → mps → cpu. No `.cuda()`, no fp16
assumptions, no CUDA-only ops. Anything that cannot run on MPS falls back to CPU explicitly
and says so in its log line.

## Scientific discipline
- Thresholds are chosen on dev. **Never** on eval.
- Hold out whole attack/generator families for unseen-attack tests, never random utterances.
- Report per-attack EER alongside pooled EER; a pooled number can hide a reversed generator.
- t-SNE/UMAP are exploratory pictures, never evidence of separability.
- No literature EER is ever compared directly with our own without a `source: literature` tag.

## Definition of done for any task
1. Its tests pass under `uv run pytest` (output pasted in the report).
2. `uv run ruff check` clean on the files it owns.
3. No `NotImplementedError` left in delivered code.
4. A **claim block** is emitted for the researcher agent (see `.claude/agents/implementer.md`).
5. The researcher's verdict is `confirmed` — or the task is not done.

## Never commit
No audio, no checkpoints, no embedding caches, no dataset files, and no run outputs
(`runs/<run id>/` is gitignored; only `runs/REGISTRY.md` and the tables in
`reports/tables/` are tracked). Check `git status` before committing and keep the repo
text-sized.

## Compute budget (measured, not assumed)

Target machine is an Apple Silicon Mac. Measured on it, for ASVspoof 2019 LA
(25,380 train / 24,844 dev / 71,237 eval = 121,461 utterances, 4 s crops at 16 kHz):

| Work | Cost |
|---|---|
| LFCC cache, D=20 | 0.81 GB train, 2.29 GB eval |
| LFCC+deltas, D=60 | 2.44 GB train, 6.86 GB eval |
| ResNet18-ish 0.31M params, batch 32 | 33.7 ms/step, 949 utts/sec on MPS |
| One training epoch over train | ~0.4 min |
| 30 epochs x 3 seeds | ~40 min |
| Six representations A-F x 3 seeds | ~4 h |
| XLS-R-shaped encoder (302M params) forward | 18.7 utts/sec -> 1.8 h for the whole corpus |

This is not a large-data problem: cost scales with utterance count x model size, and both
are modest. Do not introduce distributed training, sharding or streaming dataloaders to
solve a problem this project does not have.

**The one real blowup.** Caching un-pooled SSL hidden states costs 10.24 MB per utterance
(25 layers x 199 frames x 1024 dims, fp16) = **1.24 TB** for the corpus. Pooling over time
first costs 51.2 KB per utterance = **6.22 GB**. Pool before caching (A2's cache,
`audio_df.features.ssl`). Frame-level states are for bounded subsets only, recorded as such.

**Not feasible here:** fine-tuning XLS-R end-to-end. The project uses frozen
front-ends for feasibility and reproducibility. If a fine-tuning experiment is ever run, it
belongs on rented GPU and must report trainable parameter count and compute alongside any
result, never mixed with frozen numbers.
