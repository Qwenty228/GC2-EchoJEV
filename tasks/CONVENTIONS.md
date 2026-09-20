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
- Module purposes are fixed:
  | module | does | must not |
  |---|---|---|
  | `audio.py` | load, resample, mono, crop/pad | compute features |
  | `features.py` | STFT → filterbank → log → DCT variants | load files, cache |
  | `cache.py` | content-hashed on-disk cache | know what a feature means |
  | `data.py` | manifests, protocol parsing, splits | train anything |
  | `stats.py` | effect size, bootstrap CI, per-feature AUC | plot |
  | `visualization.py` | figures | compute statistics |
  | `metrics.py` | EER, AUCs, calibration, per-attack | choose thresholds from eval |
  | `evaluation.py` | scoring runs, registry rows | define models |
  | `training.py` | loops, seeding, early stopping | define architectures |
  | `models/` | architectures only | do IO |

## Reproducibility (every run, no exceptions)
Seed numpy/torch/python; record git commit, config path + hash, dataset split hash, seed,
device, package versions. Append one row to `reports/experiment_registry.csv` and write the
matching machine-readable JSON next to the artifact.

## Testing
- Tests run with **no dataset present**. Synthesise audio with numpy (tones, noise, chirps).
- Real-data tests are `@pytest.mark.needs_data`; model-download tests are `@pytest.mark.slow`.
- Numerical code gets a property or an analytic reference, not just a shape assertion.
  (e.g. EER of a perfectly separable score set is 0.0; an orthonormal DCT round-trips.)

## Device
Apple Silicon MPS or CPU. `device: auto` resolves cuda → mps → cpu. No `.cuda()`, no fp16
assumptions, no CUDA-only ops. Anything that cannot run on MPS falls back to CPU explicitly
and says so in its log line.

## Scientific discipline (from the handoff spec)
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
The uncompressed `image.png` (1.8 MB) is gitignored; `docs/architecture.jpg` (272 KB) is the
committed version. No audio, no checkpoints, no embedding caches, no dataset files. Check
`git status` before committing and keep the repo text-sized.
