# EchoJEV — Audio Deepfake Detection

Phase 1 implements six independent detectors with a shared training, artifact, caching, and evaluation workflow. Phase 2 (JEV/LAYA decision policies) is planned separately and is **not implemented**.

## Architectures

| CLI | Detector | Trained parameters |
|---|---|---|
| `a1` | LFCC → two GMMs | GMM parameters |
| `a2` | Frozen WavLM Base+ → neural head | Head only |
| `a4` | Mean-pooled LFCC → neural head | Head only |
| `d1` | Pooled LFCC + WavLM → concatenation | Fusion head |
| `d2` | Temporal LFCC → WavLM one-way cross-attention | Projections, attention, head |
| `d3` | Temporal LFCC ↔ WavLM bidirectional cross-attention | Projections, attention, head |

WavLM Base+ remains frozen throughout. D1 shares pooled WavLM features with A2; D2/D3 share a separate 32-token temporal cache. These representations are **not interchangeable**.

## Setup and tests

From `SystemCode/` (Python and dependencies managed by [uv](https://docs.astral.sh/uv/)):

```bash
uv sync
uv run pytest -q
uv run python -m compileall -q src/audio_df src/backend
uv run data-check --paths --verify
```

Download instructions, manifest preparation, controlled noise/codec generation and optional external datasets: **[data/README.md](data/README.md)**. The evaluation matrix and research questions are in `frontend/implementation.html`.

## Training and evaluation

```bash
uv run data-prepare --workers 4
uv run a1 train --i 2019LA
uv run a1 evaluate --i 2019LA --testsets dev,eval
uv run d2 run --i 2019LA --testsets dev,eval
```

Replace `a1` with `a2`, `a4`, `d1`, `d2` or `d3`. `train` creates `artifacts/<architecture>/<timestamp>/`; `evaluate` loads the latest completed artifact (or `--model-from PATH`) and writes scores and metrics to `runs/E-<timestamp>-<architecture>/`. `run` trains **and** evaluates. Feature caches live under `data/cache/`; do not delete them unless intentionally changing preprocessing or cache configuration.

Training uses ASVspoof 2019 LA train/dev. External test sets are held out. A dev-selected threshold is used for fixed-threshold metrics; EER and ROC AUC are score-based. `minDCF` is **not** the ASVspoof `min-tDCF`. Inspect metric settings before comparing historical runs.

**Evaluation caveat:** silence-trimmed evaluation is the intended final protocol, but the current scoring paths have not been confirmed to apply trimming. Do not label existing scores “trimmed” unless the evaluated audio was actually trimmed. The A1 reference reproduction after the EM M-step fix reported 8.97% raw eval EER; do not mix this with historical trimmed results.

## Backend

```bash
ADF_SYSTEM=a1 uv run uvicorn backend.app:app --reload
# Or load environment variables from a local .env:
uv run uvicorn backend.app:app --env-file .env
```

`ADF_SYSTEM` selects `a1`, `a2`, `a4`, `d1`, `d2`, or `d3`; `ADF_MODEL_FROM` pins an artifact (otherwise latest completed). `ADF_THRESHOLD` is optional: without a calibrated threshold, `/api/detect` returns a score but no verdict. The backend loads trained artifacts, never trains models. Keep secrets in `.env` (not Git); JEV configuration belongs to Phase 2.

## Phase 1 validation status

A1 and A4 have reported real evaluation results. The remaining models, especially D2/D3 on real temporal caches and Windows CUDA, still need end-to-end validation before claiming Phase 1 experimental completion. `pytest` and synthetic smoke tests alone do not establish model performance. The current runtime is designed around 2019 LA training; external test-set manifests must be prepared and evaluated without using them for model selection.
