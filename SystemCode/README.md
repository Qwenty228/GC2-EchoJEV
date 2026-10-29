# EchoJEV — Audio Deepfake Detection

Phase 1 implements six independent detectors with a shared training, artifact, caching, and evaluation workflow. Phase 2 tests JEV and LAYA as decision layers: first on one detector's score (FP1/FP2), now as judges that arbitrate between detectors (`modules/arbitrator.py`, `uv run arbitration`, notebook 06).

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

Download instructions, manifest preparation, controlled noise/codec generation and optional external datasets: **[data/README.md](data/README.md)**. The evaluation matrix and research questions are in `../docs/index.html`.

## Training and evaluation

```bash
uv run data-prepare --workers 4
uv run a1 train --i 2019LA
uv run a1 evaluate --i 2019LA --testsets eval
uv run d2 run --i 2019LA --testsets eval
```

Replace `a1` with `a2`, `a4`, `d1`, `d2` or `d3`. `train` creates `artifacts/<architecture>/<timestamp>/`; `evaluate` loads the latest completed artifact (or `--model-from PATH`) and writes scores and metrics to `runs/E-<timestamp>-<architecture>/`. `run` trains **and** evaluates. Feature caches live under `data/cache/`; do not delete them unless intentionally changing preprocessing or cache configuration.

Training uses ASVspoof 2019 LA train/dev. External test sets are held out. By default, `--threshold-from dev` selects the operating threshold separately from `--testsets eval` (final testing). A dev-selected threshold is used for fixed-threshold metrics; EER and ROC AUC are score-based. `minDCF` is **not** the ASVspoof `min-tDCF`. Inspect metric settings before comparing historical runs.

**Silence:** final models train and test on silence-trimmed 2019 LA audio (each run's `config.yaml` records `train_preprocessing: trimmed` and the per-test-set mode); external test sets are evaluated raw. Raw-trained runs are kept only to show the silence shortcut (notebook 02); do not compare them with trimmed runs.

## Backend

```bash
ADF_SYSTEM=a1 uv run uvicorn backend.app:app --reload
# Or load environment variables from a local .env:
uv run uvicorn backend.app:app --env-file .env
```

`ADF_SYSTEM` selects `a1`, `a2`, `a4`, `d1`, `d2`, or `d3`; `ADF_MODEL_FROM` pins an artifact (otherwise latest completed). Each detector's threshold is its dev EER threshold from `artifacts/<system>/threshold.json`, written by every evaluation; `ADF_THRESHOLD` overrides it for the start-up system only. Without a threshold, `/api/detect` returns a score but no verdict. The backend loads trained artifacts, never trains models. Keep secrets in `.env` (not Git).

### Docker

The image holds the backend, frontend and `artifacts/` (CPU PyTorch, for `linux/amd64` and `linux/arm64`, so it runs natively on Apple Silicon too). From `SystemCode/`:

```bash
docker compose pull             # ghcr.io/qwenty228/echojev:latest
docker compose up               # http://localhost:8000; `up --build` builds it from this checkout instead
```

- **Hugging Face models** (WavLM, LAYA) are not in the image: compose mounts this computer's cache, `HF_HOME` when set, else `~/.cache/huggingface` (`%USERPROFILE%\.cache\huggingface` on Windows). Missing models download into it on first use.
- **JEV key:** click 🔑 in the app. The key is saved in the `echojev-secrets` Docker volume on this computer, never in the image or Git, and loaded again at every start. `TYPESAFE_API_KEY` in the environment takes precedence. Outside Docker the same button saves it to `.secrets/`.
- Publishing a new image (no workflow): `docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/qwenty228/echojev:latest --push .`

## Results

All six detectors are trained on trimmed 2019 LA and evaluated on ten test sets; `uv run make-rankings` collects them into `artifacts/rankings.json` (the app's leaderboard) and notebook 05 tabulates them. D2 (WavLM layer 6) has the lowest mean EER, 11.0%. `pytest` and synthetic smoke tests check the code, not model performance; external test sets are never used for model selection.
