# Audio Deepfake Detection — GC2: system code

This folder is the project's code: a self-contained uv project. Run every command below from
inside `SystemCode/`.

A reproducible benchmark of spoofed-speech detectors on standardised ASVspoof protocols, with
generalisation, robustness and silence-shortcut checks built in.

**Research question.** Given an utterance *x* and label *y* ∈ {bonafide, spoof}, can we learn
a detector that captures **synthesis-related evidence** rather than dataset or generator
shortcuts? The working assumption is that for some observable representation *Z*,
`P(Z | real) ≠ P(Z | fake)`. The project investigates **where** that difference is visible
and **whether it survives unseen attacks and domains**.

## Read this first

- **[The report](../ProjectReport/report/README.md)**: the main document. It covers the question, the data,
  the three detectors and every result, as one argument in twelve short chapters.
- **[Technical reference](../ProjectReport/technical-reference.md)**: full method, maths, implementation
  pointers and result tables.
- **[Research notes](../Miscellaneous/research/README.md)**: page-referenced notes on papers and methods,
  fact-checks and architecture decision records (ADRs).

## Results

| equal error rate (EER), lower is better | 2019 LA eval, raw | 2019 LA eval, silence trimmed | In-the-Wild, raw |
|---|---|---|---|
| **A1** LFCC + GMM (official B02 recipe) | 9.56 % | 19.84 % | 58.04 % |
| **A2** frozen WavLM Base+, layer-weighted head | **0.96 %** | 13.27 % | **14.70 %** |

Tables: [`reports/tables/`](reports/tables) · run log: [`runs/REGISTRY.md`](runs/REGISTRY.md).
In short: leading silence alone predicts the label with an area under the curve (AUC) of 0.837,
and trimming it costs both detectors heavily. Both detectors also detect genuine vocoder
artefacts. On real-world audio, A2's ranking holds but its decision threshold does not, and A1
falls to chance. The [report](../ProjectReport/report/README.md) gives the full argument.

## The systems and experiments

1. **Data:** ASVspoof 2019 logical access (LA), official train/dev/eval splits; 16 kHz audio.
   In-the-Wild is the cross-domain test set and is never used for training.
2. **A1:** linear-frequency cepstral coefficients (LFCC) with Gaussian mixture models (GMM), the
   official ASVspoof 2019 B02 recipe: 60-dimensional LFCC, two 512-component GMMs, a
   log-likelihood-ratio score, whole utterances.
3. **A2:** a frozen self-supervised (SSL) model (WavLM Base+; XLS-R 300M is configured) →
   learned layer weights → mean pooling → a small multilayer perceptron (MLP). Only the head
   trains, on a 4 s centre crop.
4. **A3:** A2 plus JEV, a typed classifier service (not a large language model), asked typed
   questions over a label-free evidence state per clip. Scaffold with a dry-run client.
5. **Experiments:** E1 benchmark (2019 LA eval) · E2 ablation (analyses X4 and X6 done) · E3
   generalisation (In-the-Wild and unseen attacks A07–A19) · E4 robustness (MP3 32k, Opus 16k,
   noise at 10 dB, 8 kHz resample on a 10k stratified eval subset) · E5 A2 against A3 · E6
   efficiency.
6. **Silence:** raw audio is the main condition; trimmed audio is a diagnostic on every test set.
7. **Protocol:** thresholds are fixed on dev, and metrics are computed only from score files.
8. **Machine:** a fanless MacBook Air M4 (16 GB), so resource limits and resumable jobs are built in.

| Area | Status |
|---|---|
| Data manifests, checksums, In-the-Wild loader, stratified subsets, leakage audit | done |
| Exploratory analysis (notebooks 01–03) | done |
| A1 (`pipelines/a1.py`, notebooks 09–10) | run: E1, E3, raw and trimmed; X4 with 3 seeds per arm |
| A2 with WavLM Base+ (`pipelines/a2.py`, notebook 11) | run: E1, E3, raw and trimmed; X6 layer analysis |
| A2 with XLS-R 300M | configured, not run |
| Main model: A1 x A2 cross-attention fusion (`pipelines/fusion.py`) | scaffold: model, frame cache, training and scoring, tested on synthetic data; not yet trained |
| A3 (`pipelines/a3.py`, notebook 12) | scaffold with a dry-run client; waiting for the JEV API documentation |
| E4 degradations (`make_degraded.py`, notebook 13) | generator ready, not scored |
| E6 efficiency | not built |

---

## 1. Set up the environment

Requires [uv](https://docs.astral.sh/uv/). Everything runs through it — no `pip`, no manual
venv activation.

```bash
cd SystemCode
uv sync                 # locked env: Python 3.11, torch 2.14, ~1.3 GB
uv run pytest           # no dataset needed
uv run ruff check src tests
```

Target device is Apple Silicon **MPS** or CPU. `device: auto` resolves cuda → mps → cpu, and
an explicitly requested but unavailable backend raises rather than silently falling back —
a quiet CPU fallback would make the registry's timing columns meaningless.

```bash
uv run python -c "from audio_df.runtime.device import describe_env; print(describe_env())"
```

## 2. Get the dataset

**ASVspoof 2019 LA** is what the configs target. Hosted on Edinburgh DataShare — no login or
registration needed (the "Log In" on that page is for depositing, not downloading).

- Landing page: https://datashare.ed.ac.uk/handle/10283/3336
- `LA.zip` — **7.12 GB** ← the one you want
- `PA.zip` — 16.45 GB (replay attacks; not used here)

The handle URL redirects twice, so a naive download of the landing page gets you a 4 KB HTML
stub. Use the resolved link:

```bash
mkdir -p data/raw/ASVspoof2019
curl -L -o /tmp/LA.zip \
  "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
unzip -q /tmp/LA.zip -d data/raw/ASVspoof2019/
```

That produces the layout [`configs/data.yaml`](configs/data.yaml) expects:

```text
data/raw/ASVspoof2019/LA/
├── ASVspoof2019_LA_train/flac/          # *.flac
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/        # *.trn.txt / *.trl.txt
```

This layout is *verified*, not assumed — checked against the official AASIST and RawNet2
loaders ([`../Miscellaneous/research/knowledge/datasets/asvspoof-2019-la.md`](../Miscellaneous/research/knowledge/datasets/asvspoof-2019-la.md)).
`data/raw/` is gitignored, so none of it can reach the repo.

**In-the-Wild** (the cross-domain test set, never used for training) is the Hugging Face
dataset `mueller91/In-The-Wild` (an 8.16 GB zip), extracted to `data/raw/InTheWild`. See
[`../Miscellaneous/research/knowledge/datasets/in-the-wild.md`](../Miscellaneous/research/knowledge/datasets/in-the-wild.md).

## 3. Build manifests and run the leakage audit

Run this before any modelling. It checks whether the labels can be predicted from properties
that have nothing to do with synthesis.

```bash
uv run python scripts/prepare_data.py --config configs/data.yaml
```

Writes `data/manifests/<split>.csv` with utterance id, path, label, speaker, attack id,
sample rate, duration, codec, split and language; prints per-class / per-attack / per-speaker
counts. Missing dataset produces an actionable error naming the expected path, not a bare
traceback.

```bash
uv run python scripts/audit_leakage.py --config configs/data.yaml
```

Fits tiny classifiers (logistic regression, depth-2 tree) on **metadata only** — duration,
sample rate, codec, RMS, leading/trailing/total silence, source corpus, speaker identity —
each alone and all combined, scored strictly out-of-fold. Writes
`reports/tables/leakage_audit.csv` with AUC and EER plus bootstrap CIs, and **exits 2** if
any nuisance feature predicts the label above the threshold.

If it fires, stop and read the table before training anything. A detector that learns
`microphone noise → real` or `clean WAV → fake` is not solving the intended problem.

> The 0.65 threshold is a **project-local convention with no literature source** — the audit
> says so in its own warning text rather than implying a citation it does not have.

## 4. Run the systems

Every system trains and scores through one command, `adf`; the config's `system` key picks
the pipeline. Each heavy command applies the limits in
[`configs/resources.yaml`](configs/resources.yaml) and writes `runs/<run id>/` plus a row
in [`runs/REGISTRY.md`](runs/REGISTRY.md). Metrics come only from the shared scorer:

```bash
uv run adf --config configs/a1_lfcc_gmm.yaml timing             # size the job first
uv run adf --config configs/a1_lfcc_gmm.yaml train --exp e1
uv run adf --config configs/a2_wavlm_base_plus.yaml prepare --testsets train,dev,eval
uv run adf --config configs/a2_wavlm_base_plus.yaml train --exp e1
uv run adf --config configs/a1_lfcc_gmm.yaml score --model-from runs/<raw run id> \
    --exp e3 --silence-mode raw --testsets itw
uv run python scripts/score_experiment.py runs/<run id>
uv run python scripts/make_tables.py          # regenerates reports/tables/E1…E6
```

`uv run adf --help` lists the commands (`timing`, `prepare`, `train`, `score`, `jev`) and
which systems have each. The [technical reference §14](../ProjectReport/technical-reference.md#14-where-each-step-lives)
lists the command, code and notebook for every step. Run outputs under `runs/<run id>/`
stay local and are not tracked in git; the registry and the tables are.

## 5. Use a trained model

Every system implements the same interface: one voice sample in, one structured verdict
out. Export a scored run to a bundle, then load it anywhere:

```bash
uv run python scripts/export_bundle.py runs/E-20261001-e1-a2-raw artifacts/a2-wavlm-base-plus
uv run python scripts/export_bundle.py runs/E-20261001-e1-a2-raw artifacts/a2-jev \
    --jev configs/a3_jev.yaml          # the same model, wrapped with JEV
```

```python
from audio_df.inference import load_detector

detector = load_detector("artifacts/a2-wavlm-base-plus")
result = detector.predict(samples, sample_rate)   # any rate, mono or stereo
result.to_dict()
# {"label": "spoof", "score": -19.02, "threshold": -2.43, "system": "a2_ssl_head",
#  "model_version": "E-20261001-e1-a2-raw", "evidence": {...}, "latency_ms": 54.1}
```

`score` is higher for more bona fide; `label` compares it with the dev-set threshold. A1,
A2 and any of them wrapped with JEV are interchangeable; `scripts/check_parity.py` confirms
the detector classes reproduce the original runs' scores.

### The main model (scaffold)

See [`src/audio_df/models/README.md`](src/audio_df/models/README.md) for how detectors are
structured and how to add one.


`configs/fusion_xattn.yaml` defines the fusion system: bidirectional cross-attention between
A1's LFCC frames and A2's WavLM frames, both on one time axis in seconds
(`src/audio_df/models/fusion.py`). The first run, in order:

```bash
uv run adf --config configs/fusion_xattn.yaml prepare --testsets train,dev,eval   # ~27 GB, resumable
uv run adf --config configs/fusion_xattn.yaml train --exp e2
uv run python scripts/score_experiment.py runs/<run id>
```

It is a starting point: the block, the pooling and the layer mix are meant to change, and
`CrossAttentionFusion(..., return_attention=True)` returns every attention map for
analysis. JEV wraps it like any other detector (`JevEscalation`).

## 6. Web app

A FastAPI backend serves one detector bundle and a single-page frontend:

```bash
ADF_BUNDLE=artifacts/a2-wavlm-base-plus uv run uvicorn backend.app:app --reload
# open http://127.0.0.1:8000 — or call the API directly:
curl -F file=@clip.wav http://127.0.0.1:8000/api/detect
```

`GET /api/model` reports the loaded system and threshold; `POST /api/detect` takes a WAV,
FLAC, OGG or MP3 file (up to 60 s) and returns the `DetectionResult` JSON. The model loads on
the first request (a few seconds for WavLM), then answers in about 50 ms per clip. Change
models by exporting another bundle and pointing `ADF_BUNDLE` at it. Code:
[`src/backend/app.py`](src/backend/app.py), [`frontend/`](frontend/index.html).

## 7. Notebooks

```text
notebooks/01_dataset_audit.ipynb      manifest summary and the leakage audit
notebooks/02_spectral_eda.ipynb       spectral summaries and effect sizes, real against fake
notebooks/03_cepstral_eda.ipynb       the feat_a…feat_f feature grid
notebooks/09_a1_playground.ipynb      A1 by hand: the B02 pipeline and a small GMM
notebooks/10_a1_lfcc_gmm.ipynb        A1 runs and scores
notebooks/11_a2_ssl_head.ipynb        A2 extraction, training and scores
notebooks/12_a3_jev.ipynb             A3 evidence state and dry run
notebooks/13_e4_degradations.ipynb    E4 subset and degradations
notebooks/20_report_figures.ipynb      every figure in the report
```

Notebooks hold questions, plots and interpretation; any function used twice lives in
`src/audio_df/`. Open them in VS Code with the project's `.venv` kernel.

---

## Scientific ground rules

These are enforced in code and in review, not just documented:

- Thresholds are chosen on dev, **never** on eval.
- Unseen-attack tests hold out **whole generator families**, never random utterances.
- Every result is reported on raw audio and, as a diagnostic, with leading and trailing silence
  trimmed.
- Per-attack EER is reported alongside pooled EER — a pooled number can hide a generator that
  reverses. (On a synthetic easy/hard pair: A02=0.004, A03=0.456, pooled=0.304.)
- EER matches the official ASVspoof algorithm exactly, verified bit-for-bit against the
  official scoring package.
- t-SNE/UMAP are exploratory pictures, never evidence of separability.
- Literature EERs are labelled as literature and never compared directly with our runs.
- Every run records git commit, config hash, split hash, seed, device, params and timing.

## Layout

```text
configs/      data · features · resources · scoring · machine · one per system and experiment
data/         raw/ (gitignored) · manifests/ (CSVs gitignored, SHA256SUMS tracked) · metadata/
notebooks/    exploration, one notebook per system, report figures
reports/      tables/ (regenerated by make_tables.py)
runs/         REGISTRY.md (tracked) · <run id>/ (local only: config, scores, metrics.json, notes)
scripts/      prepare_data · audit_leakage · score_experiment · make_tables · make_degraded · get_musan
src/audio_df/ the library: data · features · models · training · evaluation · runtime · pipelines
              (the adf command) · inference.py (load_detector)
artifacts/    exported model bundles (gitignored)
src/backend/  FastAPI app (uvicorn backend.app:app)
frontend/     single static page served by the backend
tests/        run without any dataset
CONVENTIONS.md  binding coding conventions
```

The report lives in [`../ProjectReport/`](../ProjectReport/report/README.md) and the research
notes in [`../Miscellaneous/research/`](../Miscellaneous/research/README.md).

[`CONVENTIONS.md`](CONVENTIONS.md) is binding: uv only, tests without data,
dev-only thresholds, module boundaries, no silent stubs.
