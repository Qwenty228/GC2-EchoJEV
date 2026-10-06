# Audio Deepfake Detection — GC2

A reproducible benchmark of spoofed-speech detectors on standardised ASVspoof protocols, with
generalisation, robustness and silence-shortcut checks built in.

**Research question.** Given an utterance *x* and label *y* ∈ {bonafide, spoof}, can we learn
a detector that captures **synthesis-related evidence** rather than dataset or generator
shortcuts? The working assumption is that for some observable representation *Z*,
`P(Z | real) ≠ P(Z | fake)`. The project investigates **where** that difference is visible
and **whether it survives unseen attacks and domains**.

## Read this first

- **[The report](docs/report/README.md)**: the main document. It covers the question, the data,
  the three detectors and every result, as one argument in twelve short chapters.
- **[Technical reference](docs/technical-reference.md)**: full method, maths, implementation
  pointers and result tables.
- **[Research notes](research/README.md)**: page-referenced notes on papers and methods,
  fact-checks and architecture decision records (ADRs).

## Results

| equal error rate (EER), lower is better | 2019 LA eval, raw | 2019 LA eval, silence trimmed | In-the-Wild, raw |
|---|---|---|---|
| **A1** LFCC + GMM (official B02 recipe) | 9.56 % | 19.84 % | 58.04 % |
| **A2** frozen WavLM Base+, layer-weighted head | **0.96 %** | 13.27 % | **14.70 %** |

Tables: [`reports/tables/`](reports/tables/) · run log: [`experiments/REGISTRY.md`](experiments/REGISTRY.md).
In short: leading silence alone predicts the label with an area under the curve (AUC) of 0.837,
and trimming it costs both detectors heavily. Both detectors also detect genuine vocoder
artefacts. On real-world audio, A2's ranking holds but its decision threshold does not, and A1
falls to chance. The [report](docs/report/README.md) gives the full argument.

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
| A1 (`run_a1`, notebooks 09–10) | run: E1, E3, raw and trimmed; X4 with 3 seeds per arm |
| A2 with WavLM Base+ (`run_a2`, notebook 11) | run: E1, E3, raw and trimmed; X6 layer analysis |
| A2 with XLS-R 300M | configured, not run |
| A3 (`run_a3`, notebook 12) | scaffold with a dry-run client; waiting for the JEV API documentation |
| E4 degradations (`make_degraded.py`, notebook 13) | generator ready, not scored |
| E6 efficiency | not built |

---

## 1. Set up the environment

Requires [uv](https://docs.astral.sh/uv/). Everything runs through it — no `pip`, no manual
venv activation.

```bash
uv sync                 # locked env: Python 3.11, torch 2.14, ~1.3 GB
uv run pytest           # no dataset needed
uv run ruff check src tests
```

Target device is Apple Silicon **MPS** or CPU. `device: auto` resolves cuda → mps → cpu, and
an explicitly requested but unavailable backend raises rather than silently falling back —
a quiet CPU fallback would make the registry's timing columns meaningless.

```bash
uv run python -c "from audio_df.device import describe_env; print(describe_env())"
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
loaders ([`research/knowledge/datasets/asvspoof-2019-la.md`](research/knowledge/datasets/asvspoof-2019-la.md)).
`data/raw/` is gitignored, so none of it can reach the repo.

**In-the-Wild** (the cross-domain test set, never used for training) is the Hugging Face
dataset `mueller91/In-The-Wild` (an 8.16 GB zip), extracted to `data/raw/InTheWild`. See
[`research/knowledge/datasets/in-the-wild.md`](research/knowledge/datasets/in-the-wild.md).

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

Each runner starts with a 200-clip timing run, applies the limits in
[`configs/resources.yaml`](configs/resources.yaml), and writes `experiments/<run id>/` plus a
row in [`experiments/REGISTRY.md`](experiments/REGISTRY.md). Metrics come only from the shared
scorer:

```bash
uv run python -m audio_df.experiments.run_a1 train --exp e1
uv run python scripts/score_experiment.py experiments/E-20261001-e1-a1-raw
uv run python scripts/make_tables.py          # regenerates reports/tables/E1…E6
```

The [technical reference §14](docs/technical-reference.md#14-where-each-step-lives) lists the
command, code and notebook for every step. Run outputs under `experiments/<run id>/` stay
local and are not tracked in git; the registry and the tables are.

## 5. Notebooks

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

## Repository layout

```text
configs/      data · features · resources · scoring · machine · one per system and experiment
data/         raw/ (gitignored) · manifests/ (CSVs gitignored, SHA256SUMS tracked) · metadata/
docs/         report/ (main document) · technical-reference.md
experiments/  REGISTRY.md (tracked) · <run id>/ (local only: config, scores, metrics.json, notes)
notebooks/    exploration, one notebook per system, report figures
reports/      tables/ (regenerated by make_tables.py)
research/     knowledge notes, fact-checks, ADRs
scripts/      prepare_data · audit_leakage · score_experiment · make_tables · make_degraded · get_musan
src/audio_df/ audio · features · lfcc_b02 · stats · metrics · scoring · leakage · data · models/ ·
              experiments/ · jev/ · xai/ · resources · exp_io · device · seeding
tests/        run without any dataset
tasks/        CONVENTIONS.md (binding coding conventions)
```

[`tasks/CONVENTIONS.md`](tasks/CONVENTIONS.md) is binding: uv only, tests without data,
dev-only thresholds, module boundaries, no silent stubs.
