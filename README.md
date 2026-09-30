# Audio Deepfake Detection — GC2

A reproducible benchmark of three spoofed-speech detectors on standardized ASVspoof
protocols, with generalisation, robustness and silence-shortcut checks built in.

**Research question.** Given an utterance *x* and label *y* ∈ {bonafide, spoof}, can we learn
a detector that captures **synthesis-related evidence** rather than dataset or generator
shortcuts? The working assumption is that for some observable representation *Z*,
`P(Z | real) ≠ P(Z | fake)`. This project investigates **where** that difference is visible
and **whether it survives unseen attacks and domains** — not a hunt for one magic artifact.

---

## The plan (changed 2026-09-30)

1. **Data:** ASVspoof 2019 LA, official train/dev/eval; 16 kHz; 4 s crop/pad (A1 uses whole utterances, as B02 does).
2. **A1:** LFCC + GMM, the official ASVspoof 2019 B02 recipe (60-dim LFCC, two 512-component GMMs, LLR score).
3. **A2:** frozen SSL (WavLM Base+, then XLS-R 300M) → learned layer weights → mean pool → MLP; only the head trains.
4. **A3:** frozen A2 + the JEV API, asked typed questions over a JSON evidence state per clip.
5. **E1** method benchmark (2019 LA eval) · **E2** ablation (later) · **E3** generalisation (In-the-Wild + unseen attacks A07–A19).
6. **E4** robustness (MP3 32k, Opus 16k, noise 10 dB, 8 kHz resample) on a ~10k stratified eval subset.
7. **E5** JEV (A2 vs A3) on a stratified subset · **E6** efficiency on 1,000 eval clips.
8. **Silence:** raw is the main mode; trimmed is a diagnostic for A1/A2 on clean eval and In-the-Wild.
9. **Protocol:** thresholds fixed on dev; metrics only from committed score files; every run logged in [`experiments/REGISTRY.md`](experiments/REGISTRY.md).
10. **Machine:** a fanless MacBook Air M4 (16 GB), so resource limits and resumable jobs are built in.

Parked work (and why) is in [`docs/backlog.md`](docs/backlog.md). The original brief,
[`docs/handoff.md`](docs/handoff.md), is kept for history but superseded.

---

## Build status

**✅ works today and is tested; 🔜 is specified but not yet implemented.**

| Area | What | Status |
|---|---|---|
| Foundation | Environment, device resolution, seeding, resource limits and memory-pressure guard | ✅ |
| Data | Manifests + checksums, In-the-Wild loader, stratified subsets, **leakage audit** | ✅ |
| Preprocessing | Crop/pad, `trim_silence`, `preprocess(silence_mode=raw\|trimmed)` | ✅ |
| Features / EDA | Hand-crafted grid `feat_a`…`feat_f`, effect sizes, notebooks 01–03 | ✅ |
| Experiments | `experiments/<id>/`, `REGISTRY.md`, shared scoring (EER, minDCF, ROC-AUC, F1), E1–E6 tables | ✅ |
| A1 | LFCC-B02 + chunked-EM GMM (`run_a1`, notebook 10) | ✅ code · 🔜 not yet run |
| A2 | SSL pooled cache + layer-weighted head (`run_a2`, notebook 11) | ✅ code · 🔜 not yet run |
| A3 | JEV interface, cache, state builder, dry-run (`run_a3`, notebook 12) | ✅ scaffold · 🔜 awaiting JEV docs |
| E4 | Degradations on a 10k stratified eval subset (`make_degraded.py`, notebook 13) | ✅ code · 🔜 not yet run |

519 tests pass with **no dataset present** — synthetic fixtures throughout.

---

## 1. Set up the environment

Requires [uv](https://docs.astral.sh/uv/). Everything runs through it — no `pip`, no manual
venv activation.

```bash
uv sync                 # locked env: Python 3.11, torch 2.14, ~1.3 GB
uv run pytest           # 519 tests, no dataset needed
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

**Other datasets** — later, and only in this order. The handoff is explicit that a
cross-dataset generalisation claim is meaningless until the in-domain pipeline is stable.

| Dataset | Source | When |
|---|---|---|
| ASVspoof 2021 LA / DF | Zenodo `4837263` / `4835108`, keys via [`asvspoof-challenge/2021`](https://github.com/asvspoof-challenge/2021) | eval-only; DF is the deepfake track |
| ASVspoof 5 | https://www.asvspoof.org/ | newest, larger, explicitly deepfake-oriented |
| In-the-wild | Müller et al. release | cross-dataset test, last |

## 3. Build manifests and run the Stage 0 audit ✅

**Do this before any modelling.** It is the gate that decides whether the project is
measuring synthesis artifacts or dataset bookkeeping.

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

## 4. Read the knowledge base ✅

[`research/`](research/README.md) is a first-class part of this project. A dedicated
researcher agent owns it, and it has already caught real errors — including two in the task
specs themselves.

```text
research/knowledge/     notes written from primary sources (cite or delete)
research/fact-checks/   verdicts on specific implementation claims
research/decisions/     ADRs: what we chose, the evidence, the reversal condition
```

Start with [`knowledge/index.md`](research/knowledge/index.md). Worth reading before you
trust any number this repo produces:

- [`methods/eer-and-metrics.md`](research/knowledge/methods/eer-and-metrics.md) — the official
  ASVspoof EER does **not** interpolate, and higher score means **bonafide**, not spoof.
  Both of those contradicted the original task spec.
- [`methods/lfcc-mfcc-dct.md`](research/knowledge/methods/lfcc-mfcc-dct.md) — the handoff's
  DCT formula is not orthonormal as written; it needs `√(1/J)` at q=0 and `√(2/J)` above.
- [`fact-checks/`](research/fact-checks/) — four verdicts so far, three `contradicted`.

**The rule:** every implementation task ends with a *claim block* of falsifiable statements.
The researcher checks them against sources and the code. A task is not done until its verdict
is `confirmed`. Unsourced statements do not go in, and literature EERs are never compared
directly with our own runs.

## 5. Explore — notebooks ✅

```text
notebooks/01_dataset_audit.ipynb        manifest summary, leakage verdict
notebooks/02_spectral_eda.ipynb         waveform/spectral distributions, per-bin effect sizes
notebooks/03_cepstral_eda.ipynb         the feat_a…feat_f grid in feature space
notebooks/09_a1_playground.ipynb        START HERE for A1: the B02 pipeline by hand + a mini GMM to play with
```

They run end-to-end on a synthetic mini-manifest when no dataset is present. Open them in
VS Code with the project's `.venv` kernel.

**Notebook rule:** notebooks hold questions, plots and interpretation. If a function is used
twice it moves to `src/audio_df/`.

## 6. Run the benchmark 🔜

Each system gets a runner script and a notebook, each starting with a 200-clip timing run.
Every run creates `experiments/<id>/` and a row in `experiments/REGISTRY.md`; metrics come
only from the shared scorer:

```bash
uv run python scripts/score_experiment.py experiments/E-20261001-e1-a1-raw
uv run python scripts/make_tables.py          # regenerates reports/tables/E1…E6
```

---

## Scientific ground rules

These are enforced in code and in review, not just documented:

- Thresholds are chosen on dev, **never** on eval.
- Unseen-attack tests hold out **whole generator families**, never random utterances.
- Per-attack EER is reported alongside pooled EER — a pooled number can hide a generator that
  reverses. (On a synthetic easy/hard pair: A02=0.004, A03=0.456, pooled=0.304.)
- EER matches the official ASVspoof algorithm exactly, verified bit-for-bit against the
  official scoring package.
- t-SNE/UMAP are exploratory pictures, never evidence of separability.
- Literature EERs are labelled as literature and never compared directly with our runs.
- Every run records git commit, config hash, split hash, seed, device, params and timing.

## Repository layout

```text
configs/      data · features · resources · scoring · machine (+ old-plan configs)
data/         raw/ (gitignored) · manifests/ (CSVs gitignored, SHA256SUMS tracked) · cache/
experiments/  REGISTRY.md · <exp_id>/{config.yaml, env.json, notes.md, scores/, metrics.json}
notebooks/    01 dataset audit · 02 spectral EDA · 03 cepstral EDA
src/audio_df/ audio · features · stats · metrics · leakage · cache · data · visualization ·
              device · resources · seeding · exp_io · scoring · models/ · experiments/
tests/        run without any dataset
reports/      tables/ (regenerated by make_tables.py) · figures/
scripts/      prepare_data.py · audit_leakage.py · score_experiment.py · make_tables.py
research/     knowledge base, fact-checks, decisions  (see section 4)
tasks/        CONVENTIONS.md (binding) · TASKS.md, BACKLOG.md (superseded)
docs/         backlog.md · handoff.md (superseded) · assets/
```

## Contributing / continuing the build

[`tasks/CONVENTIONS.md`](tasks/CONVENTIONS.md) is binding: uv-only, tests without data,
dev-only thresholds, module boundaries, no silent stubs. Work is split into tasks with
**disjoint file ownership**, with the researcher fact-checking between waves.

## Results

_Not yet run — [`experiments/REGISTRY.md`](experiments/REGISTRY.md) has no runs yet. This
section stays empty until A1 lands, rather than being filled with placeholders._
