# Audio Deepfake Detection — GC2

A reproducible, implementation-first research workshop: from spectral EDA through
hand-engineered cepstral features to CNN/ResNet backends, self-supervised speech
representations, and a fusion + multi-task architecture — benchmarked on standardized
ASVspoof protocols.

**Research question.** Given an utterance *x* and label *y* ∈ {bonafide, spoof}, can we learn
a detector that captures **synthesis-related evidence** rather than dataset or generator
shortcuts? The working assumption is that for some observable representation *Z*,
`P(Z | real) ≠ P(Z | fake)`. This project investigates **where** that difference is visible
and **whether it survives unseen attacks and domains** — not a hunt for one magic artifact.

Full brief: [`docs/handoff.md`](docs/handoff.md).

---

## The plan, in one picture

[![Architecture and benchmarking pipeline](docs/assets/architecture.jpg)](docs/assets/architecture.jpg)

Panel 1 is the model we are building toward; panel 2 is the protocol that keeps ten systems
comparable; panel 3 is the ablation grid that has to justify it. The legend matters as much
as the boxes — fixed DSP, frozen pretrained, trainable branch, shared trunk and task-specific
parameters are four different cost categories, and the efficiency columns only mean something
if they are counted separately.

---

## Build status

The pipeline is being built in waves. **Everything below marked ✅ works today and is tested;
🔜 is specified but not yet implemented.** Commands under 🔜 will not run yet.

| Stage | What | Status |
|---|---|---|
| — | Environment, device resolution | ✅ |
| 0 | Manifests, protocol parsing, splits, **leakage audit** | ✅ |
| 1–2 | Feature extraction: the A–F representation grid | ✅ modules, 🔜 notebooks |
| 1–2 | Effect sizes, bootstrap CIs, EDA figures | ✅ modules, 🔜 notebooks |
| 3 | Linear / MLP baselines | 🔜 |
| 4 | CNN → ResNet backends | 🔜 |
| 5–10 | RawNet2, SSL probing, XLS-R, fusion + multi-task | 🔜 ([`tasks/BACKLOG.md`](tasks/BACKLOG.md)) |

163 tests pass with **no dataset present** — synthetic fixtures throughout, so the whole
scaffold is reviewable before you download anything.

---

## 1. Set up the environment

Requires [uv](https://docs.astral.sh/uv/). Everything runs through it — no `pip`, no manual
venv activation.

```bash
uv sync                 # locked env: Python 3.11, torch 2.14, ~1.3 GB
uv run pytest           # 163 tests, no dataset needed
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

## 5. Explore — notebooks 🔜

```text
notebooks/01_dataset_audit.ipynb        manifest summary, leakage verdict
notebooks/02_spectral_eda.ipynb         waveform/spectral distributions, per-bin effect sizes
notebooks/03_cepstral_eda.ipynb         the A-F grid in feature space
notebooks/04_simple_detection.ipynb     linear + MLP baselines
notebooks/05_cnn_resnet_experiments.ipynb
notebooks/06_ssl_representation_probe.ipynb
notebooks/07_fusion_error_analysis.ipynb
```

They will run end-to-end on a synthetic mini-manifest when no dataset is present.

**Notebook rule:** notebooks hold questions, plots and interpretation. If a function is used
twice it moves to `src/audio_df/`. Feature extraction, splitting, metrics, model definitions
and training loops never live in a notebook.

The modules they build on all work today:

```python
from audio_df.features import extract          # variants A-F
from audio_df.stats import cohens_d, bootstrap_ci, single_feature_auc
from audio_df.visualization import plot_mean_spectrum_with_ci
from audio_df.metrics import compute_eer, per_attack_eer
```

## 6. Run the benchmark 🔜

The point is to compare **families**, not to tune one model. Each run appends a row to
`reports/experiment_registry.csv` with git commit, config hash, split hash, seed, device,
trainable/total params, timing and metrics.

```bash
# the A-F representation grid through one identical backend
uv run python -m audio_df.experiments.run_baseline --config configs/features.yaml --variant B_lfcc_full

# Stage 4: sequence-aware CNN/ResNet over [B, 1, D, T]
uv run python -m audio_df.experiments.run_resnet --config configs/lfcc_resnet.yaml

# Stage 6: frozen SSL, one linear probe per transformer layer -> EER vs layer index
uv run python -m audio_df.experiments.run_ssl --config configs/ssl_probe.yaml
```

## 7. Train the proposed architecture 🔜

The custom model is the two-branch fusion in the diagram: a handcrafted branch (best Stage-2
representation → compact ResNet → pooling → 256-d `z_L`) and a frozen XLS-R branch (learned
softmax layer weighting → pooling → 256-d `z_X`), concatenated to 512-d, through a shared
trunk, into a detection head and an optional auxiliary head.

```bash
uv run python -m audio_df.experiments.run_fusion --config configs/fusion_multitask.yaml
```

Configured in [`configs/fusion_multitask.yaml`](configs/fusion_multitask.yaml), including the
**seven required ablations**. The project may not claim LFCC and XLS-R are complementary
until those ablations show it:

```text
handcrafted_only · ssl_only · concat_no_trunk · fusion_trunk_detection_only
fusion_trunk_multitask · ssl_final_layer_vs_learned_weights
pooling_mean_vs_stats_vs_attentive
```

The auxiliary head stays disabled unless a **defensible dataset-provided label** exists.
Inventing pseudo-ground-truth to make the architecture multi-task is explicitly out of bounds.

The headline result is **not** lowest in-domain EER. It is whether any improvement survives
unseen-generator and cross-domain evaluation, and whether the extra compute is justified.

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
configs/      data · features · lfcc_resnet · ssl_probe · fusion_multitask
data/         raw/ (gitignored) · metadata/ · manifests/ · cache/
notebooks/    01 dataset audit -> 07 fusion error analysis
src/audio_df/ audio · features · stats · metrics · leakage · cache · data ·
              visualization · device · models/ · experiments/
tests/        run without any dataset; real-data tests marked needs_data
reports/      figures/ · tables/ · experiment_registry.csv · final_report.md
scripts/      prepare_data.py · audit_leakage.py
research/     knowledge base, fact-checks, decisions  (see section 4)
tasks/        TASKS.md · CONVENTIONS.md · BACKLOG.md
docs/         handoff.md · assets/architecture.jpg · superpowers/specs/
```

## Contributing / continuing the build

[`tasks/TASKS.md`](tasks/TASKS.md) splits the work into waves of tasks with **disjoint file
ownership**, so a wave can be executed in parallel by separate agents, with the researcher
fact-checking between waves. [`tasks/CONVENTIONS.md`](tasks/CONVENTIONS.md) is binding:
uv-only, tests without data, dev-only thresholds, module boundaries, no silent stubs.

## Results

_Not yet run — `reports/experiment_registry.csv` holds only its header. This section stays
empty until Stage 3–4 baselines land, rather than being filled with placeholders._
