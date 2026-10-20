# Data setup and evaluation datasets

Run commands from `SystemCode/`. Downloaded audio, generated manifests and feature caches are local data, not source code. The evaluation protocol is defined in `implementation.html`; **only ASVspoof 2019 LA is used for Stage 1 training**.

## Dataset roles

| Dataset | Train / dev | Evaluation | Status |
|---|---|---|---|
| **ASVspoof 2019 LA** | Official train (25,380) / dev (24,844) | Official eval (71,237) | Required |
| **In-the-Wild** | Never | All 31,779 clips; cross-domain | Required |
| **ASVspoof 2021 LA / DF** | Never | Official evaluation splits | Optional; manifest builder/keys must be validated |
| **MUSAN noise** | Never | Noise source for a fixed 2019 LA evaluation subset | Required for controlled noise tests; not a labeled spoof dataset |
| **SpoofCeleb** | Conditional Stage 2 only, using its designated train/dev partitions | Held-out official evaluation | Optional adaptation; not wired into the current data CLI |
| **PhonemeDF** | Not in core protocol | Potential future external test | Deferred; not wired into the current data CLI |
| **Self-recorded audio** | Never | Optional qualitative demo | Requires provenance and both classes for quantitative detection metrics |

Controlled **codec/resampling** conditions (MP3, Opus, 8 kHz round-trip) use the same fixed ASVspoof 2019 LA evaluation subset as the MUSAN condition. “Controlled noise / codec” in the decision-layer matrix refers to these existing sets, not an additional corpus.

## Directory layout

```text
data/
├── raw/
│   ├── _downloads/                 # temporary archives
│   ├── ASVspoof2019/LA/
│   ├── InTheWild/release_in_the_wild/
│   ├── musan/noise/
│   └── ASVspoof2021/              # optional
├── manifests/                     # train.csv, dev.csv, eval.csv, itw.csv, eval_*.csv
│   └── SHA256SUMS                  # reference checksums
├── metadata/                      # leakage-audit metadata
├── cache/                         # feature and generated-audio caches
└── jev_cache/                     # decision-service cache
```

Keep `raw/`, `cache/` and generated manifest CSVs out of Git. Commit the intended manifest checksums only after verifying the dataset definitions; different local hashes should be investigated, not blindly overwritten.

## Download and prepare

Download with `curl -L -C -` to resume an interrupted transfer, **or download from the linked official page and place the archive in `data/raw/_downloads/`**. Unpack into the locations below.

### ASVspoof 2019 LA — required

Source: [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336).

```bash
mkdir -p data/raw/_downloads data/raw/ASVspoof2019
curl -L -C - -o data/raw/_downloads/LA.zip \
  'https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content'
unzip -q data/raw/_downloads/LA.zip -d data/raw/ASVspoof2019/
uv run data-prepare --workers 4
```

Expected: `data/raw/ASVspoof2019/LA/` contains `ASVspoof2019_LA_{train,dev,eval}/flac/` and `ASVspoof2019_LA_cm_protocols/`.

### In-the-Wild — required external test

Source: [Hugging Face — mueller91/In-The-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild).

```bash
mkdir -p data/raw/_downloads data/raw/InTheWild
curl -L -C - -o data/raw/_downloads/release_in_the_wild.zip \
  'https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip'
unzip -q data/raw/_downloads/release_in_the_wild.zip -d data/raw/InTheWild/
uv run data-prepare --dataset in_the_wild --workers 4
```

Expected: `data/raw/InTheWild/release_in_the_wild/`, including `meta.csv` and audio. Output: `data/manifests/itw.csv`. Test only; do not invent a training split.

### MUSAN noise — controlled robustness

Source: [OpenSLR SLR17](https://www.openslr.org/17/). Only the `musan/noise/` portion is needed.

```bash
mkdir -p data/raw/_downloads data/raw/musan
curl -L -C - -o data/raw/_downloads/musan.tar.gz \
  'https://www.openslr.org/resources/17/musan.tar.gz'
tar -xzf data/raw/_downloads/musan.tar.gz -C data/raw/ musan/noise
```

MUSAN is **not** a real/spoof classification dataset. It supplies noise for reproducible corruption of ASVspoof evaluation clips.

### Generate controlled evaluation sets

Requires ASVspoof 2019 LA; the noise condition additionally requires MUSAN. Uses the same fixed, seeded clean evaluation subset for all conditions. See `configs/e4_degradations.yaml` for exact settings.

```bash
uv run python scripts/make_degraded.py --clean-only
uv run python scripts/make_degraded.py --conditions eval_noise_10db
uv run python scripts/make_degraded.py --conditions eval_mp3_32k,eval_opus_16k,eval_resample_8k
# Or generate all conditions:
uv run python scripts/make_degraded.py --conditions all --workers 4
```

Outputs include `data/manifests/eval_clean10k.csv`, `eval_noise_10db.csv`, `eval_mp3_32k.csv`, `eval_opus_16k.csv` and `eval_resample_8k.csv`, with generated audio under `data/cache/degraded/`. The conditions must use matched source utterances when comparing robustness.

### ASVspoof 2021 LA / DF — optional external tests

Download evaluation audio from [LA (Zenodo)](https://zenodo.org/records/4837263) and [DF (Zenodo)](https://zenodo.org/records/4835108), plus official evaluation keys from [ASVspoof](https://www.asvspoof.org/). Place and extract under `data/raw/ASVspoof2021/`.

**Current limitation:** `data-prepare` does not yet build or validate 2021 LA/DF manifests. Do not include these in `--testsets` until the official keys are parsed, audio paths validated and evaluation manifests generated.

### SpoofCeleb / PhonemeDF — future work

SpoofCeleb is a **conditional Stage 2** dataset: use its designated training/development partitions only if the adaptation study is activated, and keep its evaluation partition held out. PhonemeDF is deferred from the core protocol. For both, obtain audio and metadata through their official dataset distribution and review access/licensing terms; neither has a supported manifest builder yet. Do not mix either into Stage 1 results.

## Validate and use manifests

```bash
uv run data-prepare --help
uv run data-check --help
uv run data-check --paths
uv run data-check --verify --paths
```

`--paths` checks manifest-referenced audio locations; `--verify` additionally compares manifest files with `data/manifests/SHA256SUMS`. A checksum mismatch is **not** automatically audio corruption: investigate serialization, paths, row ordering and dataset versions before updating references. Optional ASVspoof 2021 presence alone is not validation.

Models use manifest CSVs rather than scanning audio directories. Each manifest needs at least `utt_id`, `path` and `label` (`bonafide`/`spoof`); the standard builder also records split, speaker/attack IDs and audio metadata. To add a dataset, implement its builder in `src/audio_df/data/manifests.py`, expose it through `data-prepare`, validate label/path integrity and keep train/dev/test partitions disjoint. Evaluation uses names of existing manifests, for example:

```bash
uv run a1 evaluate --i 2019LA --testsets dev,eval,itw
```

This command requires an existing completed A1 artifact. Include `dev` for the current threshold-selection workflow; never select thresholds using external evaluation labels.

**Evaluation protocol:** train on the official 2019 LA train split, select checkpoints/thresholds on dev, and evaluate on held-out test sets. The agreed silence-trimmed test protocol must be enabled in the evaluation implementation before reporting any run as “trimmed”; generating manifests alone does not apply trimming. Keep preprocessing and metrics consistent across detectors.
