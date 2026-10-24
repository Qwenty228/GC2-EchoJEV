# Data setup and evaluation datasets

Run commands from `SystemCode/`. Downloaded audio, generated manifests and feature caches are local data, not source code. The evaluation protocol is defined in `implementation.html`; **only ASVspoof 2019 LA is used for Stage 1 training**.

## Dataset roles

| Dataset | Train / dev | Evaluation | Status |
|---|---|---|---|
| **ASVspoof 2019 LA** | Official train (25,380) / dev (24,844) | Official eval (71,237) | Required |
| **In-the-Wild** | Never | All 31,779 clips; cross-domain | Required |
| **ASVspoof 2021 LA / DF** | Never | Official evaluation splits | LA and DF wired in |
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
```

Keep `raw/`, `cache/` and generated manifest CSVs out of Git. Commit the intended manifest checksums only after verifying the dataset definitions; different local hashes should be investigated, not blindly overwritten.

## Download and prepare

Download URLs live in one place: [`scripts/download_data.py`](../scripts/download_data.py). It downloads into `data/raw/_downloads/` (resumable) and unpacks into the layout above:

```bash
uv run download --asvspoof2019     # or --asvspoof2021la, --asvspoof2021df, --musan, --itw, --all
```

**Manual download** works too: put the same archive (file names are in the script) in `data/raw/_downloads/` and rerun the command; existing archives are unpacked without downloading. If a server is down, the script names the archive and URL it could not fetch.

### ASVspoof 2019 LA — required

```bash
uv run download --asvspoof2019
uv run data-prepare --workers 4
```

Expected: `data/raw/ASVspoof2019/LA/` contains `ASVspoof2019_LA_{train,dev,eval}/flac/` and `ASVspoof2019_LA_cm_protocols/`.

### In-the-Wild — required external test

```bash
uv run download --itw
uv run data-prepare --dataset in_the_wild --workers 4
```

Expected: `data/raw/InTheWild/release_in_the_wild/`, including `meta.csv` and audio. Output: `data/manifests/itw.csv`. Test only; do not invent a training split.

### MUSAN noise — controlled robustness

```bash
uv run download --musan            # unpacks only musan/noise/
```

MUSAN is **not** a real/spoof classification dataset. It supplies noise for reproducible corruption of ASVspoof evaluation clips.

### Generate controlled evaluation sets

Requires ASVspoof 2019 LA; the noise condition additionally requires MUSAN. Uses the same fixed, seeded clean evaluation subset for all conditions. See `configs/e4_degradations.yaml` for exact settings.

```bash
uv run data-degrade --clean-only
uv run data-degrade --conditions eval_noise_10db
uv run data-degrade --conditions eval_mp3_32k,eval_opus_16k,eval_resample_8k
# Or generate all conditions:
uv run data-degrade --conditions all --workers 4
```

Outputs include `data/manifests/eval_clean10k.csv`, `eval_noise_10db.csv`, `eval_mp3_32k.csv`, `eval_opus_16k.csv` and `eval_resample_8k.csv`, with generated audio under `data/cache/degraded/`. The conditions must use matched source utterances when comparing robustness.

### ASVspoof 2021 LA — external test

```bash
uv run download --asvspoof2021la   # Zenodo audio + official keys (keys are only on asvspoof.org)
uv run python scripts/reencode_flac.py data/raw/ASVspoof2021/ASVspoof2021_LA_eval/flac \
    data/raw/ASVspoof2021/ASVspoof2021_LA_eval/flac_ffmpeg
uv run data-prepare --dataset asvspoof2021 --workers 4
```

The re-encode is required: libsndfile cannot decode ~30% of the original LA FLACs, although they match the official archive. It is lossless (bit-identical samples). Outputs: `asvspoof2021_eval.csv` (official 148,176-clip subset), `asvspoof2021_hidden.csv` (16,926 clips, non-speech already removed by the organisers) and `asvspoof2021_progress.csv` (leaderboard subset, not reported). Labels come from the CM key `keys/LA/CM/trial_metadata.txt`. These are scored raw (not trimmed).

### ASVspoof 2021 DF — external test

```bash
uv run download --asvspoof2021df   # Hugging Face mirror, ~34 GB + ~34 GB unpacked
uv run data-prepare --dataset asvspoof2021df --workers 4
```

Comes from a Hugging Face mirror with the official labels and protocol, audio already re-encoded for libsndfile (bit-exact). Unpacks to `data/raw/ASVspoof2021/ASVspoof2021_DF_eval/flac/` plus `labels.csv` (label, speaker, `subset` = eval/progress/hidden, codec, attack, vocoder). Outputs `asvspoof2021df_eval.csv` (official scoring subset), `asvspoof2021df_progress.csv` and `asvspoof2021df_hidden.csv`; scored raw (not trimmed).

### SpoofCeleb / PhonemeDF — future work

SpoofCeleb is a **conditional Stage 2** dataset: use its designated training/development partitions only if the adaptation study is activated, and keep its evaluation partition held out. PhonemeDF is deferred from the core protocol. For both, obtain audio and metadata through their official dataset distribution and review access/licensing terms; neither has a supported manifest builder yet. Do not mix either into Stage 1 results.

## Validate and use manifests

```bash
uv run data-prepare --help
uv run data-degrade --help
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

**Split roles:** train learns model parameters; dev is holdout validation for early stopping and threshold calibration (not a final test or cross-validation fold); eval, controlled `eval_*` and external datasets are held-out tests. JEV policy evaluation calibrates on full dev and applies the policy only to sampled test clips. **Evaluation protocol:** train on the official 2019 LA train split, select checkpoints/thresholds on dev, and evaluate on held-out test sets. The runtime trims ASVspoof 2019 LA train, dev, eval and the controlled `eval_*` sets with the same recipe (see `ProjectReport/trim_eval.md` for why training is trimmed too). External sets such as In-the-Wild remain raw. Generated manifests alone do not apply trimming. Keep preprocessing and metrics consistent across detectors.
