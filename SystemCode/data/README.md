# Data setup and evaluation datasets

Run commands from `SystemCode/`. Downloaded audio, generated manifests and feature caches are local data, not source code. The evaluation protocol is defined in `implementation.html`; **only ASVspoof 2019 LA is used for Stage 1 training**.

## Dataset roles

| Dataset | Train / dev | Evaluation | Status |
|---|---|---|---|
| **ASVspoof 2019 LA** | Official train (25,380) / dev (24,844) | Official eval (71,237) | Required |
| **In-the-Wild** | Never | All 31,779 clips; cross-domain | Required |
| **ASVspoof 2021 LA / DF** | Never | Official evaluation splits | LA and DF wired in |
| **MUSAN noise** | Never | Noise source for a fixed 2019 LA evaluation subset | Required for controlled noise tests; not a labeled spoof dataset |
| **SpoofCeleb** | Conditional Stage 2 only, using its designated train/development partitions | Held-out official evaluation | Optional adaptation; manifest builder available |
| **PhonemeDF** | Optional development dataset | Speaker-disjoint train/dev/eval for A1/A2/A4 | Derived deterministic speaker split; keep evaluation held out |
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

### SpoofCeleb — optional Stage 2 adaptation

SpoofCeleb is a **conditional Stage 2** dataset: use its designated training/development partitions only if the adaptation study is activated, and keep its evaluation partition held out. Request access to the gated [Hugging Face dataset](https://huggingface.co/datasets/jungjee/spoofceleb) and extract it under `data/raw/spoofceleb/`, preserving its `flac/`, `metadata/`, and `protocol/` directories. Build manifests with:

```bash
uv run data-prepare --dataset spoofceleb --workers 4
```

This writes `data/manifests/spoofceleb_train.csv`, `spoofceleb_development.csv`, and `spoofceleb_evaluation.csv`. Do not mix SpoofCeleb into Stage 1 results.

Check SpoofCeleb manifest structure and audio references with `uv run data-check --paths`.
Verify the three SpoofCeleb manifest checksums, as well as the other entries in `SHA256SUMS`,
with `uv run data-check --verify`.

The A1, A2 and A4 pipelines use the same deterministic SpoofCeleb training subset. By default,
each takes up to 100 clips from every `(label, attack_id)` stratum, selected with seed 1337.
This keeps the train distribution balanced across bona fide and attack families (about 1,100
training clips when all 11 train strata are present). A2/A4 use the same rule on the development
partition for training-time validation. The subset manifests are cached under
`data/cache/spoofceleb/training_subsets/`; they are regenerated when the source manifest changes.
The cap and strata are set by `spoofceleb.training_subset` in `configs/data.yaml`; `--seed` changes
the deterministic selection. The full evaluation manifest remains held out and unchanged.

Train the models with:

```bash
uv run a1 train --i spoofceleb
uv run a2 train --i spoofceleb
uv run a4 train --i spoofceleb
```

For A1, training fits only the subset of the training partition; it does not use development data
for early stopping. A2/A4 also fit against the uniform development subset for early stopping.
After training, calibrate thresholds on the full development manifest and score the full held-out
evaluation manifest:

```bash
uv run a1 evaluate --i spoofceleb --threshold-from dev --testsets eval
uv run a2 evaluate --i spoofceleb --threshold-from dev --testsets eval
uv run a4 evaluate --i spoofceleb --threshold-from dev --testsets eval
```

Keep the evaluation partition out of training and model selection. A1 is CPU-based; the GPU-enabled
PyTorch installation does not accelerate this GMM pipeline.

### PhonemeDF — speaker-disjoint training and evaluation

The local PhonemeDF release is organized under `data/raw/phonemedf/` with real speech,
TTS-generated speech, voice-converted speech, references, and TextGrid alignments. Manifest
generation uses the real WAVs and matching TTS/VC WAVs; it does not use reference recordings or
alignments. Generated attacks are named by family and generator (for example
`tts-ChatterboxTTS` and `vc-FreeVC`).

The project has no supplied train/development/evaluation protocol. The builder therefore creates
a deterministic 80/10/10 **speaker-disjoint** split using seed 1337. Matching real and generated
versions of each source utterance, and all utterances from the same source speaker, stay in one
split. The smaller XTTSv2 directory is represented where files are available.

Build and validate the manifests:

```bash
uv run data-prepare --dataset phonemedf --workers 4
uv run data-check --paths
```

This writes `phonemedf_train.csv`, `phonemedf_development.csv`, and `phonemedf_evaluation.csv`.
Training uses the same deterministic cap of up to 100 clips per `(label, attack_id)` stratum for
A1, A2, and A4. With all eight strata present, the training subset is up to 800 clips. A2/A4 use
the similarly sampled development set for early stopping. The subsets are cached under
`data/cache/phonemedf/training_subsets/`. Set `phonemedf.training_subset` in `configs/data.yaml`
to change the cap/strata; `--seed` changes the row sample, and `phonemedf.split_seed` changes the
speaker split and requires regenerating manifests.

Train and evaluate each model (evaluation remains full and held out):

```bash
uv run a1 train --i phonemedf
uv run a2 train --i phonemedf
uv run a4 train --i phonemedf

uv run a1 evaluate --i phonemedf --threshold-from dev --testsets eval
uv run a2 evaluate --i phonemedf --threshold-from dev --testsets eval
uv run a4 evaluate --i phonemedf --threshold-from dev --testsets eval
```

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
