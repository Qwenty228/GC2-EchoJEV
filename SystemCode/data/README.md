# Data: download, place, build manifests

Every command below runs from `SystemCode/`. Audio and caches never go into git: `raw/`,
`cache/` and the manifest CSVs are gitignored; only the checksums
(`manifests/SHA256SUMS`) and the leakage-audit features (`metadata/nuisance_features.csv`)
are tracked.

```text
data/
├── raw/           downloaded corpora (gitignored)
│   ├── ASVspoof2019/LA/
│   ├── InTheWild/release_in_the_wild/
│   └── musan/noise/
├── manifests/     one CSV per test set (gitignored) + SHA256SUMS (tracked)
├── metadata/      per-clip nuisance features for the leakage audit (tracked)
├── cache/         features and run caches (gitignored, rebuildable)
└── jev_cache/     cached JEV answers
```

Downloads resume if interrupted (`curl -C -`): rerun the same command. Archives go to
`raw/_downloads/` and can be deleted after unzipping.

| dataset | used for | download | unpacked |
|---|---|---|---|
| ASVspoof 2019 LA | training, E1, E2 (required) | 7.6 GB | ~8 GB |
| In-the-Wild | E3, cross-domain test | 8.2 GB | ~8 GB |
| MUSAN `noise/` | E4 noise condition | 11.1 GB (keeps ~1 GB) | ~1 GB |
| ASVspoof 2021 LA / DF | optional, not wired in yet | 7.8 GB / 4 x ~8.6 GB | |

## 1. ASVspoof 2019 LA (required)

Edinburgh DataShare, no login needed (the page's "Log In" is for depositing). The landing
page is https://datashare.ed.ac.uk/handle/10283/3336; the link below is the direct file for
`LA.zip`.

```bash
mkdir -p data/raw/_downloads data/raw/ASVspoof2019
curl -L -C - -o data/raw/_downloads/LA.zip \
  "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
unzip -q data/raw/_downloads/LA.zip -d data/raw/ASVspoof2019/
uv run python scripts/prepare_data.py                     # -> manifests/train.csv, dev.csv, eval.csv
```

Expected layout (`configs/data.yaml: dataset`):

```text
data/raw/ASVspoof2019/LA/
├── ASVspoof2019_LA_train/flac/          25,380 clips
├── ASVspoof2019_LA_dev/flac/            24,844 clips
├── ASVspoof2019_LA_eval/flac/           71,237 clips
└── ASVspoof2019_LA_cm_protocols/        *.trn.txt / *.trl.txt
```

## 2. In-the-Wild (E3)

Hugging Face `mueller91/In-The-Wild`, public, one zip.

```bash
mkdir -p data/raw/_downloads data/raw/InTheWild
curl -L -C - -o data/raw/_downloads/release_in_the_wild.zip \
  "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip"
unzip -q data/raw/_downloads/release_in_the_wild.zip -d data/raw/InTheWild/
uv run python scripts/prepare_data.py --dataset in_the_wild   # -> manifests/itw.csv
```

This gives `data/raw/InTheWild/release_in_the_wild/` with `meta.csv` (`file, speaker,
label`) and 31,779 WAVs at 16 kHz: 19,963 bona fide and 11,816 spoof. There are no attack
labels and no official split; use it for testing only. Licence: the dataset card says
CC-BY-SA 4.0, while the authors' page says Apache-2.0.

## 3. MUSAN noise (E4)

The script downloads the archive, extracts only `noise/` (929 files) into
`data/raw/musan/noise`, checks the count and deletes the archive:

```bash
uv run python scripts/get_musan.py --dry-run      # shows what it will do
uv run python scripts/get_musan.py
```

By hand instead:

```bash
mkdir -p data/raw/_downloads data/raw/musan
curl -L -C - -o data/raw/_downloads/musan.tar.gz "https://www.openslr.org/resources/17/musan.tar.gz"
tar -xzf data/raw/_downloads/musan.tar.gz -C data/raw/ musan/noise
```

Then build the E4 test sets (needs ASVspoof 2019 LA and MUSAN):

```bash
uv run python scripts/make_degraded.py subset      # 10k stratified eval subset -> manifests/eval_clean10k.csv
uv run python scripts/make_degraded.py generate    # MP3, Opus, noise, 8 kHz -> manifests/eval_*.csv
```

## 4. ASVspoof 2021 LA / DF (optional, not wired in yet)

Evaluation-only tracks from the 2021 challenge (Zenodo), with the full keys from
asvspoof.org. `prepare_data.py` has no builder for them yet; see "Adding a dataset" below.

```bash
mkdir -p data/raw/_downloads data/raw/ASVspoof2021
# LA eval audio (7.8 GB) and keys
curl -L -C - -o data/raw/_downloads/ASVspoof2021_LA_eval.tar.gz \
  "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz"
curl -L -C - -o data/raw/_downloads/LA-keys-full.tar.gz \
  "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz"
# DF eval audio (four parts, about 8.6 GB each) and keys
for p in 00 01 02 03; do
  curl -L -C - -o "data/raw/_downloads/ASVspoof2021_DF_eval_part$p.tar.gz" \
    "https://zenodo.org/records/4835108/files/ASVspoof2021_DF_eval_part$p.tar.gz"
done
curl -L -C - -o data/raw/_downloads/DF-keys-full.tar.gz \
  "https://www.asvspoof.org/asvspoof2021/DF-keys-full.tar.gz"
for f in data/raw/_downloads/ASVspoof2021_*.tar.gz data/raw/_downloads/*-keys-full.tar.gz; do
  tar -xzf "$f" -C data/raw/ASVspoof2021/
done
```

## 5. Check the manifests

```bash
uv run python scripts/prepare_data.py --verify     # every manifest matches SHA256SUMS
uv run python scripts/audit_leakage.py --config configs/data.yaml
```

`--verify` fails if a manifest is missing or differs from the checksums in git, which shows
whether your local data matches the data the results were produced on.

## Adding a dataset

Models never read audio folders directly: they read a **manifest**, a CSV with one row per
clip. A new dataset therefore needs only a manifest builder.

1. **Download** it under `data/raw/<Name>/` (add the commands to this file).
2. **Configure** it in `configs/data.yaml`: a block with its `root` and anything the builder
   needs (protocol file, meta file, expected clip count), as `in_the_wild:` does.
3. **Build** the manifest in `src/audio_df/data/manifests.py`, next to `build_itw_manifest`.
   Fail with `DatasetNotFoundError` naming the expected path when the data is missing. The
   manifest needs these columns:

   | column | meaning |
   |---|---|
   | `utt_id` | unique clip id |
   | `path` | audio file path, relative to `SystemCode/` |
   | `label` | `bonafide` or `spoof` |
   | `speaker_id` | speaker, or `-` if unknown |
   | `attack_id` | generator or attack (`A07`, ...), `-` for bona fide or unknown |
   | `source_corpus` | dataset name |
   | `sample_rate`, `duration_s`, `codec` | from the file header |
   | `split` | `train`, `dev`, `eval` or the test-set name |
   | `language` | for example `en` |

4. **Expose** it in `scripts/prepare_data.py` (`--dataset <name>`), writing
   `data/manifests/<testset>.csv`. The test-set name must match `^[a-z0-9_]+$`.
5. **Record** it: running `prepare_data.py` rewrites `manifests/SHA256SUMS`; commit that file.
6. **Use** it: add the test set to a system config (`data.testsets.<name>`), then for example
   `uv run adf --config configs/a1_lfcc_gmm.yaml score --model-from runs/<run id> --exp e3
   --silence-mode raw --testsets <name>`.
7. **Test** the builder on a tiny fake copy of the layout in `tmp_path`, as
   `tests/test_data.py` does for ASVspoof and In-the-Wild.

Never train on a test-only set, and never let one speaker or generator appear on both sides
of a train/test split (`assert_no_speaker_overlap`).
