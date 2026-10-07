---
id: asvspoof-2021
type: dataset
sources:
  - https://www.asvspoof.org/index2021.html
  - https://github.com/asvspoof-challenge/2021 (official baseline + eval-package repo)
  - https://raw.githubusercontent.com/asvspoof-challenge/2021/main/eval-package/README.md (meta-label field definitions, fetched raw)
  - https://arxiv.org/abs/2210.02437 (ASVspoof 2021 summary paper, cited by the repo, not independently re-fetched this pass)
confidence: high
last_verified: 2026-09-20
related: [asvspoof-2019-la, eer-and-metrics]
---

# ASVspoof 2021 — LA/PA/DF tracks, metadata, baselines, scoring

## Claim in one sentence
ASVspoof 2021 adds three tracks (LA channel-robustness, PA replay, DF compression/codec
deepfake) on top of the 2019 LA attack set, ships per-trial meta-label strings (not just
bonafide/spoof) documenting codec/transmission/vocoder conditions, and provides four official
reproducible baselines plus a Python `eval-package` that computes pooled and decomposed min
t-DCF and EER from the full released keys.

## Findings

### Tracks (asvspoof.org/index2021.html)
- **LA (Logical Access)**: 2019 LA challenge extended with a focus on robustness to channel
  variation (codec/transmission effects layered on top of the same underlying attack types).
- **PA (Physical Access)**: replay-attack setup similar to 2019 PA, i.e. attacks captured
  and replayed through real/simulated physical acoustic paths and devices.
- **DF (Deepfake)**: new sub-challenge dedicated to speech deepfake detection, adding
  transmission-free but *compression*-processed spoofed/bonafide audio.

Speech data hosted on Zenodo: LA (zenodo.org/record/4837263), PA (record/4834716),
DF (record/4835108).

### Meta-label fields shipped per trial (eval-package README, "On meta-labels" section)
The organizers release a `trial_metadata.txt` per track with track-specific whitespace fields,
illustrated with real example lines:

**LA**: `LA_0009 LA_E_9332881 alaw ita_tx A07 spoof notrim eval`
Fields: speaker ID, trial ID, **codec** (`none`/`alaw`/`pstn`/`g722`/`ulaw`/`gsm`/`opus`,
labelled condition classes LA-C1..LA-C7), **transmission condition**
(`ita_tx`=FR-IT, `sin_tx`=FR-SG, `loc_tx`=local, `mad_tx`=PSTN-to-Spain), **attack ID**
(reuses A07–A19 from the 2019 LA database — no new attack algorithms were introduced for
2021 LA), key (`bonafide`/`spoof`), trim flag (`notrim`/`trim`), subset
(`eval`/`progress`/`hidden`).

**PA**: `PA_0010 PA_E_1000001 R3 M3 d4 r1 m1 s4 c4 spoof notrim eval`
Fields: speaker ID, trial ID, then environment/attacker factors (ASV room/mic/distance IDs,
attacker room/mic/distance/replay-device IDs), key, trim flag, subset. Note the README
explicitly flags that the `c`/`d` letter-number meaning was **revised** between the original
challenge release and the "full keys" release (old: `d2-d4`=attacker-to-talker,
`D1-D6`=device-to-ASV; new: `c2-c4`=attacker-to-talker, `d1-d6`=device-to-ASV) — a
documented backward-incompatible metadata change to watch for if mixing old/new PA key files.

**DF**: `LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -`
Fields: speaker ID, trial ID, **compression codec**
(`nocodec`/`low_mp3`/`high_mp3`/`low_m4a`/`high_m4a`/`low_ogg`/`high_ogg`/`mp3m4a`/`oggm4a`,
condition classes DF-C1..DF-C9), **source corpus** (`asvspoof`=from ASVspoof2019,
`vcc2018`, `vcc2020`), **attack ID** (again reuses A07–A19 from 2019 LA), key, trim flag,
subset (`eval`/`progress`/`hidden`), and a trailing **vocoder type** field
(`traditional_vocoder`, or `bonafide` for genuine trials, with further fields present but
unlabelled in the fetched excerpt).

This means: for DF specifically, `source_corpus` is an explicit, first-class metadata field
in the official release — directly relevant to the Stage-0 leakage audit's `source_corpus`
nuisance-feature check (see `leakage-audit.md`).

### Official baselines (github.com/asvspoof-challenge/2021 README)
Four baseline CMs, provided for all three tracks (LA, PA, DF):
- `Baseline-CQCC-GMM` (Matlab & Python) — CQCC features + GMM classifier
- `Baseline-LFCC-GMM` (Matlab & Python) — LFCC features + GMM classifier
- `Baseline-LFCC-LCNN` (PyTorch) — LFCC features + LCNN (deep) classifier
- `Baseline-RawNet2` (PyTorch) — end-to-end raw-waveform DNN

### Official scoring/metric package
`eval-package/` in the same repo:
- A download script (`download.sh`) that fetches the **full set of keys and meta-labels**
  per track from asvspoof.org (`LA-keys-full.tar.gz`, `PA-keys-full.tar.gz`,
  `DF-keys-full.tar.gz`).
- `main.py`, run as `python main.py --cm-score-file score.txt --track LA --subset eval`,
  computing **pooled and decomposed min t-DCF and EER** given a CM score file plus the
  released keys; supports supplying custom ASV scores and recomputing the t-DCF cost
  coefficients (`C012`).
- An interactive Colab-ready notebook, `ASVspoof2021_eval_notebook.ipynb`, wrapping the same
  computation.
- Score files from each official baseline are included alongside the keys
  (`keys/<track>/CM/<baseline>/score.txt`) for direct comparison.

## Mechanism / derivation
N/A — this is a protocol/tooling note. See `eer-and-metrics.md` for the EER/t-DCF math itself
(shared with 2019).

## Implications for us

- If the project extends beyond ASVspoof 2019 LA to an unseen-generator/cross-condition
  ASVspoof 2021 evaluation (handoff §Datasets, "add a cross-dataset or unseen-generator
  evaluation if resources permit"), the codec/transmission (LA) and compression/source
  (DF) fields are exactly the kind of nuisance metadata the leakage audit (Stage 0) should
  probe — 2021 was explicitly designed to vary these, so a leakage classifier trained on 2019
  and applied to 2021 conditions must be re-checked, not assumed to transfer.
- `configs/data.yaml` currently only targets `asvspoof2019_la`; if T05/T09 are ever extended
  to 2021 data, the manifest schema will need extra columns (`codec`, `transmission_or_compression`,
  `trim_flag`, `subset` beyond eval/progress/hidden) that do not exist in the 2019 LA protocol
  format documented in `asvspoof-2019-la.md`. Do not assume the 2019 5-column parser is
  reusable as-is for 2021 trial_metadata files — the field count and meaning differ per track.
- No new LA attack IDs were introduced in 2021 LA/DF — both reuse A07–A19 from 2019. This
  means "unseen attack" claims on ASVspoof2021 LA/DF are about seen *conditions* (codec,
  compression) with the *same* generator families, not seen vs unseen generators; that is a
  materially different generalisation claim than the 2019 eval's true unseen-attack test, and
  should not be conflated in the final report.

## Open questions / what would change this
- I did not independently re-fetch the ASVspoof 2021 summary paper (arXiv:2210.02437) or the
  official evaluation plan PDF (asvspoof.org/asvspoof2021/asvspoof2021_evaluation_plan.pdf)
  in this pass — the baseline/tooling facts above come from the GitHub repo and its
  eval-package README, which I consider primary (organizer-authored, same repo referenced by
  index2021.html), but exact per-track utterance counts for 2021 (train/dev/eval sizes) were
  not found in the sources fetched and are **not asserted** in this note. Confidence would
  rise to including exact counts after fetching the evaluation plan PDF directly.
- The DF trailing fields after `traditional_vocoder` in the example line were truncated in
  the fetched excerpt (`- - - -`); their meaning is unconfirmed and not used in any claim
  above.
