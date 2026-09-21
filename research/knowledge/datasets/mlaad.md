---
id: mlaad
type: dataset
sources:
  - https://arxiv.org/pdf/2401.09512 (Müller, Kawa, Choong, Casanova, Gölge, Müller, Syga, Sperl, Böttinger, "MLAAD: The Multi-Language Audio Anti-Spoofing Dataset" — fetched and read directly, arXiv:2401.09512v11, dated 28 Aug 2026 / Table I release 01.09.2026)
  - https://huggingface.co/datasets/mueller91/MLAAD (dataset card — fetched; license, download size, meta.csv field list, `reference_speaker` note)
confidence: high
last_verified: 2026-09-21
related: [in-the-wild, leakage-audit, asvspoof-2019-la]
---

# MLAAD — versions, structure, bonafide pairing, and what a defensible subset requires

## Claim in one sentence
MLAAD is a **continuously-expanding, versioned** corpus (v1 at 63.0 h / 21 models / 22
languages, up to v11 at 1153.6 h / 205 models / 54 languages as of the copy read), built by
resynthesizing M-AILABS bonafide text with many TTS systems — its own meta.csv already
carries every field (`model_name`, `architecture`, `language`, `duration`, `reference_speaker`)
needed to build a *reproducible, family-held-out* subset, and the bonafide pairing (M-AILABS)
is fixed by construction, not a free choice we get to make.

## Findings

### Versioning (Table I, read directly from the PDF)
| version | release date | hours | # TTS models | # languages | distributed via |
|---|---|---|---|---|---|
| v1 | 17.01.2024 | 63.0 | 21 | 22 | zip |
| v2 | 28.02.2024 | 63.0 | 21 | 22 | zip |
| v3 | 16.04.2024 | 163.9 | 54 | 23 | zip |
| v4 | 24.09.2024 | 175.0 | 59 | 23 | zip |
| v5 | 02.11.2024 | 398.3 | 87 | 38 | zip + HuggingFace |
| v6 | 26.04.2025 | 420.7 | 91 | 38 | HuggingFace |
| v7 | 11.07.2025 | 485.3 | 101 | 38 | HuggingFace |
| v8 | 01.10.2025 | 570.1 | 119 | 40 | HuggingFace |
| v9 | 17.01.2026 | 687.4 | 140 | 51 | HuggingFace |
| v10 | 18.05.2026 | 1002.9 | 175 | 54 | HuggingFace |
| v11 | 01.09.2026 | 1153.6 | 205 | 54 | HuggingFace |

**This is the number the project should cite, not a single fixed "MLAAD size."** Any prior
note or plan that says "MLAAD is ~200 GB / hundreds of hours" needs a version number attached
— `concern.md`'s "hundreds of GB" framing is consistent with v9–v11, not with v1–v4, and the
practical implication (too large to use wholesale on a Mac) holds most strongly for the
current, largest versions. The HF card (checked separately) states a **183 GB** total download
size at the version it describes (v8, per the `reference_speaker`-since-v8 note on the card),
consistent with a ~500 h+ version, not the small v1.

### Construction (paper §III-A, Fig. 1, read directly)
- MLAAD is built as an extension of the **M-AILABS Speech Corpus** — 8 original languages
  (English, French, German, Italian, Polish, Russian, Spanish, Ukrainian), audiobook/speech
  recordings, 22050 Hz / 16-bit WAV.
- For each (language, TTS model) pair, **1000 utterances** are randomly sampled from M-AILABS
  as the transcript/reference source. If the target language isn't in M-AILABS, the English
  M-AILABS transcripts are machine-translated (via NMT) into the target language first.
- For multi-speaker-capable TTS models, a **reference speaker file is drawn randomly from the
  same M-AILABS baseline pool**; single-speaker models (e.g. LJSpeech-trained) use no
  reference speaker.
- **M-AILABS originals are the bonafide/training-target class**; MLAAD's own repo does **not**
  redistribute the bonafide originals — "we distribute only these synthesized samples, for the
  originals please refer to the [separate] M-AILABS dataset" (paper §III-B). This means: MLAAD
  by itself is fakes-only; the bonafide pairing (M-AILABS) is an external dependency we must
  download separately, and it is fixed by the dataset's own construction — we do not get to
  substitute a different bonafide source without breaking the intended pairing (same
  transcript source, same original recording chain as the synthesis targets).
- A Griffin-Lim resynthesis condition (1000 utterances per each of the 8 original M-AILABS
  languages) is also included as a lower-quality "drop-in" synthetic baseline, separate from
  the 205 neural TTS models.

### Structure and metadata (Table III, paper §III-B, HF card)
Directory layout: `fake/<language>/<model_name>/` each containing synthesized WAVs plus one
`meta.csv`. Fields in `meta.csv` (Table III, exact list, `reference_speaker` added since v8):
`path`, `original_file` (path into the M-AILABS tree), `language`, `is_original_language`
(flag for whether the transcript is native or machine-translated), `duration`, `training_data`
(what the TTS model itself was trained on), `model_name`, `architecture`, `transcript`,
`reference_speaker`.

The paper's own Table IV lists per-architecture total hours (e.g. Bark 123.3 h, XTTS v2
38.9 h, VITS 58.1 h, down to sub-2h entries for many smaller/rarer architectures) — **the
corpus is extremely unbalanced across architectures**: a handful of systems (Bark, Edge-TTS,
OmniVoice, Llasa, XTTS v1.1/v2, MOSS-TTS) contribute tens of hours each while dozens of
others contribute under 2 hours. A random utterance-level subsample of MLAAD would therefore
massively over-represent a small number of architectures purely by construction, independent
of any deliberate "family holdout" design.

### Evaluation protocol used by the MLAAD authors themselves (paper §IV, Table VI)
- The authors deliberately use **accuracy, not EER**, as their headline metric, explicitly
  because several of their eight cross-dataset test sets (Voc.v, WaveFake) are fake-only and
  EER cannot be computed without both classes present. This is a direct, stated methodological
  choice by the dataset's own authors — worth noting since our project's convention (per
  `eer-and-metrics.md`) is EER-first; MLAAD-derived subsets we build ourselves should still
  retain both classes so EER remains computable, rather than following MLAAD's own accuracy
  convention.
- Training and evaluating three CM models (RawGAT-ST, SSL-W2V2, WhisperDF) across four
  training sources (MLAAD v1, ASVspoof19, FoR, ITW) × eight test sets, the authors' own
  finding: **no single training dataset dominates** — ASVspoof19 and MLAAD each win on 4 of 8
  cross-dataset test cases; models trained on MLAAD are "the only ones that do not exhibit
  performance significantly below 50%" on any test set, which the authors read as MLAAD
  containing comparatively **fewer exploitable shortcut features** than ASVspoof19 or FoR (paper
  §IV-B, citing their own earlier shortcut-learning work, ref [60] = the same Müller et al.
  2106.12914 "Speech is Silver, Silence is Golden" paper already used in `leakage-audit.md`).
  This is the authors' own claim about their own dataset, not independently reproduced by us —
  tagged `source: literature`, not evidence about our pipeline.
- Note: the cross-dataset table in the paper explicitly says results "use MLAAD v1; the
  evaluation was not re-run for subsequent releases" — so even the authors' own generalization
  numbers for MLAAD are tied to the smallest (63 h) version, not the current 1153.6 h version.

### License
- HF dataset card (`mueller91/MLAAD`): **CC-BY-NC 4.0** — non-commercial only. This is
  materially different from In-the-Wild's CC-BY-SA-4.0 (see `in-the-wild.md`): MLAAD subset
  use in this project is fine for an academic/workshop deliverable, but any output that could
  be read as commercial use (e.g. redistributing a trained checkpoint commercially) would need
  separate clearance. The paper PDF itself states only the arXiv non-exclusive distribution
  notice for the *paper*, not the *dataset* — the HF card is the operative license source for
  the data.

## Mechanism / derivation
N/A — corpus-construction and versioning note.

## Implications for us

- **`revisedata.md`'s "controlled MLAAD subset instead of full MLAAD" is well-founded and the
  dataset's own metadata already supports it.** `concern.md`'s explicit requirement — "prefer
  holding out complete generator families... do not randomly sample individual utterances" —
  maps directly onto MLAAD's `architecture` field (Table III), and the severe per-architecture
  hour imbalance (Table IV) makes clear that a naive random utterance sample would already be a
  de facto (uncontrolled, undocumented) family-imbalanced sample. A defensible subset should
  therefore: (a) pick a small, fixed list of architectures to include, stratified to include
  both historically standard ones (VITS, Tacotron2, GlowTTS) and modern ones (XTTS v2,
  Bark, F5-TTS, StyleTTS2 if present in the version pulled), (b) hold out at least one whole
  architecture (not a random slice of one), (c) fix language coverage explicitly rather than
  letting the 54-language spread dominate the sample, and (d) record the exact MLAAD version
  pulled, since version determines which architectures even exist in the pool (e.g. F5-TTS,
  Higgs-Audio, Llasa, VibeVoice, VoxCPM only exist in later versions per Table IV).
- **MLAAD is fakes-only; the bonafide pairing is M-AILABS, not optional.** If this project
  generates its own TTS test set (workstream B in the task brief) using *different* text
  sources or bonafide corpora than the ones an MLAAD-derived subset uses, we will have created
  **two different bonafide populations** (M-AILABS-derived for MLAAD-subset experiments,
  whatever-we-choose for our custom TTS set) inside the same project. That is exactly the
  corpus-mismatch trap described in the task brief part B: a detector evaluated jointly across
  both could learn "which bonafide corpus" rather than "real vs fake." If MLAAD is used at all,
  M-AILABS should be the bonafide source paired with it specifically, and any custom-TTS
  workstream bonafide should be evaluated/reported separately, never pooled into one training
  set without controlling for corpus identity as a variable.
- **EER, not accuracy, should remain our metric even on an MLAAD-derived subset** — the
  authors' own switch to accuracy was a pragmatic response to some of *their* eight test sets
  being fake-only; our subset, built by us, can and should always retain both classes, so this
  is a solvable design constraint, not an inherited limitation.
- **The authors' own generalization claims about MLAAD are stale relative to the current
  dataset** (evaluated only on v1, not re-run since). Any claim this project makes of the form
  "MLAAD helps generalization" cannot lean on the MLAAD paper's own Table VI as evidence for
  whatever version/subset we actually use — that table is `source: literature`, about a
  different (older, smaller) release, and must not be presented alongside our own numbers as if
  comparable.

## Open questions / what would change this
- I have not verified the exact size-on-disk or file count for any MLAAD version other than
  what the HF card states for its own snapshot (183 GB, described alongside the v8
  `reference_speaker` field) — the paper's Table I gives hours, not GB, per version, so the
  GB-per-version mapping used by `concern.md` is an approximation, not something I confirmed
  directly against a specific version's download manifest.
- I could not access the MLAAD GitHub/HF repository's actual file tree to confirm whether
  per-architecture subsets can be downloaded independently (e.g. via HF `datasets` filtering)
  without pulling the whole multi-hundred-GB corpus first — this materially affects whether "a
  controlled subset" is cheap or expensive to obtain, and should be checked by whoever
  implements the download step before committing to a specific architecture list.
- I did not verify whether the CC-BY-NC-4.0 license found on the HF card is stated identically
  in the dataset's own included LICENSE file (not fetched) — flagged as a minor gap, low risk
  given HF dataset cards are normally sourced from the uploader's own stated terms.
