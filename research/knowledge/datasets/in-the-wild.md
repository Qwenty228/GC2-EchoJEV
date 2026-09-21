---
id: in-the-wild
type: dataset
sources:
  - https://arxiv.org/pdf/2203.16263 (Müller, Czempin, Dieckmann, Froghyar, Böttinger, "Does Audio Deepfake Detection Generalize?", Interspeech 2022 — fetched and read directly, arXiv:2203.16263v5, 27 Mar 2026 revision)
  - https://huggingface.co/datasets/mueller91/In-The-Wild (dataset card — fetched; size-on-disk, license, single `train` split)
  - https://arxiv.org/pdf/2106.12914 (Müller et al., "Speech is Silver, Silence is Golden" — same author group, cited by the In-the-Wild paper itself as ref [28] to explain why ASVspoof `eval` alone cannot measure generalization; already the primary source for `research/knowledge/methods/leakage-audit.md`)
confidence: high
last_verified: 2026-09-21
related: [leakage-audit, asvspoof-2019-la, mlaad]
---

# In-the-Wild — construction, size, and the generalization result it exists to demonstrate

## Claim in one sentence
In-the-Wild is a small (37.9 h), unsplit, English-only, 58-speaker, **manually curated**
evaluation-only corpus built specifically to show that ASVspoof-trained detectors collapse
(EER inflates 200–1000%, often to ~random) on real found audio — it ships with no attack
labels, no official train/dev/eval partition, and no guarantee that per-speaker bonafide/spoof
duration or recording condition is matched beyond the authors' manual best-effort curation.

## Findings

### What it is, exactly (paper §3, read directly)
- 37.9 hours total: **17.2 h fake / 20.7 h bonafide** (Figure 1 caption states 20.8 h bonafide
  — the abstract and Figure 1 differ by 0.1 h, evidently a rounding inconsistency in the paper
  itself, not a discrepancy in our reading).
- 58 speakers, all English-speaking celebrities/politicians, "both from present and past."
  Average bonafide audio per speaker: 23 minutes; average spoof audio per speaker: 18 minutes.
- Clips average **4.3 seconds** long.
- Construction: fake clips were extracted by **segmenting 219 publicly available video/audio
  files that explicitly advertise themselves as deepfakes** (the paper's example: "Donald
  Trump reads Star Wars") — i.e. labels are trusted from the source's own claim of being
  synthetic, not independently verified against a known generator. Corresponding bonafide
  clips were then **manually collected** for the same speakers from podcasts/speeches, with
  the authors stating they "take care to include clips where the type of speaker, style,
  emotions, etc. are similar to the fake ... and try to find similar values for background
  noise, emotions, duration, etc." — this is a **best-effort, manual, unverified** matching
  process, not a controlled/audited one.
- All audio downsampled to 16 kHz ("the highest common frequency in the original recordings")
  and converted to WAV after download.
- The dataset has **no attack/generator ID field** — the paper does not report which TTS/VC
  system produced any given fake clip, and does not claim to know. This is a direct
  consequence of sourcing fakes opportunistically from the internet rather than generating
  them under controlled conditions.
- The paper explicitly states the dataset is "intended as evaluation data" for testing
  cross-database generalization — not as a training corpus with an official split. The HF
  dataset card confirms this operationally: it ships as a **single `train` split** (i.e. no
  official train/dev/eval partition exists; any split we impose is ours to construct and
  justify).

### The headline result this dataset exists to produce (paper §5.3, Table 1)
- Twelve reimplemented architectures (LCNN family, LSTM, MesoNet family, ResNet18,
  Transformer, CRNNSpoof, RawNet2, RawPC, RawGAT-ST), each evaluated across 4 feature types ×
  2 input-length regimes, trained on ASVspoof 2019 LA train+dev, tested on ASVspoof19 eval and
  on In-the-Wild.
- Best ASVspoof19-eval performer (RawGAT-ST, full-length raw input): **1.23% EER** in-domain
  vs **37.15% EER** on In-the-Wild — a ~30x relative degradation.
- Averaged across all 24 configurations: ASVspoof19-eval EER 9.85–18.89% (full vs 4s input)
  vs In-the-Wild EER **60.10–67.25%**, i.e. close to or at chance (50%).
- Training RawNet2 on **all** ASVspoof19 splits (train+dev+eval combined) and re-testing on
  In-the-Wild still yields 33.1% EER — "no improvement over training with only the train and
  dev data" — because all ASVspoof19 splits share the same underlying VCTK recording
  chain/speaker pool, so more ASVspoof data does not buy real domain diversity (paper §5.3,
  direct quote: "all splits of ASVspoof are fundamentally based on the same dataset, VCTK").
- Paper's own explanatory framing (Conclusion): "the community has tailored its detection
  models too closely to the prevailing benchmark, ASVSpoof, and ... deepfakes are much harder
  to detect outside the lab than previously thought."

### License and hosting (HF dataset card)
- Hosted at `https://huggingface.co/datasets/mueller91/In-The-Wild`, single ZIP / single
  `train` split, **8.16 GB** on disk.
- License stated on the HF card: **CC-BY-SA-4.0**. The paper's own PDF does not restate a
  license in-text (only the arXiv non-exclusive distribution notice for the *paper*, not the
  *data*); the HF card is the only license source found. Attribution + share-alike terms
  apply — no restriction on non-commercial-only use was found, unlike MLAAD (see
  `mlaad.md`).

## Mechanism / derivation
N/A — this is a corpus-construction note, not a formula-derivation note.

## Implications for us

- **In-the-Wild has no attack/generator labels.** `revisedata.md`'s plan to hold out "at
  least one generator" for an unseen-generator test cannot be built from In-the-Wild itself —
  that experiment has to come from our own custom-generated TTS set (or from MLAAD, which
  does carry a `model_name`/`architecture` field). In-the-Wild's role can only be
  **cross-dataset/cross-domain generalization**, exactly RQ3 in `docs/handoff.md`'s numbering
  — never per-generator EER, and any table that reports a per-generator breakdown "on
  In-the-Wild" is fabricating a label the dataset does not provide.
- **There is no official split.** Any train/dev/eval partition of In-the-Wild is ours to
  construct. Given the project's own leakage-audit discipline (`leakage-audit.md`), if we
  ever train on a subset of In-the-Wild and eval on another subset of it, we must split by
  **speaker**, not utterance — 58 speakers is few enough that a bad split could put the same
  speaker's voice/recording-channel signature in both partitions, reintroducing exactly the
  channel-based shortcut the leakage audit exists to catch. The safest and best-evidenced use
  remains what the paper itself designed it for: **frozen, no-training, pure external
  evaluation** of a model trained elsewhere (ASVspoof or our own TTS set).
- **The 4.3 s average clip length is compatible with our 4 s crop/pad policy** (`docs/handoff.md`
  Stage 0 preprocessing), which is a mild point in favor of low-friction integration — but note
  the source paper's own Table 2 finding that fixed 4 s crops *hurt* EER relative to full-length
  input (19.89% → 9.85% average EER improvement from using full-length audio on ASVspoof19).
  If we crop In-the-Wild to 4 s to match our pipeline, we should expect this to work against us
  somewhat versus using full clips, and should say so if reporting In-the-Wild numbers next to
  full-length literature numbers.
- **This is not a mild "add one more eval set" decision — it is the single most informative
  sanity check available to this project for free.** Given the extremity of the reported
  degradation (up to 30x relative EER inflation for the best in-domain model), any Stage 4/5/6
  model this project builds should expect a similarly severe collapse when scored zero-shot on
  In-the-Wild, and a report that does *not* show a large gap should be treated with suspicion
  (something in the eval pipeline is probably wrong) rather than celebrated as a research win,
  unless the model was specifically designed and evidenced to generalize.
- **Class balance is close (54%/46% bonafide/fake by duration) but manually curated, not
  guaranteed matched.** Before drawing conclusions from an EER computed on In-the-Wild, our own
  Stage 0-style leakage audit should be re-run *on In-the-Wild's own metadata* (duration,
  silence, RMS) — the authors' manual speaker/style/duration matching is a good-faith effort,
  not an audited guarantee, and the project's own precedent (the Müller et al. 2106.12914
  silence-duration shortcut on ASVspoof) is a direct warning that manually-curated real/fake
  pairs from different sourcing pipelines can carry exactly this kind of unintended nuisance
  correlation.

## Open questions / what would change this
- I could not find, in the paper text or the HF card, a statement of exactly *how* the authors
  verified that a source video's "explicitly advertise audio deepfakes" claim was actually
  true (i.e. no independent forensic verification of the fake label is described) — this is
  worth flagging as a soft assumption underlying the entire dataset, though not one this
  project can resolve.
- The paper (v5, most recently revised 27 Mar 2026 per the arXiv header on the copy fetched)
  post-dates the original 2022 Interspeech publication; I did not diff v1 against v5 to check
  whether the dataset description itself changed across revisions — flagged as unverified but
  low-risk, since the HF dataset card (checked independently, same numbers) corroborates the
  size figures in this v5 text.
- No per-file metadata (duration, exact recording source, speaker ID) is exposed through the
  HF dataset viewer (permissions error encountered); if this project needs per-utterance
  metadata beyond filename, it will need to be derived locally after download, not read from
  the HF card.
