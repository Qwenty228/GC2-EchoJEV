---
id: leakage-audit
type: method
sources:
  - docs/handoff.md (§"Stage 0 - Dataset and leakage audit", this repo)
  - https://arxiv.org/abs/2308.14970 (Yi et al., "Audio Deepfake Detection: A Survey" — general generalization/interpretability framing; abstract fetched, full text not reachable as text this pass)
  - https://arxiv.org/pdf/2106.12914 (Müller et al., "Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?", fetched and read directly — the concrete, quantitative silence-shortcut evidence this note leans on)
confidence: high
last_verified: 2026-09-20
related: [asvspoof-2019-la]
---

# Leakage audit: nuisance-feature shortcuts in ASVspoof-style corpora

## Claim in one sentence
ASVspoof 2019 LA has a *documented, measured, large* shortcut in leading/trailing silence
duration alone — a single-feature classifier reaches 15.1% EER (vs 50% random) on eval, and
established deep models (RawNet2, ResNet, LSTM, CNN-GRU) lose 5–500% relative EER when
silence is trimmed from training — which is direct, quantitative precedent for exactly the
Stage-0 leakage audit this project is required to run, and sets a concrete bar for what
"suspiciously well" and "red flag" should mean numerically.

## Findings

### Why ASVspoof-style corpora are prone to this (handoff + Müller et al., mechanism)
The handoff frames the general risk: nuisance metadata (duration, sample rate, codec,
RMS/loudness, leading/trailing silence, source corpus, speaker ID) can correlate with the
label for reasons that have nothing to do with synthesis artifacts, because bonafide and
spoof audio in these corpora are typically produced by *different pipelines*, not by
perturbing the same recording. Müller et al. give the concrete causal mechanism for silence
specifically (§2, §6): bonafide LA utterances are natural VCTK recordings that retain their
original leading/trailing silence, while spoofed utterances are the output of TTS/VC systems
whose training data and synthesis process typically have silence trimmed — "arbitrary
silences introduce uncertainty into the TTS training process," so synthesis systems are
built to not produce them. The result is a systematic difference in silence duration between
the two classes that has **nothing to do with any acoustic artifact of synthesis** — it's an
artifact of how the corpus's two classes were sourced, exactly the "microphone noise -> real"
failure mode the handoff warns about.

### Quantitative evidence (Müller et al., read directly, Tables 1–4 and Fig. 1)
- A **single scalar feature** (duration of leading silence, in seconds) fed into a small FCNN
  achieves **15.12% EER** on the ASVspoof2019 LA eval set (dev: 31.09% EER), versus a
  50% EER random baseline. On the ASVspoof2021 progress-phase sets this same single-feature
  model gets **19.93% EER (LA21)** and **17.42% EER (DF21)** — i.e. the shortcut is not
  unique to 2019, it persists into 2021.
  - Note *why* dev underperforms eval here: per Fig. 1c, the eval split has more attacks
    (A07–A16) whose silence duration cleanly separates from bonafide, compared to dev's A01–A06.
- Established, considerably stronger models (ResNet, LSTM, CNN-GRU trained on real
  MFCC/CQCC features, not just the silence scalar) reach EER as low as **5.87%–8.33%** when
  they have unrestricted access to the leading/trailing silence, but **never do better than
  19.05% EER** in any configuration where the silence is either trimmed or the model is
  restricted to a random subsegment of the audio — i.e. **removing the shortcut costs these
  strong models roughly 2.5–3x their EER**, strongly suggesting a large fraction of their
  discriminative power came from the silence artifact, not from acoustic content.
- The official **RawNet2** baseline (raw-waveform, no hand-crafted features, an official
  ASVspoof-family baseline model — directly relevant since the handoff and this project also
  use RawNet2) shows the starkest effect: EER on 2019 eval goes from **3.61% → 15.50%**
  (≈+330% relative) when silence is trimmed from training; on ASVspoof2021, EER goes from
  **6.28% → 20.79%** (LA-relevant: DF) and **10.38% → 27.39%** (LA), each roughly a
  **5x relative degradation**. This is read directly from the paper's own tables (Table 4),
  not inferred — RawNet2 is exactly the model this project's own Stage 5 plans to reproduce.
- Providing silence duration explicitly as an *auxiliary input feature* gave **no additional
  benefit** over models that already had raw access to the silence (§5.4) — the models were
  already extracting the duration information implicitly from the waveform/features, so the
  shortcut is not something a model needs to be told about; it will find it on its own if
  present.
- The paper's own recommended remedies (§8, Conclusion): (a) publish/use a silence-trimmed
  version of the corpus (e.g. via `librosa.effects.trim` or a forced aligner like MFA), or
  (b) construct new train/dev/eval splits where silence duration is explicitly balanced
  across bonafide/spoof and across attack IDs, not merely present-or-absent.

### What AUC threshold constitutes a red flag, and on what basis
Neither the handoff nor Müller et al. state a single canonical numeric AUC threshold; the
handoff leaves this to the implementing agent's judgement ("if these perform suspiciously
well, flag the dataset/split"), and `tasks/TASKS.md` T09 operationalizes this as a
**configurable default of AUC > 0.65** triggering a `LEAKAGE WARNING`. I could not find an
ASVspoof-consortium or Müller-et-al.-stated numeric AUC threshold as a community standard —
this specific number (0.65) appears to be a project-local design choice, not a literature
citation. What the Müller et al. numbers **do** support is that this threshold is
conservative rather than arbitrary-and-too-loose: their weakest, single-feature shortcut
classifier alone reaches an EER of 15.1% (i.e. well above chance, corresponding to AUC well
above 0.65 by any standard EER-AUC relationship for a reasonably separable single feature),
so a 0.65 AUC gate would clearly catch the silence shortcut as documented, and is not
set so high that a known, real-world shortcut of this project's own target dataset would
slip past it.

## Mechanism / derivation
N/A — this is an empirical/methodological note, not a formula-derivation note.

## Implications for us

- **This project should treat duration/silence as the single highest-priority nuisance
  feature to test in T09**, ahead of the others listed in the handoff (sample rate, codec,
  RMS, source corpus, speaker ID) — it is the one nuisance feature with **direct, published,
  quantitative evidence of exploitation on this exact dataset** (ASVspoof 2019 LA), not a
  generic concern.
- T09's synthetic test ("a synthetic manifest where duration *is* the label → audit flags
  duration") should ideally be supplemented with a **real-data check**: fit the same
  metadata-only leakage classifier on the actual ASVspoof2019 LA manifest's leading/trailing
  silence duration and confirm it reproduces something in the neighborhood of the ~15% EER /
  well-above-0.65-AUC Müller et al. report. If the local implementation gets a substantially
  different (much lower) number on the real data, that is itself worth investigating — either
  the manifest's duration/silence field is computed differently, or something about the local
  data handling (e.g. resampling, trimming already applied upstream) has changed the
  artifact's magnitude, and that should be reported back, not silently accepted.
- Given RawNet2's own measured ~5x EER inflation from this exact shortcut (per the table
  above), **any RawNet2 result this project reports in Stage 5 should be accompanied by a
  silence-trim ablation**, or at minimum an explicit note in the final report that the
  reported RawNet2 EER may be partly attributable to the silence artifact rather than to
  learned synthesis-detection ability — this directly affects how any RawNet2 number is
  interpreted in the final benchmark table, not just the Stage-0 audit table.
- The correct response when a shortcut of this kind is found (per both the handoff and
  Müller et al.'s own recommended remedies) is **not** to simply proceed and caveat the
  result — it is to fix the split/data (trim silence consistently, or rebalance duration
  across classes/attacks) before drawing conclusions from any downstream model's EER,
  because a model that has learned the shortcut will "solve" the task in a way that
  systematically fails to generalize to future/real-world audio without exactly this same
  artifact.

## Open questions / what would change this
- The arXiv survey (2308.14970) that the handoff names as its Stage-0 reference did not, in
  the pages/abstract I could reach, contain the specific silence/duration-shortcut discussion
  — that content came from the more specific and more directly applicable Müller et al. paper
  instead, which I found via targeted search rather than being told to use it. I recommend
  the index/handoff explicitly add this paper (arXiv:2106.12914) as a named Stage-0 leakage
  reference alongside the survey, since it is materially more useful for this specific task
  than the general survey is.
- I have not verified whether the exact manifest/duration-extraction code this project will
  use (`src/audio_df/data.py`, once written) computes "leading/trailing silence duration" the
  same way Müller et al. did (`librosa.effects.trim`, `ref_db=40`); if a different silence
  detector or threshold is used, the measured leakage magnitude could differ from the
  15.1%/3.61%/etc. figures cited here, though the qualitative conclusion (silence leaks
  label) should still hold given it's reproduced across four different model families in the
  source paper.
- No numeric AUC threshold in the literature was found to independently corroborate or
  contest T09's `0.65` default; this remains a project-local convention, not a sourced fact.
