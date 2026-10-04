# Chapter 2 · Where does our data come from, and how do you score fairly when 9 of 10 clips are fake?

*Notebook cell numbers count the first cell of a notebook as cell 0.*

## A detector that is 90 % accurate and useless

Here is a detector you can write in one line: **always answer "fake"**.

On our development set it is right 89.7 % of the time. That set has 2,548 real clips and
22,296 fake ones, and 22,296 ÷ 24,844 = 89.7 % (counts from
[metrics.json](../../experiments/E-20261001-e1-a2-raw/metrics.json), `testsets.dev`). It has
learned nothing at all.

So before we build anything, we need two things: to know exactly what is in the data, and a
score that cannot be fooled by the one-line detector.

## Where the clips come from

Our main dataset is the 2019 edition of the Automatic Speaker Verification Spoofing and
Countermeasures Challenge (ASVspoof), its "logical access" (LA) part, where fakes are fed in
digitally rather than played through a loudspeaker. In this chapter "real"
means **bona fide** (genuine human speech) and "fake" means **spoof**.

- **Real clips** all come from one source, the Voice Cloning Toolkit (VCTK) corpus: 107 English
  speakers recorded in a hemi-anechoic (almost echo-free) chamber at 96 kilohertz (kHz, thousands of samples per
  second), then downsampled to 16 kHz (from paper [14, §2, p.3]).
- **Fake clips** were made by text-to-speech (TTS) and voice conversion (VC) systems, the
  two-stage generators from chapter 1, trained mainly on separate recordings of the same VCTK
  speakers, 200 utterances per speaker (from paper [14, §2.2, p.5]); a few attacks also used
  other data, for example proprietary voice data for A09 and a commercial TTS voice as the
  source for A14 (from paper [14, §3, p.8]).

Green is real, orange is fake, everywhere in this series.

![Clips per split and attack: train, dev, eval and In-the-Wild](img/ch02_clips_per_split.png)
*Source: [notebook 20](../../notebooks/20_story_figures.ipynb), cells 5–6, from the local
manifests checked against committed checksums.*

The three ASVspoof splits have three jobs: **train** (2,580 real, 22,800 fake, attacks A01–A06)
is for fitting the detectors; **dev**, the development split (2,548 real, 22,296 fake, the same
attacks), is for choosing settings and the decision threshold; **eval**, the evaluation split
(7,355 real, 63,882 fake, attacks A07–A19), is the final test.

The counts come from our own manifests (the lists of clips); the notebook cell checks those
files against their committed checksums in `data/manifests/SHA256SUMS` and checks the dev and
eval totals against the committed `metrics.json`. They agree with the database paper's own
figure (from paper [14, Fig. 2, p.4]).

Three things to notice:

1. **About 9 fakes for every real clip** in every split. In train, 22,800 ÷ 2,580 = 8.8.
2. **The eval attacks never appear in training.** Train and dev contain A01–A06 only; eval
   contains A07–A19. So the official test is already a test on generators the detector has
   never seen. That is why the project uses it as-is rather than building its own split
   (architecture decision record [ADR-001](../../research/decisions/ADR-001-generalization-data-strategy.md)).
3. **Two of those "new" attacks are old algorithms in disguise.** The paper states that "A04
   and A16 use same waveform concatenation TTS algorithm, and A06 and A19 use same VC algorithm"
   (from paper [14, Table 1 caption, p.6]), and that A07–A19 "(apart from A16 and A19) are the
   eleven unknown spoofing systems" (from paper [14, §3, p.6]). A16 and A19 are hatched in the
   figure. So eval has 11 truly unseen generators and 2 known ones applied to new speakers.

One more safety check: no speaker appears in both train and dev. The function
[`assert_no_speaker_overlap`](../../src/audio_df/data.py#L252) enforces it, and
[notebook 01](../../notebooks/01_dataset_audit.ipynb), cell 10, runs it on the real data.

### The second dataset: In-the-Wild

The right-hand panel of the same figure is a second, very different dataset, In-the-Wild
(ITW), built by Müller et al. Its real clips are celebrities and politicians from sources such
as podcasts and speeches; its fakes are cut from videos found online that advertise themselves
as deepfakes (from paper [13, §3, p.2]).

- It has **31,779 clips: 19,963 real and 11,816 fake** (our manifest, cross-checked against
  [A2 In-the-Wild metrics.json](../../experiments/E-20261001-e3-a2-raw/metrics.json)). The balance is
  reversed: here most clips are real.
- Its fakes have **no attack labels**, so we cannot break results down by generator.
- The paper reports 58 speakers (from paper [13, §3, p.2]); the dataset's own list names 54
  distinct speakers. We have not resolved this.

We never train on In-the-Wild. It is the "new world" test from chapter 0: new speakers, new
microphones, new generators, all at once.

## Two kinds of mistake

Every detector in this project (A1 and A2 from chapter 0) outputs a **score** per clip, with the convention **higher means
more real**. This is the official ASVspoof convention
([eer-and-metrics.md](../../research/knowledge/methods/eer-and-metrics.md)). To turn scores into
decisions, you pick a **threshold**: above it, call the clip real; below it, call it fake.

Any threshold makes two kinds of mistake:

- a **miss**: a real clip scored below the threshold, so a genuine person is rejected;
- a **false alarm**: a fake clip scored above the threshold, so a fake is accepted.

Here is the key move. Measure each mistake **as a rate within its own class**: misses as a
share of real clips, false alarms as a share of fake clips. Then the 1 : 9 ratio drops out. The
one-line "always fake" detector rejects every real clip, a 100 % miss rate, and is caught at
once.

## Sliding the threshold

![A2 dev scores: two score piles, miss and false-alarm curves, and where they cross](img/ch02_threshold_sweep.png)
*Source: [notebook 20](../../notebooks/20_story_figures.ipynb), cells 8–9, recomputed for the
story from the committed A2 dev scores
([scores/dev.csv](../../experiments/E-20261001-e1-a2-raw/scores/dev.csv)).*

The scores are **measured**; only the curves are drawn here.

1. **Left panel.** Two piles of scores. Fake clips pile up around −17, real clips around +13
   (medians −16.87 and +13.09, printed in cell 8). They barely overlap.
2. **Middle panel.** Slide the threshold from left to right. The green curve, the miss rate,
   starts at 0 and climbs: the further right the threshold, the more real clips fall below it.
   The orange curve, the false-alarm rate, starts at 100 % and falls. You cannot push both to
   zero; lowering one raises the other.
3. **Right panel.** Zoom in. The curves cross at a threshold of about −2.43. At that point both
   error rates are about 0.39 %.

That crossing point is the **equal error rate (EER)**: the error rate at the threshold where
misses and false alarms are equal. Written out:

$$
\text{EER} = P_\text{miss}(\theta^*) = P_\text{fa}(\theta^*), \quad \text{where } \theta^* \text{ is the threshold at which the two are equal}.
$$

In words: slide the threshold until the share of real clips rejected equals the share of fakes
accepted, and read off that share. Lower is better; 50 % is a coin flip.

For A2 on dev, the EER is **0.39 %** at threshold **−2.43**
([metrics.json](../../experiments/E-20261001-e1-a2-raw/metrics.json), `testsets.dev.eer` and
`eer_threshold`; cell 8 checks that the EER and its threshold match metrics.json).

A detail that matters when comparing with published numbers: with real data the two curves move
in small steps, so they rarely meet exactly. We use the official challenge rule: take the
threshold where the two rates are *closest* and average them, with no interpolation between
steps. Our [`compute_eer`](../../src/audio_df/metrics.py#L217) reproduces the official scoring
code for this.

## One number for every threshold: AUC

EER looks at one threshold. Sometimes we want to know how well the scores **rank** the clips,
whatever threshold is used later.

Pick one real clip and one fake clip at random. What is the chance the real one gets the higher
score? That probability is the **area under the curve (AUC)**. The curve is called the
receiver operating characteristic (ROC) curve; we won't need the curve itself, only this
probability.

- AUC 1.0: every real clip outscores every fake. Perfect ranking.
- AUC 0.5: a coin flip. The scores carry no information.
- Below 0.5: the ranking is backwards. That will matter in chapter 9, for some random seeds.

Like the two error rates, AUC is computed within classes, so it ignores the 1 : 9 ratio. The
"always fake" detector gives every clip the same score and gets AUC 0.5.

On dev, A2's AUC is 0.99995 and A1's is 0.9958
([A2 metrics.json](../../experiments/E-20261001-e1-a2-raw/metrics.json),
[A1 metrics.json](../../experiments/E-20261001-e1-a1-raw/metrics.json), `roc_auc`; the
function is [`roc_auc`](../../src/audio_df/metrics.py#L302)).

## Where the threshold comes from

EER picks its own best threshold on each test set. That is fair for comparing how well two
detectors *separate* real from fake, but a deployed detector cannot do it: it has to commit to
one threshold in advance.

So we also fix one threshold, the EER threshold **on the dev set**, and reuse it unchanged on
eval and on In-the-Wild ([`_dev_threshold`](../../src/audio_df/scoring.py#L139)). For A2 that
is −2.43 (the `threshold_source` field in
[A2 In-the-Wild metrics.json](../../experiments/E-20261001-e3-a2-raw/metrics.json) shows In-the-Wild
reuses it). Two questions, two kinds of number:

- **Can the detector tell real from fake here at all?** EER and AUC answer this.
- **Does the threshold we chose at home still work here?** The error rates at the fixed dev
  threshold answer this.

Chapter 9 shows that these two questions can have very different answers.

One last scoring rule: we also report EER **per attack**, comparing each attack's fakes against
the whole pool of real clips ([`per_attack_eer`](../../src/audio_df/metrics.py#L460)). A single
pooled number can hide one generator that the detector gets badly wrong.

> **Checkpoint**
>
> - Our main data is ASVspoof 2019 LA: real clips are VCTK studio recordings, fakes come from TTS
>   and VC systems, about 9 fakes per real clip in every split
>   ([notebook 20](../../notebooks/20_story_figures.ipynb), cell 6; from paper [14, Fig. 2, p.4]).
> - Eval uses attacks A07–A19, never seen in training; 11 are new generators and 2 (A16, A19)
>   reuse training algorithms (from paper [14, Table 1 caption and §3, p.6]).
> - In-the-Wild adds 31,779 real-world clips, mostly real, with no attack labels
>   ([In-the-Wild metrics.json](../../experiments/E-20261001-e3-a2-raw/metrics.json); from paper [13, §3, p.2]).
> - Accuracy is useless at 1 : 9 ("always fake" scores 89.7 %); misses and false alarms, measured
>   within each class, are not. EER is where they are equal: 0.39 % for A2 on dev
>   ([notebook 20](../../notebooks/20_story_figures.ipynb), cells 8–9;
>   [metrics.json](../../experiments/E-20261001-e1-a2-raw/metrics.json)).
> - AUC measures ranking over all thresholds; a fixed dev threshold tests whether a decision rule
>   transfers ([`_dev_threshold`](../../src/audio_df/scoring.py#L139)).
>
> **The question this leaves us with:** we can now score a detector fairly. But before building
> one, could something stupid, with no idea what a voice is, already score well?

## References used in this chapter

- [13] Müller et al. *Does Audio Deepfake Detection Generalize?* Interspeech 2022.
  [arXiv:2203.16263](https://arxiv.org/abs/2203.16263)
- [14] Wang et al. *ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and
  Replayed Speech.* Computer Speech & Language 64, 2020.
  [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)
