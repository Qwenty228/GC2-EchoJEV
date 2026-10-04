# Chapter 3 · Before building anything: can something stupid already "detect" fakes?

*Notebook cell numbers count the first cell of a notebook as cell 0.*

## One number per clip

Recall the pair of clips from chapter 0: the real one waited about 0.8 s before speaking, the
fake about 0.35 s. One pair proves nothing. So measure it on every clip.

For each clip we compute one number: **the fraction of the clip that is silent before the first
sound**. "Silent" has a precise meaning here. The clip is cut into overlapping frames of 2,048
samples (128 milliseconds at 16,000 samples per second), one every 512 samples. A frame counts
as silent if its energy is more than 30 decibels (dB) below the loudest frame in the same clip
([`silence_ratio`](../../src/audio_df/audio.py#L142), which follows the convention of the
`librosa` audio library). So "silent" means "much quieter than this clip's speech", not "digital
zero". Room hiss counts as silence.

Here is that number for all 121,461 clips of the ASVspoof 2019 LA dataset (the Automatic
Speaker Verification Spoofing and Countermeasures Challenge, logical access part):

![Leading-silence fraction by class (left) and median per attack (right)](img/ch03_leading_silence.png)
*Source: [notebook 20](../../notebooks/20_story_figures.ipynb), cell 11. The values are the
per-clip output of `leakage.extract_nuisance_features`, saved locally as
`data/metadata/nuisance_features.csv` (not in git; no committed script writes it to that path).
Nothing is recomputed from audio. [Notebook 01](../../notebooks/01_dataset_audit.ipynb), cell 18
would draw the same quantity if re-run with the raw audio; its saved copy has no picture.*

Look at the left panel:

- the **fake** clips (orange) form a tall spike at zero: 29.5 % of them have less than 1 % of
  leading silence;
- the **real** clips (green) form a hump around 0.3: only 0.2 % of them start that abruptly;
- the medians are 0.31 for real and 0.04 for fake (printed in cell 11).

You could almost draw a threshold by eye. Notice, though, the low orange hump sitting under the
green one: some fakes start just like real clips. The right panel shows which; we come back to
it below.

## A deliberately stupid detector

Let's make that "by eye" honest. We fit a **probe**: a classifier so small it can only answer
"is there information about real versus fake in this one number?"

- One probe is a **logistic regression** on the single number. In practice it is one
  threshold with a soft edge.
- The other is a **decision tree of depth 2**: at most three yes/no cut-offs.

Whichever of the two ranks the clips better is kept
([`_fit_and_score`](../../src/audio_df/leakage.py#L217),
[`audit_single_feature`](../../src/audio_df/leakage.py#L299)). These probes are not our
detectors and say nothing about how A1 or A2 perform. They only measure how much a shortcut
gives away.

### Scoring the probe honestly: cross-validation

A probe tested on the clips it was fitted on would look better than it is. So we use **5-fold
cross-validation**:

1. split the 121,461 clips into 5 equal parts, each with the same real/fake mix;
2. fit the probe on 4 parts, and score the 5th part, which it has not seen;
3. repeat until every part has been scored once.

Every clip then has an "out-of-fold" score from a probe that never saw it
([`_cross_val_oof_scores`](../../src/audio_df/leakage.py#L234)). We score those with the area
under the curve (AUC) and the equal error rate (EER) from chapter 2. AUC 0.5 means the number
carries no information; 1.0 means it separates the classes perfectly.

### How sure are we? The bootstrap

Whatever AUC the probe gets, is it a stable fact or luck of this particular set of clips? To
check, pretend we had drawn a different dataset of the same size:

1. draw 121,461 clips at random **with replacement** from our out-of-fold scores (some clips
   appear twice, some not at all);
2. recompute the AUC;
3. do this 1,000 times. The middle 95 % of the 1,000 values is the **95 % confidence interval
   (CI)**.

The probes are not refitted inside this loop; only the clips are resampled
([`_bootstrap_ci`](../../src/audio_df/leakage.py#L271)). A narrow interval means the number
would barely move on another sample of the same data.

## The result

Leading silence is one of several **nuisance features**: properties of a clip that have nothing
to do with how the voice was made (length, loudness, silence, who is speaking, file format). The
audit ran every one of them this way. The numbers below come from the audit's output table
`reports/tables/leakage_audit.csv`, written by
[`scripts/audit_leakage.py`](../../scripts/audit_leakage.py) and currently not in git. They match
the printed output of [notebook 01](../../notebooks/01_dataset_audit.ipynb), cell 15 (a local run;
the committed notebook has no saved output) and the
[handoff §3.5](../training-and-data-handoff.md).

| one number per clip | AUC [95 % CI] | EER |
|---|---|---|
| leading silence | **0.837** [0.835, 0.840] | 24.7 % |
| loudness (root-mean-square level) | 0.664 [0.660, 0.669] | 38.3 % |
| sample rate, codec or source corpus | 0.500 | n/a (same value for every clip) |
| all features combined | 0.876 [0.873, 0.878] | 22.9 % |

How to read the first row: pick a random real clip and a random fake clip; about 84 times in
100, the probe ranks the real one as more real, using nothing but how much silence it starts
with. That is a detector with a 24.7 % EER that knows nothing about voices.

The rest of the audit, in one breath: total silence is almost as strong (AUC 0.825), speaker
identity is borderline (0.649), duration (0.624) and trailing silence (0.611) are weaker, and
the file-format details carry nothing because every clip shares them. A feature is **flagged**
when its AUC is above 0.65. That cut-off is a project convention, not a value from the
literature ([`DEFAULT_LEAKAGE_THRESHOLD`](../../src/audio_df/leakage.py#L82)). Flagged:
leading silence, total silence, loudness, and all combined.

Notice the last row. Throwing **every** feature in only lifts AUC from 0.837 to 0.876. Leading
silence carries most of the shortcut on its own.

## Why would fakes start sooner?

Not because synthesis is detectable. Because of how the dataset was assembled.

- The real clips are VCTK (Voice Cloning Toolkit) studio recordings, and keep the quiet lead-in
  every recording has.
- Text-to-speech (TTS) systems "are usually designed such that there is no leading or trailing
  silence in the output", and "silences are usually trimmed from TTS training data" (from paper
  [27, §2, p.2]).

Müller et al. reported this in 2021. A small neural network given **only** the duration of the
leading silence reached a 15.12 % EER on the 2019 evaluation set, against 50 % for random
guessing (from paper [27, §4.2, §5.1 and Table 1, p.3]). Their set-up differs from ours: they
measured silence in seconds with a 40 dB rule and scored the evaluation split, while we
cross-validate over all clips with silence as a fraction and a 30 dB rule. So the two numbers
should not be compared directly. Both say the same thing: silence alone gets a detector a long
way.

## Not every attack is silent: the per-attack view

Now look at the right panel of the figure above: the median leading silence of each attack,
with the real-speech median drawn as a green line.

- Most eval attacks sit near zero: A10, A11, A12 and A13 have a median of exactly 0.
- **A17, A18 and A19 sit level with real speech** (medians 0.30, 0.31 and 0.29, against 0.31).
- The hatched bars, A05, A06, A17, A18 and A19, are exactly the attacks whose input is "Speech
  (human)" in the database paper (from paper [14, Table 1, p.6]). These voice conversion (VC)
  systems take a real person's recording and change the voice, so the recording's lead-in
  survives. All five sit near the real-speech line.

A second view, computed straight from the audio on a 650-clip eval sample, shows the same
split: A17, A18 and A19 near the bona fide line, A10–A13 (and A15) at zero.

![Median leading-silence fraction per eval attack, 650-clip sample](img/ch03_nb09_silence_per_attack.png)
*Source: [notebook 09](../../notebooks/09_a1_playground.ipynb), cell 16, rendered locally (the
committed notebook has no saved output; it needs the raw audio to run). This older notebook uses
default colours; the dashed line is the bona fide median.*

One bar does not fit the story: **A02**, a text-to-speech attack, also has a long lead-in
(median 0.285). Its row in the database paper gives no hint why. We don't know.

This per-attack pattern is the reason chapter 2 insisted on per-attack EER. A detector that
leaned on silence would find A10–A13 easy and A17–A19 hard, and a single pooled number would hide
that.

## What we did about it

The audit's verdict ([notebook 01](../../notebooks/01_dataset_audit.ipynb), cell 21) is that
the data "should NOT be modelled on as-is until the flagged feature(s) are investigated" (a
local run; the committed notebook has no saved output). We did not rebuild the dataset.
Instead, every result in this project is reported twice:

- on **raw** audio, the main condition;
- on **trimmed** audio, with leading and trailing silence removed
  ([`trim_silence`](../../src/audio_df/audio.py#L228), using the same 30 dB rule; pauses inside
  the clip are kept).

When a detector needs fixed-length input (A2, chapter 7), short clips are padded by
**repeating** the clip, never by adding zeros, because zero padding
would add back the very silence we are trying to control
([`preprocess`](../../src/audio_df/audio.py#L289)).

How much silence costs a real detector is a known, large effect in the literature: RawNet2, a
standard neural detector, went from 3.61 % to 15.50 % EER on the 2019 evaluation set when it was
trained on trimmed audio (from paper [27, Table 4, p.4]). Chapter 8 measures what trimming does
to our two detectors.

What we do **not** know yet: our detectors were trained on raw audio only. When we trim at test
time, a drop in performance could mean "the detector relied on silence", or simply "the test
audio no longer looks like the training audio". Training one detector on trimmed audio would
tell the two apart; that run has not been done.

> **Checkpoint**
>
> - Fake clips start much sooner than real ones: median leading-silence fraction 0.04 against
>   0.31 over all 121,461 clips ([notebook 20](../../notebooks/20_story_figures.ipynb), cell 11,
>   from the local per-clip feature file).
> - A one-number probe on leading silence ranks real above fake with AUC 0.837 [0.835, 0.840]
>   and 24.7 % EER, under 5-fold cross-validation and a 1,000-resample bootstrap
>   (`reports/tables/leakage_audit.csv`, local, not in git; matches
>   [notebook 01](../../notebooks/01_dataset_audit.ipynb), cell 15 and handoff §3.5;
>   [`leakage.py`](../../src/audio_df/leakage.py#L299)).
> - All nuisance features together reach only 0.876, so silence carries most of the shortcut
>   (same table).
> - The cause is how the dataset was built: TTS output has no lead-in (from paper [27, §2, p.2]),
>   while the five attacks that convert human recordings keep it
>   ([notebook 20](../../notebooks/20_story_figures.ipynb), cell 11; from paper [14, Table 1, p.6]).
> - We report every result on raw and trimmed audio; separating "uses silence" from "train/test
>   mismatch" needs a trimmed-training run we have not done.
>
> **The question this leaves us with:** silence is not the voice. If we look at the sound itself,
> how different are real and fake, and how much of that difference is silence in disguise?

## References used in this chapter

- [14] Wang et al. *ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and
  Replayed Speech.* Computer Speech & Language 64, 2020.
  [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)
- [27] Müller, Dieckmann, Czempin, Canals, Böttinger and Williams. *Speech is Silver, Silence is
  Golden: What do ASVspoof-trained Models Really Learn?* ASVspoof 2021 Workshop.
  [arXiv:2106.12914](https://arxiv.org/abs/2106.12914)
