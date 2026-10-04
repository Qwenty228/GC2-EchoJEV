# Chapter 11 · What do we know, what don't we, and what would tell us?

## Back to the question

Chapter 0 asked: **is there something measurable that differs between real and fake speech
because of how the fake was made, not an accident of how the dataset was collected?** And does it
survive new generators and recording conditions? The honest answer: yes, partly, and not yet.

## What we know (measured)

1. **The benchmark has a large shortcut, and it is silence.** Leading silence alone ranks real
   above fake with an area under the curve (AUC) of 0.837; every nuisance feature together reaches
   only 0.876. Text-to-speech output starts immediately; the five attacks that convert real
   recordings keep a real-looking lead-in ([chapter 3](03_the_silence_shortcut.md)).

2. **Our detectors do detect synthesis, not only silence.** A10 and A11 differ only in the
   vocoder and both lack silence, yet A1's equal error rate (EER) is 22.4 % against 0.1 %
   ([chapter 6](06_a1_two_clouds.md)); trimming moves their A1 median scores by only −0.01 (A10)
   and +0.25 (A11) ([chapter 8](08_trim_the_silence.md) figure).

3. **They also lean on silence.** Trimming it at test time doubles A1's eval EER (9.56 → 19.84 %)
   and multiplies A2's by about 14 (0.96 → 13.27 %); A2's real clips lose about 19 points of score
   while its fakes barely move ([chapter 8](08_trim_the_silence.md)).

4. **A frozen, pre-trained self-supervised (SSL) model generalises far better than the classic
   baseline.** A2 makes about ten times fewer benchmark errors than A1 (0.96 % against 9.56 %); on
   In-the-Wild it still ranks well (EER 14.70 %, AUC 0.934) while A1 is at chance (EER 46–59 %,
   AUC 0.38–0.56 over 3 seeds) ([chapters 7](07_a2_borrowed_ears.md) and
   [9](09_leaving_home.md)).

5. **Ranking travels; calibration does not.** A2's threshold from ASVspoof dev (the Automatic
   Speaker Verification Spoofing and Countermeasures Challenge 2019 development split) rejects
   48.9 % of real In-the-Wild clips ([chapter 9](09_leaving_home.md)).

6. **Linear beats mel at home, mel beats linear away.** With 3 seeds per arm, linear-frequency cepstral coefficients (LFCC) win on ASVspoof eval (9.12 against 10.61 %) and mel-frequency cepstral coefficients (MFCC) win on In-the-Wild (32.5 against
   54.2 %) ([chapters 6](06_a1_two_clouds.md) and [9](09_leaving_home.md)).

7. **A2's evidence is spread across its layers.** Dropping any one barely matters; lower layers
   (1–5) catch A19 and A18 best (A19: 2.5 % EER at layer 1, 24.3 % at layer 12), middle layers A17;
   middle layers lean least on silence ([chapter 7](07_a2_borrowed_ears.md)). Layer choice does
   not explain why A2 beats another group's use of the same frozen model
   ([chapter 9](09_leaving_home.md)).

## What we don't know yet

Each is a reading that fits our numbers, with the test that would settle it. None has been run.

- **How much of the trimming loss is the shortcut, and how much is train/test mismatch?** Both
  detectors were trained on raw audio. *Test:* train one A2 head on trimmed audio and test on
  trimmed eval (about 45 minutes of extraction, seconds of training; needs a team decision)
  ([chapter 8](08_trim_the_silence.md)).
- **Why does MFCC travel better than LFCC?** Our guess: LFCC's extra high-frequency filters model the
  recording chain of VCTK (the Voice Cloning Toolkit corpus behind every real benchmark clip). *Test (X7):* low-pass both arms at 4 kilohertz (kHz), or normalise each recording's
  average coefficients at scoring time, and see whether the gap closes ([chapter 9](09_leaving_home.md)).
- **Why did A2's real In-the-Wild scores shift?** Our guess: VCTK real clips carry a silence lead-in
  that pushes A2's score up. *Test (X8):* predict each clip's silence (and duration, high-band share)
  from each layer's averaged output and from A2's score ([chapters 7](07_a2_borrowed_ears.md),
  [9](09_leaving_home.md)).
- **What drives A1's per-attack trimming shifts, and why is A18 the exception?** *Tests (X2,
  X3):* split A1's score into per-frame and per-coefficient shares and see which frames (edges,
  exact zeros) and coefficients (c0 or the rest) carry the shift ([chapter 8](08_trim_the_silence.md)).
- **Which frequency bands does A1 use, attack by attack, and do its dynamics or pre-emphasis matter?**
  *Tests (X1, X5):* replace groups of filter channels with training averages and rescore; drop the
  Δ/ΔΔ features or the pre-emphasis ([chapters 5](05_sound_into_numbers.md), [6](06_a1_two_clouds.md)).
- **Is spectral bandwidth a real cue or silence in disguise?** *Test:* recompute it on low-silence
  clips ([chapter 4](04_looking_at_the_sound.md)).
- **Is A1's gap to A2 mostly its Gaussian mixture model (GMM) back end?** Literature on other set-ups suggests so (from paper
  [17]). *Test:* a discriminative model on our own LFCC frames ([chapter 6](06_a1_two_clouds.md)).
- **What remaining difference makes A2 beat [8]'s result with the same frozen model?** Not the
  layer; we have not identified it ([chapter 9](09_leaving_home.md)).
- **Does a second opinion help on uncertain clips?** A3 is a scaffold; it needs the JEV
  documentation, then a small cost and latency check, then experiment E5
  ([chapter 10](10_where_jev_fits.md)).

## What else is planned (not done)

- **A second backbone, XLS-R 300M** (a multilingual self-supervised speech model): configured, not extracted or trained.
- **Robustness to everyday degradations (experiment E4)**: MP3 and Opus compression, added noise
  and narrowband resampling on a fixed 10,000-clip eval subset; generator ready, nothing scored.
- **Efficiency (experiment E6)**: measuring parameters and milliseconds per clip; not built.
- **Fine-tuning the backbone (A2-alt)**: parked; it needs a graphics card we do not have.
- **The ablation experiment E2** as a formal experiment: needs a team decision; the X4 and X6 analyses
  could be reported under it.
- **Single seeds remain** for the single-layer heads, A2's original In-the-Wild run and the
  trimmed runs; small differences there are not effects.

(Status from handoff §5.3 and §13.)

## The short version

Something measurable does differ, partly because of how fakes are made: the vocoder alone moves
A1's error by 22 points, and a frozen speech model ranks well even on real-world audio. But much of what both detectors use on the benchmark is an accident of its
construction (silence and the studio recording chain), and detail that helps at home can hurt
away. The cheapest next step is not a bigger model but the tests above, most of which take minutes
on the models and caches we already have.

> **Checkpoint**
>
> - Measured: a silence shortcut (AUC 0.837 from leading silence alone), real synthesis detection
>   (A10 vs A11: 22.4 % against 0.1 %), heavy reliance on silence when it is trimmed, a frozen SSL
>   model that ranks well on real-world audio but needs re-calibration, and an LFCC/MFCC ranking that
>   flips between home and away ([chapter 3](03_the_silence_shortcut.md),
>   [6](06_a1_two_clouds.md), [8](08_trim_the_silence.md), [9](09_leaving_home.md)).
> - Not yet known: shortcut versus mismatch, why MFCC travels better, why A2's real scores shift,
>   what drives A1's per-attack shifts (A18), whether A3 helps; each has a named test that has not
>   been run (this chapter).
> - Planned and not done: X1–X3, X5, X7, X8, a trimmed-training A2 head, XLS-R, A2-alt, E2, E4, E5,
>   E6.
>
> **The question this leaves us with:** chapter 0's answer is "yes, partly", and we know which
> accidents of the dataset get in the way. Which test do we run first, so the next version of this
> story can say how much of each detector's skill is synthesis and how much is the recording?

## References used in this chapter

- [8] Pascu et al. *Towards Generalisable and Calibrated Audio Deepfake Detection with
  Self-Supervised Representations.* Interspeech 2024. [arXiv:2309.05384](https://arxiv.org/abs/2309.05384)
- [17] Mo and Wang. *Multi-task Learning Improves Synthetic Speech Detection.* ICASSP 2022.
  [doi:10.1109/ICASSP43922.2022.9746059](https://doi.org/10.1109/ICASSP43922.2022.9746059)
