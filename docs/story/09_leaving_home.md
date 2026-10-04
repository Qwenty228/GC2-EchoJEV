# Chapter 9 · Leaving home: what happens on real-world audio?

*Notebook cell numbers count the first cell of a notebook as cell 0.*

## A new world, all at once

Every number so far comes from one dataset, ASVspoof 2019 LA (the Automatic Speaker Verification
Spoofing and Countermeasures Challenge, logical access part). Its real clips all come from one
studio recording chain, the Voice Cloning Toolkit (VCTK) corpus, and its fakes were built from the
same speakers (chapter 2). Chapter 8 removed one difference between that dataset and the world:
silence. Now we remove all of them at once.

**In-the-Wild** (ITW) is 31,779 clips of celebrities and politicians: 19,963 real, 11,816 fake
(chapter 2). The real audio comes from podcasts and speeches; the fakes are cut from online videos
that advertise themselves as deepfakes. The original recordings varied, and everything was
downsampled to 16 kHz (kilohertz), "the highest common frequency in the original recordings"
(from paper [13, §3, p.2]). New speakers, new microphones, new rooms, new compression, new
generators. We never train on it.

We ask chapter 2's two questions: **ranking** (can the detector still tell real from fake? the
equal error rate, EER, and the area under the curve, AUC) and **calibration** (does the threshold
chosen on ASVspoof dev still work?). What is new is that, away from home, the two can have very
different answers.

## The picture

![In-the-Wild score histograms for A1 and A2, with each system's dev threshold and its VCTK eval real scores for reference](img/ch09_itw_scores_threshold.png)
*Source: [notebook 20](../../notebooks/20_story_figures.ipynb), cell 39, recomputed for the story
from the committed In-the-Wild scores (`E-20261001-e3-a1-raw`, `E-20261001-e3-a2-raw`), each
system's dev threshold, and its eval scores for the grey reference outline.*

## A2: the ranking survives, the threshold does not

Bottom panel. The two In-the-Wild piles are still well apart:

- **EER 14.70 %, AUC 0.934** ([E3 A2 metrics.json](../../experiments/E-20261001-e3-a2-raw/metrics.json)).
  For a model that has never heard any of these speakers, microphones or generators, that is a
  strong ranking. This is the original run of A2 (its E1 head, scored in E3). The same design
  retrained three times on an ordinary processor for the layer analysis X6 gets 14.0 ± 0.5 %:
  different runs of the same recipe, in close agreement.

But look where the green pile sits. VCTK real clips (grey outline) scored around +14.90; In-the-Wild
real clips score around **−2.19** (median). The threshold chosen on dev, −2.43, now cuts through the
middle of the real pile:

- **48.9 % of real In-the-Wild clips are rejected** as fake;
- only 1.6 % of fakes are accepted.

(Recomputed in cell 39 from the committed scores.) A deployed A2 with its home threshold would
wrongly flag about half of real podcast and speech audio. The scores still put real above fake;
the *number* that separates them has moved. That is a **calibration** failure, not a **ranking**
failure, and it would need re-calibrating on real audio from the new setting.

Why did the real pile move? One reading fits chapter 8 (**our guess**, untested): VCTK real clips
carry a studio silence lead-in that pushes A2's score up, and In-the-Wild real clips, cut from
longer recordings, do not. Trimming VCTK eval moved its real median to −4.18 (chapter 8); In-the-Wild
real sits at −2.19. It also fits chapter 8's curious exception: trimming made A2 slightly *better*
on In-the-Wild (14.70 → 13.89 %, [E3 A2 trimmed metrics.json](../../experiments/E-20261001-e3-a2-trimmed/metrics.json)).
The numbers are measured; the link between them is not. Predicting silence
from each layer (analysis X8) would test it.

How surprising is 14.7 %? For orientation only (other models, other training data): the best
ASVspoof-trained model in [13] reaches 33.94 % on In-the-Wild (from paper [13, Table 1, p.3]); frozen
wav2vec 2.0 systems reach 12.48 % with a large fusion head and 23.22 % with a simple one (from paper
[22, Table II, p.3]). The sharpest comparison is with Pascu et al., who froze the very same
WavLM Base+, took only its top layer, fitted a logistic regression (a single weighted threshold)
and report 38.6 % EER on In-the-Wild (from paper [8, Table 3, p.4]; different training data and
protocol). Chapter 7 promised the numbers for why A2 does so much better; the X6 layer analysis
supplies them ([summary.md](../../experiments/E-20261004-e2-a2-raw-layers/summary.md)): a logistic
regression on our cached **top layer** gets 24.75 %, our small network on that layer alone 16.5 %,
and the full A2 head 14.0 ± 0.5 % (the three X6 retrains). The same layer with a slightly richer
head already does much better, and even the plain logistic regression on our cache is far below
38.6 %. So the choice of layer does not explain the gap; the head matters, and some remaining
difference between their set-up and ours, which we have not identified, accounts for the rest. Calibration is a known problem there too: even [8]'s best
model has an expected calibration error (how far its probabilities are from observed frequencies)
of 16.1 % on In-the-Wild (from paper [8, Table 3, p.4]). That is a related measure, not the same
as our rejection rate at a fixed threshold.

## A1: at chance

Top panel. The In-the-Wild real and fake piles lie on top of each other, with the fakes slightly to
the right, the "more real" side:

- **EER 58.04 %, AUC 0.398** ([E3 A1 metrics.json](../../experiments/E-20261001-e3-a1-raw/metrics.json)):
  no better than a coin flip;
- medians: real +0.44, fake +0.69;
- at the home threshold, 71.8 % of real clips are rejected and 50.0 % of fakes accepted.

Is A1's ranking actually *backwards* here (AUC below 0.5)? That depends on the random seed. The
three linear-frequency cepstral coefficient (LFCC) models of chapter 6 (X4) get AUC 0.40, 0.56 and 0.38 (EER 58.0 %, 46.0 % and 58.6 %;
the committed metrics.json files, printed in cell 41). On eval those three models agree within
a point (8.7–9.6 %); on In-the-Wild they spread over 13 points. So the robust statement is: **A1 is at chance
on In-the-Wild**, and a single-seed A1 number here should not be quoted alone.

This matches the central finding of [13], qualitatively: models trained on ASVspoof fall apart on
In-the-Wild. Their spectrogram and raw-waveform models average 60.10 % EER over 28 full-length
configurations and 67.25 % over 28 four-second ones (from paper [13, Table 2, p.4]); by our count of
their Table 1, 43 of their 56 configurations reach 50 % or worse. [13] tests no LFCC or Gaussian mixture model (GMM) system,
so this is the same kind of collapse, not a replication of a number. Even retraining with every
ASVspoof split added left their RawNet2 at 33.1 %, and they point out that all ASVspoof splits are
"fundamentally based on the same dataset, VCTK" (from paper [13, §5.3, p.4]).

Why would A1 collapse? Our reading, all of it **our guess**: chapter 5 showed that level, gain and silence
all live in c0, and A1 keeps c0 with no per-recording normalisation. So A1's two GMMs describe
the **recording** as much as the voice: "real" has come to mean "sounds like the VCTK studio".
Real podcast audio is further from that studio than many web deepfakes, which are often clean,
studio-like synthetic speech.

## The flip: LFCC against MFCC, away from home

Chapter 6 found that LFCC beat mel-frequency cepstral coefficients (MFCC) on the benchmark. Here is the
same comparison, the same six models, scored on In-the-Wild:

![Eval EER and In-the-Wild EER for each of the six X4 models, LFCC (blue) and MFCC (magenta)](img/ch09_x4_home_vs_away.png)
*Source: [notebook 20](../../notebooks/20_story_figures.ipynb), cell 41, from the six committed
X4 metrics.json files (the seed-1337 LFCC model's In-the-Wild score is in `E-20261001-e3-a1-raw`).*

| mean ± sd over 3 seeds | LFCC (linear) | MFCC (mel) |
|---|---|---|
| eval EER (home) | **9.12 ± 0.44 %** | 10.61 ± 0.30 % |
| In-the-Wild EER (away) | 54.2 ± 7.1 % | **32.5 ± 1.6 %** |
| In-the-Wild AUC | 0.45 ± 0.10 | **0.73 ± 0.03** |

(Printed in cell 41.) The ranking **flips**. At home LFCC wins by 1.5 points; away, MFCC wins by
more than 20, and its three seeds agree (30.7–33.7 %) while LFCC's scatter around chance. This is a
**measured** result of this project, and it was not what we expected: before running X4 we
expected both arms to sit near chance on In-the-Wild (handoff §5.1.4).

Why? Chapter 5 gives a candidate. LFCC spends 10 of its 20 filters above 4 kHz; mel spends 5. That
top band is where recording chains differ most: VCTK is studio audio downsampled from 96 kHz,
In-the-Wild is podcasts and web video, often compressed. So the extra high-band detail that helps
LFCC spot synthesis on ASVspoof may also make it model "what the VCTK microphone sounds like up
top", which turns into noise on real-world audio. The literature has a related observation for a
different task: in speaker recognition, linear filterbanks were less robust to white noise, which
the authors attribute to the extra high-band filters (from paper [23, §III.B, p.5]; the authors'
explanation, not a separate test).

This is **our guess**. The test is cheap and not yet run (analysis X7): low-pass both arms' input
at 4 kHz, or normalise each recording's average coefficients at scoring time, and see whether the
gap closes.

So the high-frequency resolution that literature and our own benchmark favoured cuts both ways: it
sees synthesis detail and recording-chain detail alike. "LFCC is better than MFCC" is true for
ASVspoof and false on our real-world test.

> **Checkpoint**
>
> - A2 keeps a strong ranking on In-the-Wild (EER 14.70 %, AUC 0.934 for the original run;
>   14.0 ± 0.5 % over the 3 X6 retrains), but its home threshold rejects 48.9 % of real clips:
>   ranking survives, calibration does not. Layer choice does not explain why it beats another
>   group's frozen WavLM Base+ (their 38.6 %; our top-layer logistic regression 24.75 %)
>   ([E3 A2 metrics.json](../../experiments/E-20261001-e3-a2-raw/metrics.json); X6 summary;
>   [notebook 20](../../notebooks/20_story_figures.ipynb), cell 39; from paper [8, Table 3]).
> - A1 is at chance on In-the-Wild (EER 58.04 %, AUC 0.398 for seed 1337; AUC 0.56 and 0.38 for the
>   other two seeds), the same kind of collapse [13] reports for its models (from paper [13,
>   Tables 1–2]).
> - With everything else equal, MFCC beats LFCC on In-the-Wild (32.5 ± 1.6 % against 54.2 ± 7.1 %),
>   the reverse of the benchmark result (X4; notebook 20, cell 41).
> - Why A2's scores shift and why MFCC travels better are hypotheses (silence lead-in; high-band
>   recording-chain detail); the tests (X8, X7) have not been run.
>
> **The question this leaves us with:** A2 ranks well but cannot tell where to draw the line in a
> new setting. Our third system adds a second opinion on uncertain clips; is it built to fix that,
> and what would it need?

## References used in this chapter

- [8] Pascu et al. *Towards Generalisable and Calibrated Audio Deepfake Detection with
  Self-Supervised Representations.* Interspeech 2024. [arXiv:2309.05384](https://arxiv.org/abs/2309.05384)
- [13] Müller et al. *Does Audio Deepfake Detection Generalize?* Interspeech 2022.
  [arXiv:2203.16263](https://arxiv.org/abs/2203.16263)
- [22] Wang Z. et al. *Mixture of Experts Fusion for Fake Audio Detection Using Frozen wav2vec
  2.0.* arXiv preprint, 2024. [arXiv:2409.11909](https://arxiv.org/abs/2409.11909)
- [23] Zhou et al. *Linear versus Mel Frequency Cepstral Coefficients for Speaker Recognition.*
  IEEE ASRU 2011. [doi:10.1109/ASRU.2011.6163888](https://doi.org/10.1109/ASRU.2011.6163888)
