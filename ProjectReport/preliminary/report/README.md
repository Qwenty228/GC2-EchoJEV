# Detecting spoofed speech: what three detectors learn on ASVspoof 2019, and what survives outside it

This is the main document of the project. It sets out the question, the data, the three
detectors and every result, as one argument in twelve short chapters. It is written for readers
who know basic Python and statistics but not signal processing. After reading it, you should be
able to explain why the pre-trained model (A2) beats the classic baseline (A1), why silence
matters so much on this benchmark, and what is still unknown.

The [technical reference](../technical-reference.md) covers the same work at engineering depth:
full maths, implementation pointers and every result table.

## Summary

We built and measured two detectors of spoofed (fake) speech on ASVspoof 2019 logical access
(LA), the Automatic Speaker Verification Spoofing and Countermeasures Challenge dataset, and
scaffolded a third:

- **A1**: linear-frequency cepstral coefficients (LFCC) with two Gaussian mixture models (GMM),
  rebuilt from the official 2019 baseline (B02).
- **A2**: a frozen WavLM Base+ self-supervised speech model with a small trained head.
- **A3**: A2 plus JEV, a typed classifier service, asked for a second opinion. This is a scaffold
  only and has no results yet.

| equal error rate (EER), lower is better | 2019 LA eval, raw | 2019 LA eval, silence trimmed | In-the-Wild, raw |
|---|---|---|---|
| A1, LFCC + GMM | 9.56 % | 19.84 % | 58.04 % (chance) |
| A2, frozen WavLM Base+ | **0.96 %** | 13.27 % | **14.70 %** |

([reports/tables/E1.md](../../SystemCode/reports/tables/E1.md), [E3_itw.md](../../SystemCode/reports/tables/E3_itw.md))

The main findings:

1. **The benchmark has a large silence shortcut.** Leading silence alone separates real from fake
   with an area under the curve (AUC) of 0.837 ([chapter 3](03_the_silence_shortcut.md)).
2. **Both detectors also detect real synthesis artefacts.** Attacks A10 and A11 differ only in their
   vocoder, and A1's error on them is 22.4 % against 0.1 % ([chapter 6](06_a1_two_clouds.md)).
3. **Both detectors also lean on silence.** Trimming it at test time doubles A1's error and
   multiplies A2's by about 14 ([chapter 8](08_trim_the_silence.md)).
4. **On real-world audio, A2's ranking holds but its decision threshold does not.** A1 falls
   to chance ([chapter 9](09_leaving_home.md)).
5. **Linear filters beat mel filters on the benchmark, and mel beats linear on real-world audio**
   ([chapters 6](06_a1_two_clouds.md) and [9](09_leaving_home.md)).

## Chapters

**[0 · The problem: what would a detector even look for?](00_the_problem.md)**
A real and a fake spectrogram look alike, but the real one has a longer silent lead-in. The
chapter names the two traps, shortcuts and new conditions, and introduces the three systems,
A1, A2 and A3.

**[1 · What is a voice made of, and what must a fake imitate?](01_what_is_a_voice.md)**
A voice is a buzz (the source) shaped by the throat and mouth (the filter). Generators rebuild both
with an acoustic model and a vocoder. Two groups of attacks matter later: A10/A11, which differ
only in the vocoder, and A17–A19, which convert real recordings.

**[2 · Where does the data come from, and how do you score fairly when 9 of 10 clips are fake?](02_data_and_fair_scoring.md)**
ASVspoof 2019 LA is studio speech with 19 attacks, and In-the-Wild is real-world audio that is
never used for training. Accuracy is useless at 1:9, so results use the equal error rate (EER),
the area under the curve (AUC) and a threshold fixed on the development set.

**[3 · Before building anything: can something trivial already "detect" fakes?](03_the_silence_shortcut.md)**
A one-number probe on leading silence ranks real above fake with AUC 0.837, because of how the
dataset was assembled. Every later result is also reported with the silence trimmed.

**[4 · Looking at the sound: how different are real and fake?](04_looking_at_the_sound.md)**
Cohen's d and Cliff's δ measure how far apart two groups are. Spectral bandwidth separates real from
fake strongly; whether it is silence in disguise is untested.

**[5 · How do you turn a sound into a few numbers?](05_sound_into_numbers.md)**
The recipe for LFCCs, step by step: c0 carries level, c1 tilt and the higher coefficients
ripple, and a gain change moves only c0. Linear and mel filters differ only in where they sit,
which sets up the comparison of LFCC with mel-frequency cepstral coefficients (MFCC).

**[6 · A1: can two clouds of frames tell real from fake?](06_a1_two_clouds.md)**
A1 scores a clip with two Gaussian mixture models and reproduces the official baseline (9.56 %
eval EER against the published 8.09 %). A10 against A11 (22.4 % against 0.1 %) shows that A1
detects the vocoder. LFCC beats MFCC on the benchmark (analysis X4, 3 seeds).

**[7 · A2: borrow a model that has already heard 94,000 hours of speech?](07_a2_borrowed_ears.md)**
A2 trains a small head on a frozen WavLM Base+ and reaches 0.96 % eval EER. The layer analysis
(X6) shows that no single layer is essential, and that the middle layers lean least on silence.

**[8 · Trim the silence: what is left of each detector?](08_trim_the_silence.md)**
Trimming at test time doubles A1's EER and multiplies A2's by about 14. A2's real clips lose
about 19 points of score while its fakes barely move. Both detectors were trained on raw audio,
so a learned shortcut and a train/test mismatch cannot yet be told apart.

**[9 · Leaving home: what happens on real-world audio?](09_leaving_home.md)**
On In-the-Wild, A2 still ranks well (EER 14.7 %, AUC 0.93), but its benchmark threshold rejects
half the real clips: the ranking carries over, the calibration does not. A1 is at chance (AUC
0.38–0.56 over 3 seeds), and away from the benchmark MFCC beats LFCC (32.5 % against 54.2 %).

**[10 · Where does JEV fit?](10_where_jev_fits.md)**
A3 asks JEV, a typed classifier service (not a large language model), for a second opinion on
uncertain clips. It is a scaffold with a dry-run client and no results. As configured, it cannot
yet fix chapter 9's calibration problem.

**[11 · What do we know, what don't we, and what would tell us?](11_what_we_know.md)**
Back to chapter 0's question: what is measured, what is still a hypothesis, and the test that
would settle each hypothesis.

Also: [references](references.md) · [glossary](glossary.md)

## Conventions

- Green always means real (bona fide) and orange always means fake (spoof). Where linear and mel
  filters are compared, blue is linear and magenta is mel. Figures taken from notebooks 09 and 10
  keep those notebooks' default colours, as their captions note.
- Every number is labelled by its source: **measured** (an output of this project's runs or
  notebooks), **from paper [n]** (another paper's number under another protocol, context only),
  or **our guess** (an interpretation not yet tested).
- Figures are in [img/](img). Each caption names the notebook and cell that draws it, usually
  [notebook 20](../../SystemCode/notebooks/20_report_figures.ipynb). Notebook cells count from 0.
- Reference numbers [n] point to the [reference list](references.md).

## Where the numbers come from

Each run writes its outputs to `runs/<run id>/`: the configuration, the per-clip score
files, `metrics.json`, the training curve and notes. These folders stay local and are not
tracked in git. The report names runs by ID, for example `E-20261001-e1-a2-raw`. What is
tracked:

- [`runs/REGISTRY.md`](../../SystemCode/runs/REGISTRY.md): one row per run, with the
  configuration, machine, timing and peak memory;
- [`reports/tables/`](../../SystemCode/reports/tables): the experiment tables (E1–E6) and the leakage audit,
  regenerated from the run outputs by `scripts/make_tables.py`;
- the notebooks, with their outputs;
- the per-clip nuisance features behind the leakage audit,
  [`data/metadata/nuisance_features.csv`](../../SystemCode/data/metadata/nuisance_features.csv).

Main runs:

| run ID | what it is |
|---|---|
| `E-20261001-e1-a1-raw`, `E-20261001-e1-a1-trimmed` | A1 on 2019 LA dev and eval, raw and silence trimmed |
| `E-20261001-e1-a2-raw`, `E-20261001-e1-a2-trimmed` | A2 on 2019 LA dev and eval, raw and silence trimmed |
| `E-20261001-e3-a1-raw`, `-trimmed`; `E-20261001-e3-a2-raw`, `-trimmed` | A1 and A2 on In-the-Wild |
| `E-20261004-e2-a1-raw-lfcc-s{2337,3337}`, `E-20261004-e2-a1-raw-mfcc-s{1337,2337,3337}` | analysis X4: A1 with linear against mel filters, 3 seeds each (the third LFCC seed, 1337, is `E-20261001-e1-a1-raw`) |
| `E-20261004-e2-a2-raw-layers` | analysis X6: A2 layer by layer, 31 variants |
