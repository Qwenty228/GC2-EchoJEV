# Spotting fake voices: the story of our detectors

An explainer for teammates who know basic Python and statistics but not signal processing. It
retells the results handoff ([training-and-data-handoff.md](../training-and-data-handoff.md)) as a
story: it starts from a question, builds intuition with pictures, introduces each technique only
when the story needs it, and ends by tying it all together. After reading, you should be able to
explain why A2 beats A1, why silence matters, and what we still don't know.

**How to read it.** In order, from chapter 0. Each chapter opens with a question and ends with a
**Checkpoint** box: what we now know, with the evidence for each point, and the question that
opens the next chapter.

**Conventions.**

- Green always means real (bona fide), orange always means fake (spoof). Where linear and mel
  filters are compared, blue is linear and magenta is mel.
- Every number is labelled by where it comes from: **measured** (a committed file in this repo, or a local output named as such in the
  figure caption),
  **from paper [n]** (another paper's number, under another protocol, context only), or
  **our guess** (a reading not yet tested).
- Figures are embedded from [img/](img/). Each caption names its source: usually
  [notebook 20](../../notebooks/20_story_figures.ipynb), which draws every story figure from committed
  files, local caches or toy data, and otherwise a cell of an earlier notebook rendered locally.
  Notebook cell numbers count the first cell as 0.
- Reference numbers [n] follow the handoff's §15, which links every paper.

## Chapters

**[0 · The problem: what would a detector even look for?](00_the_problem.md)**
Voice clones are realistic enough to be a social risk. One real and one fake spectrogram look
alike at a glance, yet the real one starts with a longer silent lead-in, a difference that has
nothing to do with synthesis. The chapter states the project's question, names the two traps
(shortcuts and new conditions) and introduces the three systems: A1, A2 and A3.

**[1 · What is a voice made of, and what must a fake imitate?](01_what_is_a_voice.md)**
A voice is a buzz (the source: pitch and its harmonics) shaped by the throat and mouth (the
filter: formants), shown with a toy figure. Speech generators rebuild both in two stages, an
acoustic model and a vocoder. Two pairs of attacks are introduced for later: A10 and A11 differ
only in their vocoder, and A17–A19 convert recordings of real people.

**[2 · Where does our data come from, and how do you score fairly when 9 of 10 clips are fake?](02_data_and_fair_scoring.md)**
ASVspoof 2019 LA (the Automatic Speaker Verification Spoofing and Countermeasures Challenge,
logical access part): studio-recorded real speech and 19 attacks, with the eval attacks unseen in
training; plus In-the-Wild, real-world audio we never train on. Accuracy is useless at 1:9, so
we measure misses and false alarms within each class. The equal error rate (EER) is where they
meet; the area under the curve (AUC) measures ranking; a threshold fixed on dev tests whether a
decision rule travels.

**[3 · Before building anything: can something stupid already "detect" fakes?](03_the_silence_shortcut.md)**
A one-number probe on leading silence ranks real above fake with AUC 0.837, scored honestly with
cross-validation and a bootstrap. The cause is how the dataset was assembled. Attacks that convert
real recordings keep a real-looking lead-in. From here on, every result is also reported with the
silence trimmed.

**[4 · Looking at the sound: how different are real and fake?](04_looking_at_the_sound.md)**
Cohen's d and Cliff's δ measure how far apart two groups are, one by averages and one by pairwise
wins. Spectral bandwidth separates real from fake strongly; high-band energy share does not. Whether
bandwidth is silence in disguise is untested. A small sample hints that different attacks can push
the same summary in opposite directions (only a few clips per attack).

**[5 · How do you turn a sound into a few numbers?](05_sound_into_numbers.md)**
The recipe for linear-frequency cepstral coefficients (LFCC) step by step: frames, spectrum, 20 filters, logarithm, and a cosine transform that
turns 20 correlated band energies into coefficients with meanings: c0 is level, c1 tilt, higher
ones ripple. A gain change moves only c0, so A1 sees loudness and silence. Linear and mel filters
differ only in placement, which sets up the question of LFCC against mel-frequency cepstral coefficients
(MFCC).

**[6 · A1: can two clouds of frames tell real from fake?](06_a1_two_clouds.md)**
A1 models real and fake frames with two Gaussian mixture models and scores a clip by its average
log-likelihood ratio. It reproduces the official baseline (9.56 % eval EER against 8.09 %). The
A10/A11 natural experiment (22.4 % against 0.1 %) shows it detects the vocoder, not just silence.
With everything else fixed, LFCC beats MFCC on the benchmark (X4, 3 seeds).

**[7 · A2: borrow a model that has already heard 94,000 hours of speech?](07_a2_borrowed_ears.md)**
A2 freezes WavLM Base+ and trains a small head on time-averaged layer outputs. Averaging before
mixing layers is exact. A2 reaches 0.96 % eval EER. The layer analysis (X6) shows no single layer
is essential; the lower transformer layers catch two of the three conversions of real recordings
(A18, A19), while A17 is caught best by the middle layers; and middle layers lean least on
silence.

**[8 · Trim the silence: what is left of each detector?](08_trim_the_silence.md)**
Trimming silence at test time doubles A1's EER and multiplies A2's by about 14. A2's real clips
lose about 19 points of score while fakes barely move. A1 shows three per-attack patterns, plus an
unexplained exception (A18). Because both were trained on raw audio, shortcut and train/test
mismatch are not yet separated.

**[9 · Leaving home: what happens on real-world audio?](09_leaving_home.md)**
On In-the-Wild, A2 still ranks well (EER 14.7 %, AUC 0.93), but its home threshold rejects half the
real clips: ranking travels, calibration does not. Layer choice does not explain why A2 beats
another group's use of the same frozen model. A1 is at chance, with the exact value depending on
the seed. The LFCC-versus-MFCC ranking flips: MFCC wins away from home (32.5 % against 54.2 %),
possibly because extra high-band filters model the studio recording chain (untested).

**[10 · Where does JEV fit?](10_where_jev_fits.md)**
A3 adds a second opinion for uncertain clips: accept when confident, escalate when unsure. JEV is a
typed classifier API (text or JSON, a plain structured-text format, in, typed probabilities out), not a large language model. A3
sends a label-free evidence state and asks three fixed questions. It is a scaffold with a dry-run
client and no results, and as configured it cannot yet address chapter 9's calibration problem; its
JEV sources were not re-read.

**[11 · What do we know, what don't we, and what would tell us?](11_what_we_know.md)**
Back to chapter 0's question. What is measured, what is still a guess, and the tests that would
settle each guess (a trimmed-training head, X1–X3, X5, X7, X8, E5). Separately, planned work:
XLS-R, A2-alt, E2, E4 and E6. None of it has been run.
