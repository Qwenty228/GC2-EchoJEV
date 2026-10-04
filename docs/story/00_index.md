# Spotting fake voices: the story of our detectors

The results handoff ([training-and-data-handoff.md](../training-and-data-handoff.md)) retold as
a story, for teammates who know basic Python and statistics but not signal processing. After
reading, you should be able to explain why A2 beats A1, why silence matters, and what we still
don't know. Read in order; each chapter opens with a question and ends with a **Checkpoint**: what we now know, its
evidence, and the question that opens the next chapter.

**Conventions.**

- Green always means real (bona fide), orange always means fake (spoof). Where linear and mel
  filters are compared, blue is linear and magenta is mel.
- Every number is labelled by where it comes from: **measured** (a committed file in this repo, or
  a local output named as such in the figure caption), **from paper [n]** (another paper's number,
  under another protocol, context only), or **our guess** (a reading not yet tested).
- Figures live in [img/](img/); each caption names its source, usually
  [notebook 20](../../notebooks/20_story_figures.ipynb). Notebook cells count from 0.
- Reference numbers [n] follow the handoff's §15, which links every paper.

## Chapters

**[0 · The problem: what would a detector even look for?](00_the_problem.md)**
A real and a fake spectrogram look alike, but the real one has a longer silent lead-in. It names
the two traps (shortcuts and new conditions) and the three systems, A1, A2 and A3.

**[1 · What is a voice made of, and what must a fake imitate?](01_what_is_a_voice.md)**
A voice is a buzz (source) shaped by the throat and mouth (filter); generators rebuild both with an
acoustic model and a vocoder. Two attack groups are set up for later: A10/A11 (differ only in the
vocoder) and A17–A19 (convert real recordings).

**[2 · Where does our data come from, and how do you score fairly when 9 of 10 clips are fake?](02_data_and_fair_scoring.md)**
ASVspoof 2019 LA (the Automatic Speaker Verification Spoofing and Countermeasures Challenge, logical
access part) is studio speech with 19 attacks; In-the-Wild is real-world audio we never train on.
Accuracy is useless at 1:9, so we use the equal error rate (EER), the area under the curve (AUC) and
a threshold fixed on dev.

**[3 · Before building anything: can something stupid already "detect" fakes?](03_the_silence_shortcut.md)**
A one-number probe on leading silence ranks real above fake with AUC 0.837, because of how the
dataset was assembled. From here on, every result is also reported with the silence trimmed.

**[4 · Looking at the sound: how different are real and fake?](04_looking_at_the_sound.md)**
Cohen's d and Cliff's δ measure how far apart two groups are. Spectral bandwidth separates real from
fake strongly; whether it is silence in disguise is untested.

**[5 · How do you turn a sound into a few numbers?](05_sound_into_numbers.md)**
The recipe for linear-frequency cepstral coefficients (LFCC), step by step: c0 is level, c1 tilt,
higher ones ripple, and a gain change moves only c0. Linear and mel filters differ only in
placement, which sets up LFCC against mel-frequency cepstral coefficients (MFCC).

**[6 · A1: can two clouds of frames tell real from fake?](06_a1_two_clouds.md)**
A1 scores a clip with two Gaussian mixture models and reproduces the official baseline (9.56 % eval
EER against 8.09 %). A10/A11 (22.4 % against 0.1 %) shows it detects the vocoder, and LFCC beats
MFCC at home (the controlled comparison X4, 3 seeds).

**[7 · A2: borrow a model that has already heard 94,000 hours of speech?](07_a2_borrowed_ears.md)**
A2 trains a small head on a frozen WavLM Base+ and reaches 0.96 % eval EER. The layer analysis (X6)
shows no single layer is essential, and middle layers lean least on silence.

**[8 · Trim the silence: what is left of each detector?](08_trim_the_silence.md)**
Trimming at test time doubles A1's EER and multiplies A2's by about 14; A2's real clips lose about
19 points of score while its fakes barely move.
Both were trained on raw audio, so shortcut and train/test mismatch are not yet separated.

**[9 · Leaving home: what happens on real-world audio?](09_leaving_home.md)**
On In-the-Wild, A2 still ranks well (EER 14.7 %, AUC 0.93) but its home threshold rejects half the
real clips: ranking travels, calibration does not. A1 is at chance (AUC 0.38–0.56 over 3 seeds), and MFCC beats
LFCC away from home (32.5 % against 54.2 %).

**[10 · Where does JEV fit?](10_where_jev_fits.md)**
A3 asks JEV, a typed classifier service (not a large language model), for a second opinion on
uncertain clips. It is a scaffold with a dry-run client and no results, and as configured it cannot yet fix
chapter 9's calibration problem; its JEV sources were not re-read.

**[11 · What do we know, what don't we, and what would tell us?](11_what_we_know.md)**
Back to chapter 0's question: what is measured, what is still a guess, and the tests that would
settle each guess. None of those tests has been run.
