# Chapter 0 · The problem: what would a detector even look for?

## Start with two pictures

Here are two clips from our main dataset, side by side, as **spectrograms**.

![Spectrogram of a real clip (left) and a fake clip from attack A03 (right)](img/ch00_spectrogram_pair.png)
*Source: [notebook 02](../../../SystemCode/notebooks/02_spectral_eda.ipynb), cell 27.*

A spectrogram is a picture of sound:

- time runs left to right;
- **frequency** (how high or low each part of the sound is) runs bottom to top, from 0 to
  8,000 Hz (hertz, vibrations per second);
- brightness shows how much energy the sound has at that time and frequency.

The left clip is a real person reading a sentence. The right clip is a fake made by a
text-to-speech (TTS) system: attack **A03** (the dataset calls each fake-making system an
"attack" and numbers them A01–A19). Both show bright blobs where words are spoken, thin
horizontal stripes inside them and dark gaps between words. The picture does not make the fake
obvious.

Notice two small things anyway, because they come back later:

1. The real clip starts with about 0.8 s of near-silence before the first word. The fake
   starts speaking after about 0.35 s.
2. The real clip is about 3.1 s long, the fake about 1.7 s.

Neither has anything to do with *how* the fake was made. Keep that thought. The notebook itself
warns that one pair is "context only, not evidence" (notebook 02, cell 26); every *claim* after this chapter rests on
thousands of clips.

## Why anyone cares

Müller et al. write that text-to-speech "enables the cloning of politicians' voices", with
risks that include "the spread of misinformation" (from paper [13, §1, p.1]). Our dataset says
of attack A10: "synthetic speech produced by this system has high naturalness and good
similarity to target speakers perceptually" (from paper [14, §3, p.8]). A listener may not notice; we want a program
that does.

## The question, in plain words

The [README](../../../SystemCode/README.md) and [technical reference §1](../technical-reference.md) state the
project's question with a formula; in our words:

**Is there something we can measure about a clip that is distributed differently for real and
fake speech, and does that difference come from how the fake was made, rather than from some
accident of how the dataset was collected?**

Two traps hide in that sentence.

1. **The shortcut trap.** Real and fake clips can differ in ways unrelated to synthesis:
   leading silence, loudness, microphone. A detector will happily learn those instead, and fail
   when the accident goes away. The lead-in and length above are exactly this kind of accident.
2. **The new-world trap.** A detector may meet fakes from a generator it has never seen,
   recorded in conditions it has never heard. Does what it learned still apply?

## What we built to answer it

Three detectors, from old-fashioned to modern:

| system | one-line description | status |
|---|---|---|
| A1 | a classic hand-made sound summary plus a simple statistical model, rebuilt from the official 2019 challenge baseline's recipe | built and measured |
| A2 | a large pre-trained speech model, kept frozen, with a small trained decision layer on top | built and measured |
| A3 | A2 plus JEV, a classifier service: typed text or JSON (JavaScript Object Notation) in, typed probabilities out. JEV is **not** a large language model | scaffold only, no real results yet |

We test both traps along the way: chapter 3 hunts for shortcuts before any detector exists,
chapter 8 removes the silence and re-measures, and chapter 9 takes the detectors to real-world
audio. There, one does no better than a coin flip; the other still ranks real above fake well,
but its decision threshold no longer fits.

> **Checkpoint**
>
> - A spectrogram shows a sound's energy over time and frequency; a real and a fake clip look
>   alike at a glance (figure above; [notebook 02](../../../SystemCode/notebooks/02_spectral_eda.ipynb), cell 27).
> - In that one pair, the real clip has a longer silent lead-in (about 0.8 s against 0.35 s), a
>   difference unrelated to how the fake was made (same figure; one example only, not evidence).
> - Fake voices are realistic enough to be a social risk (from paper [13, §1, p.1]) and can
>   sound natural to listeners (from paper [14, §3, p.8], describing attack A10).
> - The project's question is whether some measurable property differs between real and fake
>   *because of synthesis*, and whether that survives new generators and new recording
>   conditions ([technical reference §1](../technical-reference.md)).
>
> **The question this leaves us with:** if a fake must sound like a voice, what exactly is a
> voice made of, and which parts does a generator have to rebuild?

## References used in this chapter

- [13] Müller et al. *Does Audio Deepfake Detection Generalize?* Interspeech 2022.
  [arXiv:2203.16263](https://arxiv.org/abs/2203.16263)
- [14] Wang et al. *ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and
  Replayed Speech.* Computer Speech & Language 64, 2020.
  [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)
