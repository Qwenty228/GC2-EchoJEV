# Chapter 1 · What is a voice made of, and what must a fake imitate?

## Look at the stripes

Go back to the two spectrograms from chapter 0.

![Spectrogram of a real clip (left) and a fake clip from attack A03 (right)](img/ch00_spectrogram_pair.png)
*Source: [notebook 02](../../../SystemCode/notebooks/02_spectral_eda.ipynb), cell 27.*

Inside each word are **thin horizontal stripes**, stacked evenly and most visible below about
3,000 Hz (hertz), and **broad bright bands** where several stripes light up together. The stripes
move with the pitch; the bands shift from vowel to vowel. Two different things are drawn on top
of each other, and separating them is this chapter's key idea.

## A voice is a buzz shaped by a tube

Here is the classic picture of how speech is produced. It is standard speech-science textbook
material, **not** in our reading list, so treat this section as background.

1. **The source.** Your vocal folds open and close many times a second, letting out puffs of
   air. A train of puffs at, say, 120 per second makes a buzz whose energy sits at 120 Hz and
   at every whole multiple of it: 240, 360, 480 Hz and so on. Those multiples are called
   **harmonics**. The spacing between them is the **pitch**, also called the fundamental
   frequency (F0).
2. **The filter.** The buzz then travels through your throat, mouth and lips. That tube
   resonates at some frequencies and damps others, the way a bottle rings at its own note when
   you blow across it. The resonant peaks are called **formants**. Moving your tongue and lips
   moves the formants, and that is how an "ee" differs from an "ah".

Here is a toy with made-up numbers, built only for intuition:

![Toy source-filter example: source comb, filter curve, and their product at two pitches](img/ch01_source_filter_toy.png)
*Source: [notebook 20](../../../SystemCode/notebooks/20_report_figures.ipynb), cell 3 (toy, no data).*

The source is a comb of harmonics every 120 Hz; the filter is a curve with three bumps (700,
1,220 and 2,600 Hz); the output is each tooth scaled by the curve. Bottom right keeps the **same
filter** and raises the pitch to 220 Hz: the teeth spread apart, but the curve through their tips
keeps its shape. So the stripes are the source's harmonics, the bright bands are the filter's
formants, and the two can change independently.

The arithmetic is now easy to guess. The output at each frequency is the
source's strength times the filter's gain. On a decibel (dB) scale, which is a logarithm,
multiplying becomes adding:

$$
\text{level}_\text{out}(f) = \text{level}_\text{source}(f) + \text{gain}_\text{filter}(f) \quad\text{(in dB)}
$$

In words: the loudness at a frequency is how loud the buzz is there, plus how much the mouth
boosts or cuts it. Chapter 5 uses this "multiply becomes add" trick to pull the filter out of a
recording.

## What a speech generator has to rebuild

A fake voice must produce both parts. The dataset's generators are built in two stages (from
paper [14, §3, p.6]). For attack A01, "an NN-based [neural network (NN)] acoustic model
predicts a sequence of acoustic features, including the Mel-cepstral coefficients (MCCs) of 60
dimensions, interpolated F0, and voicing flags. Finally, the WaveNet vocoder is used to convert
the MCCs and F0 into a waveform" (from paper [14, §3, p.6]). MCCs are a few numbers describing
the smooth shape of the spectrum (the filter part); chapter 5 builds a close cousin. WaveNet,
like WaveRNN and Vocaine below, is just a vocoder's name.

| stage | what it produces | in source-filter terms |
|---|---|---|
| **acoustic model** | a frame-by-frame description: F0 (pitch), voicing on/off, and a compact spectral description (here MCCs) | the source's pitch plus the filter's shape |
| **vocoder** (waveform generator) | the actual audio samples | turns that description back into sound |

The right-hand column is our reading; the paper only names the stages and their outputs. Its
Table 1 lists every attack this way, with "Outputs" (from the acoustic model) and "Waveform
generator" (the vocoder) columns (from paper [14, Table 1, p.6]). Two rows are worth memorising:

- **A10 and A11 are identical up to the vocoder.** Both turn text into mel-spectrograms
  (spectrograms on a frequency scale that mimics the ear) with the same architecture; A10 then
  uses WaveRNN, a neural vocoder, and A11 Griffin-Lim, an algorithm that needs no training (from
  paper [14, Table 1, p.6, and §3, p.8]). Any difference a detector sees between them is, by
  design, the vocoder.
- **A17, A18 and A19 start from a real person's recording** ("Speech (human)"): voice conversion
  (VC) systems that change who seems to be speaking in an existing recording, rather than generating speech
  from text (from paper [14, Table 1, p.6]).
  Whatever surrounds the speech may survive the conversion; chapter 3 shows why that matters.

## Where can a fake go wrong?

Three places to look for evidence:

1. **The filter description.** Are the formants in natural places, with natural detail?
2. **The waveform.** Even with a perfect description, the vocoder must fill in fine detail the
   description leaves open (our reading). The database authors report that attacks using the
   neural source-filter (NSF) vocoder (attack A08), Vocaine or Griffin-Lim "were not able to
   produce audio that shows more similarity to genuine speech under the current settings" (from
   paper [14, §6.1.2, p.17]).
3. **Everything that is not the voice.** Silence, background hiss, the microphone and room.
   A generator can get these "wrong" too, regardless of how good the voice is.

Which of the first two matters more? The database authors' own analysis, using their two
baseline detectors, concludes that "the acoustic model seemed to have less of an impact than
the waveform generation method", and adds that this "was also confirmed by the comparison of
performance obtained with A10 and A11 where the only difference was in the waveform generation
method" (from paper [14, §6.1.2, p.17]). Chapter 6 repeats that comparison with our own detector.

Point 3 is the dangerous one: if silence separates real from fake in training, a detector
will use silence.

> **Checkpoint**
>
> - A voice can be pictured as a buzz (the source: pitch and its harmonics) shaped by the throat
>   and mouth (the filter: formants). Pitch and filter shape can change independently
>   ([notebook 20](../../../SystemCode/notebooks/20_report_figures.ipynb), cell 3; a toy, and a textbook idea
>   not in our reading list). On a decibel scale, source times filter becomes source plus filter.
> - Speech generators work in two stages: an acoustic model predicts pitch and a spectral
>   description; a vocoder turns that into audio (from paper [14, §3, p.6], attack A01).
> - A10 and A11 differ only in their vocoder; A17–A19 convert recordings of real people (from
>   paper [14, Table 1, p.6]).
> - The database authors found the vocoder mattered more than the acoustic model for their
>   baseline detectors (from paper [14, §6.1.2, p.17]).
>
> **The question this leaves us with:** we now know what to look for. Where do our real and fake
> clips actually come from, and how do we score a detector fairly when most of them are fake?

## References used in this chapter

- Source-filter model: standard speech-science textbook material, not in our reading list and
  not checked against a specific source.
- [14] Wang et al. *ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and
  Replayed Speech.* Computer Speech & Language 64, 2020.
  [arXiv:1911.01601](https://arxiv.org/abs/1911.01601)
