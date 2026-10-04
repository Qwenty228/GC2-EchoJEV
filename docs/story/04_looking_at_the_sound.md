# Chapter 4 · Looking at the sound: how different are real and fake?

*Notebook cell numbers count the first cell of a notebook as cell 0.*

## A few numbers per clip, two piles each

Chapter 3 looked at properties that *should not* matter, such as silence and loudness. Now look
at the sound itself.

[Notebook 02](../../notebooks/02_spectral_eda.ipynb) takes a small sample of the ASVspoof 2019
LA dataset (the Automatic Speaker Verification Spoofing and Countermeasures Challenge, logical
access part): 240 clips, 40 real and 40 fake from each of train, dev and eval (cell 4). For
each clip it computes a handful of one-number summaries of the spectrum, the picture of energy
against frequency from chapter 0. Three of them matter below:

- **spectral bandwidth**: how widely the energy is spread around its centre frequency;
- **spectral rolloff**: the frequency below which 85 % of the energy sits;
- **spectral centroid**: the "centre of mass" of the frequencies; higher sounds brighter.

The figure below ([notebook 02](../../notebooks/02_spectral_eda.ipynb), cell 7) draws, for each summary, two histograms: real clips (green) and fake clips
(orange). Look at the **bandwidth** panel: the orange pile sits clearly to the left of the green
one. Now look at **band_high_ratio** (the share of energy above 4 kHz, kilohertz): the two piles
lie on top of each other.

![Real (green) and fake (orange) histograms of 13 one-number summaries](img/ch04_spectral_summaries.png)
*Source: [notebook 02](../../notebooks/02_spectral_eda.ipynb), cell 7, rendered locally (the committed notebook has no saved output); 240-clip sample.*

Some panels separate, some do not. We need a number for "how far apart are two piles?", and
we need it to be honest.

## Two ways to measure "how far apart"

### Cohen's *d*: the gap between the averages, in units of spread

Picture two piles. The obvious measure is the distance between their averages. But "300 Hz
(hertz)
apart" means a lot if each pile is 50 Hz wide and little if each is 2,000 Hz wide. So divide the
gap by the typical width:

$$
d = \frac{\text{average(fake)} - \text{average(real)}}{\text{pooled standard deviation}}
$$

In words: how many "typical spreads" apart the two averages are. The pooled standard deviation
combines the spread of both groups. A rough scale often quoted, from Cohen's 1988 textbook (not in
our reading list): 0.2 small, 0.5 medium, 0.8 large. The code is [`cohens_d`](../../src/audio_df/stats.py#L78).

### Cliff's δ: who wins, pair by pair

Averages can be dragged around by a few extreme clips. So here is a second measure that cannot
be. Take every fake clip and every real clip, all 120 × 120 pairs, and ask: in this pair, does
the fake have the higher value or the lower? Then

- δ = (share of pairs where the fake is higher) − (share where it is lower).

δ runs from −1 (every fake is lower than every real clip) through 0 (no tendency) to +1. The
code is [`cliffs_delta`](../../src/audio_df/stats.py#L105).

In the figure below ([notebook 02](../../notebooks/02_spectral_eda.ipynb), cell 13), the top row draws *d*: two
histograms, the gap between the means, and a grey ruler one pooled standard deviation long. The
bottom row draws δ: a grid of every pair, orange where the fake wins, green where the real clip
wins. Three columns:

1. **Bandwidth.** Both measures say "fake is clearly lower": d = −1.32, δ = −0.65.
2. **Leading silence.** About a quarter of the fakes sit at or near zero. That pile stretches the
   gap between averages, so *d* is bigger (−1.42), while δ ranks it about the same as
   bandwidth (−0.64).
3. **A made-up example.** Two identical piles, plus 3 extreme outliers added to the fakes. The
   outliers drag the fake average up, so *d* says "fake higher" (+0.14). δ counts pairs, and 3
   clips out of 123 barely matter, so it says "fake very slightly lower" (−0.07). The two
   disagree even on the sign.

![Cohen's d (top) and Cliff's delta (bottom) drawn for bandwidth, leading silence and a made-up example](img/ch04_d_and_delta_drawn.png)
*Source: [notebook 02](../../notebooks/02_spectral_eda.ipynb), cell 13, rendered locally (the committed notebook has no saved output).*

The rule of thumb from that cell: **if *d* and δ agree, the difference is not the work of a few
odd clips. If they disagree, look at the distribution.**

There is a bonus. δ and the area under the curve (AUC) from chapter 2 are the same information
on two scales: δ = 2 · AUC − 1 (notebook 02, cell 15 prints the check for every feature).
Bandwidth's δ of −0.65 is an AUC of 0.82 for "lower bandwidth means fake".

### Why not a p-value?

A significance test asks "could a difference this big be pure chance?" With enough clips, even
a useless difference of a few milliseconds passes. Effect sizes ask the question we care about:
*how big* is the difference.

## The result

Notebook 02, cell 10 computes both measures for every summary, with a 95 % confidence interval
(CI) on *d* from 500 bootstrap resamples (the resampling idea from chapter 3). The numbers are
the printed output of a local run; the committed notebook has no saved output.

| feature (fake against real) | Cohen's *d* [95 % CI] | Cliff's δ |
|---|---|---|
| leading silence | −1.42 [−1.73, −1.16] | −0.64 |
| spectral bandwidth | −1.32 [−1.61, −1.08] | −0.65 |
| spectral rolloff | −0.93 [−1.21, −0.69] | −0.50 |
| share of energy above 4 kHz | −0.13 [−0.37, 0.09] | −0.05 |

The next figure (notebook 02, cell 15) shows all features at once: one dot per feature, *d*
against δ. The two measures agree on the **direction** (the sign) of every feature, and the dots
follow one rising trend, so they agree broadly on which features separate strongly and which
barely do. They do swap the order of a few close neighbours: bandwidth and leading silence
(*d* ranks silence first, δ ranks bandwidth first), flatness and trailing silence, the mid- and
high-band energy shares, and the low-band share and duration. For leading silence the cause is
visible: its pile of clips at zero stretches the gap between averages more than it changes who
wins each pair. The other three swaps are between close neighbours, and we have not shown why
they swap; small differences in shape are enough to reorder features that are this close. No feature sits far off the trend, so no
feature's result is the work of a few extreme clips.

![Cohen's d against Cliff's delta, one dot per feature](img/ch04_d_vs_delta.png)
*Source: [notebook 02](../../notebooks/02_spectral_eda.ipynb), cell 15, rendered locally (the committed notebook has no saved output).*

What it says:

1. **Silence shows up again,** as strong as anything acoustic. That is chapter 3's shortcut,
   seen on a small sample.
2. **Bandwidth and rolloff are new and large.** Fake clips spread their energy less widely, and
   their energy is concentrated lower.
3. **The obvious story, "fakes lack high frequencies", is not supported.** The share of energy
   above 4 kHz barely differs, and its confidence interval includes zero.

## Is bandwidth just silence in disguise?

Here is a worry. A silent stretch is faint room hiss, and hiss spreads across all frequencies.
That *raises* a clip's average bandwidth. Real clips have more silence (chapter 3). So part of
their higher bandwidth might simply be their silence.

We don't know yet how much. The test is cheap and not yet run: recompute *d* for bandwidth using
only clips with little silence (for example under 20 %). If the gap survives, bandwidth is a
real acoustic cue. If it collapses, it was silence all along. Until then it is a **candidate**
cue (our guess, not a measured result).

## No single "fake" direction

[Notebook 02](../../notebooks/02_spectral_eda.ipynb), cell 24 splits the spectral centroid by
attack. Its overlaid histograms are unreadable with this many attacks, so use the table printed
under the figure: mean centroid per attack (printed output of a local run). Treat it as a
**hint**, not a finding: the 120 fakes in the sample are spread unevenly over the attacks, about
9–19 clips each for the training attacks A01–A06 but only 1–8 for each eval attack, and A09 has
none (counts from rebuilding notebook 02's sample with its own seed).

- Real speech: about 1,733 Hz.
- A11 (the Griffin-Lim vocoder from chapter 1): about 1,400 Hz, **below** real speech, from a
  single clip.
- A18: about 2,292 Hz, **above** real speech, from 3 clips.
- Overall, 14 of the 18 attacks in the sample sit below real speech; only A01, A17, A08 and A18
  sit above.

So fakes are mostly lower here, but not all: "fake" need not be one direction. If generators
push the same summary opposite ways, averaging them together can cancel the effect. This is why
chapter 2 insisted on per-attack results. A larger per-attack sample would turn this hint into a
measurement.

> **Checkpoint**
>
> - Spectral summaries separate real from fake to different degrees; bandwidth separates
>   strongly, high-band energy share does not ([notebook 02](../../notebooks/02_spectral_eda.ipynb),
>   cell 7).
> - Cohen's *d* measures the gap between averages in units of spread; Cliff's δ counts who wins
>   each pair and resists outliers; they agree on the direction of every feature and broadly on
>   which matter, swapping only a few close neighbours
>   (notebook 02, cells 13 and 15; [`stats.py`](../../src/audio_df/stats.py#L78)).
> - Bandwidth: d = −1.32, δ = −0.65; rolloff: d = −0.93, δ = −0.50; share of energy above
>   4 kHz: d = −0.13 with an interval that includes zero (notebook 02, cell 10; printed output of a
>   local run).
> - Whether bandwidth is independent of silence is untested; the low-silence recomputation would
>   settle it (our guess that part of it is silence).
> - Hint: attacks may move the same summary in opposite directions: A11's centroid about
>   1,400 Hz, A18's about 2,292 Hz, real speech about 1,733 Hz; 14 of 18 attacks sit below real
>   speech (notebook 02, cell 24; a hint only: 1 clip for A11, 3 for A18).
>
> **The question this leaves us with:** single summaries are blunt. How do we turn a whole
> sound into a small set of numbers that keeps the shape of the spectrum, frame by frame?
