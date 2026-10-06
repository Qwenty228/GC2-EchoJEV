---
id: FC-001-t02-silence-ratio
task: T02
claim_by: T02 implementation agent (audio.py claim block)
verdict: contradicted
checked_on: 2026-09-20
---

# FC-001 — T02's `silence_ratio` thresholded per-sample amplitude, not framed RMS, while its docstring claimed librosa's `top_db` convention

## Claim as stated
> `silence_ratio` computes leading/trailing/total silence proportion using librosa's
> `top_db` convention (frame-based RMS energy thresholding relative to the loudest frame,
> matching `librosa.effects.trim`/`split`).

## Evidence examined
- code (as originally submitted, before correction): `src/audio_df/features/audio.py`, `silence_ratio`
  — thresholded **instantaneous per-sample magnitude** against a fixed level, not framed RMS
  energy relative to a per-clip peak.
- code (current, corrected, verified in this pass): `src/audio_df/features/audio.py:133-163`
  (`silence_ratio`), now delegates to `librosa.effects.split(top_db=..., frame_length=...,
  hop_length=...)`.
- knowledge: no pre-existing knowledge note covered this; the discrepancy was caught by
  direct numeric measurement, not by a literature source.
- measurement (performed by the coordinator, reported to the researcher as settled):
  - a speech-like amplitude-modulated harmonic signal with **no true silence** scored
    `0.227` (22.7%) "silence" under the original per-sample-threshold implementation, vs.
    `0.000` under librosa's framed-RMS `top_db` convention.
  - a pure 440 Hz tone (also no true silence) scored `0.025` (2.5%) vs. `0.000`.

## Verdict
**Contradicted** (as originally submitted) — the implementation did not match its own
docstring's stated convention, and produced numerically wrong output on two synthetic signals
with zero actual silence. The function has since been reimplemented on framed RMS energy and
now matches the claimed convention; see "Required action" for what to re-verify.

## Reasoning
Any periodic waveform crosses (or comes close to) zero twice per cycle. A per-sample
amplitude threshold therefore counts ordinary zero-crossings as "silent" samples, so the
measured "silence ratio" is really a proxy for the waveform's period/shape (how much time it
spends near zero within each cycle) rather than an honest measure of actual silence
(sustained low-energy stretches). This is not a subtle numerical discrepancy — it is a
category error in what is being measured. It matters specifically because:

1. Leading/trailing silence duration is the single most important nuisance feature
   identified for this dataset (see `research/knowledge/methods/leakage-audit.md`, citing
   Müller et al. 2106.12914's measured 15.1% EER / up to ~500% relative EER degradation from
   this exact artifact in ASVspoof 2019 LA). A `silence_ratio` that is actually measuring
   waveform shape rather than silence would make T09's leakage audit test the wrong
   hypothesis — a positive or negative leakage-audit result computed from the broken
   function says nothing reliable about whether *silence* is leaking the label, only about
   whether *some entangled shape/silence proxy* is.
2. Because the corruption is systematic (period-dependent, not random noise), it could
   itself introduce a *new*, spurious, easily-detectable "signal" correlated with pitch/voicing
   characteristics — potentially even amplifying an apparent leakage effect for the wrong
   reason, or masking the real one.

This is exactly the scenario `research/README.md` describes as the reason the fact-check loop
exists: a claim ("matches librosa's top_db convention") that is plausible, well-written, and
wrong, caught only by reading the actual code and testing it against synthetic signals with
known ground truth (zero true silence) rather than trusting the docstring.

## Required action
Already applied at the time of this fact-check: `silence_ratio` (`src/audio_df/features/audio.py:133-163`)
now delegates directly to `librosa.effects.split(top_db=..., frame_length=..., hop_length=...)`
rather than re-deriving frame/dB arithmetic by hand, which the current docstring itself notes
is deliberate ("rather than re-deriving the frame/dB arithmetic by hand and risking a second,
subtly different reimplementation"). No further code change required from this fact-check.
Recommended follow-up (not blocking): when T09's leakage audit runs against the real ASVspoof
2019 LA manifest, its measured silence-duration leakage should be compared against the
Müller et al. figures in `leakage-audit.md` (~15% EER on real ASVspoof LA eval data from
silence duration alone) as a sanity check that the now-corrected `silence_ratio` reproduces a
result of the right order of magnitude — a wildly different number would suggest either a
remaining measurement bug or a genuine, reportable difference from the published result worth
investigating either way.
