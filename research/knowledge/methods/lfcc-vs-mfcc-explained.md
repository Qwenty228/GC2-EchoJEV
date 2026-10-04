---
id: lfcc-vs-mfcc-explained
type: method
sources:
  - src/audio_df/lfcc_b02.py (HEAD d6f123f; line numbers below refer to this file)
  - src/audio_df/features.py (HEAD d6f123f; hz_to_mel L76-86, mel_to_hz L88-91, filterbank L120-173)
  - src/audio_df/audio.py (trim_silence L227-285, silence_ratio)
  - configs/features.yaml; configs/a1_lfcc_gmm.yaml
  - research/knowledge/methods/b02-lfcc-gmm.md, lfcc-mfcc-dct.md, leakage-audit.md (existing notes, relied on for the B02 parameters and the silence-shortcut literature)
  - https://www.isca-archive.org/interspeech_2015/sahidullah15_interspeech.pdf (Sahidullah, Kinnunen, Hanilci 2015; fetched and read in full)
  - https://arxiv.org/pdf/2111.02813 (Frank & Schönherr, WaveFake, NeurIPS 2021 D&B; fetched and read)
  - https://www.isca-archive.org/interspeech_2016/sriskandaraja16_interspeech.pdf (Sriskandaraja et al. 2016; fetched and read)
  - https://arxiv.org/pdf/1911.01601 (Wang et al., ASVspoof 2019 database paper, arXiv v4 = Computer Speech & Language 2020; fetched and read; Table 1, p.6)
  - Zhou et al. 2011, https://doi.org/10.1109/ASRU.2011.6163888 (read in full; local note linear-vs-mel-cepstral-speaker-recognition.md, not in git)
  - experiments/E-20261001-e1-a1-raw, E-20261001-e1-a1-trimmed (committed scores/metrics); experiments/E-20261001-e1-a2-raw/layer_weights.json
  - notebooks/_executed/02_spectral_eda.ipynb, 03_cepstral_eda.ipynb (committed outputs)
confidence: medium (the maths and the code-derived numbers are high; the explanations of per-attack behaviour are hypotheses and are labelled)
last_verified: 2026-10-04
related: [b02-lfcc-gmm, lfcc-mfcc-dct, leakage-audit, linear-vs-mel-cepstral-speaker-recognition, asvspoof-2019-la]
---

# What LFCC captures, and why it can beat MFCC for synthetic-speech detection

## Reading guide: evidence tags
Every claim carries one of these tags and a confidence (H/M/L).
- **[lit]** `source: literature`. A number or statement from a paper. Never comparable to our runs.
- **[proj]** committed output of this project (file path or experiment id).
- **[calc]** arithmetic or a code evaluation anyone can reproduce from the repo at HEAD d6f123f (script logic given in the Appendix).
- **[recomp]** the researcher re-ran light code on the repo's data. Not a committed output. Sample definition given in the Appendix.
- **[hyp]** hypothesis. Not sourced, not tested. A test is specified in §6.

**Standing honesty point.** This project has **not** run a controlled LFCC-versus-MFCC comparison. "LFCC is better than
MFCC for A1" currently rests on literature plus indirect EDA. See §4.4 and §6.3.

## Claim in one sentence
A 20-filter linear bank gives the 4-8 kHz half of the spectrum 10 filters of 762 Hz where a mel bank gives it 5 filters of
1.1-1.9 kHz, so the DCT coefficients of an LFCC frame can express finer high-band shape; controlled GMM studies on other
spoofing corpora found LFCC better than MFCC [lit], but our own EDA shows only a modest, silence-confounded high-band
difference between bona fide and spoof [recomp], and under silence trimming A1's per-attack behaviour tracks frame-level energy and silence
differences, which a change of filter spacing would not obviously remove [recomp, proj, hyp].

## Findings

### F-L. What the literature says (each `source: literature`)
1. **Same GMM, LFCC vs MFCC vs inverted-mel, ASVspoof 2015 dev, average EER over attacks** [lit, H on the numbers].
   Sahidullah et al. 2015, Table 3 (p.4); 20 filters, 20 ms window with 50 % overlap, 512 DFT bins, GMM-ML with 512 mixtures
   and 10 EM iterations (Table 1 p.2, §3.2 p.3, §3.4 p.3). Columns: static / static+Δ+ΔΔ / Δ+ΔΔ only.

   | feature | static | static+ΔΔ² | ΔΔ² only |
   |---|---|---|---|
   | MFCC | 3.46 | 1.09 | 0.64 |
   | LFCC | 2.46 | 0.66 | 0.12 |
   | IMFCC (filters dense at high frequency) | 1.33 | 0.48 | 0.20 |

   Order by high-band resolution (MFCC < LFCC < IMFCC) matches the order of EER in the first two columns. Their text (§3.5, p.4):
   "IMFCC, a feature set which emphasizes high-frequency spectral information beats MFCCs that emphasize the low-frequency region."
   Evaluation set, unknown attacks only (Table 4 p.4): LFCC (ΔΔ²) 1.67 vs MFCC (ΔΔ²) 3.84.
   Corpus caveat: ASVspoof 2015, older VC/SS systems, not 2019.
2. **Their stated reason is an intuition, not a test** [lit, H]. p.2: "The finer details of speech spectrum are not preserved in
   synthetic speech as VC and SS techniques mostly focus on producing identical overall envelope of the speech spectrum." p.1:
   "Most of the synthetic speech generation techniques do not fully model temporal characteristics of speech." Conclusion (p.4):
   features carrying "information related to high-frequency region, dynamic characteristic and detailed spectral information are
   useful. Those details are not accurately modeled during voice conversion or speech synthesis process."
3. **WaveFake** [lit, H]. GMM on 20 LFCC or MFCC + Δ + ΔΔ with Sahidullah's parameters (App. C p.20). p.7: "we found them to
   outperform the MFCC features strictly. LFCC features contain a significantly higher amount of high-frequency components. We
   hypothesize that these are meaningful for achieving good overall performance." Tables 2 and 6 (pp.8, 20), average EER over ten
   test sets, seven training sets: LFCC aEER 0.062-0.222 (mean 0.129), MFCC aEER 0.214-0.360 (mean 0.295), LFCC better in all seven aEER rows
   (our arithmetic; corrected 2026-10-04). Not every test column: MelGAN-trained rows on the two JSUT sets favour MFCC (0.077/0.074 vs LFCC 0.384/0.533). WaveFake removed silences over 2 s. The vocoder spectrograms differ "especially in the higher frequencies" (p.6) and "these differences are
   visible when plotting the audio but generally inaudible when listening". Attribution (p.9): MelGAN (L) and FB-MelGAN classifiers
   rely "mostly on the higher frequencies", MB-MelGAN and PWG "also rely on low frequencies". Corpora: LJSpeech, JSUT.
4. **Counter-evidence** [lit, H]. Sriskandaraja et al. 2016, SAS corpus, 512-mixture GMM-UBM, 26 filters, 13 coefficients + Δ + ΔΔ
   (Table 1, p.4): 26-linear **8.91 %**, 26-mel **6.89 %** EER. Their sub-band study (§3, p.3) finds the most discriminative
   regions are **0-1 kHz, 2.5-5.5 kHz and 7-8 kHz**; banks with more filters in those regions score 6.09-6.57 %, better than
   both. On the 7-8 kHz region (p.3): it "may reflect the nature of most SS and VC algorithms that do not model this frequency region".
5. **Speaker-recognition precedent** [lit, H]. Zhou et al. 2011 (see `linear-vs-mel-cepstral-speaker-recognition`): 300-3400 Hz
   band, 32 filters, CMVN; LFCC better on pooled and female trials; less robust than MFCC in white noise.
6. **Generators in ASVspoof 2019 LA** [lit, H]. Wang et al. 2020, Table 1 (p.6): waveform generators are WORLD, WaveNet, WaveRNN,
   Griffin-Lim, neural source-filter, Vocaine, STRAIGHT, waveform concatenation, waveform filtering, an MFCC vocoder, and spectral
   filtering + overlap-add. **A16 uses the same algorithm as A04, and A19 the same as A06** (caption of Table 1); "A07 to A19
   (apart from A16 and A19) are the eleven unknown spoofing systems" (§2). A17-A19 convert **human** source speech; A13-A15 convert
   **TTS** output. VCTK was recorded at 96 kHz and downsampled to 16 kHz (§2, p.3).

### F-P. What our own data says
7. **Committed EDA, notebook 02** [proj, H]. 240-clip stratified sample, Hann 25 ms, no pre-emphasis. Cohen's d (spoof minus bona fide):
   spectral bandwidth -1.32 [-1.61, -1.08]; spectral rolloff -0.93 [-1.21, -0.69]; leading silence -1.42; **share of energy at
   >= 4 kHz (`band_high_ratio`) -0.126 [-0.373, 0.093], single-feature AUC 0.526**. (cells 9 and 12.)
8. **Committed EDA, notebook 03** [proj, H]. Mean absolute off-diagonal correlation on bona fide mean-pooled vectors: linear
   log-filterbank 0.851, LFCC 0.166 (cell 11). The per-coefficient effect-size plots were not saved in the executed notebook
   (figures only), so per-coefficient numbers are recomputed in item 10.
9. **A1 results** [proj, H]. Eval EER 9.56 % raw, 19.84 % trimmed; In-the-Wild 58.04 % raw, 61.36 % trimmed (handoff §10).
   Per-attack raw/trimmed in §5.
10. **Recomputed channel and coefficient effects** [recomp, M]. See Appendix R2. 600 clips (100 per split-by-label cell), B02 pipeline
    (pre-emphasis, Hamming 20 ms, log10, orthonormal DCT), whole-utterance means over frames, Cohen's d spoof minus bona fide,
    standard error about 0.08. Per linear filter channel 1..20 (centre 381 Hz ... 7619 Hz): 0.94 0.72 0.43 0.50 0.50 0.54 0.59 0.59
    0.56 0.53 0.42 0.25 0.15 0.15 0.19 0.15 0.18 0.24 0.24 0.09. Every channel is positive. Per LFCC coefficient c0..c19: 0.45 0.75
    0.07 -0.21 0.54 0.77 0.18 0.42 0.39 0.56 -0.13 0.56 0.05 0.02 -0.22 -0.26 **-0.53 -0.40 -0.63** -0.14. Mean |Δc| (frame-to-frame
    change) for c11..c19: -0.44 -0.38 -0.26 -0.12 -0.37 -0.22 -0.32 -0.54 -0.57.
11. **Gain-invariant high-band tilt** [recomp, M]. T = mean of channels 13-20 (4952-7619 Hz centres) minus mean of channels 3-11
    (1143-4190 Hz), in dB of log-energy, per clip. Spoof minus bona fide: **-2.27 dB** (d = -0.67 [-0.84, -0.52]) over all frames;
    **-1.52 dB** (d = -0.33 [-0.50, -0.17]) over the louder half of frames by c0; by split (louder half) dev -1.27, train -1.27,
    eval -2.04 dB. Mid band is +3.3 dB higher in spoof, high band +1.0 dB. T correlates with total silence fraction at Spearman 0.57.

## Mechanism / derivation

### §1 (a) The pipeline as equations, matching `src/audio_df/lfcc_b02.py`
Notation: sampling rate f_s = 16000 Hz, frame length N = 320 (20 ms), hop H = 160 (10 ms), FFT size 512, J = 20 filters.

1. **Pre-emphasis** (L246, `lfilter([1, -0.97], [1], x)`). y[n] = x[n] - 0.97 x[n-1], y[0] = x[0]. Transfer function
   H(z) = 1 - 0.97 z^-1, so |H(e^jω)|^2 = 1 + 0.97^2 - 2(0.97) cos ω with ω = 2π f / f_s.
   **High-pass gain, [calc, H]:**

   | f | 0 | 1 kHz | 2 kHz | 4 kHz | 6 kHz | 8 kHz |
   |---|---|---|---|---|---|---|
   | \|H\| | 0.030 | 0.385 | 0.754 | 1.393 | 1.820 | 1.970 |
   | dB | -30.5 | -8.3 | -2.4 | +2.9 | +5.2 | +5.9 |

   The slope is about +6 dB per octave from 1 to 2 kHz (+5.8 dB) and 2 to 4 kHz (+5.3 dB), then flattens (+3.0 dB from 4 to 8 kHz).
   So handoff §5.1's "+6 dB/octave tilt" is right below about 4 kHz and an overstatement above it. In total the 4-8 kHz band
   sits 33-36 dB above DC after this filter, and 11-14 dB above 1 kHz.
2. **Framing and window** (L250-252, `_frame_signal`, `np.hamming`). Frame m starts at sample mH; the last incomplete frame is
   zero-padded; T = 1 + ceil(max(0, n - 320)/160). w[n] = 0.54 - 0.46 cos(2πn/(N-1)) (symmetric Hamming).
3. **Power spectrum** (L255-256). P[m,k] = | Σ_{n=0}^{319} y[mH+n] w[n] e^{-j2πkn/512} |^2, k = 0..256. Bin spacing 31.25 Hz.
   Zero-padding from 320 to 512 samples only interpolates. The real resolution is set by the window: the Hamming main lobe has full
   width **201 Hz** [calc] (first null at 100.6 Hz).
4. **Triangular filterbank** (L191-214, L260-263). Edges e_i = i · 8000/21, i = 0..21 (so 381 Hz apart). For filter r = 1..20,
   Ψ_r[k] = max(0, min( (f_k - e_{r-1})/(e_r - e_{r-1}), (e_{r+1} - f_k)/(e_{r+1} - e_r) )), f_k = k · 31.25 Hz. Peak 1, not
   area-normalised. E[m,r] = Σ_k P[m,k] Ψ_r[k]. `features.filterbank(spacing="linear")` produces the identical matrix (max absolute
   difference 0.0 [calc]).
5. **Log** (L266). L[m,r] = log10(E[m,r] + ε), ε = 2.22e-16 added (not a floor).
6. **DCT-II, orthonormal** (L269). c[m,0] = √(1/J) Σ_r L[m,r]; c[m,q] = √(2/J) Σ_r L[m,r] cos(πq(r - 1/2)/J) (1-based r),
   q = 1..19. All 20 outputs are kept.
7. **Deltas** (L133-178, L272-274). Δc[t] = c[t+1] - c[t-1] (edge-replicated, no division by 2). Applying it twice gives, in the interior,
   Δ²c[t] = c[t+2] - 2c[t] + c[t-2]. Output [c, Δc, Δ²c], 60 dimensions. As frame-rate filters (100 frames/s), Δ has gain
   2|sin(2π f_m/100)| and Δ² has gain 4 sin²(2π f_m/100); both peak at modulation frequency f_m = 25 Hz and are weak at the 4 Hz
   syllable rate (Δ gain 0.50) [calc]. The context span of Δ² is ±2 frames, so a feature vector sees 60 ms of signal.

**What each coefficient means** (all [calc] from the formulas above, H on the algebra, M on the physical reading).
- **c0 = √J · mean_r L[m,r]**: √J times the average of the log band energies, i.e. √J times the log of the *geometric mean* of the 20 band energies,
  an overall level. It is not the log of total energy. A pure amplitude gain g multiplies every E by g^2 and adds the same constant 2 log10 g to every
  L; the orthonormal DCT of a constant vector is zero for q >= 1, so **a gain change moves only c0** (by √J · 2 log10 g; a 10 dB drop moves c0 by -4.47), and Δc0 is unchanged for a constant gain. Level, loudness and recording-gain differences therefore live in c0.
- **c1** has weights √(2/J)·cos(π(r - 1/2)/20) running from +0.315 (lowest channel) to -0.315 (highest): the **low-minus-high spectral tilt**.
- **c_q** is a cosine with q half-periods across the filter axis. On the linear axis (channels 381 Hz apart) the ripple period is 2·20·381/q ≈ 15,238/q Hz: c4 about 3.8 kHz,
  c10 1.5 kHz, c19 800 Hz (corrected 2026-10-04). So low-order coefficients encode the smooth envelope and tilt (formant-scale shape), higher-order ones
  finer spectral detail down to about 760 Hz ripple. Filters 381 Hz apart cannot resolve pitch harmonics (spacing 100-250 Hz), so LFCC never sees F0 structure directly.
- Truncation caveat from `lfcc-mfcc-dct`: because log of a sum is not a sum of logs, the clean source-filter reading of c_q is approximate. Hedge it.
- Not seen by LFCC at all: phase (discarded at step 3), detail finer than one filter, and anything from time-ordering beyond ±2 frames.

### §2 (b) Linear versus mel filterbank, quantified from our code [calc, H]
Mel scale: m(f) = 2595 log10(1 + f/700) (`features.hz_to_mel`, L76-86; HTK form). Its slope is
m'(f) = (2595/ln 10)/(700 + f) = 1127/(700 + f) mel per Hz: **1.61 at 0 Hz, 0.66 at 1 kHz, 0.24 at 4 kHz, 0.13 at 8 kHz**. Resolution in Hz per
mel scales as (700 + f), so 8 kHz is **12.4 times coarser than 0 Hz** and 4 kHz is 6.7 times coarser. 75.6 % of the mel axis (2146 of 2840 mel) lies below
4 kHz, so equal mel steps put 15 of 20 filter centres below 4 kHz. `features.filterbank(spacing="mel")` (L155-157) places the 22 edges
equally in mel between 0 and 8000 Hz and converts back to Hz.

20 filters, 16 kHz, 512-point FFT (31.25 Hz bins). Bandwidth = edge-to-edge (right edge minus left edge); "bins" = number of FFT bins with
non-zero weight. Linear: `lfcc_b02_filterbank(20, 512, 16000)`. Mel: `filterbank(20, 512, 16000, spacing="mel")`.

| r | linear centre Hz | linear bw Hz | linear bins | mel centre Hz | mel bw Hz | mel bins |
|---|---|---|---|---|---|---|
| 1 | 381 | 762 | 24 | 89 | 190 | 6 |
| 2 | 762 | 762 | 24 | 190 | 214 | 7 |
| 3 | 1143 | 762 | 24 | 303 | 241 | 7 |
| 4 | 1524 | 762 | 24 | 431 | 272 | 9 |
| 5 | 1905 | 762 | 25 | 575 | 307 | 10 |
| 6 | 2286 | 762 | 25 | 738 | 346 | 11 |
| 7 | 2667 | 762 | 24 | 921 | 390 | 13 |
| 8 | 3048 | 762 | 24 | 1128 | 440 | 14 |
| 9 | 3429 | 762 | 24 | 1361 | 496 | 15 |
| 10 | 3810 | 762 | 25 | 1624 | 559 | 18 |
| 11 | 4190 | 762 | 25 | 1920 | 630 | 21 |
| 12 | 4571 | 762 | 24 | 2254 | 711 | 23 |
| 13 | 4952 | 762 | 24 | 2631 | 801 | 25 |
| 14 | 5333 | 762 | 24 | 3056 | 904 | 29 |
| 15 | 5714 | 762 | 25 | 3535 | 1019 | 33 |
| 16 | 6095 | 762 | 25 | 4075 | 1149 | 36 |
| 17 | 6476 | 762 | 24 | 4683 | 1295 | 41 |
| 18 | 6857 | 762 | 24 | 5370 | 1460 | 47 |
| 19 | 7238 | 762 | 24 | 6144 | 1646 | 53 |
| 20 | 7619 | 762 | 24 | 7016 | 1856 | 60 |

| summary | linear | mel |
|---|---|---|
| filters with centre above 4 kHz | **10** | **5** |
| filters with centre above 6 kHz | 5 | 2 |
| filters with centre below 1 kHz | 2 | 7 |
| filters with centre below 2 kHz | 5 | 11 |
| filters whose support reaches above 4 kHz | 11 | 6 |
| bandwidth of the lowest filter | 762 Hz | 190 Hz |
| bandwidth of the top filter (centre; span) | 762 Hz (7619; 7238-8000) | 1856 Hz (7016; 6144-8000) |
| mean bandwidth of the filters centred above 4 kHz | 762 Hz | 1481 Hz |
| FFT bins per filter | 24-25 | 6 to 60 |

**Reading.** Handoff §5.1's counts (10 vs 5 above 4 kHz; lowest mel filter about 190 Hz; top mel filter about 1.9 kHz wide, roughly
6.1-8.0 kHz) are confirmed. Its "about 2.5 times finer resolution there" is exact for the top filter only (1856/762 = 2.4); averaged over the
five mel filters above 4 kHz the factor is **1.9**. Mel's two lowest filters (190 and 214 Hz wide) are no wider than the Hamming window's own 201 Hz
main lobe, so they probably add little independent information beyond the 6-7 bins they straddle [hyp]. A flat spectrum gives the top mel filter 2.4 times
the summed power of a linear filter (peak-1 triangles), a constant offset per channel that the GMM absorbs.

### §3 (d) How the GMM uses the 60-dimensional frame
Each class has a 512-component diagonal-covariance mixture (`src/audio_df/models/gmm.py`; handoff §6.2). For a frame x_t in R^60,
log p(x|c) = logsumexp_k [ log π_k + Σ_d ℓ_kd(x_d) ], ℓ_kd = -½[log 2πσ²_kd + (x_d - μ_kd)²/σ²_kd ].
The clip score is s(X) = (1/T) Σ_t [ log p(x_t|bona) - log p(x_t|spoof) ] (handoff §6.4; B02: `llk_genuine - llk_spoof`, `b02-lfcc-gmm`).

- **Why diagonal is acceptable** [proj, H]: DCT cuts mean |off-diagonal correlation| from 0.851 to 0.166 (item 8). The diagonal is per component;
  the 512-component mixture can still represent correlation between components.
- **An exact additive decomposition** [calc, H]. With responsibilities γ_k = π_k N_k / p, the identity log p = Σ_k γ_k (log π_k + log N_k) - Σ_k γ_k log γ_k
  holds exactly, hence log p(x|c) = Σ_d [Σ_k γ_k ℓ_kd] + R_c(x), where R_c collects the mixing terms. So the frame LLR is a sum of **60 per-dimension
  shares** s_d(x) = Σ_k γ^bona_k ℓ^bona_kd - Σ_k γ^spoof_k ℓ^spoof_kd plus a remainder R_bona - R_spoof. This is exact, but shares are not independent causal effects: the
  responsibilities γ depend on all 60 dimensions at once.
- **What can matter.** A dimension changes the score only if the bona fide and spoof mixtures differ along it, in mean or in variance, near where frames
  fall. High-variance, high-mean-difference dimensions dominate. c0 has by far the widest range in our data (about -70 for digital silence against about -4 for the louder half of speech frames, §5), so a
  level difference is cheap for the GMM to use [hyp]. A cue that exists only as a *temporal* pattern longer than ±2 frames, or in phase, **cannot** matter; the mean over t also
  means a rare, extreme frame class (digital silence, c0 about -70) can move the clip score as much as many ordinary frames.
- **Why no CMVN matters here.** Zhou et al.'s reverberation argument relies on cepstral mean subtraction (`linear-vs-mel-cepstral-speaker-recognition`); B02 has none, so
  channel, level and room differences are available to the GMM in every coefficient.

## Implications for us

### §4 (c) Why high frequencies might matter for spoofing: sourced mechanisms, and what our data says
**4.1 Mechanisms from the literature** (all [lit]; none is established as the cause in LA 2019):
- *Missing or smoothed fine spectral detail.* Sahidullah et al. 2015 p.2: synthesis "focus on producing identical overall envelope". This is the authors' stated
  intuition; their experiment shows the *ordering* in item 1, not the cause. [lit, M]
- *Vocoders reproduce high frequencies imperfectly.* WaveFake p.6: differences "especially in the higher frequencies"; inaudible when listening; MelGAN and WaveGlow
  "seem to cause a repeating horizontal pattern". [lit, H] Vocoder-specific: MB-MelGAN and PWG detectors also use low frequencies (p.9). [lit, H]
- *Algorithms that do not model the top band.* Sriskandaraja 2016 p.3: the 7-8 kHz sub-band "may reflect the nature of most SS and VC algorithms that do not model this frequency region". [lit, M, "may"]
- *Pitch and dynamics.* WaveFake p.6: "all vocoders produce a lower and less varied pitch". Sahidullah p.1: deltas help because temporal characteristics are "not fully" modelled; their Table 3
  has Δ+Δ² alone beating static features for LFCC (0.12 vs 2.46). [lit, H on numbers]
- *Phase and aperiodicity.* Table 1 of the 2019 database paper names **WORLD** (A02, A03, A05, A07) and lists band-aperiodicity (BAP/AP) and MCC/F0 as the
  parametric outputs, and **Griffin-Lim** (A11) for waveform generation. That these choices leave a *high-band or phase* artefact is **[hyp]**: the paper states the
  architecture, not the artefact. LFCC discards phase, so a pure phase artefact is invisible to it (handoff §6.1 makes the same point).
- *Band limits.* 2019 LA bona fide speech is VCTK downsampled from 96 kHz to 16 kHz (Wang et al., §2). Whether individual spoof systems were trained or run at a lower
  bandwidth, leaving the 4-8 kHz region empty or interpolated, is **[hyp]**; it was not checked in the database paper's text, and the "band-limiting above about 4 kHz" idea has **no source in this note**.

**4.2 What our EDA supports** [recomp, M]
- Item 11: after removing overall level, the 4.9-7.6 kHz channels sit **1.5 dB lower relative to the 1.1-4.2 kHz channels in spoof** (louder-half frames; -2.3 dB over all frames),
  consistent across train, dev and eval splits. This is the one statistic that agrees with a "relatively less high-band detail in synthetic speech" mechanism.
- Item 10: spoof has smaller frame-to-frame variation in the high-order cepstral coefficients (mean |Δc| d between -0.12 and -0.57 for c11-c19), which agrees with a "finer
  detail is smoother" reading and with Sahidullah's dynamics result. The static c16-c18 are also lower in spoof (d -0.4 to -0.63).

**4.3 What our EDA does not support**
- The committed `band_high_ratio` (energy share at >= 4 kHz) has d = -0.126 with a CI spanning zero and AUC 0.526 (item 7). On the 1 kHz-wide bands of the notebook-02 sample, only
  4-5 kHz differs reliably (d = -0.27 [-0.53, -0.03], **recomputed**, Appendix R1); 5-8 kHz shows small positive, non-significant offsets. **There is no large, direct
  high-band energy difference in the pooled data.**
- Item 10: the largest per-channel effect is at the **low end** (channel 1, d = 0.94; mel channel 4, d = 0.95 in the same recomputation), and all channels are positive. This is the
  signature of a *level/silence* difference (spoof has less silence, so its frame-mean log-energy is higher), not of a high-band artefact. T in item 11 correlates 0.57 with the silence
  fraction, and the louder-half version is about 33 % smaller than the all-frames version, so **part of the tilt effect is silence**.
- The strongest committed cues, bandwidth (d -1.32) and rolloff (d -0.93), are whole-spectrum descriptors. Their link to silence (silent frames have flat noise-floor spectra,
  so large bandwidth) is **[hyp]**; test: recompute d on clips with total silence fraction below 0.2.
- Pooling hides reversals: per-attack mean centroid ranges from 1400 Hz (A11) to 2292 Hz (A18) around bona fide's 1733 Hz (notebook 02 cell 19, [proj]), so the pooled tilt is an average over
  generators of opposite sign. No per-attack tilt was computed in this pass.

**4.4 What follows** [calc/lit, M]. The literature gives a consistent direction (more high-band resolution, lower EER, on 2015 and WaveFake corpora; but a counterexample on SAS), and our
own data gives a weak, partly silence-confounded signature of high-band deficit. Neither shows that A1's 9.56 % EER depends on the spacing. "LFCC rather than MFCC" is the B02 recipe, adopted
for reproduction, not a finding of this project.

### §5 (e) Explaining A1's per-attack pattern under silence trimming (F3)
Evidence assembled here:
- Per-attack leading-silence fraction, median [recomp, Appendix R3, 150 clips per attack]: bona fide **0.29**; A07-A16 between 0.00 and 0.06; A17 **0.32**, A18 **0.30**, A19 **0.27**.
  This matches handoff §3's summary and Müller et al.'s mechanism (`leakage-audit`: TTS/VC systems built from trimmed data start speech immediately). By Table 1 of the database paper, A17-A19 are
  the three VC systems with **human** source speech, which explains why their silence resembles bona fide; A13-A15 are VC from TTS output.
- Median A1 score per attack, raw and trimmed, from the committed score CSVs [proj, H] (higher = more bona fide). Bona fide: **+1.50 -> +1.19 (shift -0.31)**.
- Frame-level silence depth, from the B02 c0 [recomp, Appendix R3]: median c0 of the quietest 10 % of frames, and the fraction of frames that are exactly digital zero.
- Trimming (`trim_silence`, `audio.py` L227-285) removes **leading and trailing** frames more than 30 dB below the loudest frame only; interior pauses stay.

| attack | generator (database paper Table 1) | lead. silence | A1 EER raw -> trimmed [proj] | median score raw -> trimmed (shift) | median gap to bona fide raw -> trimmed | quietest-10 % c0 | exact-zero frames |
|---|---|---|---|---|---|---|---|
| bona fide | VCTK | 0.29 | | +1.50 -> +1.19 (-0.31) | | -20.4 | 0 |
| A07 | WORLD + GAN | 0.01 | 15.4 -> 27.3 | +0.71 -> +0.71 (0.00) | 0.78 -> 0.48 | -17.6 | 0 |
| A08 | neural source-filter | 0.06 | 0.3 -> 0.7 | -1.65 -> -1.83 (-0.19) | 3.14 -> 3.02 | -23.8 | 0 |
| A09 | Vocaine | 0.01 | 0.0 -> 0.2 | -8.38 -> -4.18 (+4.20) | 9.88 -> 5.37 | -33.8 | **5.0 %** |
| A10 | WaveRNN | 0.00 | 22.4 -> 36.7 | +0.93 -> +0.92 (-0.01) | 0.57 -> 0.27 | -17.0 | 0 |
| A11 | Griffin-Lim | 0.00 | 0.1 -> 1.5 | -1.19 -> -0.94 (+0.25) | 2.68 -> 2.13 | -16.7 | 0 |
| A12 | WaveNet (TTS) | 0.00 | 7.6 -> 36.0 | -0.70 -> +0.92 (**+1.62**) | 2.20 -> 0.27 | **-30.9** | 1.7 % |
| A13 | waveform filtering (TTS source) | 0.00 | 12.8 -> 24.9 | +0.53 -> +0.61 (+0.08) | 0.96 -> 0.58 | -14.8 | 0 |
| A14 | STRAIGHT (TTS source) | 0.01 | 1.1 -> 3.7 | -0.57 -> -0.68 (-0.11) | 2.07 -> 1.87 | -17.4 | 0 |
| A15 | WaveNet (TTS source) | 0.01 | 2.8 -> 6.1 | -0.03 -> -0.17 (-0.14) | 1.53 -> 1.36 | -19.9 | 0 |
| A16 | waveform concat. (= A04) | 0.01 | 7.8 -> 31.5 | -0.71 -> +0.78 (**+1.49**) | 2.21 -> 0.41 | **-24.4** | 0 |
| A17 | waveform filtering (human source) | **0.32** | 8.8 -> **3.6** | +0.33 -> -0.60 (**-0.92**) | 1.17 -> 1.78 | -19.9 | 0 |
| A18 | MFCC vocoder (human source) | 0.30 | 4.4 -> **20.0** | +0.43 -> +0.59 (+0.15) | 1.06 -> 0.60 | -18.3 | 0 |
| A19 | spectral filtering + OLA (= A06) | 0.27 | 12.7 -> 11.2 | +0.68 -> +0.17 (-0.52) | 0.81 -> 1.02 | -19.3 | 0 |

For comparison, the training spoofs [recomp, 100 clips each]: quietest-10 % c0 is -19.8 (A01), -22.2 (A02), **-25.8 (A03)**, **-30.2 (A04)**, -20.9 (A05), -17.8 (A06) vs bona fide -20.3.

**Three mechanisms consistent with the table** (each [hyp] at the level of "why", but the shifts themselves are [proj]):
- **M1 - the shared bona fide pool moves.** Trimming lowers the bona fide median by 0.31 (the silence frames carry a positive LLR, handoff §6.4). Per-attack EER uses the *same* bona fide pool
  (`eer-and-metrics`), so every attack whose own scores do not move loses separation. A07, A10, A13 have the smallest raw gaps (0.78, 0.57, 0.96) and their EER roughly doubles
  (15.4 -> 27.3, 22.4 -> 36.7, 12.8 -> 24.9). A08, A14, A15 have wide gaps (3.1, 2.1, 1.5) and stay low in absolute terms. This explains A07, A10, A13 with no change in the spoof scores.
- **M2 - some spoofs carry unnaturally quiet frames that A1 reads as spoof evidence; trimming removes them.** The largest changes in EER (A12 +28.4 points, A16 +23.7) come with the
  largest *rises* in the spoof score (+1.62, +1.49), not with a fall in the bona fide score. Both attacks have the deepest quiet frames in the table (A12 -30.9, A16 -24.4 vs bona fide -20.4), A12 also has
  1.7 % digital-zero frames, and A09 (5 % digital zero, c0 = log10(ε)·√20 = **-70.0** [calc]) has the largest rise (+4.20). The spoof GMM was trained on A01-A06, which include deep-quiet attacks
  (A04 -30.2, A03 -25.8), and **A16 is the same algorithm as A04**, so it is plausible that A1 learned "very low c0 means spoof" and A16 inherited it. Digital silence has L = log10(ε) in all 20 channels, hence c0 = -70 and c_q = 0 for q >= 1, a single point
  that a diagonal Gaussian with the variance floor can sit on, giving those frames a very large spoof log-likelihood [hyp]. **Counterexample inside the data:** A08's quietest decile (-23.8) is as deep as A16's
  yet its median score moves only -0.19. So depth alone is not a sufficient rule.
- **M3 - spoofs that inherit bona fide-like silence become easier when it is trimmed.** A17 and A19 have bona fide-like leading silence and quietest-decile c0, and their scores fall more than the bona fide median
  (-0.92 and -0.52 vs -0.31), so the gap widens (1.17 -> 1.78, 0.81 -> 1.02) and EER improves (8.8 -> 3.6, 12.7 -> 11.2). Reading: their silence frames earn a bona-fide-like positive LLR; once removed, only spoof-like speech frames remain.

**The exception: A18.** It has bona fide-like silence (0.30, c0 -18.3) yet its EER worsens most among the human-source attacks (4.4 -> 20.0), because its median score *rises* (+0.15) while bona fide falls (-0.31).
M1 and M3 do not predict this, and M3 predicts the opposite. **No explanation is supported.** Candidates [hyp]: trimmed frames in A18 include edge or onset frames that were strongly spoof-like; its MFCC vocoder produces
a distinctive onset; the 30 dB trim threshold relative to the loudest frame behaves differently on its level distribution. Test: frame-level LLR by position (§6.2, X3).

**What LFCC encodes about silence frames** [calc/hyp]. A low-level frame has low c0 (bona fide quietest decile -20 vs the louder-half median -4.4 [recomp]), and the shape coefficients c1..c19 of a noise-floor frame are essentially noise-floor shape,
gain-free. For white noise, the pre-emphasised spectrum is shaped by |H|^2, which rises by about 36 dB from DC to Nyquist, so such frames sit at a strongly positive high-minus-low tilt [calc]; real room noise is coloured, so this is an upper-bound picture [hyp].
The 20 ms window with 50 % overlap and Δ over ±1 frame mean that a silent stretch also has Δ about 0, i.e. a distinct (c, Δ, Δ²) cluster. All of this is available to a GMM with no CMVN.

**Evidence versus hypothesis, summarised.** Measured: the per-attack score shifts, the silence depths, the leading-silence medians, the generator types, and A16 = A04. Interpreted: M1-M3. Unexplained: A18, and why A08 does not follow M2.
Not tested: that the zero or deep frames sit at the clip ends (so are removed by trimming) rather than inside; that the GMMs assign those frames the extreme LLRs described.

## §6 (f) Explainability analyses we can add cheaply (specifications only, none has been run)
All use existing models and caches. Expectations are written **before** any result so surprises stay visible.

### X1 Band occlusion of filterbank channels on A1 (scoring only, no retraining)
- *Method.* For channel groups G in {1-4 (0-1.5 kHz), 5-8, 9-12, 13-16, 17-20 (6.1-8 kHz)} replace L[m,r], r in G, by (variant a) the bona fide training per-channel mean, (variant b) a straight line interpolated from
  the neighbouring channels; recompute DCT, Δ, Δ², score with the trained E1 GMMs. Report ΔEER (eval raw, eval trimmed) and the median LLR shift per attack.
- *Expectation.* Occluding channels 1-4 hurts most (largest EER in-domain, since item 10 puts the largest effect there). Occluding the top group hurts less than an equal-sized mid group on pooled eval, but A12/A16/A09 are unaffected (their cue is c0).
  If the occluded top group shifts A07, A10, A13 more than others, that supports a high-band story for the no-silence attacks.
- *Caveat.* Replacement makes off-distribution inputs; variant (b) reduces but does not remove this. The clean version is **retraining without the group** (5 trainings at about 20.6 min each, handoff §10.1).

### X2 Per-coefficient LLR contribution (scoring only)
- *Method.* The exact decomposition in §3: compute s_d(x_t) and the remainder for every frame of a stratified clip sample, average per clip, and group by static / Δ / Δ² and by coefficient index. Compare bona fide vs each attack,
  raw vs trimmed.
- *Expectation.* c0, Δc0 and Δ²c0 carry most of the trimmed-minus-raw LLR shift for A09, A12, A16 (more than half). For A07/A10/A13 the share of c1-c19 is larger. If instead the high-order c16-c18 carry a share on the no-silence attacks, item 10's
  coefficient pattern is mechanistic.

### X3 Frame-level LLR against frame energy and position (scoring only)
- *Method.* Bin frames by c0 decile; plot mean frame LLR per decile for bona fide and per attack; also by position (first 10 frames, last 10 frames, interior) and for exact-zero frames.
- *Expectation.* Monotone: bona fide noise-floor frames (c0 about -20) have positive LLR; frames deeper than about -28 have large negative LLR; exact-zero frames the most negative. For A18, edge frames are the place to look.
- *Cost.* One pass of the committed GMM over about 100 clips per attack; minutes.

### X4 Controlled LFCC-versus-MFCC arm of A1 (the measurement this project lacks)
- *Spec.* Identical pipeline (pre-emphasis 0.97, Hamming 20 ms/10 ms, 512 FFT, log10 + ε, orthonormal DCT, 20 coefficients incl. c0, Δ/Δ², no CMVN, 2 x 512 diagonal GMMs, 10 EM iterations,
  same random-init seeds) with **only** the edge placement changed to `features.filterbank(spacing="mel", fmin=0, fmax=8000)` (the B02 function is linear-only, so the implementer must add a spacing option and keep the existing test path).
  Run 3 EM seeds per arm; report eval EER raw/trimmed, E3 In-the-Wild, per-attack EER, with bootstrap CIs over clips. Optional third arm: inverted-mel (Sahidullah's IMFCC) to test the monotone high-band-resolution claim.
- *Expectation (written now).* Pooled eval EER differs by under 4 points; probability about 65 % that LFCC is better. In-the-Wild: both near chance. Under trimming both arms degrade by a similar factor on A09, A12, A16, because their cue is c0, not spacing.
  If MFCC is *better* on A07/A10/A13, that echoes Sriskandaraja's SAS result.
- *Cost.* About 20.6 min per training run (E1 A1, handoff §10.1), so about 2 h for 6 runs.

### X5 Δ ablation and pre-emphasis ablation (A1)
- *Method.* Drop Δ/Δ² (60 -> 20 dimensions), and separately turn off pre-emphasis; the playground already exposes `use_deltas` (notebook 09, cell 19).
- *Expectation.* Without Δ the EER rises (Sahidullah: dynamics are the strongest part for LFCC on 2015); without pre-emphasis the top channels carry 3-6 dB less relative weight (§1), and the high-band tilt cue weakens.

### X6 A2 layer-weight interpretation (head-only, seconds each)
- *Fact.* `experiments/E-20261001-e1-a2-raw/layer_weights.json` [proj, H]: index 0 (embedding output) 0.094, then 0.081 0.095 0.099 0.080 0.081 0.074 0.066 0.061 0.066 0.068 0.067 0.069 for layers 1-12. Uniform would be 1/13 = 0.077. Max/min = 1.63.
- *Warning.* These are not importances: they multiply layer states of different scale and are highly correlated; the head used 10 epochs from a uniform start (handoff §7.4), and LayerNorm after the sum partly renormalises scale.
- *Methods.* (i) 13 single-layer heads from the existing pooled cache; (ii) leave-one-layer-out (weight 0, renormalise) at scoring; (iii) per-attack per-layer EER heatmap raw and trimmed; (iv) a probe from each layer to the clip's leading-silence fraction, to see which layers encode silence.
- *Expectation.* Single-layer EER lowest at layers 1-4 and worst at 8-12; the layers that encode silence best are also the ones the head already weights most; trimmed EER is worst for the layer set that predicts silence best. If layer 3 is not near the best single layer, the weights are not reflecting usefulness.

## Open questions / what would change this
- X4 would turn §4.4 from literature into measurement. Until then, handoff §5.1 should say LFCC was **adopted from B02** and that comparative evidence is literature (Sahidullah 2015 Table 3, WaveFake Tables 2 and 6), with the SAS counter-result.
- Where exact-zero and deep-quiet frames sit in each attack (ends or interior) decides whether M2 is the mechanism; X3 settles it.
- The Δ/Δ² pooled-tilt results are from one 600-clip sample, pooled over splits and attacks, with no multiple-comparison control; treat as descriptive. A per-attack version is needed before saying which generators carry the high-band deficit.
- The 2019 database paper was read for generator descriptions only; its text does not say anything about spoof-system bandwidth, so the band-limit idea has no source here.
- DOI of Zhou et al. is from PDF metadata; see that note.

## Appendix: how the [recomp] and [calc] numbers were produced (not committed; reproducible at HEAD d6f123f)
- **R1 (notebook 02 sample).** `data.build_manifest` over `configs/data.yaml`; stratified sample 40 per (split, label), `np.random.default_rng(0)`, drawn exactly as notebook 02 cell 3 (240 clips, 120 per class). Per-clip mean power spectrum from `features.stft_power`
  with the notebook-02 STFT config. Band energy = sum of mean-power bins in [lo, hi) kHz, then log10; Cohen's d via `stats.cohens_d`, bootstrap CI 500 resamples. Bands (kHz) d [CI]: 0-0.5 +0.18 [-0.06, +0.42]; 0.5-1 +0.25 [-0.00, +0.50]; 1-2 +0.13 [-0.12, +0.38]; 2-3 -0.22 [-0.51, +0.05]; 3-4 -0.02 [-0.28, +0.22]; 4-5 -0.27 [-0.53, -0.03]; 5-6 +0.09 [-0.15, +0.33]; 6-7 +0.20 [-0.05, +0.43]; 7-8 +0.17 [-0.09, +0.44].
- **R2 (600-clip B02-pipeline sample).** 100 clips per (split, label), `default_rng(7)`. For each clip: `lfilter([1,-0.97],[1],x)`, `_frame_signal(·, 320, 160)`, `np.hamming(320)`, `rfft(n=512)`, power, `lfcc_b02_filterbank` (linear) or `features.filterbank(spacing="mel")`, `log10(E + ε)`, orthonormal DCT; whole-clip frame means; `stats.per_bin_effect_size` (spoof minus bona fide). 300 vs 300 clips, so standard error about 0.08. Tilt T: mean of channels 13-20 minus channels 3-11 of the per-clip mean log10 energy (x10 for dB); "louder half" = frames with c0-sum at or above the clip median. Mel per-channel d: 0.36 0.74 0.90 0.95 0.83 0.59 0.42 0.43 0.46 0.50 0.49 0.54 0.59 0.59 0.55 0.47 0.24 0.17 0.18 0.22.
- **R3 (silence statistics).** Eval manifest, `default_rng`-free `DataFrame.sample(random_state=...)`: 150 clips per attack (`random_state=11`) for leading and total silence fractions via `audio.silence_ratio`; 100 clips per attack (`random_state=5`) for frame statistics: exact-zero frame = all 320 samples equal 0 in `_frame_signal`; c0 from `extract_lfcc_b02`; "quietest 10 %" = median over clips of the median c0 of each clip's lowest-decile frames. Train split, same recipe, for A01-A06.
- **R4 (score shifts).** Median per attack of `experiments/E-20261001-e1-a1-raw/scores/eval.csv` and `experiments/E-20261001-e1-a1-trimmed/scores/eval.csv` joined on `clip_id` (71,237 rows each).
- **Filterbank table.** `lfcc_b02_filterbank(20,512,16000)` and `filterbank(20,512,16000,spacing="mel")`; centre = middle edge; bandwidth = outer edge difference; bins = count of non-zero weights.
