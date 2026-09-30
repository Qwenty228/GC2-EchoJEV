---
id: b02-lfcc-gmm
type: method
sources:
  - https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip (official ASVspoof 2019 baseline package, downloaded and inspected directly 2026-09-30: `ASVspoof2019_baseline_CM.m`, `LFCC/extract_lfcc.m`, `LFCC/deltas.m`, `GMM/compute_llk.m`, `Readme.txt`, `tDCF_v1/tDCF/compute_tDCF.m`, `tDCF_v1/compute_eer.m` — this is the primary source for every front-end/back-end parameter below)
  - https://ar5iv.labs.arxiv.org/html/1904.05441 (Todisco et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection", Table 1 — fetched directly, quoted verbatim below)
  - Sahidullah, Kinnunen, Hanilçi, "A Comparison of Features for Synthetic Speech Detection," Interspeech 2015 — cited by name inside `extract_lfcc.m` and `Readme.txt` as the LFCC recipe source; abstract confirmed via https://www.isca-archive.org/interspeech_2015/sahidullah15_interspeech.html, but the paper's own PDF text was not independently fetched (recipe verified instead directly from the code that implements it, which is the stronger source for exact parameter values)
  - https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-LFCC-GMM (ASVspoof 2021 baseline, `matlab/README.md` + `matlab/LFCC_GMM_ASVspoof_2021_baseline.m`, fetched directly — used only as a *contrast*, see Open questions; this is a **different, "high-frequency-resolution" LFCC recipe**, not the 2019 B02 recipe)
confidence: high
last_verified: 2026-09-30
related: [asvspoof-2019-la, lfcc-mfcc-dct, eer-and-metrics]
---

# B02 (LFCC-GMM) — the official ASVspoof 2019 baseline, exact parameters from the released code

## Claim in one sentence
The official ASVspoof 2019 B02 baseline (`ASVspoof2019_baseline_CM_v1.zip`, downloaded and read directly) uses a 20 ms Hamming-windowed, 10 ms-hop, 512-point-FFT, 20-linear-triangular-filter LFCC front-end spanning **0–8000 Hz** (not 30 Hz–8 kHz), with pre-emphasis, no CMVN, 19+0th DCT coefficients + delta + delta-delta (60-dim), scored by two independently-trained 512-component diagonal-covariance GMMs (bona fide vs spoof, train partition only, random-init EM, 10 iterations) via the mean per-frame log-likelihood-ratio over the whole utterance — and the user's working spec is correct on every point **except** the claimed 30 Hz lower frequency bound, which the actual code contradicts (the code uses 0 Hz, i.e. DC, as the lower filterbank edge).

## Findings

### Front-end — from `LFCC/extract_lfcc.m` (verbatim reading of the code)
- **Call signature used in the baseline driver** (`ASVspoof2019_baseline_CM.m` line 71/86/127):
  `[stat,delta,double_delta] = extract_lfcc(x, fs, 20, 512, 20)` — i.e. `Window_Length=20` (ms), `NFFT=512`, `No_Filter=20`.
- **Pre-emphasis: YES.** `speech = filter([1 -0.97], 1, speech);` — coefficient 0.97, applied unconditionally (not optional, no flag).
- **Window: Hamming.** `w = hamming(frame_length_inSample)`, applied multiplicatively to each frame.
- **Frame length / hop:** `frame_length_inSample = (Fs/1000)*Window_Length` = 320 samples at 16 kHz (= 20 ms). Hop is `buffer(speech, frame_length_inSample, frame_length_inSample/2, 'nodelay')`, i.e. hop = half the frame length = **10 ms (50% overlap)**. Confirms the user's 20 ms/10 ms spec exactly.
- **NFFT = 512**, magnitude-squared spectrum taken over the first `NFFT/2+1` = 257 bins (one-sided).
- **Filterbank: 20 linear-frequency triangular filters** (`trimf`), **frequency range = 0 Hz to `Fs/2` = 8000 Hz** — `f = (Fs/2)*linspace(0,1,NFFT/2+1)` then `filbandwidthsf = linspace(min(f), max(f), No_Filter+2)`, and `min(f) = 0`. **This contradicts the "30 Hz to 8 kHz" figure** that turned up in an unverified secondary summary during this research pass (a paraphrase, not a primary quote) — the code's own lower edge is DC (0 Hz), not 30 Hz. No source found (code, Readme, or the 2015 paper's title/abstract) states 30 Hz; treat that number as **incorrect** for the actual released B02 code.
- **Static coefficients: 20** (`t = dct(log10(filbanksum'+eps)); t = t(1:No_Filter,:)`) — takes the first `No_Filter=20` DCT-II outputs, i.e. the 0th cepstral coefficient (index 1 in MATLAB) plus the next 19, matching the Readme's own description: "20-channel linear filterbank, from which **19 coefficients + 0th**." There is no separate "log energy" appended — the 0th DCT coefficient (derived from the log filterbank-energy sum) plays that role; the user's phrasing "log energy... 19+0th=20 static" is functionally correct but should be read as "0th cepstral coefficient", not a distinct log-energy feature bolted on afterward.
- **DCT:** MATLAB's built-in `dct()` (DCT-II, orthonormal by default) applied along the filter axis to `log10(filterbank_energy + eps)`. Base-10 log, not natural log — worth flagging since `lfcc-mfcc-dct.md`'s general derivation should be checked against this if it assumes natural log.
- **Delta / delta-delta:** `deltas(stat', 3)` then `deltas(delta', 3)` — **window width `w=3`** (a 3-point simple linear-slope filter, explicitly passed as the second argument), not the 9-point default that `deltas.m`'s own docstring lists as its fallback. Static + delta + delta-delta = 20 + 20 + 20 = **60-dim**, confirming the user's arithmetic.
- **CMVN: NOT applied.** No mean/variance normalization step exists anywhere in `extract_lfcc.m`.
- **Whole utterance, no cropping.** `audioread(filePath)` reads the entire FLAC file; `extract_lfcc` is called on the full signal with no fixed-length crop/pad.

### Back-end — from `ASVspoof2019_baseline_CM.m` and `Readme.txt`
- **Two independent GMMs, 512 components each**, one trained on bona fide utterances, one on spoof utterances, both restricted to the **train partition only** (`ASVspoof2019.LA.cm.train.trn.txt`, `ASVspoof2019_LA_train/flac`) — the dev partition is used exclusively for scoring/evaluation in this script, never for GMM fitting.
- **Training call:** `vl_gmm(features, 512, 'verbose', 'MaxNumIterations', 10)` — VLFeat's GMM implementation, diagonal covariance (VLFeat `vl_gmm` is diagonal-covariance only), **EM with random initialisation**, **max 10 iterations** (both stated explicitly in `Readme.txt`: "512-component models, trained with an expectation-maximisation (EM) algorithm with random initialisation").
- **Scoring:** `llk_genuine = mean(compute_llk(x_fea, genuineGMM...))`, `llk_spoof = mean(compute_llk(x_fea, spoofGMM...))`, `score = llk_genuine - llk_spoof`. `compute_llk` = `logsumexp(lgmmprob(data, mu, sigma, w), 1)`, i.e. the total-mixture log-likelihood per frame; `mean(...)` averages this **across all frames of the utterance**. So the CM score is the **mean per-frame log-likelihood-ratio (bona fide GMM − spoof GMM), computed once per whole utterance** — confirms the user's "score = LLR" and rules out any per-segment/max-pooling variant.
- **Primary metric:** minimum normalized t-DCF, from `tDCF_v1/tDCF/compute_tDCF.m`: `tDCF = C1.*Pmiss_cm + C2.*Pfa_cm; tDCF_norm = tDCF./min(C1,C2)`, minimized over CM thresholds — this is the **2019-specific tandem t-DCF** (function of both CM and a fixed ASV system's error rates), a different quantity from ASVspoof 5's minDCF (see `min-dcf.md`); do not conflate the two normalizations.

### Published LA results — Table 1, arXiv:1904.05441 (fetched and read directly, ar5iv mirror)
- Table 1 has exactly **4 columns**: rank (`#`), system `ID`, `t-DCF`, `EER [%]` — **pooled over all attacks, eval set only**. There is **no separate Dev column** in this table; it is the challenge's eval-set leaderboard (50 rows: T01–T62 submitted systems plus B01/B02 baselines).
- **B02 (LFCC-GMM), LA, eval, pooled: min t-DCF = 0.2116, EER = 8.09%** — row 28 of 50, i.e. below the median submitted system.
- **B01 (CQCC-GMM), LA, eval, pooled: min t-DCF = 0.2366, EER = 9.57%** — row 33 of 50.
- **Dev-set EER/t-DCF for B02 was not found in this paper** — Table 1 in 1904.05441 reports eval only. Any dev-set number for B02 quoted elsewhere in this project must be traced to a different source (e.g. re-running the baseline code ourselves) before being used, not attributed to 1904.05441.
- PA scenario, for completeness (not requested but present in the same table): B01 t-DCF=0.2454/EER=11.04%, B02 t-DCF=0.3017/EER=13.54%.

## Mechanism / derivation
See `lfcc-mfcc-dct.md` for the general STFT→filterbank→log→DCT derivation; this note only pins down the *parameter values* the official code uses at each stage (window/hop/NFFT/filter count/range/static-coeff-count/delta-width/log-base), which the general note leaves as free variables.

## Implications for us
- **Adopt exactly:** 20 ms Hamming / 10 ms hop / 512-pt FFT / 20 linear triangular filters / 0–8000 Hz / pre-emphasis 0.97 / log10 / DCT-II orthonormal / 20 static (incl. 0th) + Δ + ΔΔ = 60-dim / no CMVN / delta window width 3 / 512-component diagonal-covariance GMMs per class / train-partition-only / random-init EM, 10 iterations / score = mean-per-frame LLR over the whole utterance, if the goal is a faithful B02 reproduction.
- **Correct the frequency floor:** the implementer's spec should say **0 Hz**, not 30 Hz, as the lower filterbank edge. This changes the two lowest triangular filters' shapes near DC and is a real (if probably small) numerical difference from what was planned.
- **Do not borrow the ASVspoof 2021 baseline's parameters by mistake.** The 2021 "Baseline-LFCC-GMM" repo (same EURECOM authors, same `compute_llk`/scoring logic) uses a *different* front-end — 30 ms window, 15 ms hop, 1024-pt FFT, 70 filters, 0–4000 Hz — explicitly labelled "high-frequency resolution LFCC" in its own README. It is a different, later baseline, not a restatement of B02. If any code in this repo cites "the ASVspoof baseline" without saying 2019 vs 2021, check which parameter set it actually uses.
- **Only eval-set B02 numbers (t-DCF 0.2116, EER 8.09%) are traceable to 1904.05441.** Any reported "B02 dev EER" needs its own citation (most likely: running the official code ourselves against the dev protocol, which the script supports natively) before it can appear in `reports/final_report.md`.
- **t-DCF ≠ ASVspoof 5 minDCF.** They share the letters "DCF" and a similar Pmiss/Pfa structure but different cost models, different normalizations, and t-DCF additionally depends on a fixed ASV system's error rates (Pfa_asv, Pmiss_asv, Pmiss_spoof_asv) that minDCF has no equivalent of. Never plot them on the same axis without saying so.

## Open questions / what would change this
- The Sahidullah et al. 2015 Interspeech paper's own PDF text was not independently fetched in this pass (only its abstract, via ISCA archive) — the code (`extract_lfcc.m`) is the primary source actually used above, since it is what participants ran; if the 2015 paper's own recipe differs from what EURECOM's 2018-dated `extract_lfcc.m` implements, that is a fact about the paper vs. the code, not about B02 itself, and does not change any claim in this note.
- VLFeat's `vl_gmm` covariance type was inferred as diagonal from general VLFeat documentation knowledge, not re-derived from the `.m` source in this pass (VLFeat GMM is a compiled MEX binary, not inspectable as MATLAB source in this package) — flagged **medium confidence** on "diagonal covariance" specifically, though it is very unlikely to be full covariance given 60-dim features × 512 components × ASVspoof-2019-train-scale data.
