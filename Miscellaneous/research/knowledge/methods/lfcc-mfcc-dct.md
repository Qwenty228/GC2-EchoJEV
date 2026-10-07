---
id: lfcc-mfcc-dct
type: method
sources:
  - the original project brief (§"Stage 2 - LFCC, MFCC, and DCT as controlled feature engineering")
  - https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html (DCT-II definition and norm="ortho" scaling, fetched directly)
confidence: high
last_verified: 2026-09-20
related: []
---

# LFCC/MFCC pipeline and the orthonormal DCT

## Claim in one sentence
The handoff's DCT formula `c[m,q] = Σ_r L[m,r] cos(πq/J (r+½))` is the *unnormalised*
DCT-II kernel; to make it match `scipy`'s (and any standard) orthonormal DCT-II, the q=0 row
must be scaled by `√(1/J)` and every q>0 row by `√(2/J)` — the handoff's bare formula as
written is **not** orthonormal, and code claiming "orthonormal DCT matching the handoff
formula" must apply this scaling explicitly rather than treating the formula as already
normalised.

## Findings

### Pipeline stages (handoff §Stage 2, verbatim structure)
```
x[n] -> STFT X[m,k] -> power P[m,k] -> filterbank E[m,r] -> log L[m,r] -> DCT c[m,q]
E[m,r] = Σ_k P[m,k] Ψ_r[k]
c[m,q] = Σ_r L[m,r] cos(πq/J (r+½))
```
where `R` = STFT hop, `J` = number of filterbank channels, `Ψ_r[k]` = filterbank weights.

### DCT-II definition and orthonormal scaling (scipy.fft.dct docs, fetched directly)
Standard ("backward"/unnormalised) DCT-II: `y_k = 2 Σ_{n=0}^{N-1} x_n cos(πk(2n+1)/(2N))`.
For `norm="ortho"`, scipy applies a per-row scale factor `f(k)` to this unnormalised sum so
that the resulting transform matrix is orthonormal (`O @ O.T = I`): `f(0) = √(1/(4N))` and
`f(k>0) = √(1/(2N))`, applied to the factor-of-2 unnormalised sum above. Working this through:
for `k=0`, `f(0)·2 = √(1/N)`; for `k>0`, `f(k)·2 = √(2/N)`. So the orthonormal DCT-II is:

```
c[q=0] =  √(1/J) · Σ_r L[r]
c[q>0] =  √(2/J) · Σ_r L[r] cos(πq/J (r+½))
```

The handoff's formula `c[m,q] = Σ_r L[m,r] cos(πq/J (r+½))` has **no leading scale factor at
all** — it is the raw (unnormalised) cosine-kernel sum, identical in structure to scipy's
`norm=None`/"backward" DCT-II divided by 2 (scipy's unnormalised form carries an explicit
leading `2`; the handoff's formula, and most classic speech-processing textbook presentations
of LFCC/MFCC, drop that leading constant since it is absorbed into whatever downstream scale
is used). Either way, the handoff's bare formula is **not orthonormal** as written; making it
orthonormal (invertible via a matching IDCT, energy-preserving) requires exactly the `√(1/J)`
/ `√(2/J)` row scaling derived above.

### What each stage discards (handoff, explicit)
- **Power** `P[m,k] = |X[m,k]|²` discards **phase** entirely — only magnitude information
  about the STFT bin survives past this point.
- **Filterbank pooling** `E[m,r] = Σ_k P[m,k]Ψ_r[k]` discards **fine FFT bin-level detail**,
  replacing `K` frequency bins with `J ≪ K` smoothed channel energies — any spectral feature
  narrower than a filterbank channel's bandwidth is averaged away and cannot be recovered.
- **Log compression** `L = log(E)` is a monotone, positive-domain transform — it reshapes
  dynamic range/geometry (compressing large energies, expanding small ones) but is invertible
  given `E`; it does not by itself discard information, only re-scales it.
- **DCT truncation** (`n_coeff < J`) discards information: dropping high-index DCT
  coefficients removes the components of `L[m,·]` that vary fastest across filterbank
  channels `r` — i.e. the *finer-grained shape* of the smoothed log-filterbank spectrum, not
  the original signal's fine spectral detail (which was already lost at the filterbank
  stage). A **full**, untruncated orthonormal DCT is exactly invertible (`IDCT(DCT(L)) = L`);
  truncating to fewer coefficients than `J` is the only genuinely lossy step in the DCT stage
  itself.

### Source-filter identity does not survive log-sum (handoff, explicit; math self-consistent)
The classical source-filter model gives the exact identity `log|X(f)| = log|G(f)| + log|H(f)|`
at each individual spectral frequency `f`, where `G` is the source (excitation) spectrum and
`H` is the filter (vocal-tract) response. This identity is a statement about `log` of a
**pointwise product** at a single frequency (`|X(f)| = |G(f)||H(f)|` pointwise, and
`log(ab) = log a + log b` exactly). The filterbank stage, however, computes
`L[m,r] = log(Σ_k P[m,k]Ψ_r[k]) = log(Σ_k |G[k]|²|H[k]|²Ψ_r[k])` — a **log of a sum** of
per-bin products across the channel's bandwidth, not a log of a single product. Because
`log(Σ a_i b_i) ≠ Σ log(a_i) + Σ log(b_i)` in general (log does not distribute over
summation), the clean additive source/filter decomposition is **broken** by the filterbank
pooling step. It survives only approximately, under the assumption that `H[k]` (or `G[k]`) is
roughly constant/smooth across the bins spanned by a single filter channel — i.e. the
filter's magnitude response varies slowly relative to filterbank channel bandwidth, so it can
be approximately pulled outside the sum. This is an approximation that degrades as filterbank
channels widen (fewer, broader channels) or as the underlying spectral envelope has sharp
structure within a channel's bandwidth.

### Historical DCT/GMM rationale (handoff, explicit)
The DCT was historically valuable because filterbank channel outputs `E[m,r]` (and hence
`L[m,r]`) are strongly correlated across adjacent channels `r`, and classical GMM
back-ends commonly used **diagonal covariance matrices** for tractability — a diagonal-
covariance Gaussian implicitly assumes feature dimensions are uncorrelated. Applying an
(approximately) decorrelating transform like the DCT before a diagonal-covariance GMM
compensates for that independence assumption. Modern neural network backends (CNN/ResNet,
attention-based) can model correlated, non-diagonal feature interactions directly, so the
DCT/no-DCT choice is now an **empirical representation choice** (does it help this specific
backend on this specific task) rather than a structural necessity imposed by the backend.

## Mechanism / derivation
Orthonormal DCT-II, in the notation used in `src/audio_df/`:
```
c[m, 0] = sqrt(1/J) * Σ_{r=0}^{J-1} L[m, r]
c[m, q] = sqrt(2/J) * Σ_{r=0}^{J-1} L[m, r] * cos( (π*q/J) * (r + 1/2) )     for q = 1..J-1
```
This is the transform whose matrix `O` (rows indexed by `q`, columns by `r`) satisfies
`O @ O.T = I`, i.e. `IDCT(DCT(L)) = L` exactly for the full (`n_coeff = J`) case, and whose
Parseval-style energy relation `Σ_r L[r]^2 = Σ_q c[q]^2` holds exactly — this is the concrete,
checkable property that should back the T06 test "orthonormal full DCT round-trips
(`idct(dct(L)) ≈ L`)."

## Implications for us

- **`tasks/TASKS.md` T06** requires `dct_coeffs(L, n_coeff=None, orthonormal=True)` to
  "match the handoff formula ... up to the stated normalisation — state which normalisation
  you used and why." Per this note, the handoff's literal formula is unnormalised; T06's
  implementation must apply the `√(1/J)` (q=0) / `√(2/J)` (q>0) scaling on top of the
  handoff's cosine kernel to satisfy `orthonormal=True` and to pass the round-trip test
  (`idct(dct(L)) ≈ L`). If T06 implements the handoff's formula verbatim with no scale factor
  and calls that "orthonormal," that claim is **incorrect** — flag as contradicted if seen in
  a T06 claim block. Equivalently, T06 can simply call `scipy.fft.dct(L, type=2, norm='ortho')`
  along the filterbank axis and rely on scipy's own orthonormalization, which implements
  precisely this scaling — this is almost certainly the simplest correct implementation path
  and avoids hand-deriving the constants.
- `configs/features.yaml` variants B (`lfcc_full`, `n_coeff: 20`) and E (`mfcc`, `n_coeff: 13`)
  use `dct: true`; whatever scaling `extract()` applies must be **consistent across all DCT
  variants** (B, C, E, F) — comparing "with DCT" vs "without DCT" (A, D) representations is
  only a fair ablation if the DCT itself doesn't introduce an arbitrary, unexplained gain
  scale relative to the log-filterbank features it's compared against.
- The source-filter caveat matters directly for any interpretive claim in the EDA/report
  that "LFCC coefficient q captures formant/pitch structure directly" — per this note, that
  claim is only approximately true and specifically degrades for wideband filters or sharp
  spectral structure; the report should hedge claims about what high-order LFCC coefficients
  "mean" physically, per the handoff's own explicit warning.

## Open questions / what would change this
- I verified the orthonormal scaling against scipy's own documentation (a standard, widely
  trusted reference for exactly this transform), not against an ASVspoof-specific or
  speech-processing-textbook primary source; if the project's `features.py` uses a different
  DCT convention (e.g. HTK-style or librosa's MFCC DCT, which by default also uses
  `scipy.fftpack.dct(type=2, norm='ortho')`), the same scaling applies and no correction is
  needed, but this hasn't been independently checked against librosa's source in this pass.
