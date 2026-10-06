---
id: min-dcf
type: method
sources:
  - https://arxiv.org/html/2601.03944v1 ("ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Systems" — fetched directly, Eq. 1 and surrounding text quoted below)
  - https://www.eurecom.fr/publication/7829/download/sec-publi-7829.pdf (attempted fetch; PDF text extraction failed — not usable as an independent source this pass, listed for completeness only)
  - https://www.asvspoof.org/asvspoof2019/ASVspoof_2019_baseline_CM_v1.zip, `tDCF_v1/tDCF/compute_tDCF.m` (contrast source — the older, unrelated ASVspoof2019 t-DCF; kept here only to prevent the two metrics being conflated, see `b02-lfcc-gmm.md`)
confidence: high
last_verified: 2026-09-30
related: [b02-lfcc-gmm, eer-and-metrics, asvspoof-5]
---

# ASVspoof 5 Track 1 minDCF — exact definition, parameters, and normalization

## Claim in one sentence
ASVspoof 5's Track 1 primary metric is `minDCF = min_τ [β·P_miss(τ) + P_fa(τ)]` where `β := C_miss(1−π_spf)/(C_fa·π_spf)`, with **C_miss=1, C_fa=10, π_spf=0.05 giving β≈1.90**; "miss" is a bona fide utterance wrongly rejected and "false alarm" is a spoof utterance wrongly accepted; the paper's own "DCF" formula is **already normalized by `C_fa·π_spf`** (not literally written as "divide by `min(C_miss(1−π), C_fa·π)`"), but for these specific parameter values the two prescriptions are numerically identical, because `C_fa·π_spf` (=0.5) happens to be the smaller of the two terms (`C_miss(1−π_spf)`=0.95).

## Findings

### The formula, quoted (arXiv:2601.03944v1, Eq. 1 and surrounding text, fetched directly)
- `DCF(τ_cm) = β · P_miss^cm(τ_cm) + P_fa^cm(τ_cm)` — this quantity, called simply "DCF" in the paper, is what gets minimized to define minDCF.
- `P_miss^cm` = **"the miss rate (false rejection rate of bona fide data)"** — i.e. a bona fide trial incorrectly rejected (called spoof). This is a **cost on rejecting real speech**, not on missing an attack.
- `P_fa^cm` = **"the false alarm rate (false acceptance rate of spoofed data)"** — i.e. a spoof trial incorrectly accepted (called bona fide).
- `β := C_miss(1−π_spf) / (C_fa·π_spf)`. The paper states: "Parameters for ASVspoof 5 give β≈1.90," and explains the interpretation directly: "missed detections of bona fide utterances are penalized nearly twice as much as false accepts" is *not* quite the right reading — rather, β≈1.9 means the **miss term is weighted ~1.9× relative to the false-alarm term** inside the already-normalized DCF, given the paper's own stated priors and costs.
- `minDCF := min_{τ_cm} DCF(τ_cm)` — "the value of the DCF at the threshold that minimizes [the DCF] for evaluation data," i.e. a minimum over all CM thresholds, exactly analogous to how ASVspoof2019's min t-DCF is a minimum over CM thresholds.
- **Parameter values** C_miss=1, C_fa=10, π_spf=0.05 were not found stated as a single verbatim sentence inside the fetched full-text extraction of 2601.03944v1 itself (the tool's text extraction returned the β-only form); they were corroborated via a WebSearch result citing the same paper's Track 1 parameters (see Open questions). **Arithmetic check:** β = C_miss(1−π)/(C_fa·π) = 1×0.95/(10×0.05) = 0.95/0.5 = **1.90** — exactly matches the paper's own stated β≈1.90, so the parameter values are internally consistent with the one number independently confirmed inside the paper's own text.

### Normalization — what the paper actually divides by, versus what a general NIST-style normalization would do
- The paper's `DCF(τ)` (Eq. 1) is **already the normalized quantity** — it is derived from a *raw*, un-normalized cost `raw_DCF(τ) = C_miss(1−π_spf)·P_miss(τ) + C_fa·π_spf·P_fa(τ)` divided through by `C_fa·π_spf`: dividing every term of `raw_DCF` by `C_fa·π_spf` gives `[C_miss(1−π_spf)/(C_fa·π_spf)]·P_miss + P_fa = β·P_miss + P_fa`, which is exactly Eq. 1. So **the paper's normalizer is `C_fa·π_spf`, unconditionally** — not "whichever of `C_miss(1−π_spf)` or `C_fa·π_spf` is smaller."
- **The implementer's stated plan** — `DCF(τ) = C_miss(1−π)P_miss(τ) + C_fa·π·P_fa(τ)`, normalized by `min(C_miss(1−π), C_fa·π)` — is the classical NIST SRE-style "normalized DCF" convention (dividing by the cost of the trivial always-accept/always-reject system). **For the official ASVspoof 5 parameter values this gives an identical numeric result to the paper's own formula**, because `min(C_miss(1−π_spf), C_fa·π_spf) = min(0.95, 0.5) = 0.5 = C_fa·π_spf` (the false-alarm term is the smaller one whenever β≥1, which it is here). Dividing by 0.5 in both cases yields the same `β·P_miss + P_fa` expression.
- **This equivalence is parameter-value-specific, not general.** If `C_miss`, `C_fa`, or `π_spf` were ever changed such that `β<1` (i.e. `C_miss(1−π) < C_fa·π`), the paper's rule ("always divide by `C_fa·π_spf`") and the implementer's general "divide by `min(...)`" rule would diverge — the paper's own Eq. 1 has no branch on which term is smaller, it fixes the denominator to `C_fa·π_spf` regardless. Code should therefore either (a) hardcode the ASVspoof5-specific normalizer `C_fa·π_spf` per the paper's actual Eq. 1, or (b) if implementing the general `min(...)` form intentionally, document that this is a *different, generalized* convention that happens to coincide with the paper's for the official parameters, and will silently diverge if the parameters ever change.

## Mechanism / derivation
```
raw_DCF(τ)   = C_miss·(1−π_spf)·P_miss(τ) + C_fa·π_spf·P_fa(τ)
DCF(τ)       = raw_DCF(τ) / (C_fa·π_spf)
             = β·P_miss(τ) + P_fa(τ),      β = C_miss(1−π_spf)/(C_fa·π_spf)
minDCF       = min_τ DCF(τ)
```
With C_miss=1, C_fa=10, π_spf=0.05: β = (1×0.95)/(10×0.05) = 1.90.
`min(C_miss(1−π_spf), C_fa·π_spf) = min(0.95, 0.5) = 0.5 = C_fa·π_spf` — so dividing `raw_DCF` by this `min(...)` term is arithmetically the same operation as dividing by `C_fa·π_spf` for these specific values, confirming the implementer's planned code will produce numerically correct minDCF values under the official ASVspoof 5 parameters, for the reason given above (not for the reason the implementer may have assumed — see the fact-check below).

## Implications for us
- Implement `DCF(τ) = β·P_miss(τ) + P_fa(τ)` directly with `β` computed from `C_miss=1, C_fa=10, π_spf=0.05` — this is literally the paper's Eq. 1 and avoids any ambiguity about which normalizer applies.
- If the `min(C_miss(1−π), C_fa·π)` form is kept for readability/generality, add a code comment noting it coincides with the paper's fixed `C_fa·π_spf` normalizer **only because β≥1 for the official parameters** — this protects against a silent bug if parameters are ever swept.
- `P_miss` = bona fide rejected, `P_fa` = spoof accepted — same polarity as ASVspoof2019's CM error naming, so no sign-flip risk relative to existing `eer-and-metrics.md` conventions, but this should still be asserted explicitly in code/tests since the words "miss" and "false alarm" are given a specific, non-obvious-to-a-newcomer meaning here (miss = reject a *bona fide*, not "miss an attack").

## Open questions / what would change this
- I could not get a single verbatim sentence out of the paper's full-text extraction stating "C_miss=1, C_fa=10, π_spf=0.05" together in one place — this is corroborated by (a) a WebSearch summary citing the same arXiv ID and (b) the arithmetic consistency with the paper's own stated β≈1.90, but a re-fetch with a different extraction tool (or downloading the PDF and grepping it directly) would raise this from "high confidence via consistency" to "directly quoted." Flagging so the implementer knows this number is corroborated, not verbatim-quoted.
- The EURECOM PDF mirror (`sec-publi-7829.pdf`) could not be read as text in this pass (compressed-stream extraction failure) — if a cleaner copy is found later, it should be checked for the same numbers as a second primary source.
