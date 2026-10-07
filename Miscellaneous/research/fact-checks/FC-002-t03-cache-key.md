---
id: FC-002-t03-cache-key
task: T03
claim_by: T03 implementation agent (cache.py claim block)
verdict: contradicted
checked_on: 2026-09-20
---

# FC-002 — T03's `cache_key` collided a fractional float with the equal-looking string, and separately made a birthday-bound claim about 128-bit truncation

## Claim as stated
> `cache_key(utt_id, config)` produces a stable, collision-resistant hash over
> `(utt_id, config)` such that two configs differing only in cosmetic representation
> (dict-key order, int-vs-float for an unchanged numeric value) hash identically, while any
> substantive difference — including a change buried in a nested dict, and including a
> difference in value *type* — hashes differently. Separately: 128-bit (32-hex-char) SHA-256
> truncation is safe at this project's scale (thousands of utterances × a handful of feature
> variants), with collision probability astronomically below any other source of pipeline
> error.

## Evidence examined
- code (as originally submitted, before correction): `src/audio_df/features/cache.py`, canonicalisation
  logic — converted every scalar leaf via `repr(value)`-style string serialisation without a
  type tag, so a fractional float `20.5` and the string `"20.5"` both canonicalised to the
  JSON string `"20.5"` and therefore hashed identically.
- code (current, corrected, verified in this pass): `src/audio_df/features/cache.py:41-108`
  (`_canonicalise`), now encodes every scalar leaf as a type-tagged 2-element
  `["n"|"s"|"b"|"z", payload]` pair — `["n", "20.5"]` for the float, `["s", "20.5"]` for the
  string — making the collision structurally impossible (different first element).
  `cache_key` itself: `src/audio_df/features/cache.py:110-125`, SHA-256 of the canonical JSON, truncated
  to 32 hex chars (128 bits).
- probe (performed by the coordinator, reported to the researcher as settled): re-ran a
  cross-type probe after the fix — all cross-type cases (`20` vs `20.0` vs `"20"` vs `"20.0"`
  vs `True`/`1`, `None`/`"None"`/`"null"`) now produce distinct keys except the one pair the
  design *intends* to unify (`20 == 20.0`, both `["n", 20]`), which still collides
  deliberately and correctly.
- birthday-bound arithmetic (checked independently by the coordinator): confirmed —
  128-bit truncation gives a collision probability on the order of 10⁻²⁹ at 10⁵ distinct keys
  (standard birthday approximation `n²/2^(b+1)` with `n=10⁵`, `b=128`), which is indeed
  astronomically below any other source of error in this pipeline (disk corruption, config
  drift, human error in re-running an experiment).

## Verdict
**Contradicted** (as originally submitted) on the type-collision claim — the implementation
did not actually guarantee "any substantive difference... hashes differently," since a
float/string type difference at the same textual value was a substantive difference that
collided. **Confirmed** on the 128-bit birthday-bound claim — the probability figure is
correct and the truncation choice is sound at this project's scale. The type-collision defect
has since been fixed (type-tagged canonicalisation) and independently re-verified; the module
as it now stands satisfies the original claim.

## Reasoning
A cache keyed on a config hash is only safe if "same key implies same features" and "same
features imply same key" both hold. The pre-fix collision broke the second direction: a
config with `n_coeff: 20.5` (a float, plausible if some upstream code promotes an int
division result) and a hypothetical config with `n_coeff: "20.5"` (a string, plausible if a
YAML value was quoted, or a value round-tripped through a different loader) would silently
share a cache entry despite being semantically different configs — one asks for a numeric
truncation count, the other for whatever downstream code does with a string in a field that
expects a number (likely an error, but possibly a silent Python coercion). A shared cache
entry in that situation returns whichever config wrote first, for both callers, with no error
raised — exactly the "stale cache returns features computed under an entirely different
config" failure mode the module's own docstring says it exists to prevent. This is a real,
plausible collision mode for this project specifically, because `configs/features.yaml`'s
`n_coeff` field is sometimes `null` and sometimes an integer across variants (A–F), and this
is exactly the kind of value that could arrive as a float or a string depending on how a
future caller constructs the config dict passed to `extract()`/`cache_key()`.

The birthday-bound claim is a separate, independently checkable mathematical claim (not
contingent on the collision bug above) and holds regardless of the type-tagging fix: 128 bits
of hash space is enormously more than sufficient headroom for `O(10^5)` keys, so truncating
SHA-256 to 32 hex characters was never the risk here — the risk was entirely in what got
hashed (the canonicalisation step), not how much of the hash was kept.

## Required action
Already applied at the time of this fact-check: type-tagged canonicalisation
(`src/audio_df/features/cache.py:41-108`), independently re-probed and confirmed to separate all
cross-type cases while preserving the intended `20 == 20.0` unification. No further code
change required from this fact-check. No action required on the 128-bit truncation design —
it is confirmed correct as originally claimed.
