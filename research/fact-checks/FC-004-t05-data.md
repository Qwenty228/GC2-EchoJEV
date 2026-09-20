---
id: FC-004-t05-data
task: T05
claim_by: T05 implementation agent (data.py claim block)
verdict: confirmed
checked_on: 2026-09-20
---

# FC-004 — T05's ASVspoof 2019 LA protocol column order, and its self-flagged directory-layout assumption, are both correct

## Claim as stated
> "The ASVspoof 2019 LA protocol column order as parsed matches the official file format":
> 5 whitespace-separated fields per line, `[speaker_id, utt_id, unused("-"), attack_id,
> label]`, bonafide rows keep the literal `"-"` in `attack_id`. Separately, self-flagged as
> an *unverified* assumption: on-disk audio layout is
> `<root>/ASVspoof2019_LA_<split>/flac/<utt_id>.flac`.

## Evidence examined
- code: `src/audio_df/data.py:75-102` (`_PROTOCOL_N_FIELDS`, `_PROTOCOL_FIELD_INDEX`,
  `_split_protocol_line`), `data.py:113-135` (`parse_asvspoof_protocol`), `data.py:138-152`
  (`_resolve_audio_path`), and `build_manifest`'s `split_audio_dir = root /
  f"ASVspoof2019_LA_{split}" / "flac"` construction.
- knowledge: `research/knowledge/datasets/asvspoof-2019-la.md` (this researcher's note),
  §"Protocol file format" and the newly added §"On-disk directory layout".
- source (protocol column order): `github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/data_utils.py`
  (`genSpoof_list`: `_, key, _, _, label = line.strip().split(' ')`) and
  `github.com/clovaai/aasist/blob/main/data_utils.py` (byte-identical parsing pattern) — two
  independent official/near-official parsers of the real 2019 LA protocol files, read
  directly.
- source (directory layout): the same two repos' `__getitem__` methods
  (`base_dir + 'flac/' + key + '.flac'` and `base_dir / f"flac/{key}.flac"` respectively),
  plus `clovaai/aasist/main.py`'s `get_loader`, which constructs
  `database_path / "ASVspoof2019_{track}_{train,dev,eval}/"` as each split's `base_dir`.

## Verdict
**Confirmed**, on both the protocol column-order claim and the directory-layout assumption
the implementing agent itself flagged as unverified.

## Reasoning
- **Protocol format.** `_PROTOCOL_FIELD_INDEX` in `data.py` (speaker_id=0, utt_id=1,
  attack_id=3, label=4, index 2 unused) matches field-for-field the parsing performed by two
  independent official/near-official baseline loaders on the real protocol files. This is not
  a case of the implementing agent's convention merely being *plausible* — it is byte-for-byte
  the same field-splitting logic (`line.split()` into exactly 5 fields) as the ASVspoof
  challenge organization's own 2021 RawNet2 baseline.
- **Directory layout.** The implementing agent correctly identified this as a separate,
  independently-fixable assumption and isolated it in one function
  (`_resolve_audio_path`) rather than entangling it with the protocol-parsing logic — good
  practice that made this fact-check straightforward. The assumption
  (`<root>/ASVspoof2019_LA_<split>/flac/<utt_id>.flac`) is exactly what two independent
  official-baseline data loaders construct and read from, composing the per-split directory
  name (`ASVspoof2019_LA_train`/`_dev`/`_eval`, confirmed from `aasist/main.py`'s
  `get_loader`) with a fixed `flac/` subdirectory and `.flac` extension. No code change
  required.

Both claims check out against sources external to this project and external to the
implementing agent's own reasoning, which is what "confirmed" requires here — the agent's
confidence in its own code was not the basis for this verdict.

## Required action
Nothing. Both the protocol format and the directory-layout assumption stand as implemented.
