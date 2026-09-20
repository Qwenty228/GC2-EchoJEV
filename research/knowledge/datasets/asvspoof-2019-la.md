---
id: asvspoof-2019-la
type: dataset
sources:
  - https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf (ASVspoof 2019 Evaluation Plan, v0.4, 15 Jan 2019)
  - https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/data_utils.py (official baseline `genSpoof_list`, confirms protocol column order and `base_dir/flac/<key>.flac` path convention)
  - https://github.com/clovaai/aasist/blob/main/data_utils.py (independent confirmation of same column order/parsing)
  - https://github.com/clovaai/aasist/blob/main/main.py (official-style loader; confirms per-split directory naming `ASVspoof2019_{track}_{train,dev,eval}/`)
confidence: high
last_verified: 2026-09-20
related: [asvspoof-2021, eer-and-metrics]
---

# ASVspoof 2019 LA — protocol, splits, attacks

## Claim in one sentence
ASVspoof 2019 LA ships three disjoint-speaker partitions (train/dev/eval) with a fixed
5-field whitespace-separated protocol line format `SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY`;
train/dev share 6 seen attacks (A01–A06), eval contains 13 unseen attacks (A07–A19).

## Findings

### Split sizes (Evaluation Plan, Table 1 / Table 5, p.3 & p.13)
LA logical-access condition, non-overlapping speakers per subset:

| Subset | #speakers (M/F) | #bonafide | #spoof | #total utterances | protocol file |
|---|---|---|---|---|---|
| Training | 8/12 (20) | 2,580 | 22,800 | 25,380 | `ASVspoof2019.LA.cm.train.trn.txt` |
| Development | 8/12 (20) | 2,548 | 22,296 | 24,844 | `ASVspoof2019.LA.cm.dev.trl.txt` |
| Evaluation | not listed in Table 1 (utterance count omitted from public plan; "-" in Table 5) | — | — | — (external corroboration below) | `ASVspoof2019.LA.cm.eval.trl.txt` |

The evaluation plan's own Table 5 leaves the eval utterance count as "-" (not disclosed in
the plan text I could fetch). Independent, converging secondary reports (arXiv literature
search, multiple re-used in downstream papers) consistently cite the eval set as containing
71,237 trials in total, with 20 speakers in train and dev and 67 speakers in eval — but I did
not find this number inside an ASVspoof-hosted primary document, only in third-party papers
citing it. **Confidence on the exact eval utterance/speaker count is therefore medium, not
high** — flagged below as an open question.

Source data: all LA speech (train/dev/eval) is derived from the VCTK corpus (107 speakers:
46 male, 61 female), no significant channel/background noise, no speaker overlap across the
three subsets for target/adaptation speakers (Evaluation Plan §3, p.3).

### Protocol file format (CRITICAL for T05)
Confirmed independently from two official/near-official parsers (ASVspoof-challenge org's own
2021 RawNet2 baseline `data_utils.py`, and the widely used AASIST implementation, both of
which parse the *2019* LA protocol files identically):

```python
# whitespace-separated, 5 fields per line
_, key, _, _, label = line.strip().split(' ')
```

i.e. the field order is:

1. `SPEAKER_ID` — e.g. `LA_0079`
2. `AUDIO_FILE_NAME` (the trial/utterance ID, no extension) — e.g. `LA_T_1138215`
3. a fixed placeholder field, literal `-` in the LA train/dev/eval files (not used by either
   baseline parser; in the 2021 challenge's extended meta-label files the analogous position
   carries the codec/environment tag, but for the pure 2019 LA files it is `-`)
4. `SYSTEM_ID` — one of `A01`–`A19` for spoof trials, or `-` for bonafide trials
5. `KEY` — literal string `bonafide` or `spoof`

The eval protocol file (`ASVspoof2019.LA.cm.eval.trl.txt`) has the identical 5-column
format; only the labels are withheld to participants during the live challenge, not in the
released post-challenge protocol file used for reproducible research today.

### Attack IDs and seen/unseen split (Evaluation Plan §3.1–3.2, p.3–4)
- Training and development sets are generated with the **same set of 6 spoofing algorithms**
  (2 voice-conversion, 4 text-to-speech) — by convention labelled **A01–A06**. The dev set
  uses the *same* algorithms as train, not variants.
- The evaluation set uses **13 additional, different spoofing algorithms**, labelled
  **A07–A19**, which are described as "variants of the spoofing algorithms used to generate
  the development dataset" but are explicitly *not* the same algorithms — the plan states
  this is "intentionally different" so as to test generalisation to unseen attacks (§3.2, p.4).
- This A07–A19 numbering for the eval-only unseen attacks is corroborated independently by
  the AASIST evaluation script (`clovaai/aasist/evaluation.py`), which hard-codes
  `attack_types = [f'A{_id:02d}' for _id in range(7, 20)]` when building the per-attack EER
  breakdown for the LA eval set — i.e. the official baseline tooling itself assumes exactly
  A07 through A19 (13 attacks) appear in eval.
- Net effect: **A01–A06 are seen** (appear in train and dev only), **A07–A19 are unseen**
  (appear in eval only). No attack ID is shared between train/dev and eval.

### On-disk directory layout (CRITICAL for T05, independent fact from the protocol *file* format above)
This is a separate, independently-verifiable fact from the protocol file's column format
above, and must not be corrected or invalidated together with it if either turns out wrong.

Confirmed from two independent official/near-official loaders that both read the *real*
ASVspoof 2019 LA release (not a re-hosted mirror with a different layout):

- `asvspoof-challenge/2021`'s own RawNet2 baseline (`LA/Baseline-RawNet2/data_utils.py`):
  ```python
  X, fs = librosa.load(self.base_dir + 'flac/' + key + '.flac', sr=16000)
  ```
  i.e. audio path = `<base_dir>/flac/<utt_id>.flac`.
- `clovaai/aasist` (`data_utils.py`, byte-identical pattern using `soundfile`):
  ```python
  X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
  ```
- `clovaai/aasist`'s own `main.py` shows exactly how `base_dir` is constructed per split:
  ```python
  trn_database_path  = database_path / "ASVspoof2019_{}_train/".format(track)
  dev_database_path  = database_path / "ASVspoof2019_{}_dev/".format(track)
  eval_database_path = database_path / "ASVspoof2019_{}_eval/".format(track)
  ```
  where `track = "LA"` for this project and `database_path` is the dataset root.

Composed together, the confirmed on-disk layout for ASVspoof 2019 LA is:

```
<root>/ASVspoof2019_LA_train/flac/<utt_id>.flac
<root>/ASVspoof2019_LA_dev/flac/<utt_id>.flac
<root>/ASVspoof2019_LA_eval/flac/<utt_id>.flac
<root>/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{train.trn,dev.trl,eval.trl}.txt
```

This is a direct match for `src/audio_df/data.py:_resolve_audio_path`'s own stated assumption
(`<root>/ASVspoof2019_LA_<split>/flac/<utt_id>.flac`, `data.py:141-144`) and for
`build_manifest`'s construction of `split_audio_dir = root / f"ASVspoof2019_LA_{split}" /
"flac"` (`data.py:~180`) — **the assumption is correct as implemented, no fix required.**

A secondary, independently-searched source (third-party GitHub/HuggingFace re-hostings, not
independently verified against the ASVspoof-hosted archive itself) additionally states file
counts matching Table 5 above plus an eval count of 71,237 `.flac` files, consistent with
(though not itself proof of) the "71,237 trials" eval-size figure flagged as medium-confidence
in the split-sizes section above. This raises that figure's plausibility but does not move it
to high confidence, since it is still a third-party report, not an ASVspoof-hosted document
stating the number directly.

### Per-attack EER convention (from AASIST's official-style evaluation script)
```python
bona_cm = cm_scores[cm_keys == 'bonafide']                       # the FULL bonafide pool
spoof_cm_breakdown = {atk: cm_scores[cm_sources == atk] for atk in attack_types}
eer_cm_breakdown = {atk: compute_eer(bona_cm, spoof_cm_breakdown[atk])[0] for atk in attack_types}
```
Each attack's spoof scores are pooled against the **same, full bonafide score set** — not a
per-attack bonafide subset. See `research/knowledge/methods/eer-and-metrics.md` for the EER
mechanism itself; this note only establishes that the bonafide pool used in the per-attack
breakdown is the shared/full one, confirmed by reading the actual evaluation code.

## Mechanism / derivation
N/A (protocol/metadata note, not a formula).

## Implications for us

- `configs/data.yaml` and `tasks/TASKS.md` (T05) should parse the LA protocol files with
  exactly this 5-field format: columns
  `[speaker_id, utt_id, unused("-"), attack_id("-"/"A01".."A19"), label]`.
  `src/audio_df/data.py:parse_asvspoof_protocol` must **not** assume 4 columns — the `-`
  placeholder in position 3 is real and must be consumed/ignored, not treated as the
  attack_id column. Getting the column index wrong (e.g. treating column 3 as attack_id)
  would silently swap `-`/attack_id and corrupt every downstream leakage/attack-conditioned
  analysis.
- `configs/data.yaml:manifest.columns` includes `attack_id` — this must be populated from
  field 4 (0-indexed: index 3), and `"-"` for bonafide rows should be kept literally as `"-"`
  per the config's own comment ("`-` for bonafide"), which matches the official convention
  exactly — no translation needed.
- `holdout_attack_families` (T05) has a clean, literature-grounded partition to test against:
  holding out A07–A19 entirely (i.e., using the stock eval split) already *is* the canonical
  unseen-attack evaluation; a *custom* unseen-attack split must instead partition within
  train+dev (A01–A06) if the project wants an additional in-house generalisation test, since
  the official partition already exhausts the seen/unseen boundary at the train/dev vs eval
  line.
- T04's per-attack EER convention ("scores each spoof generator against the shared bonafide
  pool") is **confirmed correct** by this note — see the exact code excerpt above. This
  should be treated as settled, not merely "our reading."

## Open questions / what would change this
- Exact eval-set utterance/speaker counts (71,237 trials / 67 speakers is a widely repeated
  secondary-source figure, not one I could confirm inside an ASVspoof-hosted primary
  document in this pass). Confidence on this specific number: **medium**. This would be
  raised to high by fetching the actual `ASVspoof2019.LA.cm.eval.trl.txt` file line count
  directly (requires the gated dataset download, which this note-writing pass did not have
  access to) or finding an ASVspoof-hosted table stating the number explicitly.
- I did not find an ASVspoof-hosted document that names the literal placeholder in protocol
  column 3; its identity as "unused" is inferred from the fact that two independent official
  parsers discard it (`_`) rather than from an explicit column-name table in the evaluation
  plan PDF (which does not show a header row for the protocol file in the pages I could
  render).
