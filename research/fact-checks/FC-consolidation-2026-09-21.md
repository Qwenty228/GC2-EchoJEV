---
id: FC-consolidation-2026-09-21
task: consolidation of always-ignore/{SESSION-NOTES,concern,research,revisedata,remaining-tasks}.md into one handoff file
claim_by: orchestrator (pre-merge fact-check request)
verdict: mixed — see per-claim verdicts below
checked_on: 2026-09-21
---

# FC-consolidation-2026-09-21 — verdicts on 12 claims proposed for the merged handoff

Twelve claims, checked by running commands and reading code/data/sources directly, not by
trusting the always-ignore prose. Verdicts: CONFIRMED / WRONG / STALE / UNVERIFIABLE.

## Claim 1 — `uv run pytest` passes; "194 tests passing"; "11 leakage tests pass"

**Verdict: STALE (count), CONFIRMED (leakage subset, pass status).**

- `uv run pytest` → `196 passed, 3 warnings in 29.16s` (full run, current tree, no dataset
  dependency issues).
- `uv run pytest tests/test_leakage.py -q` → `11 passed`. The "11 leakage tests" figure is
  correct.
- The **194** figure in `remaining-tasks.md`/`SESSION-NOTES.md` is stale: it predates the
  uncommitted `tests/test_leakage.py` diff, which adds a new `TestAudioPathResolution` class
  with 2 tests (11 leakage tests now vs presumably 9 before). 194 + 2 = 196, consistent with
  the diff being the entire delta.

Corrected statement for the merged doc: **"196 tests pass (11 in `tests/test_leakage.py`),
current as of committing the `leakage.py` path-resolution fix."** Do not carry "194" forward
without the caveat that it predates that commit.

## Claim 2 — `uv run ruff check src tests notebooks` is clean

**Verdict: CONFIRMED.**

```
$ uv run ruff check src tests notebooks
All checks passed!
```

## Claim 3 — manifests and `leakage_audit.csv` "untracked and correctly gitignored"; status of `nuisance_features.csv`

**Verdict: WRONG for the manifests and for `nuisance_features.csv`; CONFIRMED only for `leakage_audit.csv`.**

```
$ git check-ignore -v data/manifests/train.csv   # exit 1 — NOT ignored
$ git check-ignore -v data/manifests/dev.csv     # exit 1 — NOT ignored
$ git check-ignore -v data/manifests/eval.csv    # exit 1 — NOT ignored
$ git check-ignore -v reports/tables/leakage_audit.csv
.gitignore:23:reports/tables/*    reports/tables/leakage_audit.csv   # exit 0 — IS ignored
$ git check-ignore -v data/metadata/nuisance_features.csv   # exit 1 — NOT ignored
$ git status --short
?? data/manifests/dev.csv
?? data/manifests/eval.csv
?? data/manifests/train.csv
?? data/metadata/nuisance_features.csv
```

`remaining-tasks.md` §0 is factually wrong about the manifests: they are untracked but **not**
gitignored — no `.gitignore` rule matches `data/manifests/*.csv`. They currently sit as
untracked files in `git status`, which means the next `git add -A` (or careless `git add .`)
would stage them. Only `reports/tables/leakage_audit.csv` is actually covered, by the
`reports/tables/*` glob.

`data/metadata/nuisance_features.csv` (16 MB, new this session, holds every audio-derived
nuisance feature per utterance including `silence_leading`/`silence_total`/`rms`) is likewise
untracked and **not** ignored — same exposure.

**Required action for the merged doc:** state plainly that `data/manifests/*.csv`,
`data/metadata/nuisance_features.csv`, and any other regenerable-but-large per-utterance
artifact must either (a) get an explicit `.gitignore` rule, mirroring the existing
`reports/tables/*` pattern, or (b) be committed deliberately. Do not repeat "correctly
gitignored" — it is not true today and committing 16 MB of feature cache by accident is
exactly the kind of failure this project's own conventions warn against.

## Claim 4 — the uncommitted `leakage.py`/`test_leakage.py` diff mechanism

**Verdict: CONFIRMED.**

`git diff src/audio_df/leakage.py` shows `_resolve_audio_path` previously did:
```python
p = Path(row["path"])
if audio_root is not None and not p.is_absolute():
    p = Path(audio_root) / p          # unconditional join
```
and now tries `p` as-written first, falling back to `audio_root / p` only if the as-written
path doesn't exist (`src/audio_df/leakage.py:117-137` post-diff). The docstring added in the
diff states the mechanism explicitly and correctly: `build_manifest` already joins
`dataset.root` onto `path`, while `run_audit` passes that same root in again as `audio_root`,
so the old code produced `<root>/<root>/...`.

Confirmed the NaN conversion and downstream drop independently by reading the surrounding
code, not just the diff's own claim:
- `_extract_rms`/silence extractors catch `(FileNotFoundError, OSError, KeyError)` and leave
  the cell `NaN` — `src/audio_df/leakage.py:192`.
- `audit_single_feature` drops NaN rows before fitting (`leakage.py:322-324`), and
  `_audit_combined` drops any column that is **entirely** NaN before fitting the combined
  model (`leakage.py:359-364`).
- The public report-builder drops all-NaN columns from the output table
  (`leakage.py:431-437`, `if col.isna().all(): ...`).

So the described chain — double-join → `FileNotFoundError` → `NaN` → column entirely NaN →
silently dropped from the audit table — is accurate end to end, verified by reading the code
at each step, not by trusting the diff's own comment.

The new tests (`TestAudioPathResolution`, `tests/test_leakage.py`) exercise exactly this
regression and pass (`11 passed` overall, confirmed under Claim 1).

## Claim 5 — protocol counts: train 25,380 / dev 24,844 / eval 71,237

**Verdict: CONFIRMED.**

```
$ wc -l data/raw/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
   25380
$ wc -l .../ASVspoof2019.LA.cm.dev.trl.txt
   24844
$ wc -l .../ASVspoof2019.LA.cm.eval.trl.txt
   71237
```
Exact match, from the protocol files themselves (not third-party citation). This also settles
the "eval count has no ASVspoof-hosted primary source" caveat in `SESSION-NOTES.md` §7 — it's
now directly counted from the extracted, ASVspoof-hosted protocol file, so that caveat is
stale and should be dropped in the merged doc, as `remaining-tasks.md` §2 already says.

## Claim 6 — attack partition: train/dev = A01–A06, eval = A07–A19

**Verdict: CONFIRMED.**

```
$ awk '{print $4}' .../ASVspoof2019.LA.cm.train.trn.txt | sort -u
-  A01 A02 A03 A04 A05 A06
$ awk '{print $4}' .../ASVspoof2019.LA.cm.dev.trl.txt | sort -u
-  A01 A02 A03 A04 A05 A06
$ awk '{print $4}' .../ASVspoof2019.LA.cm.eval.trl.txt | sort -u
-  A07 A08 A09 A10 A11 A12 A13 A14 A15 A16 A17 A18 A19
```
Exact match, counted from the protocol's 4th field directly.

## Claim 7 — class balance: train 2,580 bonafide / 22,800 spoof; eval 7,355 / 63,882

**Verdict: CONFIRMED.**

```python
>>> pd.read_csv('data/manifests/train.csv')['label'].value_counts()
spoof       22800
bonafide     2580
>>> pd.read_csv('data/manifests/eval.csv')['label'].value_counts()
spoof       63882
bonafide     7355
```
Exact match. (Note as an aside, not part of the claim: dev is 22,296 spoof / 2,548 bonafide,
~1:8.75 — also heavily skewed, consistent with the same argument for PR-AUC.)

## Claim 8 — Stage 0 audit table (combined 0.876, silence_leading 0.837, silence_total 0.825, rms 0.664, speaker_id 0.649, duration_s 0.624, silence_trailing 0.611)

**Verdict: CONFIRMED.**

`reports/tables/leakage_audit.csv` (current, full 121,461-utterance run, `n=121461` on every
row) gives, rounded to 3 dp: combined 0.876 (CI 0.873–0.878, EER 0.229, flagged=True),
silence_leading 0.837 (0.835–0.840, EER 0.247, flagged=True), silence_total 0.825
(0.822–0.827, EER 0.255, flagged=True), rms 0.664 (0.660–0.669, EER 0.383, flagged=True),
speaker_id 0.649 (0.643–0.654, EER 0.414, flagged=False), duration_s 0.624 (0.620–0.628, EER
0.486, flagged=False), silence_trailing 0.611 (0.606–0.616, EER 0.437, flagged=False). Every
number, CI, and flag in `remaining-tasks.md`'s table matches the current CSV exactly, to the
rounding shown. sample_rate/codec/source_corpus all sit at AUC 0.500 / EER 0.600, also
matching.

## Claim 9 — new per-split silence_leading stats from `data/metadata/nuisance_features.csv`

**Verdict: CONFIRMED, all sub-claims, computed independently via `audio_df.leakage.audit_single_feature(..., seed=1337)` and pandas groupbys.**

- Per-split AUC (logreg, `seed=1337`, `n`=25380/24844/71237): train **0.7460**, dev
  **0.7890**, eval **0.8867**. Matches 0.746 / 0.789 / 0.887 exactly.
- Median `silence_leading` by split+label: bonafide train/dev/eval = 0.2978 / 0.3160 /
  0.3089 (claimed ~0.298/0.316/0.309 — matches); spoof train/dev/eval = 0.1681 / 0.1674 /
  0.0142 (claimed 0.168/0.167/0.014 — matches).
- Fraction of eval spoof with `silence_leading <= 0`: **0.3564** (claimed 0.356 — matches, n =
  63,882). Fraction of eval bonafide: **0.00095** (claimed 0.001 — matches, n = 7,355).
- Per-attack median at eval: A10/A11/A12/A13 = **0.0000** each (matches). A17 = **0.3026**,
  A18 = **0.3101**, A19 = **0.2886**, bonafide (`attack_id == "-"`) = **0.3089** (all match
  to 4 dp).

**On the interpretation** ("the silence shortcut is attack-dependent, absent for A17/A18/A19,
and strongest at eval"): **supported by these numbers, with one precision fix.** A17/A18/A19
medians (0.286–0.310) sit within the same range as the bonafide median (0.309), so the
single-feature shortcut genuinely does not separate those three attacks from bonafide —
"absent" is accurate for them specifically. A07–A16 medians are near 0 (well below bonafide),
so the pooled-eval AUC of 0.887 is being driven almost entirely by A07–A16, not uniformly
across all 13 eval attacks — the merged doc should say *which* attacks carry the shortcut,
not just that it's "attack-dependent" in the abstract, since that phrase alone doesn't convey
that it's roughly a 10-vs-3 split among eval attacks. "Strongest at eval" is also correct by
AUC (0.887 > 0.789 > 0.746), but note this is confounded with attack composition, not just
split identity: eval contains 13 attacks including the near-zero-silence A07–A16 group, while
train/dev only ever see A01–A06 — so "strongest at eval" is really "strongest on the specific
attacks that happen to be eval-only," and that causal attribution should be made explicit
rather than left implying something about the eval split itself.

## Claim 10 — "15.1% EER from a single silence-duration scalar on ASVspoof 2019 LA (arXiv:2106.12914)" vs our eval EER 0.1825

**Verdict: CONFIRMED (the literature number), CONFIRMED (our number), WRONG framing if presented as directly comparable — this needs an explicit caveat in the merged doc.**

- Literature number: `research/knowledge/methods/leakage-audit.md` (already `confidence:
  high`, primary source fetched and read directly) states Müller et al. 2021
  (arXiv:2106.12914) get **15.12% EER on the ASVspoof2019 LA eval set** (and 31.09% on dev)
  from a single scalar — leading-silence duration in seconds — fed into a small FCNN. This is
  correctly attributed: single feature, eval set, ASVspoof2019 LA. Confirmed as an accurate
  restatement of the existing knowledge note.
- Our number: recomputed independently, `audit_single_feature(eval_df['silence_leading'],
  eval_df['label'], seed=1337)` → **AUC 0.8867, EER 0.1825** (`n=71237`). This matches the
  0.1825 figure exactly.
- **These are not the same quantity and must not be presented side by side without a
  caveat.** Two protocol differences, both material:
  1. **Train/test regime.** Müller et al.'s FCNN is fit on the *train* split and evaluated on
     the held-out *eval* split (a genuine cross-split generalization number — consistent with
     them also reporting a separate, worse dev number, 31.09%, from the same fitted model).
     Our `audit_single_feature` figure is a **5-fold stratified cross-validation number
     computed entirely within the eval split** (`_cross_val_oof_scores`,
     `src/audio_df/leakage.py:234-260`) — it never sees train or dev at all. It answers "how
     separable is silence_leading within eval's own attack mix," not "does a
     silence-duration classifier fit on train generalize to eval."
  2. **Classifier.** Müller et al. use a small FCNN; ours reports whichever of {logreg, tiny
     decision tree} scores higher out-of-fold (here, logreg won — `res['model'] ==
     'logreg'`). Different model family, different capacity.
  Given (1), the fact that our number (18.25%) is *higher* than theirs (15.12%) despite being
  an in-split CV estimate (which would usually be easier, not harder, than true
  cross-split generalization) is itself worth flagging as a genuine point of note in the
  merged doc, not smoothing over — it suggests eval's attack mix is intrinsically harder for
  this single feature than train→eval generalization was for their FCNN, or that the specific
  feature computation/thresholding differs slightly between the two pipelines. Neither this
  note nor the always-ignore files resolve which; it should be flagged as open, not asserted.
  Per `research/README.md`'s rule, tag Müller et al.'s number `source: literature` and never
  put it in the same column/table as `0.1825`; report them as two separate, independently
  interesting numbers that corroborate the same qualitative finding (silence is a large,
  real, single-feature shortcut on this corpus) without claiming they measure the same thing.

## Claim 11 — In-the-Wild: CC-BY-SA-4.0, 8.16 GB, single split, no generator labels, 58 speakers

**Verdict: CONFIRMED**, already recorded at `confidence: high` in
`research/knowledge/datasets/in-the-wild.md` (sources: arXiv:2203.16263 fetched and read
directly, plus the HuggingFace dataset card `mueller91/In-The-Wild`). All five sub-claims
(license, size-on-disk, single unsplit `train` partition, absence of an attack/generator-ID
field, 58 speakers) are stated verbatim in that note with primary-source citations. No
discrepancy found on re-reading the note against its cited sources.

## Claim 12 — ADR-002 naming collision

**Verdict: CONFIRMED there is a real collision; both `research/decisions/ADR-002-*` and `ADR-003-*` are currently unclaimed.**

```
$ ls research/decisions/
ADR-001-generalization-data-strategy.md
_TEMPLATE.md
```
Only `ADR-001` exists. `remaining-tasks.md` §4 reserves `ADR-002` for "which Stage-2
representation carries into Stage 9." A separately-drafted `always-ignore/HANDOFF.md` (found
in the same directory, not one of the five files named in this task, dated later than the
others) already resolves this the same way I would recommend: give `ADR-002` to whichever
decision **gates Wave 4 chronologically** — the silence-control decision, since Wave 4
baselines cannot be responsibly speced without it — and push the Stage-2-representation
decision to `ADR-003`. Flagging for the merge: the merged handoff must pick one and state it
explicitly (`ADR-002` = silence-control decision, `ADR-003` = representation choice), because
`remaining-tasks.md`'s unmodified §4 language, if carried forward unedited, would collide with
whatever the silence-control ADR is filed as.

**Note on `always-ignore/HANDOFF.md`:** this file exists in the same directory as the five
named for merging but was not in the task's file list. It appears to already be a draft of
the merged document (626 lines, most recently modified). I did not fact-check its contents
beyond the ADR-numbering passage quoted above, since it is outside this task's scope — flag it
to the orchestrator explicitly rather than silently ignoring or silently treating it as
authoritative.

---

## Summary table

| # | Claim | Verdict |
|---|---|---|
| 1 | 194 tests / 11 leakage tests | STALE (194 → 196) / CONFIRMED (11) |
| 2 | ruff clean | CONFIRMED |
| 3 | manifests + audit csv "correctly gitignored" | WRONG (manifests, nuisance_features.csv not ignored); CONFIRMED only for leakage_audit.csv |
| 4 | leakage.py double-join → NaN → dropped-column mechanism | CONFIRMED |
| 5 | protocol counts 25,380/24,844/71,237 | CONFIRMED |
| 6 | attack partition A01–A06 / A07–A19 | CONFIRMED |
| 7 | class balance 2,580/22,800, 7,355/63,882 | CONFIRMED |
| 8 | Stage 0 audit table | CONFIRMED |
| 9 | new silence_leading per-split/per-attack stats | CONFIRMED (all sub-numbers); interpretation supported with precision caveat |
| 10 | 15.1% EER (lit.) vs our 0.1825 (eval) | CONFIRMED (both numbers individually); WRONG if presented as directly comparable — different train/test regime and classifier, needs explicit caveat |
| 11 | In-the-Wild dataset facts | CONFIRMED |
| 12 | ADR-002 naming collision | CONFIRMED (real collision); recommend ADR-002 = silence-control, ADR-003 = representation |

## Required action

1. Fix the test count (196, not 194) and note it's post-commit of the pending `leakage.py`
   diff.
2. Correct §0's gitignore claim: manifests and `nuisance_features.csv` are **not** ignored
   today; add explicit `.gitignore` rules or commit deliberately before the next broad `git
   add`.
3. Carry Claim 9's attack-level nuance (A07–A16 shortcut-bearing, A17–A19 not) into the
   merged doc rather than the flatter "attack-dependent" phrasing.
4. Tag the Müller et al. 15.1% figure `source: literature`, keep it in a separate
   column/table from our 0.1825, and state the cross-split-vs-in-split-CV difference
   explicitly wherever both appear together.
5. Resolve the ADR-002/ADR-003 numbering before the merged doc is finalized, and surface
   `always-ignore/HANDOFF.md`'s existence to the orchestrator — it wasn't in my task's input
   list and may itself need a fact-check pass if it's intended to become the merged file.
