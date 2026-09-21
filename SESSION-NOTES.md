# Session notes — Audio Deepfake Detection (GC2)

Written 2026-09-21. This is the state of the project and everything a fresh session needs to
continue it. The data-strategy research discussion (ADR-001, In-the-Wild / MLAAD / custom-TTS)
is **not** in here and is being carried over separately.

---

## 0. Read this first if you just moved the folder

The project now lives at `~/LocalDocs/Learning/College/NUS/MTech-AIS/PS/GC2`
(it was under `PS/pattern understanding/GC2`). The move was clean — all 12 commits, clean
working tree, `origin` still `github.com/Qwenty228/DeepFakeDetectionGC2.git`.

Two things the move broke or left pending:

1. **The virtualenv had absolute paths baked in** and stopped working. Fix:
   ```bash
   rm -rf .venv && uv sync
   ```
2. **The dataset is still a zip.** `data/raw/ASVspoof 2019 Database.zip` (7.1 GB) has not been
   extracted. It must end up at `data/raw/ASVspoof2019/LA/...`:
   ```bash
   unzip -q "data/raw/ASVspoof 2019 Database.zip" -d data/raw/ASVspoof2019/
   ls data/raw/ASVspoof2019/LA/     # expect ASVspoof2019_LA_{train,dev,eval}/flac/ + _cm_protocols/
   ```
   If the archive's internal structure differs, only `_resolve_audio_path` in
   `src/audio_df/data.py` needs adjusting — that assumption is isolated there on purpose.
   Check `configs/data.yaml:5` points at wherever it actually lands.

---

## 1. What this project is

A reproducible audio deepfake detection workshop, built from `docs/handoff.md`. It progresses
from spectral EDA → hand-engineered cepstral features → CNN/ResNet → self-supervised
representations → a fusion + multi-task architecture, benchmarked on ASVspoof protocols.

The research question: can we learn a detector that captures **synthesis evidence** rather
than dataset or generator **shortcuts**? The architecture diagram is
`docs/assets/architecture.jpg` (the full-res PNG is gitignored — never commit it).

---

## 2. Current state

**Waves 0–3 complete. 194 tests pass with no dataset present.**

| Module | Lines | What it does |
|---|---|---|
| `device.py` | 96 | MPS/CPU resolution, env capture for provenance |
| `audio.py` | 194 | load, resample, crop/pad, RMS, silence ratio |
| `cache.py` | 208 | content-hashed feature cache, atomic writes |
| `metrics.py` | 433 | EER (official ASVspoof algorithm), AUCs, per-attack |
| `data.py` | 276 | manifests, protocol parsing, whole-family attack holdout |
| `features.py` | 315 | STFT → filterbank → log → DCT, variants A–F |
| `stats.py` | 236 | effect sizes, bootstrap CIs, deliberately no p-values |
| `visualization.py` | 352 | EDA figure vocabulary, aggregate-by-default |
| `leakage.py` | 493 | Stage 0 audit — the gate before any modelling |
| `synthetic.py` | — | synthetic ASVspoof-shaped fixture for tests/notebooks |
| `training.py` / `evaluation.py` / `registry.py` | — | training loop, scoring, experiment registry |

Notebooks `01_dataset_audit`, `02_spectral_eda`, `03_cepstral_eda` exist and execute
end-to-end **with no dataset**, switching to real data automatically once it's extracted.

**Not built:** Stage 3–4 baselines (T12, T13) and everything in `tasks/BACKLOG.md`
(RawNet2, AASIST, SSL probing, XLS-R, fusion, multi-task).

### Commands that work today
```bash
uv sync
uv run pytest                                   # 194 tests, no data needed
uv run ruff check src tests notebooks           # note: ruff lints .ipynb cells too
uv run python scripts/prepare_data.py --config configs/data.yaml
uv run python scripts/audit_leakage.py --config configs/data.yaml
```

---

## 3. How the work is being done

`tasks/TASKS.md` splits the build into **waves** of tasks with **disjoint file ownership**, so
a whole wave can run in parallel as separate Sonnet subagents, with a researcher agent
fact-checking between waves. `tasks/CONVENTIONS.md` is binding on every agent.

The loop that made this work:

1. Each implementer ends with a **claim block** — falsifiable statements, `file.py:L##`
   evidence, and what it relied on, with its own derivations flagged separately from sourced
   facts.
2. The researcher checks claims against primary sources and writes a verdict into
   `research/fact-checks/`.
3. **The orchestrator independently verifies** — runs the tests, reads the diffs, and writes
   adversarial scripts that try to break the thing rather than confirm it.
4. A task isn't done until it survives all three.

**This is the part worth keeping.** In every single case, the agent's own "least confident
claim" was where the real defect was. The claim blocks pointed straight at the problems
instead of requiring a blind audit of hundreds of lines.

Agent role definitions are in `.claude/agents/{implementer,researcher}.md`.

---

## 4. Defects this caught (the reason for the ceremony)

Each of these produced plausible-looking numbers and would have silently corrupted results.

**`silence_ratio` measured zero-crossings, not silence.** It thresholded per-sample amplitude
while its docstring claimed librosa's `top_db` convention, which thresholds *framed RMS*.
Every periodic waveform crosses near zero twice a cycle, so a speech-like signal with no
silence at all scored 0.227 against librosa's 0.000. Silence is a nuisance feature in the
Stage 0 leakage audit, so this would have produced a confident, wrong verdict about whether
the dataset was safe to model on. Now delegates to `librosa.effects.split`.

**The EER didn't match official ASVspoof scoring — and that was my spec's fault.** I specified
interpolation between bracketing operating points. The official `compute_eer` does
`argmin|FRR−FAR|` then the *mean of the two rates at that discrete index*, with no
interpolation; the 2019 evaluation plan's footnote 9 calls this a deliberate "simplified
nearest-neighbor" choice. An interpolated EER is not comparable with any published number.

**Score polarity was backwards in my spec too.** I wrote "higher score = spoof". The
evaluation plan §5 p.8 says the opposite, verbatim: *"high detection score should indicate
bona fide"*. That error alone inverts every EER to `1 − EER` — a number that still looks
entirely plausible. `metrics.py` now requires `higher_is_spoof` as an explicit argument with
no default, and a conformance test vendors the official code verbatim so the official
algorithm is the executable spec rather than a docstring claim.

**The handoff's DCT formula is not orthonormal as written.** `c[m,q] = Σ_r L[m,r] cos(πq/J
(r+½))` needs DCT-II ortho scaling — √(1/J) at q=0, √(2/J) above — or the round-trip test
can't pass. Caught by the researcher *before* implementation, which matters: the natural
response to a failing round-trip test is to loosen the test, not to suspect the spec.

**A cache key collision.** Fractional floats were canonicalised via `repr()`, a string, so
`20.5` and `"20.5"` hashed identically. A stale cache silently returning the wrong features is
the most dangerous failure mode in the project. Now type-tagged.

**A one-sided bootstrap coverage bound.** The test asserted coverage in `[0.88, 1.0]`, which
passes an interval that is far too *wide* — a bootstrap returning `[-inf, +inf]` covers 100%
and looks perfect. Now two-sided, derived from the binomial SE.

**Silently dropped registry keys.** `append_run` discarded any key not in the CSV header
without a sound, so a typo'd metric name would never be recorded and would surface as a blank
column in the final report, after the experiments had run. Now warns, naming each key.

**A fixture that drifted before it was even committed.** The synthetic corpus builder was
inlined in three notebooks; one copy had already diverged. That fixture *is* the ground truth
the notebooks make claims about, so three copies meant three corpora all described as "the
synthetic mini-manifest". Now single-source in `src/audio_df/synthetic.py` with property tests.

---

## 5. Scientific rules that are enforced, not just documented

- **Thresholds come from dev, never eval.** This is enforced structurally: `evaluate()`
  contains no aggregation over its own scores, takes a required `split` argument, and stamps
  it into a frozen `ScoreSet`. `pick_threshold` is the only function that picks an operating
  point and refuses anything not tagged `split="dev"`.
- **Per-attack EER alongside pooled EER.** A pooled number hides a reversed generator — on a
  synthetic easy/hard pair: A02 = 0.004, A03 = 0.456, pooled = 0.304.
- **Whole attack families held out**, never random utterances.
- **No p-values per frequency bin** — effect sizes with CIs instead, reasoning in the module
  docstring so the next reader knows the obvious thing was skipped deliberately.
- **Literature EERs never share a column with our runs.**
- **Stage 0 leakage audit runs before any modelling.** Verified adversarially: zero false
  positives on clean random-label data across four configs (including the pathological
  200-speakers × 1-utterance case), while the true-positive control fires at AUC 1.000.

---

## 6. Compute budget — measured on this Mac, not estimated

| Work | Cost |
|---|---|
| LFCC cache D=20 | 0.81 GB train, 2.29 GB eval |
| ResNet18-ish, batch 32 | 33.7 ms/step → 949 utts/sec on MPS |
| One epoch over 25,380 train utts | ~0.4 min |
| 30 epochs × 3 seeds | ~40 min |
| **All six A–F representations × 3 seeds** | **~4 hours** (one overnight run) |
| XLS-R-shaped 302M encoder, frozen forward | 18.7 utts/sec → 1.8 h for all 121,461 utts |

**This is not a big-data problem.** 7 GB is mostly FLAC of ~121k short clips; cost scales with
utterance count × model size, both modest. Do not introduce distributed training, sharding or
streaming dataloaders.

**The one real blowup:** caching un-pooled XLS-R hidden states costs 10.24 MB/utt → **1.24 TB**
for the corpus. Pooling over time first costs 51.2 KB/utt → **6.22 GB**. A 200× difference that
decides whether Stage 6 runs at all. Guarded in `configs/ssl_probe.yaml` with the measurement
attached, because a config value without a reason is one refactor away from being "optimised"
into a bug.

**Not feasible here:** fine-tuning XLS-R end-to-end. Frozen only. If it's ever run on rented
GPU, report trainable params and compute rather than mixing it into the frozen column.

---

## 7. Facts about ASVspoof 2019 LA worth knowing

- Train/dev share attacks **A01–A06**; eval uses **A07–A19**. The official eval split is
  therefore *already an unseen-generator test*. A dev/eval EER gap is generalisation, not
  necessarily overfitting.
- Protocol line format: `SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY`, five fields, verified
  against two official loaders.
- On-disk layout `<root>/ASVspoof2019_LA_<split>/flac/<utt_id>.flac`, verified likewise.
- Train = 25,380 and dev = 24,844 utterances, confirmed from the evaluation plan. The eval
  count (commonly quoted as 71,237) has **no ASVspoof-hosted primary source** and is recorded
  at medium confidence — running `prepare_data.py` on the real data will settle it.

---

## 8. What to do next

1. **Extract the zip**, then run `prepare_data.py` and `audit_leakage.py`. If the audit fires,
   stop and read `reports/tables/leakage_audit.csv` before training anything.
2. **Wave 4** — the first real science, and where numbers start existing:
   - **T12**: linear + MLP baselines over pooled features, all six A–F representations through
     one identical backend, 3 seeds. Notebook 04.
   - **T13**: CNN → ResNet over `[B,1,D,T]`. Document what every kernel and stride means
     *temporally and spectrally* — audio axes are not interchangeable image axes. Notebook 05.
   Both briefs are in `tasks/TASKS.md`. Dispatch as parallel implementer subagents.
3. **After Wave 4**: ADR recording which Stage-2 representation carries forward, then
   `reports/final_report.md` sections 1–6.

### Open items carried forward
- A `spectral_descriptors(x, sr, cfg) -> dict` helper for Stage-1 scalar descriptors
  (centroid/bandwidth/rolloff/flatness/ZCR) — currently one-off `librosa.feature.*` calls in
  notebook 02. T12 should decide whether it belongs in `src/`.
- The README's build-status table and empty Results section will need a pass once Wave 4 lands.
- Four unverified facts are recorded in `research/knowledge/index.md` rather than left
  implicit — including that the **0.65 leakage threshold is a project-local convention with no
  literature source**, which the audit states in its own warning text.

---

## 9. Things that will bite you

- `uv run` only — never bare `pip` or `python`, never activate the venv by hand.
- `ruff check src tests notebooks` — the `notebooks` part is easy to forget and it does lint
  `.ipynb` cells.
- **Never commit** `docs/assets/architecture-full.png`, audio, checkpoints or embedding caches.
  Check `git status` before every commit.
- Moving the project again breaks `.venv` — `rm -rf .venv && uv sync`.
- Subagent reports are not automatically true. Two claims in this session were wrong on
  inspection (a test count, and "duplicated verbatim" when one copy had already drifted).
  Verify before aggregating.
