# TASKS — orchestration plan

Scope of this plan: **scaffold + Stages 0–4** of the handoff spec (dataset/leakage audit →
spectral & cepstral EDA → controlled feature ablations → linear baselines → CNN/ResNet).
Stages 5–10 (RawNet2, SSL probing, XLS-R, fusion, multi-task) are in `tasks/BACKLOG.md`,
deliberately unscheduled until Stage 0–4 evidence exists.

## How this runs

Each task below is a complete brief for one **`implementer`** subagent (Sonnet).
Dispatch a whole wave in parallel — tasks inside a wave share no files. Wait for the wave to
finish before starting the next; later waves import earlier waves' interfaces.

```
Agent(subagent_type="implementer", prompt="Execute task T02 from tasks/TASKS.md. <paste the task entry>")
```

After each wave, dispatch the **`researcher`** agent with the wave's claim blocks:

```
Agent(subagent_type="researcher", prompt="Fact-check the claim blocks from wave 1: <paste>")
```

A task is done only when its researcher verdict is `confirmed`. `contradicted` or
`unsupported` re-dispatches the implementer with the verdict attached.

Read `tasks/CONVENTIONS.md` before dispatching anything. Every agent reads it too.

---

## Wave 0 — foundation (1 agent, sequential)

### T01 — uv environment and repo smoke test
**Owns:** `pyproject.toml`, `uv.lock`, `tests/test_env.py`, `src/audio_df/__init__.py`
**Goal:** a working, locked environment that imports the package and resolves a device.
- `uv sync` succeeds on Apple Silicon, Python 3.11; commit `uv.lock`.
- Add `src/audio_df/device.py` with `resolve_device(spec: str = "auto") -> torch.device`
  (cuda → mps → cpu) and `describe_env() -> dict` returning python/torch/numpy versions,
  device, and git commit (`git rev-parse HEAD`, tolerating a dirty tree).
- `tests/test_env.py`: package imports, `resolve_device("auto")` returns a real device,
  `describe_env()` has non-empty keys.
**Verify:** `uv sync && uv run pytest tests/test_env.py && uv run ruff check src tests`
**Claim:** torch+torchaudio versions actually resolve and MPS is detected on this machine.

---

## Wave 1 — primitives (4 agents in parallel, + researcher)

### T02 — `audio.py`: loading and shaping
**Owns:** `src/audio_df/audio.py`, `tests/test_audio.py`
**Interface:**
```python
load_audio(path, target_sr=16000, mono=True) -> tuple[np.ndarray, int]
fix_length(x, n_samples, mode="center"|"random"|"repeat_pad", rng=None) -> np.ndarray
rms(x) -> float
silence_ratio(x, top_db=30) -> float          # leading/trailing + total
```
- Resampling via `librosa`/`torchaudio`; assert output sr equals `target_sr`.
- `fix_length` handles both crop and pad; `repeat_pad` tiles short audio, never zero-pads
  silently without recording that it did.
**Tests (no data):** synthesised sine/noise; round-trip length invariants; a 1 s clip padded
to 4 s has the expected energy; resample 8 k→16 k preserves tone frequency within tolerance.
**Claim:** the resampler preserves tone frequency; the crop is deterministic under a seed.

### T03 — `cache.py`: content-hashed feature cache
**Owns:** `src/audio_df/cache.py`, `tests/test_cache.py`
**Interface:**
```python
cache_key(utt_id: str, config: dict) -> str   # stable hash over sorted config
FeatureCache(root: Path).get(key) -> np.ndarray | None
FeatureCache(root: Path).put(key, array, meta: dict) -> Path
```
- Key must change when *any* relevant config field changes and must be stable across
  processes and dict ordering (sort keys; hash the canonical JSON).
- Store `.npy` + a sidecar `.json` with the full config that produced it.
**Tests:** reordered dicts give the same key; changed `n_fft` gives a different key;
put→get round-trips exactly; a cache miss returns `None`, never a stale array.
**Claim:** the key is order-independent and collision-resistant for the configs we use.

### T04 — `metrics.py`: EER and friends
**Owns:** `src/audio_df/metrics.py`, `tests/test_metrics.py`
**Interface:**
```python
compute_eer(scores, labels) -> tuple[float, float]        # (eer, threshold)
roc_auc(scores, labels) -> float
pr_auc(scores, labels) -> float
prf1_at_threshold(scores, labels, thr) -> dict
brier_score(probs, labels) -> float
per_attack_eer(scores, labels, attack_ids) -> dict[str, float]
```
- EER is the primary anti-spoofing metric: find where FAR and FRR cross, interpolating
  between the bracketing operating points. Document the convention (higher score = spoof)
  in the docstring and enforce it with an argument, not a comment.
- `per_attack_eer` scores each spoof generator **against the shared bonafide pool** — not
  against a subset — and returns a macro mean alongside the per-attack dict.
**Tests:** perfectly separable scores → EER 0.0; identical distributions → EER ≈ 0.5;
inverted labels → 1 − EER; AUC agrees with `sklearn.roc_auc_score`; a known 4-point case
computed by hand.
**Claim:** the EER interpolation and the per-attack bonafide-pooling convention are correct.

### T05 — `data.py`: manifests, protocols, splits
**Owns:** `src/audio_df/data.py`, `tests/test_data.py`, `scripts/prepare_data.py`
**Interface:**
```python
parse_asvspoof_protocol(path, split) -> pd.DataFrame
build_manifest(cfg) -> pd.DataFrame        # all columns in configs/data.yaml
split_hash(df) -> str
holdout_attack_families(df, families) -> tuple[pd.DataFrame, pd.DataFrame]
assert_no_speaker_overlap(train, dev) -> None
```
- Manifest carries every column listed in `configs/data.yaml:manifest.columns`. Missing
  metadata is an explicit `None`/`"-"`, never silently dropped.
- Absent dataset → a clear error naming the expected path and the download page. Never a
  bare `FileNotFoundError` traceback.
- `holdout_attack_families` removes whole generator families, never random utterances.
- `scripts/prepare_data.py` writes `data/manifests/<split>.csv` and prints class/attack counts.
**Tests (no data):** synthesise a tiny fake protocol file in `tmp_path`; verify column set,
split hash stability, whole-family holdout, and that speaker overlap raises.
**Claim:** the ASVspoof 2019 LA protocol column order as parsed matches the official file format.

### R01 — researcher: seed the knowledge base
**Owns:** `research/knowledge/**`, `research/knowledge/index.md`
Write `verified` notes for: `asvspoof-2019-la`, `asvspoof-2021`, `eer-and-metrics`,
`lfcc-mfcc-dct`, `leakage-audit`. Fetch the primary sources listed in `index.md`.
Then fact-check the T02–T05 claim blocks. `eer-and-metrics` must nail down the official EER
definition used by ASVspoof scoring so T04 can be checked against it, not against intuition.

---

## Wave 2 — representations and the Stage-0 audit (4 agents in parallel)

### T06 — `features.py`: the controlled Stage-2 ablation grid
**Owns:** `src/audio_df/features.py`, `tests/test_features.py`
**Depends on:** T02, T03
**Interface:**
```python
stft_power(x, cfg) -> np.ndarray                  # [M, K]
filterbank(n_filters, n_fft, sr, spacing="linear"|"mel", fmin, fmax) -> np.ndarray  # Psi[r, k]
apply_filterbank(P, Psi) -> np.ndarray            # E[m, r]
dct_coeffs(L, n_coeff=None, orthonormal=True) -> np.ndarray   # c[m, q]
add_deltas(C, order=2) -> np.ndarray
extract(x, variant: str, cfg) -> np.ndarray       # [T, D], dispatches variants A–F
```
- Implement the pipeline exactly as the handoff states:
  `x[n] → STFT X[m,k] → power P[m,k] → filterbank E[m,r] = Σ_k P[m,k]Ψ_r[k] → log L → DCT c`.
- All six variants A–F from `configs/features.yaml` must run through `extract`.
- Cache through `cache.py`; do not reimplement caching.
**Tests:** orthonormal full DCT round-trips (`idct(dct(L)) ≈ L`); truncation loses energy
monotonically; linear and mel filterbanks differ in centre frequencies but both sum to
near-unity coverage; output shape is `[T, D]` for every variant; a pure tone lands in the
expected filterbank channel.
**Claim:** the DCT is orthonormal and matches the handoff formula
`c[m,q] = Σ_r L[m,r] cos(πq/J (r+½))` up to the stated normalisation — state which
normalisation you used and why.

### T07 — `stats.py`: effect sizes done honestly
**Owns:** `src/audio_df/stats.py`, `tests/test_stats.py`
**Depends on:** T04
**Interface:**
```python
describe_by_class(df, feature, label_col) -> pd.DataFrame   # mean/median/std/IQR/n
cohens_d(a, b) -> float
cliffs_delta(a, b) -> float                                 # non-parametric alternative
bootstrap_ci(stat_fn, a, b, n_boot=2000, alpha=0.05, seed=0) -> tuple[float, float]
single_feature_auc(values, labels) -> float
per_bin_effect_size(spec_a, spec_b) -> np.ndarray           # standardized, per frequency bin
```
- `per_bin_effect_size` returns standardized differences for plotting. It must **not** return
  p-values: the handoff explicitly forbids thousands of uncorrected pointwise tests as evidence.
  Put that reasoning in the module docstring.
**Tests:** Cohen's d on two unit-variance normals separated by 1.0 ≈ 1.0; Cliff's delta is
±1 for disjoint samples and ~0 for identical; bootstrap CI covers the true value on synthetic
data; `single_feature_auc` matches `metrics.roc_auc`.
**Claim:** the effect-size estimators are unbiased on the synthetic cases tested.

### T08 — `visualization.py`: the EDA figure vocabulary
**Owns:** `src/audio_df/visualization.py`, `tests/test_visualization.py`
**Depends on:** T06 (shapes only — code to the interface, do not edit it)
**Interface:** `plot_waveform`, `plot_spectrogram`, `plot_mean_spectrum_with_ci`,
`plot_class_distributions`, `plot_effect_size_per_bin`, `plot_feature_trajectory`,
`plot_eer_vs_layer`, `save_figure(fig, name, out_dir="reports/figures")`.
- Every function takes an `ax=None` and returns the axis, so notebooks can compose.
- Aggregate views by default; no function encourages cherry-picking a single pair.
- `plot_*_by_attack` variants overlay generator-conditioned curves — a cue that reverses
  across generators must be *visible*, not averaged away.
**Tests:** headless `matplotlib` Agg backend; each function returns an Axes and writes a
non-empty PNG to `tmp_path`.
**Claim:** figures render headless with no display, on this machine.

### T09 — Stage 0 leakage audit
**Owns:** `src/audio_df/leakage.py`, `tests/test_leakage.py`, `scripts/audit_leakage.py`
**Depends on:** T04, T05
**Goal:** prove (or disprove) that labels are predictable from nuisance metadata *before*
any modelling.
- Fit tiny classifiers (logistic regression / depth-2 tree) on metadata-only features:
  duration, sample rate, codec, RMS/loudness, leading+trailing silence, source corpus,
  speaker identity — each alone, and all together.
- Report EER/AUC per nuisance feature, with bootstrap CIs, into
  `reports/tables/leakage_audit.csv` and a figure.
- Emit a loud `LEAKAGE WARNING` when any single nuisance feature exceeds a configurable AUC
  threshold (default 0.65), naming the feature.
**Tests:** a synthetic manifest where duration *is* the label → audit flags duration;
a synthetic manifest with random durations → no flag.
**Claim:** the audit detects a planted shortcut and does not fire on a clean synthetic set.

---

## Wave 3 — EDA notebooks and the training spine (2 agents)

### T10 — notebooks 01–03 (Stages 0–2 EDA)
**Owns:** `notebooks/01_dataset_audit.ipynb`, `02_spectral_eda.ipynb`, `03_cepstral_eda.ipynb`
**Depends on:** T05–T09
- Each notebook: a stated question at the top, imports from `src/audio_df` only, aggregate
  distributions (never a cherry-picked pair), generator-conditioned plots, and a written
  interpretation cell that separates *observation* from *inference*.
- 01: manifest summary, class/attack/speaker counts, leakage audit results, verdict on
  whether the split is safe to model on.
- 02: waveform/RMS/duration/silence distributions, spectra with CIs, centroid/bandwidth/
  rolloff/flatness/ZCR, band-energy ratios, per-bin effect sizes, single-feature AUC table.
- 03: the A–F variant grid in feature space — per-dimension distributions, effect sizes,
  correlation matrices, PCA, attack-conditioned views, temporal trajectories.
- Notebooks must execute end-to-end on a **synthetic mini-manifest** when no dataset is
  present, so they are reviewable today. Gate the real-data cells on a path check.
- Strip outputs before committing (`uv run nbstripout`).
**Claim:** each notebook runs top-to-bottom with no dataset present.

### T11 — `training.py` + `evaluation.py` + the registry
**Owns:** `src/audio_df/training.py`, `src/audio_df/evaluation.py`,
`src/audio_df/registry.py`, `tests/test_training.py`, `tests/test_registry.py`
**Depends on:** T03, T04
**Interface:**
```python
set_seed(seed) -> None                       # python, numpy, torch, cudnn/mps determinism
train_model(model, loaders, cfg) -> TrainResult   # early stopping on dev_eer
evaluate(model, loader, cfg) -> ScoreSet          # per-utterance scores, never thresholds from eval
pick_threshold(dev_scores) -> float
append_run(row: dict, path="reports/experiment_registry.csv") -> str  # returns run_id
write_run_json(result, out_dir) -> Path
count_params(model) -> tuple[int, int]            # (trainable, total)
measure_rtf(model, loader) -> float
```
- `append_run` writes **every** column in the existing registry header, filling unknowns with
  `""` rather than shifting columns. It is append-only and never rewrites history.
- Training records wall time, peak memory (MPS/CPU aware), params, seed, config hash, git commit.
**Tests:** a 2-layer toy model on synthetic tensors trains for 2 epochs on CPU; seeding twice
gives identical losses; early stopping fires on a plateau; `append_run` round-trips through
pandas with the header intact; threshold selection reads dev only (assert eval is untouched).
**Claim:** seeding is genuinely reproducible on this machine, and eval scores never influence
the threshold.

---

## Wave 4 — Stage 3 and Stage 4 baselines (2 agents in parallel)

### T12 — Stage 3: simple detection baselines
**Owns:** `src/audio_df/models/{linear.py,mlp.py,pooling.py}`,
`src/audio_df/experiments/run_baseline.py`, `notebooks/04_simple_detection.ipynb`,
`tests/test_models_simple.py`
**Depends on:** T06, T11
- `pooling.py`: `mean_pool`, `stats_pool` (mean+std), `attentive_stats_pool` — shared by
  Stage 3 and Stage 4, so keep it dependency-free.
- Logistic regression and a small MLP over pooled features; `z = wᵀx + b`,
  `p(fake|x) = σ(z)`. The notebook must make the point that detection is a boundary in
  feature space, not a hard-coded artifact detector.
- `run_baseline.py --config configs/features.yaml --variant <A..F>` trains one variant and
  appends a registry row. Run all six variants through the *same* backend.
- Report EER, ROC-AUC, PR-AUC, F1 at a dev-chosen threshold, Brier, per-attack EER, 3 seeds
  with mean ± std.
**Claim:** the six variants were run under an identical backend, split and seed set — so any
difference is attributable to the representation.

### T13 — Stage 4: CNN → ResNet backends
**Owns:** `src/audio_df/models/{cnn.py,resnet.py}`,
`src/audio_df/experiments/run_resnet.py`, `notebooks/05_cnn_resnet_experiments.ipynb`,
`tests/test_models_cnn.py`
**Depends on:** T06, T11 (uses T12's `pooling.py` read-only — do not edit it)
- Input tensor `[B, 1, D, T]`. Small 2D CNN first, then a ResNet18-style 1-channel backbone
  (`y = x + F(x)`), then pooling → embedding (256-d) → linear head.
- **Document every kernel shape and stride in terms of what it means temporally and
  spectrally.** Do not inherit ImageNet conventions unexamined — the handoff is explicit that
  audio axes are not interchangeable image axes. Put this in the module docstring.
- Run the same backbone over variants A–F where dimensions permit, to separate backend
  strength from representation strength.
- Must train on MPS and on CPU; log which device and the wall time.
**Tests:** forward pass shape on synthetic `[2,1,20,100]`; residual block is identity when
`F(x)` weights are zeroed; parameter count is stable; one training step decreases the loss
on a memorisable 8-sample batch.
**Claim:** the residual block implements `y = x + F(x)` with a correct projection shortcut on
stride/channel change, and the receptive field spans the time extent claimed in the docstring.

---

## Wave 5 — consolidation (researcher + 1 agent)

### R02 — researcher: full fact-check sweep and performance explanation
Fact-check every outstanding claim from waves 2–4. Write `research/knowledge/benchmarks/`
notes for each model run: number, protocol, efficiency cost, and a mechanistic expectation.
Open `research/decisions/ADR-001-representation-choice.md` recording which Stage-2 variant the
project will carry forward into Stage 9, with the evidence and the reversal condition.

### T14 — interim report
**Owns:** `reports/final_report.md` (sections 1–6), `README.md` results section
**Depends on:** all of the above
Sections 1–6 of the handoff's report structure: problem and RQs, dataset/protocol and leakage
audit, spectral/cepstral EDA, feature-engineering experiments, simple baselines, CNN/ResNet
baselines. Every number traceable to a `run_id` in the registry. State limitations plainly;
mark Stages 5–10 as not yet run rather than implying coverage.

---

## Dependency graph

```
T01
 ├─> T02 ─┐
 ├─> T03 ─┼─> T06 ─┬─> T10        (EDA notebooks)
 ├─> T04 ─┼─> T07 ─┤
 └─> T05 ─┴─> T08 ─┤
        └─> T09 ───┘
     T03,T04 ─────> T11 ─┬─> T12 ─┐
                         └─> T13 ─┴─> T14
     R01 (wave 1) ──> R02 (wave 5)   [researcher checks every wave in between]
```

## Orchestrator checklist
- [ ] Wave 0: T01 alone. Confirm `uv sync` and `uv.lock` before anything else.
- [ ] Wave 1: T02, T03, T04, T05 in parallel → then R01.
- [ ] Wave 2: T06, T07, T08, T09 in parallel → researcher fact-check.
- [ ] Wave 3: T10, T11 in parallel → researcher fact-check.
- [ ] Wave 4: T12, T13 in parallel → researcher fact-check.
- [ ] Wave 5: R02, then T14.
- [ ] After each wave: `uv run pytest && uv run ruff check src tests`, then commit.
- [ ] `git status` before every commit — no audio, no checkpoints, no `docs/assets/architecture-full.png`.
