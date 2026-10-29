# `runs/` layout

Each evaluation writes one directory, `E-<UTC timestamp>-<system>` (for example
`E-20261024-214524-879357-d2`), created by `audio_df.runtime`. Older runs from the first experiments use
`E-YYYYMMDD-<exp>-<system>-<silence_mode>` (for example `E-20261001-e1-a1-raw`).

```
runs/<run>/
  config.yaml          # system, model_artifact, train_preprocessing, per-test-set silence mode, test sets
  calibration.json     # dev EER and its threshold (the operating point for every test set)
  metrics.json         # all test sets
  metrics_<testset>.json  # eer, min_dcf, roc_auc, f1 at the dev threshold, dev_eer, threshold
  scores/<testset>.csv    # clip_id, label, score[, attack_id]
```

Every evaluation also replaces `artifacts/<system>/threshold.json` with its calibration, so the web app
uses the newest threshold.

## Score convention

**Higher `score` means more bona fide** (ASVspoof convention). Labels are `bonafide` or `spoof`. Metrics
come from `audio_df.evaluation.evaluate_scores`, which uses `audio_df.helpers.metrics`; the threshold is
always chosen on `dev` (or `--threshold-from`) and reused unchanged on every other test set.

## What is tracked

Run directories stay local except the whitelisted ones in `.gitignore` (runs that a tracked notebook,
script or `threshold.json` reads). Large per-clip score files stay local unless something reads them;
`metrics*.json` and `calibration.json` carry the reported numbers.

`REGISTRY.md` is the hand-kept log of the first experiments (up to 2026-10-12). Later runs are not added
to it; their `config.yaml`, `calibration.json` and `metrics*.json` describe them, and
`uv run make-rankings` collects them into the app's leaderboard.
