# `runs/` layout

Each experiment run lives in its own directory, named by its experiment ID
(`E-YYYYMMDD-<exp>-<system>-<silence_mode>[-<tag>]`, e.g. `E-20261001-e1-a1-raw`;
built and validated by `audio_df.runtime.exp_io.make_exp_id`). `<exp>` is `e1`..`e6`
(the experiments listed in the [README](../README.md)) or `dev`/`smoke` for tuning/smoke runs that are never scored as
an experiment result. Directories are created by `audio_df.runtime.exp_io.create_experiment`
and never hand-edited in place except for `notes.md`.

Run directories stay local and are not tracked in git (see `.gitignore`); only this file and
`REGISTRY.md` are. The experiment tables built from them are tracked in `reports/tables/`.

```
runs/<exp_id>/
  config.yaml    # system, experiment, data (splits/manifests), preprocessing
                 # (sample_rate, crop_seconds, silence_mode), seed, features, ...
  env.json       # machine + software provenance (audio_df.runtime.exp_io.capture_env)
  notes.md       # free-text log: created-at stamp, any force-overwrite events,
                 # anything a human wants attached to this run
  scores/
    <testset>.csv   # clip_id,label,score[,attack_id] — see convention below
  run_summary.json  # optional: {wall_time_s, peak_rss_mb, mps_peak_mb,
                     # swap_start_gb, swap_end_gb, swap_peak_gb, pressure_max};
                     # written by whatever measured the run's resource usage.
                     # score_experiment copies it verbatim into metrics.json's
                     # "resources" key when present (audio_df.runtime.exp_io.format_resources
                     # renders the same dict into REGISTRY.md's `resources` column).
  metrics.json   # written by scoring.py / scripts/score_experiment.py, never
                 # by training code directly
```

## Score file convention

`scores/<testset>.csv` columns: `clip_id`, `label` (`bonafide` or `spoof`), `score`,
and an optional `attack_id`. **Higher `score` means more bona fide** (ASVspoof
convention; matches `audio_df.evaluation.metrics`'s default `higher_is_spoof=False`). Test set
names match `^[a-z0-9_]+$`, e.g. `dev`, `eval`, `itw`, `eval_mp3_32k`,
`eval_opus_16k`, `eval_noise_10db`, `eval_resample_8k`, `jev_dev500`, `eff_eval1k`.

Score files are produced once and never re-run in place: `write_scores` refuses to
overwrite an existing file unless `force=True`, and a forced overwrite is always
logged in that run's `notes.md`.

## Metrics are computed once, downstream, from score files only

Metrics are **never** computed inside training code. `scripts/score_experiment.py`
(via `audio_df.evaluation.scoring.score_experiment`) reads every `scores/*.csv` in a run
directory and writes `metrics.json`, reusing `audio_df.evaluation.metrics` functions
(`compute_eer`, `roc_auc`, `prf1_at_threshold`, `per_attack_eer`, `min_dcf`) rather
than reimplementing anything. Thresholds are always selected on `dev` (this run's own
`scores/dev.csv`, or another run's via `--threshold-from` for dev-less subset runs)
and reused unchanged on every other test set — never fit on eval/itw/etc.

## The registry

`runs/REGISTRY.md` is the single run log across all experiment directories —
one row per run, including failures. See its own header for column meanings.
`scripts/make_tables.py` reads it (plus every `config.yaml`/`metrics.json`) to build
`reports/tables/E1.md` .. `E6.md`, including only rows with `status: done` and
excluding anything later marked `superseded`.
