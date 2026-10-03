# Experiment registry

Single run log for this project. One row per run, including failed runs — a
failure is recorded (with what changed in `notes.md`) rather than deleted, so
the log stays a true history. Timings in any linked `metrics.json` are measured
under throttling on a fanless machine (see `configs/machine.yaml`). `status` is
one of `running`, `done`, `failed`, `superseded`. Only `done` rows (excluding any
later marked `superseded`) feed `scripts/make_tables.py`. `resources` is a compact
rendering of {wall_time_s, peak_rss_mb, mps_peak_mb, swap_start_gb, swap_end_gb,
swap_peak_gb, pressure_max} — e.g. `3.1 h · RSS 1.8 GB · MPS 2.9 GB · swap
0.6→0.7 GB (peak 0.8) · pressure max warn` — see `exp_io.registry_append`/
`registry_update`'s `resources` argument; the same dict is stored verbatim under
`metrics.json`'s `"resources"` key when a `run_summary.json` is present.

| id | date | system | experiment | config summary | key EERs | status | machine | resources | notes |
|---|---|---|---|---|---|---|---|---|---|
| E-20261001-e1-a1-raw | 2026-10-01 | a1 | e1 | a1 lfcc-gmm train, 512-component diag GMM x2 | dev=2.62; eval=9.56 | done | Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0 | 20.6 m · RSS 3.8 GB · MPS 0.0 GB · swap 0.6→0.6 GB (peak 0.6) · pressure max 1 | A1 uses whole utterances (official B02 baseline), not this project's usual 4s crop (configs/data.yaml: audio.crop_seconds).; train-frame cache: data/cache/a1/06c95e9be4b94e0f (gitignored, resumable); train frames: bonafide=873016, spoof=7809362 (dim=60); bona fide GMM final avg log-lik=-31.812, spoof GMM final avg log-lik=-29.123 |
| E-20261001-e1-a2-raw | 2026-10-01 | a2 | e1 |  | dev=0.39; eval=0.96 | done | Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0 | 16.9 s · RSS 4.9 GB · MPS 8.3 GB · swap 0.6→2.4 GB (peak 0.6) · pressure max 1 | best dev loss at epoch 8 (dev loss checkpoint selection) |
| E-20261001-e3-a1-raw | 2026-10-01 | a1 | e1 | a1 score, silence_mode=raw, model_from=experiments/E-20261001-e1-a1-raw | itw=58.04 | done | Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0 | 3.2 m · RSS 0.6 GB · MPS 0.0 GB · swap 0.6→0.6 GB (peak 0.6) · pressure max 1 |  |
| E-20261001-e1-a1-trimmed | 2026-10-01 | a1 | e1 | a1 score, silence_mode=trimmed, model_from=experiments/E-20261001-e1-a1-raw | eval=19.84 | done | Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0 | 5.4 m · RSS 0.6 GB · MPS 0.0 GB · swap 0.6→0.6 GB (peak 0.6) · pressure max 1 | TRIMMED DIAGNOSTIC: no trimmed training exists for A1 — these are the raw-trained GMMs from experiments/E-20261001-e1-a1-raw scoring trimmed test audio. The threshold must come from the RAW experiment's dev scores, not this one's. |
| E-20261001-e3-a1-trimmed | 2026-10-01 | a1 | e1 | a1 score, silence_mode=trimmed, model_from=experiments/E-20261001-e1-a1-raw | itw=61.36 | done | Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0 | 3.1 m · RSS 0.6 GB · MPS 0.0 GB · swap 0.6→0.6 GB (peak 0.6) · pressure max 1 | TRIMMED DIAGNOSTIC: no trimmed training exists for A1 — these are the raw-trained GMMs from experiments/E-20261001-e1-a1-raw scoring trimmed test audio. The threshold must come from the RAW experiment's dev scores, not this one's. |
| E-20261001-e3-a2-raw | 2026-10-01 | a2 | e3 | a2 score, silence_mode=raw, testsets=itw, model_from=experiments/E-20261001-e1-a2-raw | itw=14.70 | done | Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0 |  | row added 2026-10-03: the score run predates run_a2 score using experiment_run (no row was written); resource summary not recorded |
| E-20261001-e1-a2-trimmed | 2026-10-01 | a2 | e1 | a2 score, silence_mode=trimmed, testsets=eval, model_from=experiments/E-20261001-e1-a2-raw | eval=13.27 | done | Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0 |  | row added 2026-10-03: the score run predates run_a2 score using experiment_run (no row was written); resource summary not recorded |
| E-20261001-e3-a2-trimmed | 2026-10-01 | a2 | e3 | a2 score, silence_mode=trimmed, testsets=itw, model_from=experiments/E-20261001-e1-a2-raw | itw=13.89 | done | Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0 |  | row added 2026-10-03: the score run predates run_a2 score using experiment_run (no row was written); resource summary not recorded |
