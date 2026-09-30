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

