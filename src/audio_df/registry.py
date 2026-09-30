"""Append-only writer for `reports/experiment_registry.csv` and per-run JSON sidecars.

Superseded 2026-09-30: `experiments/REGISTRY.md` (one row per run, written by
`audio_df.exp_io`) is now the single run log for the A1/A2/A3 benchmark. This CSV registry
is kept for history and for the existing tests below; it is not the place new runs get
recorded going forward.

The CSV header is the contract (`tasks/TASKS.md`, `reports/experiment_registry.csv`). This
module never hardcodes that header: it reads whatever header is currently on disk and writes
exactly those columns, in that order, for every row. That means:

- A row dict missing a key the header has gets `""` in that column, never a shifted value.
- A row dict with extra keys the header does not have simply does not write those keys to the
  CSV (they belong in the JSON sidecar via `write_run_json` instead) — this is `csv.DictWriter`
  with `extrasaction="ignore"`, not a silent column shift. Dropping is **audible**, not silent:
  `append_run` emits a `UserWarning` naming every dropped key and the registry path, so a
  typo'd or renamed metric key does not vanish without a trace. It still only warns, never
  raises — a long training run's final write should not die because of one bad key.
- If the registry ever gains a column, old rows already on disk are untouched (append-only,
  file opened in `"a"` mode) and new rows just pick up the new column via the same read-header
  mechanism, with no code change required here.
"""

from __future__ import annotations

import csv
import json
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["append_run", "write_run_json"]


def _read_header(path: Path) -> list[str]:
    with path.open("r", newline="") as f:
        header = next(csv.reader(f), None)
    if not header:
        raise ValueError(f"{path} has no header row; cannot append column-exact rows to it")
    return header


def _generate_run_id() -> str:
    """`{UTC timestamp, microsecond}-{8 hex chars}` — sortable (timestamp prefix) and unique
    (uuid4 suffix, so two runs started in the same microsecond still cannot collide)."""
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    return f"{ts}-{uuid.uuid4().hex[:8]}"


def append_run(row: dict[str, Any], path: str = "reports/experiment_registry.csv") -> str:
    """Append one row to the registry, returning the generated `run_id`.

    Reads the header from `path` (never hardcoded), fills any column absent from `row` with
    `""`, drops any key in `row` that is not a registry column (warning about it — see below),
    and always assigns a freshly generated `run_id` (overwriting any `run_id` the caller may
    have supplied) so the return value is authoritative and guaranteed unique. Opens the file
    in append mode only — never reads-then-rewrites the rows already there.

    If `row` contains keys that are not columns in the current header, a `UserWarning` is
    raised (not a hard error — the append still succeeds and the run's data is not otherwise
    lost) naming every dropped key and the registry path, so a typo'd or renamed metric key
    does not disappear silently. Write the full `row` (including any dropped keys) via
    `write_run_json` if you want it preserved somewhere despite not fitting the current CSV
    header.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"registry {p} does not exist; expected a file with the header row already present"
        )
    header = _read_header(p)
    header_set = set(header)

    dropped = sorted(k for k in row if k not in header_set)
    if dropped:
        warnings.warn(
            f"append_run: {len(dropped)} key(s) not in the registry header were dropped: "
            f"{dropped} (registry: {p})",
            UserWarning,
            stacklevel=2,
        )

    run_id = _generate_run_id()
    full_row = {col: row.get(col, "") for col in header}
    if "run_id" in full_row:
        full_row["run_id"] = run_id

    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writerow({k: ("" if v is None else v) for k, v in full_row.items()})

    return run_id


def write_run_json(result: dict[str, Any], out_dir: str | Path) -> Path:
    """Write a full machine-readable JSON sidecar for one run, named by its `run_id`.

    `result` should include everything `append_run`'s row does plus anything too large or
    structured for a CSV cell (e.g. the full per-attack EER breakdown, the resolved config).
    `result["run_id"]` must be present (typically the value returned by `append_run`) so the
    JSON file and its registry row can always be matched up.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = result.get("run_id")
    if not run_id:
        raise ValueError(
            "result must contain a non-empty 'run_id' key (the value append_run returned) "
            "so the JSON sidecar can be matched to its registry row"
        )
    out_path = out_dir / f"{run_id}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    return out_path
