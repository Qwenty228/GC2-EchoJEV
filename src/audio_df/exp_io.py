"""Experiment directory lifecycle: IDs, config/env capture, score files, the registry.

This module owns everything about *how an experiment directory comes into being and
is written to* — it does not compute a single metric (that is `scoring.py`'s job) and
it does not choose what system/config to run (that is whatever training code calls
`create_experiment`). Score files are produced once and never re-run in place:
`write_scores` refuses to overwrite an existing `scores/<testset>.csv` unless the
caller explicitly passes `force=True`, and even then the overwrite is logged in
`notes.md` so a metrics.json computed from the old file is not silently orphaned
without a trace.

## Experiment ID

`E-YYYYMMDD-<exp>-<system>-<silence_mode>[-<tag>]`, e.g. `E-20261001-e1-a1-raw`.
`<exp>` is one of `e1..e6` (an experiment from research doc §5.2) or `dev`/`smoke`
for tuning/smoke runs that are not part of any numbered experiment (dev is explicitly
for tuning, never treated as an experiment result — research doc §5.2). `<system>`
identifies the benchmarked system (`a1`, `a2`, `a3`, optionally with a model tag like
`a2-xlsr`). `<silence_mode>` records whether raw or trimmed audio was used, which is
a diagnostic attached to E1/E3, never its own experiment.

## Score file convention (project-wide, see task brief)

`scores/<testset>.csv` has columns `clip_id,label,score[,attack_id]`.
`label` is `bonafide` or `spoof`. Higher `score` means more bona-fide (ASVspoof
convention, matches `metrics.py`'s `higher_is_spoof=False` default). Test set names
match `^[a-z0-9_]+$` (`dev`, `eval`, `itw`, `eval_mp3_32k`, `eval_opus_16k`,
`eval_noise_10db`, `eval_resample_8k`, `jev_dev500`, `eff_eval1k`, ...).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from audio_df.device import git_commit

__all__ = [
    "make_exp_id",
    "create_experiment",
    "capture_env",
    "write_scores",
    "registry_append",
    "registry_update",
    "format_resources",
    "VALID_LABELS",
    "TESTSET_NAME_RE",
]

VALID_LABELS = {"bonafide", "spoof"}
TESTSET_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_EXP_RE = re.compile(r"^(e[1-6]|dev|smoke)$")
_SLUG_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_EXP_ID_RE = re.compile(
    r"^E-(?P<date>\d{8})-(?P<exp>[a-z0-9]+)-(?P<system>[a-z0-9-]+)-(?P<silence>[a-z0-9-]+)"
    r"(?:-(?P<tag>[a-z0-9-]+))?$"
)
_VALID_STATUSES = {"running", "done", "failed", "superseded"}
_REQUIRED_CONFIG_KEYS = (
    "system",
    "experiment",
    "data",
    "preprocessing",
    "seed",
    "features",
)

_REGISTRY_HEADER_PARAGRAPH = (
    "# Experiment registry\n\n"
    "Single run log for this project. One row per run, including failed runs — a\n"
    "failure is recorded (with what changed in `notes.md`) rather than deleted, so\n"
    "the log stays a true history. Timings in any linked `metrics.json` are measured\n"
    "under throttling on a fanless machine (see `configs/machine.yaml`). `status` is\n"
    "one of `running`, `done`, `failed`, `superseded`. Only `done` rows (excluding any\n"
    "later marked `superseded`) feed `scripts/make_tables.py`. `resources` is a compact\n"
    "rendering of {wall_time_s, peak_rss_mb, mps_peak_mb, swap_start_gb, swap_end_gb,\n"
    "swap_peak_gb, pressure_max} — e.g. `3.1 h · RSS 1.8 GB · MPS 2.9 GB · swap\n"
    "0.6→0.7 GB (peak 0.8) · pressure max warn` — see `exp_io.registry_append`/\n"
    "`registry_update`'s `resources` argument; the same dict is stored verbatim under\n"
    "`metrics.json`'s `\"resources\"` key when a `run_summary.json` is present.\n\n"
)
_REGISTRY_COLUMNS = [
    "id",
    "date",
    "system",
    "experiment",
    "config summary",
    "key EERs",
    "status",
    "machine",
    "resources",
    "notes",
]


def _registry_table_header() -> str:
    header = "| " + " | ".join(_REGISTRY_COLUMNS) + " |\n"
    sep = "|" + "|".join(["---"] * len(_REGISTRY_COLUMNS)) + "|\n"
    return header + sep


# ---------------------------------------------------------------------------
# Experiment ID
# ---------------------------------------------------------------------------


def make_exp_id(
    date: str | _dt.date,
    exp: str,
    system: str,
    silence_mode: str,
    tag: str | None = None,
) -> str:
    """Build and validate `E-YYYYMMDD-<exp>-<system>-<silence_mode>[-<tag>]`.

    `date` may be an 8-digit `YYYYMMDD` string or a `datetime.date`/`datetime.datetime`
    (formatted via `strftime("%Y%m%d")`). `exp` must be `e1`..`e6`, `dev`, or `smoke`.
    `system`, `silence_mode`, and `tag` (if given) must each be a lowercase
    alphanumeric slug, optionally hyphen-segmented (e.g. `a2-xlsr`), matching
    `^[a-z0-9]+(-[a-z0-9]+)*$`. Raises `ValueError` naming the offending part.
    """
    date_str = date.strftime("%Y%m%d") if isinstance(date, _dt.date) else str(date)
    if not re.fullmatch(r"\d{8}", date_str):
        raise ValueError(f"date must be an 8-digit YYYYMMDD string or a date object; got {date!r}")

    if not _EXP_RE.fullmatch(exp):
        raise ValueError(f"exp must be one of e1..e6, dev, smoke; got {exp!r}")
    if not _SLUG_RE.fullmatch(system):
        raise ValueError(f"system must match {_SLUG_RE.pattern!r}; got {system!r}")
    if not _SLUG_RE.fullmatch(silence_mode):
        raise ValueError(f"silence_mode must match {_SLUG_RE.pattern!r}; got {silence_mode!r}")
    if tag is not None and not _SLUG_RE.fullmatch(tag):
        raise ValueError(f"tag must match {_SLUG_RE.pattern!r}; got {tag!r}")

    parts = [f"E-{date_str}", exp, system, silence_mode]
    if tag is not None:
        parts.append(tag)
    exp_id = "-".join(parts)
    if not _EXP_ID_RE.fullmatch(exp_id):
        raise ValueError(f"constructed id {exp_id!r} failed its own format check")
    return exp_id


def _validate_exp_id_format(exp_id: str) -> None:
    if not _EXP_ID_RE.fullmatch(exp_id):
        raise ValueError(
            f"exp_id {exp_id!r} does not match E-YYYYMMDD-<exp>-<system>-<silence>[-<tag>]"
        )


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        ).stdout.strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _machine_label() -> str | None:
    try:
        with open("configs/machine.yaml") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("label")
    except OSError:
        return None


def capture_env() -> dict[str, Any]:
    """Everything needed to say what machine/software produced an experiment.

    Best-effort on fields that are macOS-specific (`sysctl`, `platform.mac_ver`):
    missing values become `None` rather than raising, so this still runs on CPU-only
    CI/Linux, it just records less.
    """
    commit_raw = git_commit()
    dirty = commit_raw.endswith("-dirty")
    commit = commit_raw[: -len("-dirty")] if dirty else commit_raw

    mac_ver = platform.mac_ver()[0] or None
    cpu_brand = _sysctl("machdep.cpu.brand_string") or platform.processor() or None
    hw_model = _sysctl("hw.model")
    logical_cores = os.cpu_count()
    physical_cores_raw = _sysctl("hw.physicalcpu")
    physical_cores = int(physical_cores_raw) if physical_cores_raw else None
    ram_bytes_raw = _sysctl("hw.memsize")
    ram_gb = round(int(ram_bytes_raw) / (1024**3), 2) if ram_bytes_raw else None

    mps_available = bool(torch.backends.mps.is_available())
    cuda_available = bool(torch.cuda.is_available())
    cuda_version = torch.version.cuda if cuda_available else None

    return {
        "os": platform.system(),
        "os_version": mac_ver or platform.version(),
        "machine_model": hw_model,
        "cpu_brand": cpu_brand,
        "cores_logical": logical_cores,
        "cores_physical": physical_cores,
        "ram_gb": ram_gb,
        "gpu_mps_available": mps_available,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_commit": commit,
        "git_dirty": dirty,
        "machine_label": _machine_label(),
        "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------


def create_experiment(
    exp_id: str,
    config: dict[str, Any],
    root: str | Path = "experiments",
    resume: bool = False,
) -> Path:
    """Create `experiments/<exp_id>/{config.yaml, env.json, notes.md, scores/}`.

    Refuses to create over an existing directory unless `resume=True`, in which case
    the existing `config.yaml` must match `config` exactly (else `ValueError` — an
    experiment directory is not a place to quietly change the recipe partway through).
    On a fresh (non-resume) creation, `env.json` is (re)written via `capture_env()`
    every time `resume=True` is used against an already-existing, matching directory,
    since the environment (git commit, versions) may have moved on between resumes.
    """
    _validate_exp_id_format(exp_id)

    missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in config]
    if missing:
        raise ValueError(f"config is missing required key(s): {missing}")

    exp_dir = Path(root) / exp_id
    config_path = exp_dir / "config.yaml"

    if exp_dir.exists():
        if not resume:
            raise FileExistsError(
                f"{exp_dir} already exists; pass resume=True to reuse it "
                "(config.yaml must match exactly)"
            )
        if not config_path.exists():
            raise ValueError(f"{exp_dir} exists but has no config.yaml to resume from")
        with config_path.open() as f:
            existing_config = yaml.safe_load(f)
        if existing_config != config:
            raise ValueError(
                f"resume=True but config differs from the existing {config_path}; "
                "an experiment directory's config must not change under a run"
            )
    else:
        exp_dir.mkdir(parents=True, exist_ok=False)
        with config_path.open("w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        (exp_dir / "notes.md").write_text(
            f"# {exp_id}\n\ncreated: {_dt.datetime.now(_dt.UTC).isoformat()}\n"
        )

    (exp_dir / "scores").mkdir(parents=True, exist_ok=True)

    with (exp_dir / "env.json").open("w") as f:
        json.dump(capture_env(), f, indent=2, default=str)

    return exp_dir


# ---------------------------------------------------------------------------
# Score files
# ---------------------------------------------------------------------------


def write_scores(
    exp_dir: str | Path,
    testset: str,
    df: pd.DataFrame,
    force: bool = False,
) -> Path:
    """Validate and write `exp_dir/scores/<testset>.csv`, sorted by `clip_id`.

    Required columns: `clip_id`, `label` (values in `{"bonafide", "spoof"}`), `score`
    (finite, no NaN/inf). An optional `attack_id` column is passed through unchanged.
    `clip_id` must be unique. Refuses to overwrite an existing score file unless
    `force=True`; when forced, the overwrite (old file's sha256-independent fact,
    just the event) is appended to `notes.md` so it is never silent.
    """
    if not TESTSET_NAME_RE.fullmatch(testset):
        raise ValueError(f"testset name {testset!r} must match {TESTSET_NAME_RE.pattern!r}")

    required_cols = {"clip_id", "label", "score"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"scores dataframe missing required column(s): {sorted(missing_cols)}")

    bad_labels = sorted(set(df["label"].unique()) - VALID_LABELS)
    if bad_labels:
        raise ValueError(f"unrecognised label value(s) {bad_labels}; expected {VALID_LABELS}")

    score_col = pd.to_numeric(df["score"], errors="coerce")
    if score_col.isna().any() or not np.isfinite(score_col.to_numpy(dtype=float)).all():
        raise ValueError("score column contains NaN, inf, or non-numeric values")

    if df["clip_id"].duplicated().any():
        dupes = df.loc[df["clip_id"].duplicated(), "clip_id"].unique().tolist()
        raise ValueError(f"duplicate clip_id value(s): {dupes[:10]}")

    exp_dir = Path(exp_dir)
    scores_dir = exp_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    out_path = scores_dir / f"{testset}.csv"

    if out_path.exists() and not force:
        raise FileExistsError(
            f"{out_path} already exists; scores are produced once and never re-run in "
            "place. Pass force=True if you really mean to overwrite it."
        )

    overwriting = out_path.exists() and force
    ordered_cols = ["clip_id", "label", "score"] + (
        ["attack_id"] if "attack_id" in df.columns else []
    )
    df.loc[:, ordered_cols].sort_values("clip_id").to_csv(out_path, index=False)

    if overwriting:
        notes_path = exp_dir / "notes.md"
        stamp = _dt.datetime.now(_dt.UTC).isoformat()
        with notes_path.open("a") as f:
            f.write(f"\n- {stamp}: force-overwrote scores/{testset}.csv\n")

    return out_path


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


_RESOURCE_KEYS = (
    "wall_time_s",
    "peak_rss_mb",
    "mps_peak_mb",
    "swap_start_gb",
    "swap_end_gb",
    "swap_peak_gb",
    "pressure_max",
)


def _format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.1f} m"
    return f"{seconds:.1f} s"


def format_resources(resources: dict[str, Any] | None) -> str:
    """Render a resource-summary dict into one compact registry-cell string.

    Expected keys (all optional; missing/`None` values are simply omitted from
    the rendering, never rendered as an error or placeholder):
      - `wall_time_s`: total wall time, rendered as hours/minutes/seconds
        depending on magnitude (e.g. `3.1 h`).
      - `peak_rss_mb`: peak resident set size in MB, rendered as GB (`RSS 1.8 GB`).
      - `mps_peak_mb`: peak MPS (GPU) memory in MB, rendered as GB (`MPS 2.9 GB`);
        commonly `None` on CPU-only runs, in which case this segment is omitted.
      - `swap_start_gb`, `swap_end_gb`: swap usage in GB at the start/end of the
        run, rendered together as `swap 0.6→0.7 GB` (only if both are present).
      - `swap_peak_gb`: peak swap in GB during the run, appended as `(peak 0.8)`
        to the swap segment (only if the swap segment itself is present).
      - `pressure_max`: the maximum observed memory-pressure level (e.g. `warn`,
        `critical`, `normal`), rendered as `pressure max <value>`.

    `None` or an empty dict renders to `""` (an empty registry cell), matching
    what a run with no attached resource summary should show.

    Example (every field present):
    `"3.1 h · RSS 1.8 GB · MPS 2.9 GB · swap 0.6→0.7 GB (peak 0.8) · pressure max warn"`
    """
    if not resources:
        return ""

    parts: list[str] = []

    wall_time_s = resources.get("wall_time_s")
    if wall_time_s is not None:
        parts.append(_format_duration(float(wall_time_s)))

    peak_rss_mb = resources.get("peak_rss_mb")
    if peak_rss_mb is not None:
        parts.append(f"RSS {float(peak_rss_mb) / 1024:.1f} GB")

    mps_peak_mb = resources.get("mps_peak_mb")
    if mps_peak_mb is not None:
        parts.append(f"MPS {float(mps_peak_mb) / 1024:.1f} GB")

    swap_start_gb = resources.get("swap_start_gb")
    swap_end_gb = resources.get("swap_end_gb")
    if swap_start_gb is not None and swap_end_gb is not None:
        swap_part = f"swap {float(swap_start_gb):.1f}\u2192{float(swap_end_gb):.1f} GB"
        swap_peak_gb = resources.get("swap_peak_gb")
        if swap_peak_gb is not None:
            swap_part += f" (peak {float(swap_peak_gb):.1f})"
        parts.append(swap_part)

    pressure_max = resources.get("pressure_max")
    if pressure_max is not None:
        parts.append(f"pressure max {pressure_max}")

    return " \u00b7 ".join(parts)


def _read_registry_rows(path: Path) -> tuple[str, list[dict[str, str]]]:
    """Returns (paragraph_text, rows). `rows` preserve column order via dicts."""
    text = path.read_text()
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("| id ")), None)
    if header_idx is None:
        # No table yet; everything is paragraph.
        return text if text.endswith("\n") else text + "\n", []

    raw_paragraph = "\n".join(lines[:header_idx]).strip("\n")
    paragraph = (raw_paragraph + "\n\n") if raw_paragraph else ""

    rows = []
    for ln in lines[header_idx + 2 :]:
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != len(_REGISTRY_COLUMNS):
            continue
        rows.append(dict(zip(_REGISTRY_COLUMNS, cells, strict=True)))
    return paragraph, rows


def _write_registry(path: Path, paragraph: str, rows: list[dict[str, str]]) -> None:
    body = paragraph if paragraph.endswith("\n\n") or paragraph == "" else paragraph + "\n"
    if not body:
        body = _REGISTRY_HEADER_PARAGRAPH
    out = body + _registry_table_header()
    for row in rows:
        out += "| " + " | ".join(_escape_cell(row.get(c, "")) for c in _REGISTRY_COLUMNS) + " |\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".registry-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(out)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def registry_append(
    exp_id: str,
    date: str,
    system: str,
    experiment: str,
    config_summary: str = "",
    key_eers: str = "",
    status: str = "running",
    machine: str = "",
    resources: dict[str, Any] | None = None,
    notes: str = "",
    root: str | Path = "experiments",
) -> None:
    """Append one row to `<root>/REGISTRY.md`, creating the file if absent.

    `status` must be one of `running`, `done`, `failed`, `superseded`. Creates the
    file with the standard header paragraph + empty table first if it does not exist
    yet (so tests and fresh checkouts both work without a separate init step).

    `resources` is an optional resource-summary dict (see `format_resources` for
    the expected keys); it is rendered into the `resources` column via
    `format_resources`, never stored as a raw dict in the markdown table. Pass
    `None` (default) to leave the cell empty (e.g. a `running` row before the
    run has finished measuring resources).
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {_VALID_STATUSES}; got {status!r}")

    path = Path(root) / "REGISTRY.md"
    if path.exists():
        paragraph, rows = _read_registry_rows(path)
    else:
        paragraph, rows = _REGISTRY_HEADER_PARAGRAPH, []

    if any(r["id"] == exp_id for r in rows):
        raise ValueError(f"{exp_id} already has a row in {path}; use registry_update instead")

    rows.append(
        {
            "id": exp_id,
            "date": date,
            "system": system,
            "experiment": experiment,
            "config summary": config_summary,
            "key EERs": key_eers,
            "status": status,
            "machine": machine,
            "resources": format_resources(resources),
            "notes": notes,
        }
    )
    _write_registry(path, paragraph, rows)


def registry_update(exp_id: str, root: str | Path = "experiments", **fields: Any) -> None:
    """Edit `exp_id`'s row in place, keeping `<root>/REGISTRY.md` a valid table.

    `fields` keys must be registry columns other than `id` (e.g. `status="done"`,
    key_eers=... note: use the exact column names with spaces, e.g.
    `**{"key EERs": "eval=3.2"}`, or the convenience snake_case aliases
    `config_summary`, `key_eers` which are mapped onto their spaced column names).
    `resources=<dict>` is rendered via `format_resources` before being written
    (a plain string is also accepted as-is, for round-tripping an already-rendered
    cell without reformatting).
    """
    path = Path(root) / "REGISTRY.md"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; nothing to update")

    alias = {"config_summary": "config summary", "key_eers": "key EERs"}
    normalized = {alias.get(k, k): v for k, v in fields.items()}

    unknown = set(normalized) - set(_REGISTRY_COLUMNS)
    if unknown:
        raise ValueError(f"unknown registry column(s): {sorted(unknown)}")
    if "status" in normalized and normalized["status"] not in _VALID_STATUSES:
        raise ValueError(f"status must be one of {_VALID_STATUSES}; got {normalized['status']!r}")
    if "resources" in normalized and isinstance(normalized["resources"], dict):
        normalized["resources"] = format_resources(normalized["resources"])

    paragraph, rows = _read_registry_rows(path)
    for row in rows:
        if row["id"] == exp_id:
            row.update({k: str(v) for k, v in normalized.items()})
            break
    else:
        raise ValueError(f"{exp_id} has no row in {path}; use registry_append first")

    _write_registry(path, paragraph, rows)
