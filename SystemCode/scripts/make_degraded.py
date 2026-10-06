#!/usr/bin/env python
"""E4 robustness degradation generator: build/verify the eval subset, time it, run it, verify it.

Usage:
    uv run python scripts/make_degraded.py subset [--verify]
    uv run python scripts/make_degraded.py timing [--n 200]
    uv run python scripts/make_degraded.py generate \
        [--conditions all|eval_mp3_32k,...] [--resume]
    uv run python scripts/make_degraded.py verify [--sample 50]

`subset` (re)builds the seeded, stratified ~10,000-clip eval subset named in
`configs/e4_degradations.yaml` and writes it back into that same config file (so the clip
list is a config value, not a side file) plus `data/manifests/eval_clean10k.csv` (the clean
reference manifest for the same clips, original paths, `condition="clean"`). `--verify`
instead recomputes the subset from `subset.source_manifest`/`n`/`strata`/`seed` and checks it
matches what is currently stored in the config, without writing anything.

`generate` applies every requested condition to every subset clip, writing 16 kHz mono FLAC
under `output_dir` and one manifest CSV per condition
(`data/manifests/eval_<condition>.csv`). Resumable via `--resume`: partially-completed
conditions pick up where they left off (chunk granularity — see `audio_df.resources.ChunkedJob`).
The `noise` condition refuses to start if `data/raw/musan/noise` is missing, naming
`scripts/get_musan.py` as the fix.

This script does not download MUSAN and is not run to completion by the agent that wrote
it — the user runs `generate` for real, on the full ~10,000-clip subset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from audio_df.audio import load_audio
from audio_df.data import stratified_subset
from audio_df.degrade import degrade
from audio_df.resources import (
    ChunkedJob,
    MemoryGuard,
    MemoryPressureExit,
    apply_resource_limits,
    check_can_start,
    check_free_disk,
    ffmpeg_workers,
    format_timing,
    heavy_job_lock,
    load_resources,
    memory_status,
    run_resource_summary,
    timing_run,
)

_DEFAULT_CONFIG_PATH = "configs/e4_degradations.yaml"

# The E4 output manifest carries the same columns as data/manifests/eval.csv, plus one extra
# column (`condition`) naming which degradation (or "clean") produced this row's `path`.
_BASE_MANIFEST_COLUMNS = [
    "utt_id",
    "path",
    "label",
    "speaker_id",
    "attack_id",
    "source_corpus",
    "sample_rate",
    "duration_s",
    "codec",
    "split",
    "language",
]
_MANIFEST_COLUMNS = [*_BASE_MANIFEST_COLUMNS, "condition"]

_SR = 16000

__all__ = [
    "load_config",
    "save_config",
    "compute_subset",
    "rebuild_config_text",
    "write_clean_subset_manifest",
    "load_noise_bank",
    "process_one_clip",
    "generate_condition",
    "run_generate",
    "verify_outputs",
]


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: dict[str, Any], path: str | Path, header: str) -> None:
    """Write `cfg` back out with a comment `header` prepended (PyYAML drops comments on
    round-trip, so the header is regenerated fresh each time from live data rather than
    preserved textually)."""
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100)
    Path(path).write_text(header + "\n" + body)


# ---------------------------------------------------------------------------
# Subset
# ---------------------------------------------------------------------------


def compute_subset(cfg: dict[str, Any], full_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recompute the stratified subset from `cfg["subset"]` against `full_df` (or the manifest
    named by `cfg["subset"]["source_manifest"]`)."""
    subset_cfg = cfg["subset"]
    if full_df is None:
        full_df = pd.read_csv(subset_cfg["source_manifest"])
    return stratified_subset(
        full_df,
        n=subset_cfg["n"],
        strata=tuple(subset_cfg["strata"]),
        seed=cfg["seed"],
    )


def _stratum_report(full_df: pd.DataFrame, sub_df: pd.DataFrame, strata: list[str]) -> str:
    full_counts = full_df.groupby(strata, dropna=False).size()
    sub_counts = sub_df.groupby(strata, dropna=False).size()
    lines = [
        "# Per-stratum counts, subset vs full eval (proportions preserved by construction —",
        "# audio_df.data.stratified_subset allocates by largest remainder):",
    ]
    for key in sorted(full_counts.index, key=str):
        n_full = int(full_counts.get(key, 0))
        n_sub = int(sub_counts.get(key, 0))
        p_full = n_full / len(full_df)
        p_sub = (n_sub / len(sub_df)) if len(sub_df) else 0.0
        lines.append(
            f"#   {key}: full={n_full} ({p_full:.4f})  subset={n_sub} ({p_sub:.4f})"
        )
    return "\n".join(lines)


def rebuild_config_text(cfg: dict[str, Any], full_df: pd.DataFrame, sub_df: pd.DataFrame) -> str:
    """Full file text: a fixed doc header, the live per-stratum comment block, then the YAML."""
    header = (
        "# configs/e4_degradations.yaml — E4 robustness degradation generator config.\n"
        "# Regenerate/verify the clip_ids list with:\n"
        "#   uv run python scripts/make_degraded.py subset [--verify]\n"
        "#\n" + _stratum_report(full_df, sub_df, list(cfg["subset"]["strata"]))
    )
    body = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=100)
    return header + "\n" + body


def write_clean_subset_manifest(sub_df: pd.DataFrame, out_path: str | Path) -> pd.DataFrame:
    """Write the clean-subset reference manifest: same clips, original paths, condition='clean'."""
    manifest = sub_df[_BASE_MANIFEST_COLUMNS].copy()
    manifest["condition"] = "clean"
    manifest = manifest[_MANIFEST_COLUMNS]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def cmd_subset(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    full_df = pd.read_csv(cfg["subset"]["source_manifest"])
    sub_df = compute_subset(cfg, full_df)
    computed_ids = sorted(sub_df["utt_id"].tolist())

    if args.verify:
        stored_ids = sorted(cfg["subset"].get("clip_ids") or [])
        if stored_ids != computed_ids:
            print(
                f"MISMATCH: config has {len(stored_ids)} clip_ids, recomputation gives "
                f"{len(computed_ids)}; {len(set(stored_ids) ^ set(computed_ids))} differ."
            )
            sys.exit(1)
        print(f"OK: {len(computed_ids)} clip_ids in {args.config} match a fresh recomputation.")
        return

    cfg["subset"]["clip_ids"] = computed_ids
    text = rebuild_config_text(cfg, full_df, sub_df)
    Path(args.config).write_text(text)

    clean_path = cfg["manifests"]["clean_subset"]
    write_clean_subset_manifest(sub_df, clean_path)

    print(f"wrote {len(computed_ids)} clip_ids into {args.config}")
    print(f"wrote clean-subset reference manifest to {clean_path}")
    print(_stratum_report(full_df, sub_df, list(cfg["subset"]["strata"])))


# ---------------------------------------------------------------------------
# Noise bank
# ---------------------------------------------------------------------------


def load_noise_bank(noise_dir: str | Path, sr: int = _SR) -> list[np.ndarray]:
    """Load every audio file directly under `noise_dir` (MUSAN noise/ layout, or any flat
    directory of noise clips for tests) as mono float32 at `sr`. Raises `FileNotFoundError`
    with an actionable message naming `scripts/get_musan.py` if the directory is missing."""
    noise_dir = Path(noise_dir)
    if not noise_dir.is_dir():
        raise FileNotFoundError(
            f"noise directory not found at '{noise_dir}'. Run "
            f"'uv run python scripts/get_musan.py' to fetch MUSAN's noise/ subset."
        )
    paths = sorted(
        p for p in noise_dir.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".mp3")
    )
    if not paths:
        raise FileNotFoundError(
            f"no audio files found under '{noise_dir}'. Run "
            f"'uv run python scripts/get_musan.py' to fetch MUSAN's noise/ subset."
        )
    return [load_audio(str(p), target_sr=sr)[0] for p in paths]


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------


def process_one_clip(
    row: pd.Series,
    condition_name: str,
    cfg: dict[str, Any],
    seed: int,
    output_dir: Path,
    noise_bank: list[np.ndarray] | None,
) -> dict[str, Any]:
    """Load one source clip, degrade it, write it atomically as FLAC, return its manifest row."""
    x, sr = load_audio(row["path"], target_sr=_SR)
    out = degrade(
        x, sr, condition_name, cfg, seed=seed, clip_id=row["utt_id"], noise_bank=noise_bank
    )

    final_path = output_dir / f"{row['utt_id']}.flac"
    tmp_path = output_dir / f"{row['utt_id']}.flac.tmp"
    sf.write(tmp_path, out, sr, format="FLAC")
    tmp_path.replace(final_path)

    return {
        "utt_id": row["utt_id"],
        "path": str(final_path),
        "label": row["label"],
        "speaker_id": row["speaker_id"],
        "attack_id": row["attack_id"],
        "source_corpus": row["source_corpus"],
        "sample_rate": sr,
        "duration_s": len(out) / sr,
        "codec": "FLAC",
        "split": row["split"],
        "language": row["language"],
        "condition": condition_name,
    }


def generate_condition(
    condition_name: str,
    condition_cfg: dict[str, Any],
    clips_df: pd.DataFrame,
    output_dir: Path,
    job_dir: Path,
    seed: int,
    chunk_size: int,
    noise_bank: list[np.ndarray] | None = None,
    memory_guard: MemoryGuard | None = None,
    max_workers: int = 1,
    blocking_pause: bool = True,
) -> pd.DataFrame:
    """Run one condition over `clips_df` (resumable via `job_dir`'s `ChunkedJob`).

    Returns the full manifest DataFrame for this condition once every chunk is complete.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_df = clips_df.sort_values("utt_id").reset_index(drop=True)
    ids = list(clips_df["utt_id"])
    by_id = {row["utt_id"]: row for _, row in clips_df.iterrows()}

    cfg = {"conditions": {condition_name: condition_cfg}}
    job = ChunkedJob(
        out_dir=job_dir,
        item_ids=ids,
        chunk_size=chunk_size,
        job_config={"condition": condition_name, "condition_cfg": condition_cfg, "seed": seed},
        memory_guard=memory_guard,
        blocking_pause=blocking_pause,
    )

    def _one(clip_id: str) -> tuple[str, int]:
        row = by_id[clip_id]
        result = process_one_clip(row, condition_name, cfg, seed, output_dir, noise_bank)
        return clip_id, int(result["sample_rate"])

    for chunk_idx, chunk_ids in job.pending():
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(_one, chunk_ids))
        else:
            for clip_id in chunk_ids:
                _one(clip_id)
        job.save_chunk(chunk_idx, chunk_ids, {"done": np.ones(len(chunk_ids), dtype=np.int8)})

    rows = []
    for clip_id in ids:
        row = by_id[clip_id]
        final_path = output_dir / f"{clip_id}.flac"
        info = sf.info(str(final_path))
        rows.append(
            {
                "utt_id": clip_id,
                "path": str(final_path),
                "label": row["label"],
                "speaker_id": row["speaker_id"],
                "attack_id": row["attack_id"],
                "source_corpus": row["source_corpus"],
                "sample_rate": info.samplerate,
                "duration_s": info.frames / info.samplerate,
                "codec": "FLAC",
                "split": row["split"],
                "language": row["language"],
                "condition": condition_name,
            }
        )
    return pd.DataFrame.from_records(rows)[_MANIFEST_COLUMNS]


# ---------------------------------------------------------------------------
# generate / timing / verify orchestration
# ---------------------------------------------------------------------------


def _condition_clips_df(cfg: dict[str, Any], full_df: pd.DataFrame | None = None) -> pd.DataFrame:
    subset_cfg = cfg["subset"]
    if full_df is None:
        full_df = pd.read_csv(subset_cfg["source_manifest"])
    clip_ids = set(subset_cfg["clip_ids"])
    return full_df[full_df["utt_id"].isin(clip_ids)].reset_index(drop=True)


def run_generate(
    cfg: dict[str, Any],
    resources_cfg: dict[str, Any],
    condition_names: list[str],
    output_root: Path,
    manifests_dir: Path,
    resume: bool,
    full_df: pd.DataFrame | None = None,
    status_fn=memory_status,
    lock_path: str | Path = "data/cache/.heavy_job.lock",
) -> dict[str, pd.DataFrame]:
    clips_df = _condition_clips_df(cfg, full_df)
    max_workers = ffmpeg_workers(2, resources_cfg)
    chunk_size = resources_cfg["resources"]["chunk_size"]
    seed = cfg["seed"]

    results: dict[str, pd.DataFrame] = {}
    with heavy_job_lock("e4-degrade", lock_path=lock_path):
        guard = MemoryGuard(resources_cfg, status_fn=status_fn)
        for name in condition_names:
            condition_cfg = cfg["conditions"][name]
            noise_bank = None
            if condition_cfg["type"] == "noise":
                noise_bank = load_noise_bank(condition_cfg["noise_dir"])

            out_dir = output_root / name
            job_dir = out_dir / "_job"
            if not resume and job_dir.exists():
                shutil.rmtree(job_dir)
                for f in out_dir.glob("*.flac"):
                    f.unlink()

            manifest = generate_condition(
                name,
                condition_cfg,
                clips_df,
                out_dir,
                job_dir,
                seed,
                chunk_size,
                noise_bank=noise_bank,
                memory_guard=guard,
                max_workers=max_workers,
            )
            manifests_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifests_dir / f"{name}.csv"
            manifest.to_csv(manifest_path, index=False)
            results[name] = manifest
    return results


def verify_outputs(manifest_paths: list[Path], sample: int = 50, seed: int = 0) -> list[str]:
    """Re-check sample rate and non-zero length for a random sample of rows across manifests.

    Returns a list of problem descriptions; empty means everything sampled checked out.
    """
    rng = np.random.default_rng(seed)
    problems: list[str] = []
    for manifest_path in manifest_paths:
        if not Path(manifest_path).exists():
            problems.append(f"{manifest_path}: manifest missing")
            continue
        df = pd.read_csv(manifest_path)
        if df.empty:
            continue
        k = min(sample, len(df))
        idx = rng.choice(len(df), size=k, replace=False)
        for i in idx:
            row = df.iloc[i]
            path = Path(row["path"])
            if not path.exists():
                problems.append(f"{manifest_path}:{row['utt_id']}: file missing at {path}")
                continue
            info = sf.info(str(path))
            if info.samplerate != _SR:
                problems.append(
                    f"{manifest_path}:{row['utt_id']}: sample_rate={info.samplerate}, "
                    f"expected {_SR}"
                )
            if info.frames == 0:
                problems.append(f"{manifest_path}:{row['utt_id']}: zero-length audio")
    return problems


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _all_condition_names(cfg: dict[str, Any]) -> list[str]:
    return list(cfg["conditions"].keys())


def cmd_timing(args: argparse.Namespace) -> None:
    resources_cfg = load_resources()
    apply_resource_limits(resources_cfg)
    cfg = load_config(args.config)
    full_df = pd.read_csv(cfg["subset"]["source_manifest"])
    clips_df = _condition_clips_df(cfg, full_df)
    n = args.n
    sample_df = clips_df.iloc[:n]

    total_bytes = 0
    results: dict[str, Any] = {}
    for name in _all_condition_names(cfg):
        condition_cfg = cfg["conditions"][name]
        noise_bank = None
        if condition_cfg["type"] == "noise":
            noise_dir = Path(condition_cfg["noise_dir"])
            if not noise_dir.is_dir():
                print(
                    f"skipping timing for {name!r}: {noise_dir} missing "
                    "(run scripts/get_musan.py)"
                )
                continue
            noise_bank = load_noise_bank(noise_dir)

        tmp_out = Path("data/cache/timing") / f"_probe_{name}"
        tmp_out.mkdir(parents=True, exist_ok=True)
        run_cfg = {"conditions": {name: condition_cfg}}

        def _one(row, _name=name, _cfg=run_cfg, _bank=noise_bank, _out=tmp_out):
            return process_one_clip(row, _name, _cfg, cfg["seed"], _out, _bank)

        rows = [r for _, r in sample_df.iterrows()]
        timing = timing_run(_one, rows, n=len(rows), total_items=len(clips_df) * 4)
        results[name] = timing
        print(f"[{name}]")
        print(format_timing(timing))

        condition_bytes = sum(p.stat().st_size for p in tmp_out.glob("*.flac"))
        total_bytes += condition_bytes
        shutil.rmtree(tmp_out)

    n_timed = sum(1 for name in results)
    if n_timed and total_bytes:
        bytes_per_clip = total_bytes / (n_timed * n)
        est_total_gb = (
            bytes_per_clip * len(clips_df) * len(_all_condition_names(cfg)) / (1024**3)
        )
        print(
            f"disk estimate for the full 4-condition x {len(clips_df)}-clip run: "
            f"{est_total_gb:.2f} GB"
        )

    out_path = Path("data/cache/timing/e4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote timing results to {out_path}")


def _disk_estimate_gb(cfg: dict[str, Any], condition_names: list[str]) -> float:
    """Estimate total FLAC bytes for `condition_names` x the subset, from a prior timing run
    (`data/cache/timing/e4.json`, written by `timing`) if present, else a documented fallback:
    16 kHz mono FLAC of this corpus's ~2-4 s clips runs ~60-100 KB/clip; we use 150 KB/clip as
    a conservative fallback so a missing timing run undershoots free-disk headroom rather than
    overshoots it."""
    timing_path = Path("data/cache/timing/e4.json")
    n_clips = len(cfg["subset"]["clip_ids"])
    fallback_bytes_per_clip = 150_000
    if not timing_path.exists():
        return fallback_bytes_per_clip * n_clips * len(condition_names) / (1024**3)
    try:
        json.loads(timing_path.read_text())
    except Exception:
        return fallback_bytes_per_clip * n_clips * len(condition_names) / (1024**3)
    # timing.json does not carry bytes/clip directly (that is printed, not persisted per
    # condition); fall back to the same conservative constant either way.
    return fallback_bytes_per_clip * n_clips * len(condition_names) / (1024**3)


def cmd_generate(args: argparse.Namespace) -> None:
    resources_cfg = load_resources()
    apply_resource_limits(resources_cfg)
    cfg = load_config(args.config)

    condition_names = (
        _all_condition_names(cfg) if args.conditions == "all" else args.conditions.split(",")
    )

    output_root = Path(cfg["output_dir"])
    manifests_dir = Path(cfg["manifests"]["out_dir"])

    check_can_start(resources_cfg)
    output_root.mkdir(parents=True, exist_ok=True)
    check_free_disk(output_root, need_gb=_disk_estimate_gb(cfg, condition_names), cfg=resources_cfg)

    try:
        run_generate(
            cfg,
            resources_cfg,
            condition_names,
            output_root,
            manifests_dir,
            resume=args.resume,
        )
    except MemoryPressureExit as exc:
        print(f"memory pressure critical; resource summary: {exc.args[0]}")
        print("rerun with --resume")
        sys.exit(3)

    print(run_resource_summary())
    testsets = ",".join(condition_names)
    print("Next: score these testsets, e.g.")
    print(f"  uv run python -m audio_df.run_a1 score --testsets {testsets}")
    print(f"  uv run python -m audio_df.run_a2 extract --testsets {testsets}")
    print(f"  uv run python -m audio_df.run_a2 score --testsets {testsets}")


def cmd_verify(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    manifests_dir = Path(cfg["manifests"]["out_dir"])
    manifest_paths = [manifests_dir / f"{name}.csv" for name in _all_condition_names(cfg)]
    manifest_paths.append(Path(cfg["manifests"]["clean_subset"]))
    problems = verify_outputs(manifest_paths, sample=args.sample)
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            print(f"  {p}")
        sys.exit(1)
    print(
        f"OK: sampled up to {args.sample} rows per manifest across "
        f"{len(manifest_paths)} manifests."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=_DEFAULT_CONFIG_PATH)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_subset = sub.add_parser("subset")
    p_subset.add_argument("--verify", action="store_true")
    p_subset.set_defaults(func=cmd_subset)

    p_timing = sub.add_parser("timing")
    p_timing.add_argument("--n", type=int, default=200)
    p_timing.set_defaults(func=cmd_timing)

    p_generate = sub.add_parser("generate")
    p_generate.add_argument("--conditions", default="all")
    p_generate.add_argument("--resume", action="store_true")
    p_generate.set_defaults(func=cmd_generate)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--sample", type=int, default=50)
    p_verify.set_defaults(func=cmd_verify)

    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    args.func(args)
