#!/usr/bin/env python
"""Build manifest CSVs and print the Stage-0 summary table.

Usage:
    uv run python scripts/prepare_data.py --config configs/data.yaml
    uv run python scripts/prepare_data.py --dataset in_the_wild
    uv run python scripts/prepare_data.py --verify

Writes `data/manifests/<split>.csv` for every split of the chosen `--dataset`
(`asvspoof2019_la`, the default, unchanged from before; or `in_the_wild`, which writes a
single `data/manifests/itw.csv`) and prints the per-class/per-attack/per-speaker/per-split
counts from `audio_df.data.manifests.summarise_manifest`. Raises the same actionable
`DatasetNotFoundError` as `build_manifest`/`build_itw_manifest` if the dataset has not been
downloaded yet — this script does not swallow it.

After writing manifests, (re)writes `data/manifests/SHA256SUMS` — the reproducibility record
for the gitignored manifest CSVs (see `audio_df.data.manifests.write_checksums`) — covering
every CSV
currently in `data/manifests/`. `--verify` instead recomputes checksums for every file listed
in `data/manifests/SHA256SUMS` and exits non-zero if any file is missing or has changed,
without writing or rebuilding any manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from audio_df.data.manifests import (
    build_itw_manifest,
    build_la2021_manifest,
    build_df2021_manifest,
    build_manifest,
    manifest_checksums,
    split_hash,
    summarise_manifest,
    verify_checksums,
    write_checksums,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHECKSUM_FILENAME = "SHA256SUMS"


def _write_manifests(cfg: dict, dataset: str, workers: int = 4) -> Path:
    """Build and write manifest CSV(s) for `dataset`; returns the manifest output directory."""
    out_dir = Path(cfg["manifest"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "asvspoof2019_la":
        manifest = build_manifest(cfg, workers=workers)
        for split, split_df in manifest.groupby("split"):
            out_path = out_dir / f"{split}.csv"
            split_df.to_csv(out_path, index=False)
            print(f"wrote {len(split_df)} rows to {out_path} (split_hash={split_hash(split_df)})")
        print()
        print("Manifest summary (counts by split / class / attack, with speaker counts):")
        print(summarise_manifest(manifest).to_string(index=False))
    elif dataset == "in_the_wild":
        manifest = build_itw_manifest(cfg, workers=workers)
        out_path = out_dir / "itw.csv"
        manifest.to_csv(out_path, index=False)
        print(f"wrote {len(manifest)} rows to {out_path} (split_hash={split_hash(manifest)})")
        print()
        print("Manifest summary (counts by split / class / attack, with speaker counts):")
        print(summarise_manifest(manifest).to_string(index=False))
    elif dataset in ("asvspoof2021", "asvspoof2021df"):
        build = build_la2021_manifest if dataset == "asvspoof2021" else build_df2021_manifest
        manifest = build(cfg, workers=workers)
        # One CSV per key phase/subset; <dataset>_eval.csv is the official scoring subset.
        for split, split_df in manifest.groupby("split"):
            out_path = out_dir / f"{split}.csv"
            split_df.to_csv(out_path, index=False)
            print(f"wrote {len(split_df)} rows to {out_path} (split_hash={split_hash(split_df)})")
        print()
        print("Manifest summary (counts by split / class / attack, with speaker counts):")
        print(summarise_manifest(manifest).to_string(index=False))
    else:
        raise ValueError(f"unknown --dataset {dataset!r}")

    return out_dir


def _write_checksums_for_dir(out_dir: Path) -> Path:
    """Write/update `<out_dir>/SHA256SUMS` covering every `*.csv` currently in `out_dir`."""
    csv_paths = sorted(out_dir.glob("*.csv"))
    entries = manifest_checksums(csv_paths, base_dir=_REPO_ROOT)
    sums_path = out_dir / _CHECKSUM_FILENAME
    write_checksums(entries, sums_path)
    print(f"\nwrote checksums for {len(entries)} file(s) to {sums_path}")
    return sums_path


def _verify(sums_path: Path) -> int:
    mismatches = verify_checksums(sums_path, base_dir=_REPO_ROOT)
    if mismatches:
        print(f"CHECKSUM VERIFICATION FAILED ({len(mismatches)} problem(s)):")
        for msg in mismatches:
            print(f"  - {msg}")
        return 1
    print(f"checksum verification OK: all files listed in {sums_path} match.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the dataset config (default: configs/data.yaml)",
    )
    parser.add_argument(
        "--dataset",
        choices=["asvspoof2019_la", "in_the_wild", "asvspoof2021", "asvspoof2021df"],
        default="asvspoof2019_la",
        help="Which dataset to build a manifest for (default: asvspoof2019_la)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Recompute and compare checksums against data/manifests/SHA256SUMS instead of "
            "(re)building any manifest; exits non-zero on any mismatch."
        ),
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Parallel audio-header readers (default: 4; use 1 for sequential)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["manifest"]["out_dir"])

    if args.verify:
        sums_path = out_dir / _CHECKSUM_FILENAME
        if not sums_path.exists():
            print(f"no checksum file found at {sums_path} — nothing to verify.")
            sys.exit(1)
        sys.exit(_verify(sums_path))

    out_dir = _write_manifests(cfg, args.dataset, workers=args.workers)
    _write_checksums_for_dir(out_dir)


if __name__ == "__main__":
    main()
