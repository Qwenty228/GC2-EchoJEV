#!/usr/bin/env python
"""Build the ASVspoof manifest CSVs and print the Stage-0 summary table.

Usage:
    uv run --no-sync python scripts/prepare_data.py --config configs/data.yaml

Writes `data/manifests/<split>.csv` for every split in the config and prints the per-class/
per-attack/per-speaker/per-split counts from `audio_df.data.summarise_manifest`. Raises the
same actionable `DatasetNotFoundError` as `build_manifest` if the dataset has not been
downloaded yet — this script does not swallow it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from audio_df.data import build_manifest, split_hash, summarise_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the dataset config (default: configs/data.yaml)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    manifest = build_manifest(cfg)

    out_dir = Path(cfg["manifest"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, split_df in manifest.groupby("split"):
        out_path = out_dir / f"{split}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"wrote {len(split_df)} rows to {out_path} (split_hash={split_hash(split_df)})")

    print()
    print("Manifest summary (counts by split / class / attack, with speaker counts):")
    summary = summarise_manifest(manifest)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
