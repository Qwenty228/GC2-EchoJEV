#!/usr/bin/env python
"""Fetch MUSAN's noise/ subset from OpenSLR SLR17, extract only what E4 needs, then clean up.

Usage:
    uv run python scripts/get_musan.py --dry-run
    uv run python scripts/get_musan.py
    uv run python scripts/get_musan.py --keep-tarball

Downloads `musan.tar.gz` (11 GB; see `Miscellaneous/research/knowledge/datasets/musan.md`) from
OpenSLR SLR17 with resume support (`curl -C -`, so a partial download from a previous interrupted
run is continued rather than restarted), checks free disk against `configs/resources.yaml`'s
floor before starting (the archive plus its extracted `noise/` subset need roughly 11 GB +
1 GB = 12 GB of headroom on top of that floor), extracts ONLY the `musan/noise/` member
prefix into `data/raw/musan/noise` (a `tarfile` member filter — the archive's `speech/` and
`music/` subtrees, which this project never needs, are never written to disk), verifies the
extracted file count against the 929 files documented in
`Miscellaneous/research/knowledge/datasets/musan.md`, then deletes the tarball (kept only if
`--keep-tarball` is passed).

This script is written but NOT run to completion by the agent that wrote it (the user
downloads MUSAN themselves) and its network path has no test coverage here — tests cover
only argument parsing and the tar-member filter, against a tiny synthetic tar in `tmp_path`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

from audio_df.runtime.resources import check_free_disk, load_resources

# OpenSLR SLR17 ("A corpus of music, speech, and noise") download page:
# https://www.openslr.org/17/ — the single-archive URL pattern for SLR resources, confirmed
# for this resource id by the OpenSLR page linked in
# Miscellaneous/research/knowledge/datasets/musan.md.
_MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"

# The technical report (Snyder, Chen, Povey, arXiv:1510.08484) states the noise/ subset
# contains exactly 929 files; see Miscellaneous/research/knowledge/datasets/musan.md.
_EXPECTED_NOISE_FILE_COUNT = 929

_NOISE_MEMBER_PREFIX = "musan/noise/"

_DEFAULT_TARBALL_PATH = "data/raw/musan.tar.gz"
_DEFAULT_EXTRACT_ROOT = "data/raw"
_DEFAULT_NOISE_DIR = "data/raw/musan/noise"

# Rough disk budget: 11 GB compressed archive + ~1 GB for the extracted noise/ subset alone
# (see Miscellaneous/research/knowledge/datasets/musan.md — no official per-subdirectory size
# figure exists, so this is a documented over-estimate, not a measured one).
_TARBALL_GB = 11.0
_EXTRACTED_NOISE_GB = 1.0

__all__ = [
    "noise_members",
    "extract_noise_only",
    "count_noise_files",
    "download_with_resume",
]


def noise_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Filter a `TarFile`'s members to only those under `musan/noise/`."""
    return [m for m in tar.getmembers() if m.name.startswith(_NOISE_MEMBER_PREFIX)]


def extract_noise_only(tarball_path: Path, extract_root: Path) -> list[Path]:
    """Extract only `musan/noise/*` members from `tarball_path` into `extract_root`.

    Returns the list of extracted file paths (directories excluded). Uses `filter="data"`
    (Python 3.12+ `tarfile` extraction hardening — rejects absolute paths / path traversal
    ('..') and device files) so a corrupted or malicious archive cannot write outside
    `extract_root`.
    """
    extract_root = Path(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with tarfile.open(tarball_path, "r:gz") as tar:
        members = noise_members(tar)
        for member in members:
            tar.extract(member, path=extract_root, filter="data")
            if member.isfile():
                extracted.append(extract_root / member.name)
    return extracted


def count_noise_files(noise_dir: Path) -> int:
    noise_dir = Path(noise_dir)
    if not noise_dir.is_dir():
        return 0
    return sum(1 for p in noise_dir.rglob("*") if p.is_file())


def download_with_resume(url: str, dest: Path, dry_run: bool = False) -> None:
    """`curl -C - -o dest url`: resumes a partial download rather than restarting it.

    `-C -` asks curl to auto-detect the resume offset from `dest`'s current size; curl itself
    handles the case where `dest` does not exist yet (starts from zero) or is already complete
    (server returns 416 / curl reports "already downloaded", both non-fatal here).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "-C", "-", "-o", str(dest), url]
    if dry_run:
        print(f"[dry-run] would run: {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True)


def _cfg() -> dict:
    return load_resources()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=_MUSAN_URL)
    parser.add_argument("--tarball-path", default=_DEFAULT_TARBALL_PATH)
    parser.add_argument("--extract-root", default=_DEFAULT_EXTRACT_ROOT)
    parser.add_argument("--noise-dir", default=_DEFAULT_NOISE_DIR)
    parser.add_argument("--keep-tarball", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    tarball_path = Path(args.tarball_path)
    extract_root = Path(args.extract_root)
    noise_dir = Path(args.noise_dir)

    need_gb = _TARBALL_GB + _EXTRACTED_NOISE_GB
    if args.dry_run:
        print(f"[dry-run] would download {args.url} -> {tarball_path} ({_TARBALL_GB:.0f} GB)")
        print(f"[dry-run] would extract '{_NOISE_MEMBER_PREFIX}*' -> {noise_dir}")
        print(f"[dry-run] would verify {_EXPECTED_NOISE_FILE_COUNT} extracted files")
        print(
            f"[dry-run] would delete {tarball_path} afterwards"
            + (" (skipped: --keep-tarball)" if args.keep_tarball else "")
        )
        return 0

    cfg = _cfg()
    check_free_disk(extract_root if extract_root.exists() else Path("."), need_gb, cfg)

    print(f"downloading {args.url} -> {tarball_path} (resumable)")
    download_with_resume(args.url, tarball_path, dry_run=False)

    print(f"extracting '{_NOISE_MEMBER_PREFIX}*' -> {noise_dir}")
    extract_noise_only(tarball_path, extract_root)

    n_files = count_noise_files(noise_dir)
    if n_files != _EXPECTED_NOISE_FILE_COUNT:
        print(
            f"WARNING: extracted {n_files} files under {noise_dir}, expected "
            f"{_EXPECTED_NOISE_FILE_COUNT} "
            "(see Miscellaneous/research/knowledge/datasets/musan.md). "
            "Not deleting the tarball so this can be investigated.",
            file=sys.stderr,
        )
        return 1
    print(f"verified {n_files} files under {noise_dir}")

    if args.keep_tarball:
        print(f"keeping tarball at {tarball_path} (--keep-tarball)")
    else:
        tarball_path.unlink(missing_ok=True)
        print(f"deleted {tarball_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
