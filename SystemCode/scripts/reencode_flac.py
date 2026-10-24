"""Losslessly re-encode a folder of FLAC files with ffmpeg so libsndfile can decode them.

libsndfile 1.2.2 (via soundfile) fails on ~30% of the ASVspoof 2021 LA eval FLACs
("unknown error in flac decoder", "psf_fseek() failed") although ffmpeg decodes them and
they match the official archive byte for byte. Re-encoding gives files soundfile reads
with bit-identical samples. Resumable: finished files are skipped; each file is written
to a temp name, checked with soundfile, then renamed.

    uv run python scripts/reencode_flac.py SRC_DIR DST_DIR [--workers 4]
"""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import soundfile as sf
from tqdm.auto import tqdm


def reencode(job: tuple[Path, Path]) -> str | None:
    src, dst = job
    if dst.exists():
        return None
    tmp = dst.with_name(dst.stem + ".tmp.flac")
    result = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "flac", str(tmp)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        return f"{src.name}: ffmpeg: {result.stderr.strip()}"
    try:
        sf.read(str(tmp), dtype="int16")
    except Exception as exc:  # noqa: BLE001 - report any decode failure, keep going
        return f"{src.name}: still unreadable: {exc}"
    os.replace(tmp, dst)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    jobs = [(p, args.dst / p.name) for p in sorted(args.src.glob("*.flac"))]
    with ProcessPoolExecutor(args.workers) as pool:
        errors = [e for e in tqdm(pool.map(reencode, jobs, chunksize=64), total=len(jobs),
                                  desc="re-encode", unit="file") if e]
    print(f"{len(jobs) - len(errors)}/{len(jobs)} files ok in {args.dst}")
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
