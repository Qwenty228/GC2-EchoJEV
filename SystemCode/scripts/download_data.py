"""Download and unpack raw datasets into data/raw/ (resumable; finished downloads are skipped).

    uv run download --asvspoof2019      # or --asvspoof2021la, --asvspoof2021df, --musan, --itw, --all

Manual alternative: download the same archives, put them in data/raw/_downloads/ and rerun;
existing archives are unpacked without downloading. Then run `uv run data-prepare`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
DOWNLOADS = RAW / "_downloads"

# name: [(archive file, url, unpack into data/raw/<dir>, only this archive member or None)]
# hf:// urls are Hugging Face parquet datasets: clips are written to <dir>/flac/ plus <dir>/labels.csv.
DATASETS = {
    "asvspoof2019": [
        ("LA.zip", "https://datashare.ed.ac.uk/server/api/core/bitstreams/"
                   "a9f87c35-f055-4015-80e2-2fdff0d46269/content", "ASVspoof2019", None),
    ],
    "asvspoof2021la": [  # audio + official keys (keys are only hosted on asvspoof.org)
        ("ASVspoof2021_LA_eval.tar.gz",
         "https://zenodo.org/records/4837263/files/ASVspoof2021_LA_eval.tar.gz", "ASVspoof2021", None),
        ("LA-keys-full.tar.gz", "https://www.asvspoof.org/asvspoof2021/LA-keys-full.tar.gz", "ASVspoof2021", None),
    ],
    "asvspoof2021df": [  # labels included; audio already re-encoded so libsndfile can read it
        ("ASVspoof2021_DF", "hf://SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF",
         "ASVspoof2021/ASVspoof2021_DF_eval", None),
    ],
    "musan": [  # only the noise portion is used
        ("musan.tar.gz", "https://www.openslr.org/resources/17/musan.tar.gz", ".", "musan/noise"),
    ],
    "itw": [
        ("release_in_the_wild.zip",
         "https://huggingface.co/datasets/mueller91/In-The-Wild/resolve/main/release_in_the_wild.zip",
         "InTheWild", None),
    ],
}


def unpack_hf(snapshot: Path, dest: Path) -> None:
    """Write each parquet clip to dest/flac/ and its label + notes to dest/labels.csv."""
    import pyarrow.parquet as pq

    (dest / "flac").mkdir(parents=True, exist_ok=True)
    rows = []
    for shard in sorted(snapshot.glob("data/test-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["path", "audio", "label", "notes"]):
            for r in batch.to_pylist():
                (dest / "flac" / r["path"]).write_bytes(r["audio"]["bytes"])
                rows.append({"path": f"flac/{r['path']}", "label": ("bonafide", "spoof")[r["label"]],
                             **json.loads(r["notes"])})
    pd.DataFrame(rows).to_csv(dest / "labels.csv", index=False)


def fetch(archive: str, url: str, dest: str, only: str | None) -> None:
    path, done = DOWNLOADS / archive, RAW / dest / f".{archive}.unpacked"
    if done.exists():
        print(f"{archive}: already unpacked")
        return
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    if url.startswith("hf://"):
        from huggingface_hub import snapshot_download
        print(f"{archive}: downloading from Hugging Face (resumable)")
        for attempt in range(5):  # Hub storage drops connections on long downloads; finished files are kept
            try:
                snapshot_download(url[5:], repo_type="dataset", local_dir=path,
                                  allow_patterns=["data/*", "LICENSE.txt"], max_workers=4)
                break
            except Exception as exc:  # noqa: BLE001 - retry any transfer error, re-raise the last
                if attempt == 4:
                    raise
                print(f"{archive}: attempt {attempt + 1} failed ({exc}); retrying")
    elif not path.exists():
        part = path.with_name(archive + ".part")
        print(f"{archive}: downloading (resumes {part.name} if present)")
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "-o", str(part), url], check=True)
        part.rename(path)
    print(f"{archive}: unpacking into data/raw/{dest}")
    (RAW / dest).mkdir(parents=True, exist_ok=True)
    if url.startswith("hf://"):
        unpack_hf(path, RAW / dest)
    elif only is None:
        shutil.unpack_archive(path, RAW / dest, filter="data")
    else:
        with tarfile.open(path) as tar:
            tar.extractall(RAW / dest, members=[m for m in tar if m.name.startswith(only)], filter="data")
    done.touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choice = parser.add_mutually_exclusive_group(required=True)
    for name in [*DATASETS, "all"]:
        choice.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    failed = []
    for name, archives in DATASETS.items():
        if args.all or getattr(args, name):
            for item in archives:
                try:
                    fetch(*item)
                except Exception as exc:  # noqa: BLE001 - report every failed source, keep going
                    failed.append((*item, exc))
    for archive, url, _, _, exc in failed:  # e.g. asvspoof.org down: LA keys are only hosted there
        print(f"FAILED {archive} ({exc}): download it manually from {url} into {DOWNLOADS} and rerun")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
