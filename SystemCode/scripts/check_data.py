"""Read-only dataset inventory and manifest validation for EchoJEV."""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"
AUDIO_EXT = {".flac", ".wav", ".mp3", ".opus", ".ogg", ".m4a"}
DOWNLOAD = (
    "mkdir -p data/raw/_downloads data/raw/ASVspoof2019\n"
    'curl -L -C - -o data/raw/_downloads/LA.zip '
    '"https://datashare.ed.ac.uk/server/api/core/bitstreams/'
    'a9f87c35-f055-4015-80e2-2fdff0d46269/content"\n'
    "unzip -q data/raw/_downloads/LA.zip -d data/raw/ASVspoof2019/\n"
    "uv run data-prepare"
)
SPLITS = {
    "train": ("ASVspoof2019_LA_train/flac", 25380),
    "dev": ("ASVspoof2019_LA_dev/flac", 24844),
    "eval": ("ASVspoof2019_LA_eval/flac", 71237),
}


def audio_files(folder: Path) -> set[Path]:
    if not folder.is_dir():
        return set()
    return {p.resolve() for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT}


def check_manifest(name: str, *, paths: bool, expected: int | None = None,
                   audio_dir: Path | None = None) -> bool:
    file = MANIFESTS / name
    if not file.is_file():
        print(f"  MISSING manifest: {file.relative_to(ROOT)} (run uv run data-prepare)")
        return False
    with file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not {"path", "label", "utt_id"}.issubset(reader.fieldnames or []):
            print(f"  INVALID manifest columns: {name}")
            return False
        rows = list(reader)

    labels = Counter(row["label"] for row in rows)
    ids = [row["utt_id"] for row in rows]
    ok = bool(rows) and len(ids) == len(set(ids)) and set(labels).issubset({"bonafide", "spoof"})
    if expected is not None:
        ok &= len(rows) == expected
    referenced = {(ROOT / row["path"]).resolve() for row in rows}
    # Required ASVspoof splits always check paths; --paths extends this to optional manifests.
    missing = sum(not p.is_file() for p in referenced) if (paths or audio_dir is not None) else 0
    ok &= missing == 0
    extra = len(audio_files(audio_dir) - referenced) if audio_dir is not None else 0
    summary = f"  {name}: {len(rows):,} rows; labels={dict(labels)}"
    if expected is not None:
        summary += f"; expected={expected:,}"
    if paths or audio_dir is not None:
        summary += f"; missing audio={missing:,}"
    print(summary + f' [{"OK" if ok else "ERROR"}]')
    if extra:
        print(f"    WARNING: {extra:,} extra audio files not referenced by this manifest (ignored)")
    if len(ids) != len(set(ids)):
        print("    ERROR: duplicate utterance IDs")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", action="store_true", help="Also check optional manifest audio paths")
    parser.add_argument("--verify", action="store_true", help="Verify SHA256SUMS with existing implementation")
    args = parser.parse_args()
    required_ok = True
    base = RAW / "ASVspoof2019" / "LA"
    print("ASVspoof 2019 LA (required)")
    for split, (subdir, expected) in SPLITS.items():
        folder = base / subdir
        required_ok &= check_manifest(f"{split}.csv", paths=True, expected=expected, audio_dir=folder)
    protocol = base / "ASVspoof2019_LA_cm_protocols"
    protocols_ok = protocol.is_dir() and any(protocol.rglob("*.txt"))
    required_ok &= protocols_ok
    print(f'  protocols: {"OK" if protocols_ok else "MISSING"}')
    if not required_ok:
        print("\nRequired data missing or invalid. Download/repair from SystemCode/:\n" + DOWNLOAD)

    print("\nOptional datasets")
    for name, folder, expected in [
        ("In-the-Wild", RAW / "InTheWild" / "release_in_the_wild", 31779),
        ("MUSAN noise", RAW / "musan" / "noise", 929),
    ]:
        actual = len(audio_files(folder))
        print(f"  {name}: {actual:,}" + (f"/{expected:,}" if folder.is_dir() else " (not installed)"))
    print(f'  ASVspoof 2021: {"present (not validated)" if (RAW / "ASVspoof2021").is_dir() else "not installed"}')
    if (MANIFESTS / "itw.csv").exists():
        print("\nOptional manifest")
        check_manifest("itw.csv", paths=args.paths, expected=31779)

    if args.verify:
        print("\nManifest SHA-256 verification (existing implementation)", flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "prepare_data.py"), "--verify"],
            cwd=ROOT, check=False,
        )
        required_ok &= result.returncode == 0
    else:
        print("\nFor SHA-256 verification: uv run data-check --verify")
        print("For optional audio path validation: uv run data-check --paths")
    if not required_ok:
        raise SystemExit(1)
    print("\nRequired data checks passed.")


if __name__ == "__main__":
    main()
