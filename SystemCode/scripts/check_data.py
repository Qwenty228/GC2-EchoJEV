"""Read-only dataset inventory and manifest validation for EchoJEV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"
AUDIO_EXT = {".flac", ".wav", ".mp3", ".opus", ".ogg", ".m4a"}
DOWNLOAD = "uv run download --asvspoof2019\nuv run data-prepare"
SPLITS = {
    "train": ("ASVspoof2019_LA_train/flac", 25380),
    "dev": ("ASVspoof2019_LA_dev/flac", 24844),
    "eval": ("ASVspoof2019_LA_eval/flac", 71237),
}
SPOOFCELEB_SPLITS = {
    "train": ("spoofceleb_train.csv", 2_540_421),
    "development": ("spoofceleb_development.csv", 55_741),
    "evaluation": ("spoofceleb_evaluation.csv", 91_130),
}
SPOOFCELEB_CHECKSUMS = "SHA256SUMS"
PHONEMEDF_SPLITS = {
    "train": "phonemedf_train.csv",
    "development": "phonemedf_development.csv",
    "evaluation": "phonemedf_evaluation.csv",
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


def check_spoofceleb_manifest(name: str, expected: int, *, paths: bool) -> bool:
    """Stream rows and retain IDs only for duplicate detection in large SpoofCeleb manifests."""
    file = MANIFESTS / name
    if not file.is_file():
        print(f"  MISSING manifest: {file.relative_to(ROOT)} (run data-prepare --dataset spoofceleb)")
        return False

    count = 0
    labels: Counter[str] = Counter()
    ids: set[str] = set()
    duplicate_ids = False
    missing_audio = 0
    with file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "label", "utt_id"}.issubset(reader.fieldnames or []):
            print(f"  INVALID manifest columns: {name}")
            return False
        for row in reader:
            count += 1
            labels[row["label"]] += 1
            utt_id = row["utt_id"]
            duplicate_ids |= utt_id in ids
            ids.add(utt_id)
            if paths and not (ROOT / row["path"]).resolve().is_file():
                missing_audio += 1

    ok = (
        count == expected
        and count > 0
        and not duplicate_ids
        and set(labels).issubset({"bonafide", "spoof"})
        and labels["bonafide"] > 0
        and labels["spoof"] > 0
        and missing_audio == 0
    )
    summary = f"  {name}: {count:,} rows; labels={dict(labels)}; expected={expected:,}"
    if paths:
        summary += f"; missing audio={missing_audio:,}"
    print(summary + f' [{"OK" if ok else "ERROR"}]')
    if duplicate_ids:
        print("    ERROR: duplicate utterance IDs")
    return ok


def verify_spoofceleb_checksums(
    manifests_dir: Path = MANIFESTS, repo_root: Path = ROOT
) -> bool:
    """Require and verify checksum entries for all prepared SpoofCeleb split manifests."""
    manifest_paths = [manifests_dir / filename for filename, _ in SPOOFCELEB_SPLITS.values()]
    sums_path = manifests_dir / SPOOFCELEB_CHECKSUMS
    manifests_present = any(path.exists() for path in manifest_paths)
    if not manifests_present:
        print("  SpoofCeleb checksums: not prepared (optional dataset)")
        return True
    if not sums_path.is_file():
        print(f"  SpoofCeleb checksums: MISSING {sums_path}")
        return False

    entries: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, separator, relative_path = line.partition("  ")
        if separator and relative_path.startswith("data/manifests/spoofceleb_"):
            entries[relative_path] = digest

    ok = True
    for manifest_path in manifest_paths:
        relative_path = manifest_path.relative_to(repo_root).as_posix()
        expected_digest = entries.get(relative_path)
        if expected_digest is None:
            print(f"  SpoofCeleb checksum: MISSING entry for {relative_path}")
            ok = False
            continue
        if not manifest_path.is_file():
            print(f"  SpoofCeleb checksum: MISSING manifest {manifest_path}")
            ok = False
            continue
        digest = hashlib.sha256()
        with manifest_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != expected_digest:
            print(
                f"  SpoofCeleb checksum: MISMATCH {relative_path} "
                f"(expected {expected_digest}, got {actual_digest})"
            )
            ok = False
        else:
            print(f"  SpoofCeleb checksum: OK {relative_path}")
    return ok


def check_phonemedf_manifests(*, paths: bool) -> bool:
    """Validate PhonemeDF split manifests and ensure speakers are held out across splits."""
    split_speakers: dict[str, set[str]] = {}
    ok = True
    for split, filename in PHONEMEDF_SPLITS.items():
        file = MANIFESTS / filename
        if not file.is_file():
            print(f"  MISSING manifest: {file.relative_to(ROOT)} (run data-prepare --dataset phonemedf)")
            ok = False
            continue

        count = 0
        labels: Counter[str] = Counter()
        ids: set[str] = set()
        speakers: set[str] = set()
        duplicate_ids = False
        missing_audio = 0
        with file.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"utt_id", "path", "label", "speaker_id"}
            if not required.issubset(reader.fieldnames or []):
                print(f"  INVALID manifest columns: {filename}")
                ok = False
                continue
            for row in reader:
                count += 1
                labels[row["label"]] += 1
                duplicate_ids |= row["utt_id"] in ids
                ids.add(row["utt_id"])
                speakers.add(row["speaker_id"])
                if paths and not (ROOT / row["path"]).resolve().is_file():
                    missing_audio += 1

        split_speakers[split] = speakers
        valid = (
            count > 0
            and not duplicate_ids
            and set(labels).issubset({"bonafide", "spoof"})
            and labels["bonafide"] > 0
            and labels["spoof"] > 0
            and missing_audio == 0
        )
        ok &= valid
        summary = f"  {filename}: {count:,} rows; labels={dict(labels)}; speakers={len(speakers)}"
        if paths:
            summary += f"; missing audio={missing_audio:,}"
        print(summary + f' [{"OK" if valid else "ERROR"}]')
        if duplicate_ids:
            print("    ERROR: duplicate utterance IDs")

    split_names = list(split_speakers)
    for index, split in enumerate(split_names):
        for other in split_names[index + 1 :]:
            overlap = split_speakers[split] & split_speakers[other]
            if overlap:
                print(
                    f"    ERROR: PhonemeDF speaker overlap between {split} and {other} "
                    f"({len(overlap)} speakers)"
                )
                ok = False
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
    all_ok = required_ok

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
    spoofceleb_manifests = any(
        (MANIFESTS / filename).exists() for filename, _ in SPOOFCELEB_SPLITS.values()
    )
    if spoofceleb_manifests:
        print("\nSpoofCeleb manifests")
        for filename, expected in SPOOFCELEB_SPLITS.values():
            all_ok &= check_spoofceleb_manifest(filename, expected, paths=args.paths)
    phonemedf_manifests = any(
        (MANIFESTS / filename).exists() for filename in PHONEMEDF_SPLITS.values()
    )
    if phonemedf_manifests:
        print("\nPhonemeDF manifests")
        all_ok &= check_phonemedf_manifests(paths=args.paths)
    if args.verify:
        print("\nSpoofCeleb manifest checksum verification")
        all_ok &= verify_spoofceleb_checksums()

    if args.verify:
        print("\nManifest SHA-256 verification (existing implementation)", flush=True)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "prepare_data.py"), "--verify"],
            cwd=ROOT, check=False,
        )
        all_ok &= result.returncode == 0
    else:
        print("\nFor SHA-256 verification: uv run data-check --verify")
        print("For optional audio path validation: uv run data-check --paths")
    if not all_ok:
        raise SystemExit(1)
    print("\nData checks passed.")


if __name__ == "__main__":
    main()
