from __future__ import annotations

import hashlib

from scripts import check_data
from scripts.check_data import verify_spoofceleb_checksums


def test_spoofceleb_checksum_verification_checks_all_split_manifests(tmp_path):
    repo_root = tmp_path
    manifests_dir = repo_root / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    files = [
        "spoofceleb_train.csv",
        "spoofceleb_development.csv",
        "spoofceleb_evaluation.csv",
    ]
    checksum_lines = []
    for filename in files:
        path = manifests_dir / filename
        path.write_text(filename, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  data/manifests/{filename}")
    (manifests_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    assert verify_spoofceleb_checksums(manifests_dir, repo_root)


def test_spoofceleb_checksum_verification_fails_for_missing_entry(tmp_path):
    repo_root = tmp_path
    manifests_dir = repo_root / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    files = [
        "spoofceleb_train.csv",
        "spoofceleb_development.csv",
        "spoofceleb_evaluation.csv",
    ]
    checksum_lines = []
    for filename in files:
        path = manifests_dir / filename
        path.write_text(filename, encoding="utf-8")
        if filename == "spoofceleb_evaluation.csv":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  data/manifests/{filename}")
    (manifests_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")

    assert not verify_spoofceleb_checksums(manifests_dir, repo_root)


def test_spoofceleb_checksum_verification_fails_for_modified_manifest(tmp_path):
    repo_root = tmp_path
    manifests_dir = repo_root / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    files = [
        "spoofceleb_train.csv",
        "spoofceleb_development.csv",
        "spoofceleb_evaluation.csv",
    ]
    checksum_lines = []
    for filename in files:
        path = manifests_dir / filename
        path.write_text(filename, encoding="utf-8")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  data/manifests/{filename}")
    (manifests_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    (manifests_dir / "spoofceleb_train.csv").write_text("changed", encoding="utf-8")

    assert not verify_spoofceleb_checksums(manifests_dir, repo_root)


def test_phonemedf_manifest_validation_requires_speaker_disjoint_splits(tmp_path, monkeypatch):
    repo_root = tmp_path
    manifests_dir = repo_root / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    monkeypatch.setattr(check_data, "ROOT", repo_root)
    monkeypatch.setattr(check_data, "MANIFESTS", manifests_dir)
    for split, filename in check_data.PHONEMEDF_SPLITS.items():
        path = manifests_dir / filename
        path.write_text(
            "utt_id,path,label,speaker_id\n"
            f"{split}-real,data/raw/{split}-real.wav,bonafide,speaker-{split}-real\n"
            f"{split}-spoof,data/raw/{split}-spoof.wav,spoof,speaker-{split}-spoof\n",
            encoding="utf-8",
        )

    assert check_data.check_phonemedf_manifests(paths=False)

    evaluation = manifests_dir / check_data.PHONEMEDF_SPLITS["evaluation"]
    evaluation.write_text(
        "utt_id,path,label,speaker_id\n"
        "evaluation-real,data/raw/evaluation-real.wav,bonafide,speaker-train-real\n"
        "evaluation-spoof,data/raw/evaluation-spoof.wav,spoof,speaker-evaluation-spoof\n",
        encoding="utf-8",
    )
    assert not check_data.check_phonemedf_manifests(paths=False)
