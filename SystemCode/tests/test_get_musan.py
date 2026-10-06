"""Tests for `scripts/get_musan.py`: argument parsing and tar-member filtering only.

No network access: the download path is never exercised here. A tiny synthetic tar (a
handful of `musan/noise/*`, `musan/speech/*`, `musan/music/*` members) stands in for the
real 11 GB archive.
"""

from __future__ import annotations

import importlib.util
import sys
import tarfile
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "get_musan.py"
_spec = importlib.util.spec_from_file_location("get_musan", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
get_musan_mod = importlib.util.module_from_spec(_spec)
sys.modules["get_musan"] = get_musan_mod
_spec.loader.exec_module(get_musan_mod)

noise_members = get_musan_mod.noise_members
extract_noise_only = get_musan_mod.extract_noise_only
count_noise_files = get_musan_mod.count_noise_files
main = get_musan_mod.main


def _make_synthetic_tar(tmp_path: Path) -> Path:
    tar_path = tmp_path / "musan.tar.gz"
    src_dir = tmp_path / "src"
    members = {
        "musan/noise/free-sound/noise-free-sound-0000.wav": b"noise-a",
        "musan/noise/free-sound/noise-free-sound-0001.wav": b"noise-b",
        "musan/noise/sound-bible/noise-sound-bible-0000.wav": b"noise-c",
        "musan/speech/us-gov/speech-us-gov-0000.wav": b"speech-a",
        "musan/music/fma/music-fma-0000.wav": b"music-a",
        "musan/README": b"readme",
    }
    with tarfile.open(tar_path, "w:gz") as tar:
        for name, content in members.items():
            file_path = src_dir / name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(content)
            tar.add(file_path, arcname=name)
    return tar_path


def test_noise_members_filters_to_noise_prefix_only(tmp_path):
    tar_path = _make_synthetic_tar(tmp_path)
    with tarfile.open(tar_path, "r:gz") as tar:
        members = noise_members(tar)
    names = sorted(m.name for m in members)
    assert names == [
        "musan/noise/free-sound/noise-free-sound-0000.wav",
        "musan/noise/free-sound/noise-free-sound-0001.wav",
        "musan/noise/sound-bible/noise-sound-bible-0000.wav",
    ]


def test_extract_noise_only_writes_only_noise_files(tmp_path):
    tar_path = _make_synthetic_tar(tmp_path)
    extract_root = tmp_path / "extracted"
    extracted = extract_noise_only(tar_path, extract_root)

    assert len(extracted) == 3
    for p in extracted:
        assert p.exists()
        assert "noise" in str(p)

    # speech/ and music/ subtrees must never be written.
    assert not (extract_root / "musan" / "speech").exists()
    assert not (extract_root / "musan" / "music").exists()
    assert not (extract_root / "musan" / "README").exists()


def test_count_noise_files_counts_extracted_files(tmp_path):
    tar_path = _make_synthetic_tar(tmp_path)
    extract_root = tmp_path / "extracted"
    extract_noise_only(tar_path, extract_root)
    n = count_noise_files(extract_root / "musan" / "noise")
    assert n == 3


def test_count_noise_files_missing_dir_returns_zero(tmp_path):
    assert count_noise_files(tmp_path / "does_not_exist") == 0


def test_extract_noise_only_rejects_path_traversal(tmp_path):
    """A malicious/corrupt member name with '..' must never escape extract_root."""
    tar_path = tmp_path / "evil.tar.gz"
    src_dir = tmp_path / "src"
    evil_path = src_dir / "evil.wav"
    evil_path.parent.mkdir(parents=True, exist_ok=True)
    evil_path.write_bytes(b"evil")

    with tarfile.open(tar_path, "w:gz") as tar:
        info = tar.gettarinfo(evil_path, arcname="musan/noise/../../../evil.wav")
        with open(evil_path, "rb") as f:
            tar.addfile(info, f)

    extract_root = tmp_path / "extracted"
    with pytest.raises(tarfile.TarError):
        extract_noise_only(tar_path, extract_root)


def test_dry_run_does_not_touch_network_or_disk(tmp_path, capsys):
    exit_code = main(
        [
            "--dry-run",
            "--tarball-path",
            str(tmp_path / "musan.tar.gz"),
            "--extract-root",
            str(tmp_path / "extract"),
            "--noise-dir",
            str(tmp_path / "extract" / "musan" / "noise"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "[dry-run]" in captured.out
    assert not (tmp_path / "musan.tar.gz").exists()
    assert not (tmp_path / "extract").exists()


def test_dry_run_mentions_keep_tarball_flag(tmp_path, capsys):
    main(["--dry-run", "--keep-tarball", "--tarball-path", str(tmp_path / "musan.tar.gz")])
    captured = capsys.readouterr()
    assert "skipped: --keep-tarball" in captured.out


def test_argument_defaults():
    parser_args = main(["--dry-run"])
    assert parser_args == 0
