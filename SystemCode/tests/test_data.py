"""Tests for manifest building, protocol parsing and split hygiene (audio_df/data/manifests.py).

No dataset is required: a tiny fake ASVspoof-shaped protocol file plus fake audio files are
synthesised under `tmp_path` for every test. Real-data tests would be marked
`@pytest.mark.needs_data`, but none of the required behaviour here needs the real corpus.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from audio_df.data.manifests import (
    DatasetNotFoundError,
    assert_no_speaker_overlap,
    build_itw_manifest,
    build_manifest,
    holdout_attack_families,
    manifest_checksums,
    parse_asvspoof_protocol,
    split_hash,
    stratified_subset,
    summarise_manifest,
    verify_checksums,
    write_checksums,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _real_manifest_columns() -> list[str]:
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data.yaml").read_text())
    return list(cfg["manifest"]["columns"])


def _write_protocol(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _write_fake_audio(path: Path, seconds: float = 0.5, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    x = rng.standard_normal(int(seconds * sr)).astype(np.float32) * 0.01
    sf.write(str(path), x, sr)


# A handful of protocol lines in the real 5-field format:
# SPEAKER_ID AUDIO_FILE_NAME - SYSTEM_ID KEY
_TRAIN_LINES = [
    "LA_0001 LA_T_0001 - - bonafide",
    "LA_0001 LA_T_0002 - A01 spoof",
    "LA_0002 LA_T_0003 - A05 spoof",
    "LA_0002 LA_T_0004 - A05 spoof",
    "LA_0003 LA_T_0005 - - bonafide",
]


def _build_fake_dataset(tmp_path: Path, splits: dict[str, list[str]]) -> dict:
    """Build a tmp_path ASVspoof-shaped tree and return a cfg dict pointing at it."""
    la_root = tmp_path / "LA"
    protocol_dir = la_root / "ASVspoof2019_LA_cm_protocols"
    split_filenames = {}

    for split, lines in splits.items():
        filename = f"ASVspoof2019.LA.cm.{split}.trn.txt"
        split_filenames[split] = filename
        _write_protocol(protocol_dir / filename, lines)

        flac_dir = la_root / f"ASVspoof2019_LA_{split}" / "flac"
        for line in lines:
            utt_id = line.split(" ")[1]
            _write_fake_audio(flac_dir / f"{utt_id}.wav")

    cfg = {
        "dataset": {
            "name": "asvspoof2019_la",
            "root": str(la_root),
            "protocol_dir": str(protocol_dir),
            "splits": split_filenames,
        },
        "manifest": {"columns": _real_manifest_columns()},
        "seed": 1337,
    }
    return cfg


# ------------------------------------------------------------------ parse_asvspoof_protocol


def test_parse_protocol_bonafide_gets_dash_attack_id(tmp_path):
    protocol_path = tmp_path / "proto.txt"
    _write_protocol(
        protocol_path,
        [
            "LA_0001 LA_T_0001 - - bonafide",
            "LA_0001 LA_T_0002 - A01 spoof",
        ],
    )
    df = parse_asvspoof_protocol(protocol_path, split="train")
    bonafide_row = df[df["utt_id"] == "LA_T_0001"].iloc[0]
    spoof_row = df[df["utt_id"] == "LA_T_0002"].iloc[0]

    assert bonafide_row["label"] == "bonafide"
    assert bonafide_row["attack_id"] == "-"
    assert spoof_row["label"] == "spoof"
    assert spoof_row["attack_id"] == "A01"


def test_parse_protocol_missing_file_is_actionable(tmp_path):
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(DatasetNotFoundError) as excinfo:
        parse_asvspoof_protocol(missing, split="train")
    assert str(missing) in str(excinfo.value)
    assert "asvspoof" in str(excinfo.value).lower()


# --------------------------------------------------------------------------- build_manifest


def test_build_manifest_has_exact_column_set(tmp_path):
    cfg = _build_fake_dataset(tmp_path, {"train": _TRAIN_LINES})
    df = build_manifest(cfg)
    assert list(df.columns) == _real_manifest_columns()


def test_build_manifest_missing_dataset_root_is_actionable(tmp_path):
    cfg = {
        "dataset": {
            "name": "asvspoof2019_la",
            "root": str(tmp_path / "nope" / "LA"),
            "protocol_dir": str(tmp_path / "nope" / "LA" / "ASVspoof2019_LA_cm_protocols"),
            "splits": {"train": "ASVspoof2019.LA.cm.train.trn.txt"},
        },
        "manifest": {"columns": _real_manifest_columns()},
    }
    with pytest.raises(DatasetNotFoundError) as excinfo:
        build_manifest(cfg)
    message = str(excinfo.value)
    assert str(tmp_path / "nope" / "LA") in message
    assert "asvspoof.org" in message.lower()


def test_build_manifest_row_values(tmp_path):
    cfg = _build_fake_dataset(tmp_path, {"train": _TRAIN_LINES})
    df = build_manifest(cfg)
    assert set(df["label"]) == {"bonafide", "spoof"}
    assert (df.loc[df["label"] == "bonafide", "attack_id"] == "-").all()
    assert set(df.loc[df["label"] == "spoof", "attack_id"]) == {"A01", "A05"}
    assert (df["split"] == "train").all()
    assert (df["sample_rate"] == 16000).all()
    assert (df["duration_s"] > 0).all()


# --------------------------------------------------------------------------- split_hash


def test_split_hash_stable_under_row_shuffle():
    df = pd.DataFrame(
        {
            "utt_id": ["a", "b", "c"],
            "label": ["bonafide", "spoof", "spoof"],
            "attack_id": ["-", "A01", "A02"],
        }
    )
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    assert split_hash(df) == split_hash(shuffled)


def test_split_hash_changes_with_content():
    df = pd.DataFrame(
        {
            "utt_id": ["a", "b", "c"],
            "label": ["bonafide", "spoof", "spoof"],
            "attack_id": ["-", "A01", "A02"],
        }
    )
    changed = df.copy()
    changed.loc[0, "label"] = "spoof"
    assert split_hash(df) != split_hash(changed)


def test_split_hash_is_a_string():
    df = pd.DataFrame({"utt_id": ["a"], "label": ["bonafide"]})
    assert isinstance(split_hash(df), str)
    assert len(split_hash(df)) > 0


# ------------------------------------------------------------------ holdout_attack_families


def test_holdout_attack_families_removes_whole_family():
    df = pd.DataFrame(
        {
            "utt_id": ["a", "b", "c", "d", "e"],
            "label": ["bonafide", "spoof", "spoof", "spoof", "spoof"],
            "attack_id": ["-", "A01", "A05", "A05", "A06"],
        }
    )
    remainder, holdout = holdout_attack_families(df, ["A05"])

    assert (holdout["attack_id"] == "A05").all()
    assert len(holdout) == 2
    assert "A05" not in set(remainder["attack_id"])
    assert set(remainder["attack_id"]) == {"-", "A01", "A06"}


def test_holdout_attack_families_multiple_families():
    df = pd.DataFrame(
        {
            "utt_id": ["a", "b", "c", "d"],
            "label": ["spoof", "spoof", "spoof", "spoof"],
            "attack_id": ["A01", "A02", "A03", "A04"],
        }
    )
    remainder, holdout = holdout_attack_families(df, ["A01", "A02"])
    assert set(holdout["attack_id"]) == {"A01", "A02"}
    assert set(remainder["attack_id"]) == {"A03", "A04"}


# --------------------------------------------------------------- assert_no_speaker_overlap


def test_assert_no_speaker_overlap_raises_and_names_speaker():
    train = pd.DataFrame({"speaker_id": ["LA_0001", "LA_0002"]})
    dev = pd.DataFrame({"speaker_id": ["LA_0002", "LA_0003"]})
    with pytest.raises(ValueError) as excinfo:
        assert_no_speaker_overlap(train, dev)
    assert "LA_0002" in str(excinfo.value)


def test_assert_no_speaker_overlap_passes_when_disjoint():
    train = pd.DataFrame({"speaker_id": ["LA_0001", "LA_0002"]})
    dev = pd.DataFrame({"speaker_id": ["LA_0003", "LA_0004"]})
    assert assert_no_speaker_overlap(train, dev) is None


# --------------------------------------------------------------------------- summarise_manifest


def test_summarise_manifest_counts_by_class_attack_split(tmp_path):
    cfg = _build_fake_dataset(tmp_path, {"train": _TRAIN_LINES})
    df = build_manifest(cfg)
    summary = summarise_manifest(df)

    assert "split" in summary.columns
    assert "label" in summary.columns
    assert "attack_id" in summary.columns
    assert summary["n_utterances"].sum() == len(df)


# ------------------------------------------------------------------------- build_itw_manifest

_ITW_META_ROWS = [
    ("0.wav", "trump", "bona-fide"),
    ("1.wav", "trump", "spoof"),
    ("2.wav", "biden", "spoof"),
    ("3.wav", "biden", "bona-fide"),
]


def _write_itw_fixture(root: Path, *, nested: bool, meta_file: str = "meta.csv") -> None:
    audio_dir = (root / "release_in_the_wild") if nested else root
    audio_dir.mkdir(parents=True, exist_ok=True)
    for fname, _speaker, _label in _ITW_META_ROWS:
        _write_fake_audio(audio_dir / fname)
    meta_df = pd.DataFrame(_ITW_META_ROWS, columns=["file", "speaker", "label"])
    meta_df.to_csv(audio_dir / meta_file, index=False)


def _itw_cfg(root: Path, *, expected_clips: int | None = None) -> dict:
    return {
        "in_the_wild": {
            "root": str(root),
            "meta_file": "meta.csv",
            "expected_clips": expected_clips,
        },
        "manifest": {"columns": _real_manifest_columns()},
    }


def test_build_itw_manifest_flat_layout(tmp_path):
    root = tmp_path / "InTheWild"
    _write_itw_fixture(root, nested=False)
    df = build_itw_manifest(_itw_cfg(root))

    assert list(df.columns) == _real_manifest_columns()
    assert len(df) == len(_ITW_META_ROWS)
    assert set(df["source_corpus"]) == {"InTheWild"}
    assert set(df["split"]) == {"itw"}
    assert set(df["language"]) == {"en"}
    assert all(u.startswith("ITW_") for u in df["utt_id"])


def test_build_itw_manifest_nested_release_layout(tmp_path):
    root = tmp_path / "InTheWild"
    _write_itw_fixture(root, nested=True)
    df = build_itw_manifest(_itw_cfg(root))
    assert len(df) == len(_ITW_META_ROWS)


def test_build_itw_manifest_label_normalisation(tmp_path):
    root = tmp_path / "InTheWild"
    _write_itw_fixture(root, nested=False)
    df = build_itw_manifest(_itw_cfg(root))

    bonafide = df[df["utt_id"] == "ITW_0"].iloc[0]
    spoof = df[df["utt_id"] == "ITW_1"].iloc[0]
    assert bonafide["label"] == "bonafide"
    assert bonafide["attack_id"] == "-"
    assert spoof["label"] == "spoof"
    assert spoof["attack_id"] == "unknown"


def test_build_itw_manifest_missing_meta_is_actionable(tmp_path):
    root = tmp_path / "InTheWild"
    root.mkdir(parents=True)
    _write_fake_audio(root / "0.wav")  # audio present, meta missing
    with pytest.raises(DatasetNotFoundError) as excinfo:
        build_itw_manifest(_itw_cfg(root))
    message = str(excinfo.value)
    assert "meta.csv" in message
    assert "huggingface.co" in message.lower()


def test_build_itw_manifest_missing_root_is_actionable(tmp_path):
    root = tmp_path / "does_not_exist"
    with pytest.raises(DatasetNotFoundError):
        build_itw_manifest(_itw_cfg(root))


def test_build_itw_manifest_warns_on_clip_count_mismatch(tmp_path):
    root = tmp_path / "InTheWild"
    _write_itw_fixture(root, nested=False)
    with pytest.warns(UserWarning, match="expected_clips"):
        build_itw_manifest(_itw_cfg(root, expected_clips=31779))


def test_itw_manifest_column_parity_with_la_manifest(tmp_path):
    la_cfg = _build_fake_dataset(tmp_path / "la", {"train": _TRAIN_LINES})
    la_df = build_manifest(la_cfg)

    itw_root = tmp_path / "InTheWild"
    _write_itw_fixture(itw_root, nested=False)
    itw_df = build_itw_manifest(_itw_cfg(itw_root))

    assert list(la_df.columns) == list(itw_df.columns)


# ----------------------------------------------------------------------------- stratified_subset


def _stratified_fixture() -> pd.DataFrame:
    rows = []
    # 80 bonafide, 15 spoof/A01, 5 spoof/A02 -> 100 rows total
    for i in range(80):
        rows.append({"utt_id": f"b{i:03d}", "label": "bonafide", "attack_id": "-"})
    for i in range(15):
        rows.append({"utt_id": f"s1{i:03d}", "label": "spoof", "attack_id": "A01"})
    for i in range(5):
        rows.append({"utt_id": f"s2{i:03d}", "label": "spoof", "attack_id": "A02"})
    return pd.DataFrame(rows)


def test_stratified_subset_preserves_proportions_approximately():
    df = _stratified_fixture()
    subset = stratified_subset(df, n=20, seed=1337)

    assert len(subset) == 20
    counts = subset.groupby(["label", "attack_id"]).size()
    # 80/15/5 percent of 20 -> ~16/3/1, largest-remainder allocation should land close.
    assert counts.get(("bonafide", "-"), 0) in (15, 16, 17)
    assert counts.get(("spoof", "A01"), 0) in (2, 3, 4)
    assert counts.get(("spoof", "A02"), 0) >= 1  # smallest stratum still represented


def test_stratified_subset_gives_every_stratum_at_least_one_when_n_allows():
    df = _stratified_fixture()
    subset = stratified_subset(df, n=3, seed=1337)
    assert len(subset) == 3
    strata_present = set(subset[["label", "attack_id"]].itertuples(index=False, name=None))
    assert strata_present == {("bonafide", "-"), ("spoof", "A01"), ("spoof", "A02")}


def test_stratified_subset_is_deterministic():
    df = _stratified_fixture()
    a = stratified_subset(df, n=20, seed=1337)
    b = stratified_subset(df, n=20, seed=1337)
    pd.testing.assert_frame_equal(a, b)


def test_stratified_subset_different_seeds_differ():
    df = _stratified_fixture()
    a = stratified_subset(df, n=20, seed=1337)
    b = stratified_subset(df, n=20, seed=7)
    assert list(a["utt_id"]) != list(b["utt_id"])


def test_stratified_subset_sorted_by_utt_id():
    df = _stratified_fixture()
    subset = stratified_subset(df, n=20, seed=1337)
    assert list(subset["utt_id"]) == sorted(subset["utt_id"])


def test_stratified_subset_n_larger_than_df_returns_all_rows():
    df = _stratified_fixture()
    subset = stratified_subset(df, n=1000, seed=1337)
    assert len(subset) == len(df)


# ------------------------------------------------------------------- checksums


def test_manifest_checksums_write_and_verify_roundtrip(tmp_path):
    csv_path = tmp_path / "data" / "manifests" / "train.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame({"utt_id": ["a", "b"], "label": ["bonafide", "spoof"]}).to_csv(
        csv_path, index=False
    )

    entries = manifest_checksums([csv_path], base_dir=tmp_path)
    assert entries[0]["rows"] == 2
    assert len(entries[0]["sha256"]) == 64
    assert entries[0]["file"] == "data/manifests/train.csv"

    sums_path = tmp_path / "data" / "manifests" / "SHA256SUMS"
    write_checksums(entries, sums_path)

    mismatches = verify_checksums(sums_path, base_dir=tmp_path)
    assert mismatches == []


def test_checksum_tamper_detection(tmp_path):
    csv_path = tmp_path / "data" / "manifests" / "train.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame({"utt_id": ["a", "b"], "label": ["bonafide", "spoof"]}).to_csv(
        csv_path, index=False
    )

    entries = manifest_checksums([csv_path], base_dir=tmp_path)
    sums_path = tmp_path / "data" / "manifests" / "SHA256SUMS"
    write_checksums(entries, sums_path)

    # tamper with the manifest after the checksum was written
    pd.DataFrame({"utt_id": ["a", "b", "c"], "label": ["bonafide", "spoof", "spoof"]}).to_csv(
        csv_path, index=False
    )

    mismatches = verify_checksums(sums_path, base_dir=tmp_path)
    assert len(mismatches) == 1
    assert "train.csv" in mismatches[0]


def test_sha256sums_format_verified_by_shasum_binary(tmp_path):
    shasum_bin = shutil.which("shasum") or shutil.which("sha256sum")
    if shasum_bin is None:
        pytest.skip("no shasum/sha256sum binary available on this machine")

    csv_path = tmp_path / "data" / "manifests" / "train.csv"
    csv_path.parent.mkdir(parents=True)
    pd.DataFrame({"utt_id": ["a", "b"], "label": ["bonafide", "spoof"]}).to_csv(
        csv_path, index=False
    )

    entries = manifest_checksums([csv_path], base_dir=tmp_path)
    sums_path = tmp_path / "data" / "manifests" / "SHA256SUMS"
    write_checksums(entries, sums_path)

    if "shasum" in shasum_bin:
        cmd = [shasum_bin, "-a", "256", "-c", str(sums_path)]
    else:
        cmd = [shasum_bin, "-c", str(sums_path)]

    result = subprocess.run(
        cmd, cwd=str(tmp_path), capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
