"""Tests for manifest building, protocol parsing, and split hygiene (src/audio_df/data.py).

No dataset is required: a tiny fake ASVspoof-shaped protocol file plus fake audio files are
synthesised under `tmp_path` for every test. Real-data tests would be marked
`@pytest.mark.needs_data`, but none of the required behaviour here needs the real corpus.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from audio_df.data import (
    DatasetNotFoundError,
    assert_no_speaker_overlap,
    build_manifest,
    holdout_attack_families,
    parse_asvspoof_protocol,
    split_hash,
    summarise_manifest,
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
