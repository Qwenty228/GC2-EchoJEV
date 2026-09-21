"""Tests for the synthetic ASVspoof-2019-LA-shaped fixture (src/audio_df/synthetic.py).

These tests exist so that if `synthetic.py` ever changes, a test fails loudly instead of three
notebooks quietly telling different stories about the same "synthetic mini-manifest" -- per the
module's own "fixture's contract" docstring section, notebooks 01-03 make specific claims (a
centroid reversal across attacks, a leakage-audit flag, a passing speaker-overlap check) that
are facts about *this generator*, not about ASVspoof, and must stay true.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_df import data
from audio_df.synthetic import build_synthetic_asvspoof_tree


def test_build_manifest_accepts_the_synthetic_tree(tmp_path):
    """data.build_manifest (the real manifest-building code path, not a hand-built DataFrame)
    must accept the tree this fixture writes."""
    cfg = build_synthetic_asvspoof_tree(tmp_path, seed=0)
    manifest = data.build_manifest(cfg)

    assert len(manifest) > 0
    assert list(manifest.columns) == cfg["manifest"]["columns"]
    assert set(manifest["label"].unique()) == {"bonafide", "spoof"}
    assert set(manifest["split"].unique()) == {"train", "dev", "eval"}


def test_attack_family_split_train_dev_vs_eval(tmp_path):
    """Planted property 2: train/dev only ever contain train_dev_attacks; eval only ever
    contains eval_attacks; the two attack sets are disjoint -- mirroring (in miniature) the
    real corpus's A01-A06 (train/dev) vs A07-A19 (eval) asymmetry that notebook 01's
    "unseen-attack generalisation" discussion depends on."""
    cfg = build_synthetic_asvspoof_tree(tmp_path, seed=1)
    manifest = data.build_manifest(cfg)

    train_dev_attacks = set(cfg["train_dev_attacks"])
    eval_attacks = set(cfg["eval_attacks"])
    assert train_dev_attacks.isdisjoint(eval_attacks)

    for split in ("train", "dev"):
        spoof_attacks = set(manifest.loc[manifest["split"] == split, "attack_id"]) - {"-"}
        assert spoof_attacks == train_dev_attacks, f"{split} attacks changed unexpectedly"

    eval_spoof_attacks = set(manifest.loc[manifest["split"] == "eval", "attack_id"]) - {"-"}
    assert eval_spoof_attacks == eval_attacks

    # the defining property: no attack family leaks across the train/dev vs eval boundary
    assert eval_spoof_attacks.isdisjoint(train_dev_attacks)


def test_speaker_partition_no_overlap(tmp_path):
    """Planted property 3: speakers are disjoint across every pair of splits, so
    `assert_no_speaker_overlap` genuinely passes on this fixture rather than passing by luck."""
    cfg = build_synthetic_asvspoof_tree(tmp_path, seed=2)
    manifest = data.build_manifest(cfg)

    train_df = manifest[manifest["split"] == "train"]
    dev_df = manifest[manifest["split"] == "dev"]
    eval_df = manifest[manifest["split"] == "eval"]

    data.assert_no_speaker_overlap(train_df, dev_df)  # must not raise

    train_speakers = set(train_df["speaker_id"])
    dev_speakers = set(dev_df["speaker_id"])
    eval_speakers = set(eval_df["speaker_id"])
    assert train_speakers.isdisjoint(dev_speakers)
    assert train_speakers.isdisjoint(eval_speakers)
    assert dev_speakers.isdisjoint(eval_speakers)


def test_per_attack_gain_produces_intended_centroid_ordering(tmp_path):
    """Planted property 1: attacks with a positive high-shelf gain get a HIGHER mean spectral
    centroid than bonafide, attacks with a negative gain get a LOWER one, and the neutral
    (0 dB) attack stays close to bonafide -- this is exactly the reversal notebook 02's
    by-attack centroid plot shows, and exactly what notebook 01's `rms` leakage flag fires on.
    """
    librosa = pytest.importorskip("librosa")

    cfg = build_synthetic_asvspoof_tree(tmp_path, seed=3)
    manifest = data.build_manifest(cfg)

    def _mean_centroid(paths) -> float:
        vals = []
        for p in paths:
            x, sr = librosa.load(p, sr=16000, mono=True)
            vals.append(librosa.feature.spectral_centroid(y=x, sr=sr).mean())
        return float(np.mean(vals))

    bonafide_centroid = _mean_centroid(manifest.loc[manifest["label"] == "bonafide", "path"])

    attack_gain_db = cfg["attack_gain_db"]
    boosted_attacks = [a for a, g in attack_gain_db.items() if g > 0]
    cut_attacks = [a for a, g in attack_gain_db.items() if g < 0]
    neutral_attacks = [a for a, g in attack_gain_db.items() if g == 0]
    assert boosted_attacks and cut_attacks and neutral_attacks, "gain cycle must plant all 3 cases"

    attack_col = manifest["attack_id"]
    boosted_centroid = _mean_centroid(manifest.loc[attack_col.isin(boosted_attacks), "path"])
    cut_centroid = _mean_centroid(manifest.loc[attack_col.isin(cut_attacks), "path"])
    neutral_centroid = _mean_centroid(manifest.loc[attack_col.isin(neutral_attacks), "path"])

    assert boosted_centroid > bonafide_centroid > cut_centroid
    # neutral attack should sit much closer to bonafide than either boosted or cut extreme
    assert abs(neutral_centroid - bonafide_centroid) < abs(boosted_centroid - bonafide_centroid)
    assert abs(neutral_centroid - bonafide_centroid) < abs(cut_centroid - bonafide_centroid)


def test_gain_cycle_none_disables_engineered_spectral_tilt(tmp_path):
    """`gain_cycle=None` is the escape hatch documented in the module docstring: every attack
    gets 0 dB gain, so nothing about this fixture's spectrum is engineered to differ by class."""
    cfg = build_synthetic_asvspoof_tree(tmp_path, seed=4, gain_cycle=None)
    assert all(g == 0.0 for g in cfg["attack_gain_db"].values())


def test_deterministic_given_seed(tmp_path):
    """Same seed -> identical manifest content (same utt_ids, labels, attacks, durations)."""
    cfg_a = build_synthetic_asvspoof_tree(tmp_path / "a", seed=42)
    cfg_b = build_synthetic_asvspoof_tree(tmp_path / "b", seed=42)
    manifest_a = data.build_manifest(cfg_a)
    manifest_b = data.build_manifest(cfg_b)

    compare_cols = ["utt_id", "label", "speaker_id", "attack_id", "split", "duration_s"]
    a = manifest_a[compare_cols].reset_index(drop=True)
    b = manifest_b[compare_cols].reset_index(drop=True)
    assert a.equals(b)
