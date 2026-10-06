"""Tests for the Stage 0 leakage audit (src/audio_df/evaluation/leakage.py).

No dataset is required: every test builds a synthetic manifest with numpy/pandas. The
question each test answers is purely "does the audit's statistics behave correctly on data
whose leakage properties we constructed and therefore know exactly" — never "does it work on
the real ASVspoof corpus" (that would need `@pytest.mark.needs_data`, and is out of scope for
this module's tests).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from audio_df.evaluation.leakage import (
    audit_all,
    audit_single_feature,
    extract_nuisance_features,
)

SEED = 0


def _rng():
    return np.random.default_rng(SEED)


def _planted_shortcut_manifest(n: int = 200) -> pd.DataFrame:
    """duration_s perfectly predicts label: bonafide short, spoof long, no overlap."""
    rng = _rng()
    n_bona = n // 2
    n_spoof = n - n_bona
    bona_duration = rng.uniform(1.0, 2.0, size=n_bona)
    spoof_duration = rng.uniform(5.0, 6.0, size=n_spoof)
    labels = ["bonafide"] * n_bona + ["spoof"] * n_spoof
    duration = np.concatenate([bona_duration, spoof_duration])
    order = rng.permutation(n)
    return pd.DataFrame(
        {
            "utt_id": [f"u{i}" for i in range(n)],
            "label": np.array(labels)[order],
            "duration_s": duration[order],
            "sample_rate": np.full(n, 16000),
            "codec": ["FLAC"] * n,
            "source_corpus": ["X"] * n,
            "speaker_id": [f"spk{i % 20}" for i in range(n)],
        }
    )


def _clean_manifest(n: int = 200) -> pd.DataFrame:
    """Random duration, balanced classes, no relationship to label at all."""
    rng = _rng()
    labels = rng.choice(["bonafide", "spoof"], size=n)
    duration = rng.uniform(1.0, 6.0, size=n)
    # speakers assigned independently of label -> every speaker should mix classes
    speaker = rng.integers(0, 20, size=n)
    return pd.DataFrame(
        {
            "utt_id": [f"u{i}" for i in range(n)],
            "label": labels,
            "duration_s": duration,
            "sample_rate": np.full(n, 16000),
            "codec": rng.choice(["FLAC", "WAV"], size=n),
            "source_corpus": ["X"] * n,
            "speaker_id": [f"spk{s}" for s in speaker],
        }
    )


def _partial_shortcut_manifest(n: int = 400, p_correct: float = 0.75) -> pd.DataFrame:
    """duration_s predicts label correctly a fraction `p_correct` of the time."""
    rng = _rng()
    labels = rng.choice(["bonafide", "spoof"], size=n)
    duration = np.empty(n)
    for i, lab in enumerate(labels):
        follows_rule = rng.random() < p_correct
        is_bonafide_short = (lab == "bonafide") == follows_rule
        if is_bonafide_short:
            duration[i] = rng.uniform(1.0, 2.0)
        else:
            duration[i] = rng.uniform(5.0, 6.0)
    return pd.DataFrame(
        {
            "utt_id": [f"u{i}" for i in range(n)],
            "label": labels,
            "duration_s": duration,
            "sample_rate": np.full(n, 16000),
            "codec": ["FLAC"] * n,
            "source_corpus": ["X"] * n,
            "speaker_id": [f"spk{i % 40}" for i in range(n)],
        }
    )


def _disjoint_speaker_manifest(n_speakers: int = 20, per_speaker: int = 10) -> pd.DataFrame:
    """Each speaker appears in exactly one class; duration is pure noise (no shortcut there)."""
    rng = _rng()
    rows = []
    speaker_labels = {f"spk{i}": ("bonafide" if i % 2 == 0 else "spoof") for i in range(n_speakers)}
    idx = 0
    for spk, lab in speaker_labels.items():
        for _ in range(per_speaker):
            rows.append(
                {
                    "utt_id": f"u{idx}",
                    "label": lab,
                    "duration_s": rng.uniform(1.0, 6.0),
                    "sample_rate": 16000,
                    "codec": "FLAC",
                    "source_corpus": "X",
                    "speaker_id": spk,
                }
            )
            idx += 1
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


class TestExtractNuisanceFeatures:
    def test_extracts_metadata_columns_present_in_manifest(self):
        manifest = _clean_manifest(50)
        feats = extract_nuisance_features(manifest)
        assert "duration_s" in feats.columns
        assert "sample_rate" in feats.columns
        assert "codec" in feats.columns
        assert "source_corpus" in feats.columns
        assert "speaker_id" in feats.columns
        assert len(feats) == len(manifest)


class TestAuditSingleFeature:
    def test_planted_shortcut_is_caught(self):
        manifest = _planted_shortcut_manifest()
        result = audit_single_feature(manifest["duration_s"], manifest["label"], seed=SEED)
        assert result["auc"] > 0.95
        assert result["eer"] < 0.1

    def test_no_false_alarm_on_random_duration(self):
        manifest = _clean_manifest()
        result = audit_single_feature(manifest["duration_s"], manifest["label"], seed=SEED)
        lo, hi = result["auc_ci"]
        assert lo < 0.5 < hi or abs(result["auc"] - 0.5) < 0.12

    def test_partial_shortcut_lands_in_the_middle(self):
        manifest = _partial_shortcut_manifest()
        result = audit_single_feature(manifest["duration_s"], manifest["label"], seed=SEED)
        assert 0.55 < result["auc"] < 0.95
        lo, hi = result["auc_ci"]
        # the interesting assertion: CI must exclude chance (0.5)
        assert lo > 0.5

    def test_speaker_identity_disjoint_across_classes_is_flagged(self):
        manifest = _disjoint_speaker_manifest()
        result = audit_single_feature(manifest["speaker_id"], manifest["label"], seed=SEED)
        assert result["auc"] > 0.65


class TestAuditAll:
    def test_flags_planted_shortcut_and_reports_combined_row(self):
        manifest = _planted_shortcut_manifest()
        feats = extract_nuisance_features(manifest)
        with pytest.warns(UserWarning, match="LEAKAGE WARNING"):
            report = audit_all(feats, manifest["label"], threshold=0.65, seed=SEED)
        assert "duration_s" in report["feature"].values
        assert "combined" in report["feature"].values
        row = report.loc[report["feature"] == "duration_s"].iloc[0]
        assert bool(row["flagged"])

    def test_no_flags_on_clean_manifest(self):
        manifest = _clean_manifest()
        feats = extract_nuisance_features(manifest)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            report = audit_all(feats, manifest["label"], threshold=0.65, seed=SEED)
        assert not report["flagged"].any()

    def test_warning_names_the_offending_feature(self):
        manifest = _planted_shortcut_manifest()
        feats = extract_nuisance_features(manifest)
        with pytest.warns(UserWarning) as records:
            audit_all(feats, manifest["label"], threshold=0.65, seed=SEED)
        messages = [str(r.message) for r in records.list]
        assert any("duration_s" in m and "LEAKAGE WARNING" in m for m in messages)

    def test_speaker_identity_flagged_in_combined_report(self):
        manifest = _disjoint_speaker_manifest()
        feats = extract_nuisance_features(manifest)
        with pytest.warns(UserWarning, match="LEAKAGE WARNING"):
            report = audit_all(feats, manifest["label"], threshold=0.65, seed=SEED)
        row = report.loc[report["feature"] == "speaker_id"].iloc[0]
        assert bool(row["flagged"])


class TestAudioPathResolution:
    """Regression: audio-derived features must actually populate on a real dataset tree.

    `build_manifest` writes `path` already joined onto `dataset.root`, while `run_audit`
    passes that same root in as `audio_root`. Prepending it a second time produced
    `<root>/<root>/...`, whose `FileNotFoundError` `extract_nuisance_features` converts to
    NaN and `audit_all` then drops — so rms and every silence feature vanished from the
    report in silence. Silence is the one nuisance feature with a published shortcut result
    (15.1% EER, arXiv:2106.12914), so dropping it unnoticed defeats the module's purpose.
    """

    def _tree(self, tmp_path, monkeypatch):
        import soundfile as sf

        monkeypatch.chdir(tmp_path)
        root = Path("data/raw/ds/LA")
        flac_dir = root / "ASVspoof2019_LA_train" / "flac"
        flac_dir.mkdir(parents=True)
        rng = np.random.default_rng(SEED)
        rows = []
        for i in range(4):
            rel = flac_dir / f"u{i}.wav"
            sf.write(str(rel), rng.normal(0, 0.1, 16000).astype(np.float32), 16000)
            rows.append(
                {
                    "utt_id": f"u{i}",
                    "path": str(rel),  # as build_manifest writes it: root already joined
                    "label": "bonafide" if i % 2 == 0 else "spoof",
                    "speaker_id": f"s{i}",
                    "attack_id": "-",
                    "source_corpus": "ds",
                    "sample_rate": 16000,
                    "duration_s": 1.0,
                    "codec": "WAV",
                    "split": "train",
                    "language": "en",
                }
            )
        return pd.DataFrame(rows), str(root)

    def test_audio_features_populate_when_root_is_already_in_path(self, tmp_path, monkeypatch):
        manifest, audio_root = self._tree(tmp_path, monkeypatch)
        feats = extract_nuisance_features(manifest, audio_root=audio_root)
        for col in ("rms", "silence_leading", "silence_trailing", "silence_total"):
            assert feats[col].notna().all(), f"{col} is NaN — audio was not found"

    def test_audio_features_survive_audit_all(self, tmp_path, monkeypatch):
        manifest, audio_root = self._tree(tmp_path, monkeypatch)
        feats = extract_nuisance_features(manifest, audio_root=audio_root)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = audit_all(feats, manifest["label"], threshold=0.65, seed=SEED)
        audited = set(report["feature"].values)
        for col in ("rms", "silence_leading", "silence_trailing", "silence_total"):
            assert col in audited, f"{col} was dropped from the audit report"
