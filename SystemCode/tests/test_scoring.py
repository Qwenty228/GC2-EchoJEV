"""Tests for `audio_df.evaluation.scoring.score_experiment` on synthetic experiment dirs.

No real dataset or model is needed: score files are hand-built numpy/pandas data with
known EER (cross-checked against `audio_df.evaluation.metrics.compute_eer` directly).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from audio_df.evaluation.metrics import compute_eer
from audio_df.evaluation.scoring import score_experiment
from audio_df.runtime.exp_io import create_experiment, write_scores


@pytest.fixture
def scoring_cfg_path(tmp_path):
    cfg = {
        "min_dcf": {"c_miss": 1, "c_fa": 10, "p_spoof": 0.05, "source": "test"},
        "threshold": {"selected_on": "dev", "rule": "eer"},
        "f1": {"positive_class": "spoof"},
    }
    path = tmp_path / "scoring.yaml"
    with path.open("w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _minimal_config():
    return {
        "system": "a1",
        "experiment": "e1",
        "data": {"eval": "data/manifests/eval.csv"},
        "preprocessing": {"sample_rate": 16000, "crop_seconds": 4.0, "silence_mode": "raw"},
        "seed": 0,
        "features": {"type": "lfcc"},
    }


def _known_scores_df(seed=0, n=100, separation=3.0, prefix="clip"):
    rng = np.random.default_rng(seed)
    bonafide = rng.normal(loc=separation / 2, scale=1.0, size=n)
    spoof = rng.normal(loc=-separation / 2, scale=1.0, size=n)
    rows = []
    for i, s in enumerate(bonafide):
        rows.append({"clip_id": f"{prefix}_bona_{i:04d}", "label": "bonafide", "score": s})
    for i, s in enumerate(spoof):
        rows.append({"clip_id": f"{prefix}_spoof_{i:04d}", "label": "spoof", "score": s})
    return pd.DataFrame(rows)


def _make_experiment(tmp_path, exp_id="E-20261001-e1-a1-raw", with_dev=True, with_attack=False):
    exp_dir = create_experiment(exp_id, _minimal_config(), root=tmp_path)
    if with_dev:
        write_scores(exp_dir, "dev", _known_scores_df(seed=1, prefix="dev"))
    eval_df = _known_scores_df(seed=2, prefix="eval")
    if with_attack:
        n_spoof = (eval_df["label"] == "spoof").sum()
        attack_ids = ["A07" if i % 2 == 0 else "A08" for i in range(n_spoof)]
        eval_df.loc[eval_df["label"] == "spoof", "attack_id"] = attack_ids
        eval_df.loc[eval_df["label"] == "bonafide", "attack_id"] = "-"
    write_scores(exp_dir, "eval", eval_df)
    return exp_dir


# ---------------------------------------------------------------------------
# basic correctness, cross-checked against metrics.compute_eer directly
# ---------------------------------------------------------------------------


def test_score_experiment_eer_matches_compute_eer_directly(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)

    eval_df = pd.read_csv(exp_dir / "scores" / "eval.csv")
    expected_eer, _ = compute_eer(eval_df["score"], eval_df["label"], higher_is_spoof=False)
    assert result["testsets"]["eval"]["eer"] == pytest.approx(expected_eer, abs=1e-12)

    dev_df = pd.read_csv(exp_dir / "scores" / "dev.csv")
    expected_dev_eer, expected_dev_thr = compute_eer(
        dev_df["score"], dev_df["label"], higher_is_spoof=False
    )
    assert result["testsets"]["dev"]["eer"] == pytest.approx(expected_dev_eer, abs=1e-12)
    assert result["testsets"]["dev"]["eer_threshold"] == pytest.approx(expected_dev_thr, abs=1e-12)
    # eer_threshold only reported for dev
    assert "eer_threshold" not in result["testsets"]["eval"]


def test_score_experiment_writes_metrics_json(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    metrics_path = exp_dir / "metrics.json"
    assert metrics_path.exists()
    with metrics_path.open() as f:
        on_disk = json.load(f)
    assert on_disk["exp_id"] == exp_dir.name
    assert "eval" in on_disk["testsets"]
    assert "dev" in on_disk["testsets"]


def test_score_experiment_records_score_file_hashes(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    assert set(result["score_file_hashes"]) == {"dev", "eval"}
    for h in result["score_file_hashes"].values():
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64


def test_score_experiment_records_scoring_config_hash(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    assert result["scoring_config_hash"].startswith("sha256:")


# ---------------------------------------------------------------------------
# threshold reuse (dev-fit, applied unchanged elsewhere) and dev-less handling
# ---------------------------------------------------------------------------


def test_dev_threshold_is_reused_unchanged_on_eval(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    dev_thr = result["testsets"]["dev"]["eer_threshold"]
    assert result["threshold_source"]["value"] == pytest.approx(dev_thr, abs=1e-12)

    eval_df = pd.read_csv(exp_dir / "scores" / "eval.csv")
    from audio_df.evaluation.metrics import prf1_at_threshold

    expected = prf1_at_threshold(
        eval_df["score"], eval_df["label"], thr=dev_thr, higher_is_spoof=False
    )
    assert result["testsets"]["eval"]["f1"] == pytest.approx(expected["f1"], abs=1e-12)
    assert result["testsets"]["eval"]["precision"] == pytest.approx(
        expected["precision"], abs=1e-12
    )


def test_f1_orientation_predicted_spoof_when_score_below_threshold(tmp_path, scoring_cfg_path):
    """Explicit orientation check: with higher=bonafide, spoof is predicted for
    scores <= threshold, not >=. A flipped rule would give a very different F1 on
    this asymmetric, well-separated synthetic set."""
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    thr = 0.0
    # Well separated: all bonafide score high (above thr), all spoof score low (below thr).
    df = pd.DataFrame(
        {
            "clip_id": [f"c{i}" for i in range(10)],
            "label": ["bonafide"] * 5 + ["spoof"] * 5,
            "score": [5, 4, 3, 2, 1, -1, -2, -3, -4, -5],
        }
    )
    write_scores(exp_dir, "dev", df)
    write_scores(exp_dir, "eval", df)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    # dev EER threshold should land such that this perfectly-separated set gives F1 == 1.0
    assert result["testsets"]["eval"]["f1"] == pytest.approx(1.0, abs=1e-9)
    assert result["testsets"]["eval"]["precision"] == pytest.approx(1.0, abs=1e-9)
    assert result["testsets"]["eval"]["recall"] == pytest.approx(1.0, abs=1e-9)
    _ = thr


def test_missing_dev_reports_null_f1_with_reason(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path, with_dev=False)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    assert result["testsets"]["eval"]["f1"] is None
    assert "reason" not in result["testsets"]["eval"] or result["testsets"]["eval"].get("f1_reason")
    assert result["threshold_source"]["value"] is None
    assert result["threshold_source"]["reason"]


def test_threshold_from_other_experiment_dev(tmp_path, scoring_cfg_path):
    dev_source = _make_experiment(tmp_path, exp_id="E-20261001-dev-a1-raw", with_dev=True)
    subset = _make_experiment(tmp_path, exp_id="E-20261001-e5-a1-raw", with_dev=False)

    result = score_experiment(subset, scoring_cfg_path=scoring_cfg_path, threshold_from=dev_source)
    assert result["threshold_source"]["value"] is not None
    assert result["threshold_source"]["exp_dir"] == str(dev_source)
    assert result["testsets"]["eval"]["f1"] is not None


# ---------------------------------------------------------------------------
# per-attack EER via score-file column or manifest join
# ---------------------------------------------------------------------------


def test_per_attack_eer_from_score_file_column(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path, with_attack=True)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    per_attack = result["testsets"]["eval"]["per_attack_eer"]
    assert per_attack is not None
    assert set(per_attack["per_attack"]) == {"A07", "A08"}


def test_per_attack_eer_via_manifest_join(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path, with_attack=False)
    eval_df = pd.read_csv(exp_dir / "scores" / "eval.csv")

    manifest_rows = []
    for _, row in eval_df.iterrows():
        attack = "-" if row["label"] == "bonafide" else "A19"
        manifest_rows.append({"utt_id": row["clip_id"], "attack_id": attack})
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    result = score_experiment(
        exp_dir, scoring_cfg_path=scoring_cfg_path, manifest_paths=[manifest_path]
    )
    per_attack = result["testsets"]["eval"]["per_attack_eer"]
    assert per_attack is not None
    assert set(per_attack["per_attack"]) == {"A19"}


def test_per_attack_eer_null_without_column_or_manifest(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path, with_attack=False)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    assert result["testsets"]["eval"]["per_attack_eer"] is None
    assert result["testsets"]["eval"]["per_attack_eer_reason"]


def test_per_attack_eer_missing_clip_in_manifest_gives_reason(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path, with_attack=False)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([{"utt_id": "not_a_real_clip", "attack_id": "A01"}]).to_csv(
        manifest_path, index=False
    )
    result = score_experiment(
        exp_dir, scoring_cfg_path=scoring_cfg_path, manifest_paths=[manifest_path]
    )
    assert result["testsets"]["eval"]["per_attack_eer"] is None
    assert "not found" in result["testsets"]["eval"]["per_attack_eer_reason"]


# ---------------------------------------------------------------------------
# min_dcf reported per testset with params
# ---------------------------------------------------------------------------


def test_min_dcf_present_with_params(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    eval_result = result["testsets"]["eval"]
    assert 0.0 <= eval_result["min_dcf"] <= 1.0 + 1e-9
    assert eval_result["min_dcf_params"] == {"c_miss": 1, "c_fa": 10, "p_spoof": 0.05}


def test_score_experiment_raises_when_no_score_files(tmp_path, scoring_cfg_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    with pytest.raises(FileNotFoundError):
        score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)


# ---------------------------------------------------------------------------
# resources: run_summary.json -> metrics.json["resources"]
# ---------------------------------------------------------------------------


def test_resources_copied_from_run_summary_json(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    resources = {
        "wall_time_s": 11160.0,
        "peak_rss_mb": 1843.2,
        "mps_peak_mb": 2969.6,
        "swap_start_gb": 0.6,
        "swap_end_gb": 0.7,
        "swap_peak_gb": 0.8,
        "pressure_max": "warn",
    }
    with (exp_dir / "run_summary.json").open("w") as f:
        json.dump(resources, f)

    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    assert result["resources"] == resources

    with (exp_dir / "metrics.json").open() as f:
        on_disk = json.load(f)
    assert on_disk["resources"] == resources


def test_resources_null_when_run_summary_missing(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    assert result["resources"] is None


def test_resources_with_null_mps_peak_round_trips(tmp_path, scoring_cfg_path):
    exp_dir = _make_experiment(tmp_path)
    resources = {
        "wall_time_s": 60.0,
        "peak_rss_mb": 512.0,
        "mps_peak_mb": None,
        "swap_start_gb": 0.0,
        "swap_end_gb": 0.0,
        "swap_peak_gb": 0.0,
        "pressure_max": "normal",
    }
    with (exp_dir / "run_summary.json").open("w") as f:
        json.dump(resources, f)
    result = score_experiment(exp_dir, scoring_cfg_path=scoring_cfg_path)
    assert result["resources"]["mps_peak_mb"] is None
