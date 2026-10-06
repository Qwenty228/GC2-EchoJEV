"""Tests for `audio_df.exp_io`: experiment IDs, dirs, env capture, scores, registry.

All tests use `tmp_path` as the experiment root so nothing here touches the real
`runs/` directory. No dataset or model download is needed.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from audio_df.exp_io import (
    _REGISTRY_COLUMNS,
    capture_env,
    create_experiment,
    format_resources,
    make_exp_id,
    registry_append,
    registry_update,
    write_scores,
)

# ---------------------------------------------------------------------------
# make_exp_id
# ---------------------------------------------------------------------------


def test_make_exp_id_basic_format():
    assert make_exp_id("20261001", "e1", "a1", "raw") == "E-20261001-e1-a1-raw"


def test_make_exp_id_accepts_date_object():
    d = dt.date(2026, 10, 1)
    assert make_exp_id(d, "e1", "a1", "raw") == "E-20261001-e1-a1-raw"


def test_make_exp_id_with_tag():
    assert (
        make_exp_id("20261001", "e2", "a2-xlsr", "trimmed", tag="ablation")
        == "E-20261001-e2-a2-xlsr-trimmed-ablation"
    )


def test_make_exp_id_dev_and_smoke_allowed():
    assert make_exp_id("20261001", "dev", "a1", "raw") == "E-20261001-dev-a1-raw"
    assert make_exp_id("20261001", "smoke", "a1", "raw") == "E-20261001-smoke-a1-raw"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"date": "2026100", "exp": "e1", "system": "a1", "silence_mode": "raw"},
        {"date": "20261001", "exp": "e7", "system": "a1", "silence_mode": "raw"},
        {"date": "20261001", "exp": "e1", "system": "A1", "silence_mode": "raw"},
        {"date": "20261001", "exp": "e1", "system": "a1", "silence_mode": "RAW"},
        {"date": "20261001", "exp": "e1", "system": "a1", "silence_mode": "raw", "tag": "bad tag"},
    ],
)
def test_make_exp_id_rejects_invalid_parts(kwargs):
    with pytest.raises(ValueError):
        make_exp_id(**kwargs)


# ---------------------------------------------------------------------------
# capture_env
# ---------------------------------------------------------------------------


def test_capture_env_has_expected_keys():
    env = capture_env()
    expected_keys = {
        "os",
        "os_version",
        "machine_model",
        "cpu_brand",
        "cores_logical",
        "cores_physical",
        "ram_gb",
        "gpu_mps_available",
        "cuda_available",
        "cuda_version",
        "python",
        "torch",
        "numpy",
        "git_commit",
        "git_dirty",
        "machine_label",
        "timestamp_utc",
    }
    assert expected_keys <= set(env.keys())
    assert env["python"]
    assert env["torch"]
    assert env["numpy"]
    assert env["git_commit"] and env["git_commit"] != "unknown"
    assert isinstance(env["git_dirty"], bool)


# ---------------------------------------------------------------------------
# create_experiment
# ---------------------------------------------------------------------------


def _minimal_config(**overrides):
    cfg = {
        "system": "a1",
        "experiment": "e1",
        "data": {"eval": "data/manifests/eval.csv"},
        "preprocessing": {"sample_rate": 16000, "crop_seconds": 4.0, "silence_mode": "raw"},
        "seed": 0,
        "features": {"type": "lfcc"},
    }
    cfg.update(overrides)
    return cfg


def test_create_experiment_writes_expected_files(tmp_path):
    exp_id = "E-20261001-e1-a1-raw"
    exp_dir = create_experiment(exp_id, _minimal_config(), root=tmp_path)

    assert exp_dir == tmp_path / exp_id
    assert (exp_dir / "config.yaml").exists()
    assert (exp_dir / "env.json").exists()
    assert (exp_dir / "notes.md").exists()
    assert (exp_dir / "scores").is_dir()

    with (exp_dir / "env.json").open() as f:
        env = json.load(f)
    assert "git_commit" in env


def test_create_experiment_rejects_missing_config_keys(tmp_path):
    bad_cfg = _minimal_config()
    del bad_cfg["seed"]
    with pytest.raises(ValueError, match="seed"):
        create_experiment("E-20261001-e1-a1-raw", bad_cfg, root=tmp_path)


def test_create_experiment_rejects_bad_id(tmp_path):
    with pytest.raises(ValueError):
        create_experiment("not-an-id", _minimal_config(), root=tmp_path)


def test_create_experiment_refuses_overwrite_without_resume(tmp_path):
    exp_id = "E-20261001-e1-a1-raw"
    create_experiment(exp_id, _minimal_config(), root=tmp_path)
    with pytest.raises(FileExistsError):
        create_experiment(exp_id, _minimal_config(), root=tmp_path)


def test_create_experiment_resume_with_matching_config_succeeds(tmp_path):
    exp_id = "E-20261001-e1-a1-raw"
    cfg = _minimal_config()
    create_experiment(exp_id, cfg, root=tmp_path)
    exp_dir = create_experiment(exp_id, cfg, root=tmp_path, resume=True)
    assert exp_dir == tmp_path / exp_id


def test_create_experiment_resume_with_mismatched_config_raises(tmp_path):
    exp_id = "E-20261001-e1-a1-raw"
    create_experiment(exp_id, _minimal_config(), root=tmp_path)
    with pytest.raises(ValueError):
        create_experiment(exp_id, _minimal_config(seed=1), root=tmp_path, resume=True)


# ---------------------------------------------------------------------------
# write_scores
# ---------------------------------------------------------------------------


def _toy_scores_df(n_bonafide=5, n_spoof=5, with_attack=False):
    rows = []
    for i in range(n_bonafide):
        rows.append({"clip_id": f"bona_{i}", "label": "bonafide", "score": 1.0 + i * 0.1})
    for i in range(n_spoof):
        row = {"clip_id": f"spoof_{i}", "label": "spoof", "score": -1.0 - i * 0.1}
        if with_attack:
            row["attack_id"] = f"A0{i % 3 + 1}"
        rows.append(row)
    return pd.DataFrame(rows)


def test_write_scores_round_trips(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    df = _toy_scores_df()
    out_path = write_scores(exp_dir, "eval", df)
    assert out_path == exp_dir / "scores" / "eval.csv"

    read_back = pd.read_csv(out_path)
    assert set(read_back.columns) == {"clip_id", "label", "score"}
    assert list(read_back["clip_id"]) == sorted(read_back["clip_id"])
    assert len(read_back) == len(df)


def test_write_scores_keeps_attack_id_column(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    df = _toy_scores_df(with_attack=True)
    out_path = write_scores(exp_dir, "eval", df)
    read_back = pd.read_csv(out_path)
    assert "attack_id" in read_back.columns


def test_write_scores_rejects_bad_testset_name(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    with pytest.raises(ValueError):
        write_scores(exp_dir, "Eval-Set", _toy_scores_df())


def test_write_scores_rejects_missing_columns(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    df = _toy_scores_df().drop(columns=["score"])
    with pytest.raises(ValueError):
        write_scores(exp_dir, "eval", df)


def test_write_scores_rejects_bad_labels(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    df = _toy_scores_df()
    df.loc[0, "label"] = "fake_but_not_spoof"
    with pytest.raises(ValueError):
        write_scores(exp_dir, "eval", df)


def test_write_scores_rejects_nan_or_inf(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    df = _toy_scores_df()
    df.loc[0, "score"] = float("nan")
    with pytest.raises(ValueError):
        write_scores(exp_dir, "eval", df)

    df2 = _toy_scores_df()
    df2.loc[0, "score"] = float("inf")
    with pytest.raises(ValueError):
        write_scores(exp_dir, "eval", df2)


def test_write_scores_rejects_duplicate_clip_id(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    df = _toy_scores_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        write_scores(exp_dir, "eval", df)


def test_write_scores_refuses_overwrite_without_force(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    write_scores(exp_dir, "eval", _toy_scores_df())
    with pytest.raises(FileExistsError):
        write_scores(exp_dir, "eval", _toy_scores_df())


def test_write_scores_force_overwrite_logs_to_notes(tmp_path):
    exp_dir = create_experiment("E-20261001-e1-a1-raw", _minimal_config(), root=tmp_path)
    write_scores(exp_dir, "eval", _toy_scores_df())
    write_scores(exp_dir, "eval", _toy_scores_df(n_bonafide=6), force=True)

    notes = (exp_dir / "notes.md").read_text()
    assert "force-overwrote scores/eval.csv" in notes


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_append_creates_valid_table(tmp_path):
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        config_summary="lfcc d=20",
        key_eers="eval=3.20",
        status="done",
        machine="MBA M4",
        root=tmp_path,
    )
    text = (tmp_path / "REGISTRY.md").read_text()
    assert "| E-20261001-e1-a1-raw |" in text
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    # header + separator + 1 data row
    assert len(lines) == 3
    for ln in lines:
        assert ln.count("|") == 11  # 10 columns -> 11 pipes


def test_registry_append_rejects_bad_status(tmp_path):
    with pytest.raises(ValueError):
        registry_append(
            "E-20261001-e1-a1-raw",
            date="2026-10-01",
            system="a1",
            experiment="e1",
            status="bogus",
            root=tmp_path,
        )


def test_registry_append_rejects_duplicate_id(tmp_path):
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        status="running",
        root=tmp_path,
    )
    with pytest.raises(ValueError):
        registry_append(
            "E-20261001-e1-a1-raw",
            date="2026-10-01",
            system="a1",
            experiment="e1",
            status="running",
            root=tmp_path,
        )


def test_registry_update_edits_row_in_place(tmp_path):
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        status="running",
        root=tmp_path,
    )
    registry_update(
        "E-20261001-e1-a1-raw", root=tmp_path, status="done", key_eers="eval=3.20"
    )
    text = (tmp_path / "REGISTRY.md").read_text()
    assert "done" in text
    assert "eval=3.20" in text
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    assert len(lines) == 3  # still exactly one data row


def test_registry_update_unknown_column_raises(tmp_path):
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        status="running",
        root=tmp_path,
    )
    with pytest.raises(ValueError):
        registry_update("E-20261001-e1-a1-raw", root=tmp_path, not_a_column="x")


def test_registry_update_missing_row_raises(tmp_path):
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        status="running",
        root=tmp_path,
    )
    with pytest.raises(ValueError):
        registry_update("E-not-a-real-id", root=tmp_path, status="done")


# ---------------------------------------------------------------------------
# format_resources
# ---------------------------------------------------------------------------


def test_format_resources_none_and_empty_dict_give_empty_string():
    assert format_resources(None) == ""
    assert format_resources({}) == ""


def test_format_resources_all_fields_present():
    resources = {
        "wall_time_s": 3.1 * 3600,
        "peak_rss_mb": 1.8 * 1024,
        "mps_peak_mb": 2.9 * 1024,
        "swap_start_gb": 0.6,
        "swap_end_gb": 0.7,
        "swap_peak_gb": 0.8,
        "pressure_max": "warn",
    }
    rendered = format_resources(resources)
    assert rendered == (
        "3.1 h · RSS 1.8 GB · MPS 2.9 GB · swap 0.6→0.7 GB "
        "(peak 0.8) · pressure max warn"
    )


def test_format_resources_null_mps_peak_is_omitted():
    resources = {
        "wall_time_s": 120.0,
        "peak_rss_mb": 512.0,
        "mps_peak_mb": None,
        "swap_start_gb": 0.0,
        "swap_end_gb": 0.0,
        "swap_peak_gb": 0.0,
        "pressure_max": "normal",
    }
    rendered = format_resources(resources)
    assert "MPS" not in rendered
    assert rendered.startswith("2.0 m")
    assert "pressure max normal" in rendered


def test_format_resources_partial_dict_omits_missing_keys():
    rendered = format_resources({"wall_time_s": 45.0})
    assert rendered == "45.0 s"
    assert "RSS" not in rendered
    assert "swap" not in rendered
    assert "pressure" not in rendered


def test_format_resources_swap_without_peak_omits_peak_segment():
    rendered = format_resources({"swap_start_gb": 0.1, "swap_end_gb": 0.2})
    assert rendered == "swap 0.1→0.2 GB"
    assert "peak" not in rendered


def test_format_resources_minutes_scale():
    rendered = format_resources({"wall_time_s": 150.0})
    assert rendered == "2.5 m"


# ---------------------------------------------------------------------------
# registry resources column
# ---------------------------------------------------------------------------


def test_registry_append_with_resources_dict_renders_compactly(tmp_path):
    resources = {
        "wall_time_s": 3.1 * 3600,
        "peak_rss_mb": 1.8 * 1024,
        "mps_peak_mb": 2.9 * 1024,
        "swap_start_gb": 0.6,
        "swap_end_gb": 0.7,
        "swap_peak_gb": 0.8,
        "pressure_max": "warn",
    }
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        status="done",
        resources=resources,
        root=tmp_path,
    )
    text = (tmp_path / "REGISTRY.md").read_text()
    assert "RSS 1.8 GB" in text
    assert "MPS 2.9 GB" in text
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    assert len(lines) == 3
    for ln in lines:
        assert ln.count("|") == 11  # 10 columns -> 11 pipes


def test_registry_append_without_resources_leaves_cell_empty(tmp_path):
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        status="running",
        root=tmp_path,
    )
    text = (tmp_path / "REGISTRY.md").read_text()
    data_line = [ln for ln in text.splitlines() if ln.startswith("| E-")][0]
    cells = [c.strip() for c in data_line.strip().strip("|").split("|")]
    resources_idx = _REGISTRY_COLUMNS.index("resources")
    assert cells[resources_idx] == ""


def test_registry_update_accepts_resources_dict(tmp_path):
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        status="running",
        root=tmp_path,
    )
    registry_update(
        "E-20261001-e1-a1-raw",
        root=tmp_path,
        status="done",
        resources={"wall_time_s": 60.0, "peak_rss_mb": 1024.0, "pressure_max": "normal"},
    )
    text = (tmp_path / "REGISTRY.md").read_text()
    assert "RSS 1.0 GB" in text
    assert "pressure max normal" in text


def test_registry_update_accepts_already_rendered_resources_string(tmp_path):
    registry_append(
        "E-20261001-e1-a1-raw",
        date="2026-10-01",
        system="a1",
        experiment="e1",
        status="running",
        root=tmp_path,
    )
    registry_update("E-20261001-e1-a1-raw", root=tmp_path, resources="1.0 m · RSS 0.5 GB")
    text = (tmp_path / "REGISTRY.md").read_text()
    assert "1.0 m · RSS 0.5 GB" in text
