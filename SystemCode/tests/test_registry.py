"""Tests for `audio_df.registry`.

Uses a throwaway copy of the real registry header (never the real `reports/experiment_registry.csv`
file, so tests cannot pollute it) written to a pytest `tmp_path`.
"""

from __future__ import annotations

import csv
import json
import warnings

import pandas as pd
import pytest

from audio_df.registry import append_run, write_run_json

_HEADER = [
    "run_id",
    "timestamp",
    "git_commit",
    "dataset",
    "split",
    "preprocessing",
    "representation",
    "backbone",
    "ssl_checkpoint",
    "ssl_layers",
    "ssl_frozen",
    "pooling",
    "loss_weights",
    "augmentation",
    "seed",
    "trainable_params",
    "total_params",
    "eer",
    "roc_auc",
    "pr_auc",
    "f1",
    "per_attack_eer_json",
    "train_wall_s",
    "peak_mem_mb",
    "rtf",
    "artifact_path",
    "notes",
]


@pytest.fixture
def registry_path(tmp_path):
    path = tmp_path / "experiment_registry.csv"
    with path.open("w", newline="") as f:
        csv.writer(f).writerow(_HEADER)
    return path


def test_append_run_round_trips_through_pandas_with_header_intact(registry_path):
    row = {
        "dataset": "asvspoof2019la",
        "split": "dev",
        "representation": "feat_b_lfcc_full",
        "backbone": "toy",
        "seed": 42,
        "eer": 0.1234,
    }
    run_id = append_run(row, path=str(registry_path))

    df = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    assert list(df.columns) == _HEADER
    assert len(df) == 1
    assert df.loc[0, "run_id"] == run_id
    assert df.loc[0, "dataset"] == "asvspoof2019la"
    assert df.loc[0, "eer"] == "0.1234"


def test_two_appends_produce_two_rows_with_distinct_run_ids_and_no_drift(registry_path):
    run_id_1 = append_run({"dataset": "asvspoof2019la", "seed": 1}, path=str(registry_path))
    run_id_2 = append_run({"dataset": "asvspoof2019la", "seed": 2}, path=str(registry_path))

    assert run_id_1 != run_id_2

    df = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    assert list(df.columns) == _HEADER
    assert len(df) == 2
    assert set(df["run_id"]) == {run_id_1, run_id_2}
    assert df.loc[df["run_id"] == run_id_1, "seed"].iloc[0] == "1"
    assert df.loc[df["run_id"] == run_id_2, "seed"].iloc[0] == "2"


def test_append_run_is_append_only_never_touches_prior_rows(registry_path):
    run_id_1 = append_run({"dataset": "first"}, path=str(registry_path))
    df_after_first = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    first_row_before = df_after_first.loc[df_after_first["run_id"] == run_id_1].to_dict()

    append_run({"dataset": "second"}, path=str(registry_path))

    df_after_second = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    first_row_after = (
        df_after_second.loc[df_after_second["run_id"] == run_id_1]
        .reset_index(drop=True)
        .to_dict()
    )
    first_row_before = (
        df_after_first.loc[df_after_first["run_id"] == run_id_1].reset_index(drop=True).to_dict()
    )
    assert first_row_before == first_row_after


def test_append_run_with_missing_keys_fills_empty_string_without_shifting_columns(registry_path):
    # Only a handful of keys supplied; most columns absent.
    run_id = append_run(
        {"representation": "feat_a_linear_logfb", "eer": 0.5}, path=str(registry_path)
    )

    df = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    row = df.loc[df["run_id"] == run_id].iloc[0]

    # The known column present in the input lands in the right place...
    assert row["representation"] == "feat_a_linear_logfb"
    assert row["eer"] == "0.5"
    # ...and every column NOT supplied is an empty string, not shifted data from a
    # neighbouring column.
    for col in _HEADER:
        if col in {"run_id", "representation", "eer"}:
            continue
        assert row[col] == "", f"column {col!r} was not empty: {row[col]!r} (possible column drift)"


def test_append_run_ignores_unknown_extra_keys_without_corrupting_columns(registry_path):
    with pytest.warns(UserWarning):
        run_id = append_run(
            {"dataset": "asvspoof2019la", "some_future_field_not_in_header_yet": "xyz"},
            path=str(registry_path),
        )
    df = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    assert list(df.columns) == _HEADER
    row = df.loc[df["run_id"] == run_id].iloc[0]
    assert row["dataset"] == "asvspoof2019la"


def test_append_run_warns_naming_the_dropped_key(registry_path):
    with pytest.warns(UserWarning, match=r"eer_typo_column") as record:
        append_run({"eer": 0.33, "eer_typo_column": 0.99}, path=str(registry_path))

    messages = [str(w.message) for w in record.list]
    assert any("eer_typo_column" in m for m in messages)
    assert any("append_run" in m and "1 key" in m for m in messages)
    assert any(str(registry_path) in m for m in messages)

    # the run still succeeded and the known column landed correctly -- dropping is audible,
    # not fatal.
    df = pd.read_csv(registry_path, dtype=str, keep_default_na=False)
    assert len(df) == 1
    assert df.loc[0, "eer"] == "0.33"
    assert "eer_typo_column" not in df.columns


def test_append_run_warning_names_all_dropped_keys_when_several(registry_path):
    with pytest.warns(UserWarning) as record:
        append_run(
            {"eer": 0.1, "typo_one": 1, "typo_two": 2},
            path=str(registry_path),
        )
    messages = [str(w.message) for w in record.list]
    assert any("typo_one" in m and "typo_two" in m for m in messages)


def test_append_run_does_not_warn_when_every_key_is_known(registry_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        append_run({"dataset": "asvspoof2019la", "eer": 0.1}, path=str(registry_path))


def test_append_run_raises_if_registry_file_missing(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        append_run({"dataset": "x"}, path=str(missing))


def test_write_run_json_uses_run_id_as_filename(tmp_path):
    result = {"run_id": "20260101T000000000000-abcd1234", "eer": 0.05, "notes": "toy run"}
    out_path = write_run_json(result, tmp_path / "artifacts")

    assert out_path.exists()
    assert out_path.name == "20260101T000000000000-abcd1234.json"
    loaded = json.loads(out_path.read_text())
    assert loaded == result


def test_write_run_json_requires_run_id(tmp_path):
    with pytest.raises(ValueError, match="run_id"):
        write_run_json({"eer": 0.05}, tmp_path / "artifacts")
