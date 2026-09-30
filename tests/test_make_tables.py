"""Tests for `scripts/make_tables.py` against synthetic `experiments/` trees.

`scripts/` is not a package (no `__init__.py`, matching the other scripts in this
repo), so the module is loaded by file path via `importlib`, exactly as
`uv run python scripts/make_tables.py` would run it, then its functions are called
directly against `tmp_path` fixtures. No real dataset or model is needed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_tables.py"
_spec = importlib.util.spec_from_file_location("make_tables", _SCRIPT_PATH)
make_tables_mod = importlib.util.module_from_spec(_spec)
sys.modules["make_tables"] = make_tables_mod
_spec.loader.exec_module(make_tables_mod)

load_done_experiments = make_tables_mod.load_done_experiments
build_e1_table = make_tables_mod.build_e1_table
build_e2_table = make_tables_mod.build_e2_table
build_e3_itw_table = make_tables_mod.build_e3_itw_table
build_e3_per_attack_table = make_tables_mod.build_e3_per_attack_table
build_e4_table = make_tables_mod.build_e4_table
build_e5_table = make_tables_mod.build_e5_table
build_e6_table = make_tables_mod.build_e6_table
make_tables = make_tables_mod.make_tables


_REGISTRY_HEADER = (
    "# Experiment registry\n\nsome header text\n\n"
    "| id | date | system | experiment | config summary | key EERs | status | machine | notes |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def _write_registry(root: Path, rows: list[dict]) -> None:
    lines = [_REGISTRY_HEADER.rstrip("\n")]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                row.get(c, "")
                for c in [
                    "id",
                    "date",
                    "system",
                    "experiment",
                    "config summary",
                    "key EERs",
                    "status",
                    "machine",
                    "notes",
                ]
            )
            + " |"
        )
    (root / "REGISTRY.md").write_text("\n".join(lines) + "\n")


def _write_experiment(
    root: Path,
    exp_id: str,
    system: str,
    experiment: str,
    silence_mode: str,
    testsets: dict[str, dict],
    efficiency: dict | None = None,
) -> None:
    exp_dir = root / exp_id
    exp_dir.mkdir(parents=True)
    config = {
        "system": system,
        "experiment": experiment,
        "data": {},
        "preprocessing": {"sample_rate": 16000, "crop_seconds": 4.0, "silence_mode": silence_mode},
        "seed": 0,
        "features": {"type": "lfcc"},
    }
    with (exp_dir / "config.yaml").open("w") as f:
        yaml.safe_dump(config, f)

    metrics = {"exp_id": exp_id, "testsets": testsets}
    if efficiency is not None:
        metrics["efficiency"] = efficiency
    with (exp_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f)


def _eval_result(eer=0.05, min_dcf=0.2, roc_auc=0.95, f1=0.9, per_attack=None):
    result = {"n": 100, "eer": eer, "min_dcf": min_dcf, "roc_auc": roc_auc, "f1": f1}
    if per_attack is not None:
        result["per_attack_eer"] = {"per_attack": per_attack}
    return result


# ---------------------------------------------------------------------------
# load_done_experiments: status filtering
# ---------------------------------------------------------------------------


def test_load_done_experiments_excludes_non_done_status(tmp_path):
    _write_experiment(
        tmp_path, "E-1-e1-a1-raw", "a1", "e1", "raw", {"eval": _eval_result()}
    )
    _write_experiment(
        tmp_path, "E-2-e1-a1-raw", "a1", "e1", "raw", {"eval": _eval_result()}
    )
    _write_registry(
        tmp_path,
        [
            {"id": "E-1-e1-a1-raw", "status": "done"},
            {"id": "E-2-e1-a1-raw", "status": "superseded"},
        ],
    )
    loaded = load_done_experiments(tmp_path)
    assert [e["exp_id"] for e in loaded] == ["E-1-e1-a1-raw"]


def test_load_done_experiments_skips_missing_metrics_file(tmp_path):
    exp_dir = tmp_path / "E-1-e1-a1-raw"
    exp_dir.mkdir(parents=True)
    with (exp_dir / "config.yaml").open("w") as f:
        yaml.safe_dump({"system": "a1", "experiment": "e1"}, f)
    _write_registry(tmp_path, [{"id": "E-1-e1-a1-raw", "status": "done"}])
    assert load_done_experiments(tmp_path) == []


def test_load_done_experiments_missing_registry_gives_empty_list(tmp_path):
    assert load_done_experiments(tmp_path) == []


# ---------------------------------------------------------------------------
# E1
# ---------------------------------------------------------------------------


def test_build_e1_table_and_ordering(tmp_path):
    _write_experiment(
        tmp_path, "E-1-e1-a2-trimmed", "a2", "e1", "trimmed", {"eval": _eval_result(eer=0.03)}
    )
    _write_experiment(
        tmp_path, "E-2-e1-a1-raw", "a1", "e1", "raw", {"eval": _eval_result(eer=0.10)}
    )
    _write_registry(
        tmp_path,
        [
            {"id": "E-1-e1-a2-trimmed", "status": "done"},
            {"id": "E-2-e1-a1-raw", "status": "done"},
        ],
    )
    experiments = load_done_experiments(tmp_path)
    df = build_e1_table(experiments)
    assert list(df["system"]) == ["a1", "a2"]  # sorted by system
    assert df.loc[df["system"] == "a1", "eer"].iloc[0] == pytest.approx(0.10)
    assert df.loc[df["system"] == "a2", "silence_mode"].iloc[0] == "trimmed"


def test_build_e1_table_skips_experiments_without_eval(tmp_path):
    _write_experiment(tmp_path, "E-1-e1-a1-raw", "a1", "e1", "raw", {"dev": _eval_result()})
    _write_registry(tmp_path, [{"id": "E-1-e1-a1-raw", "status": "done"}])
    experiments = load_done_experiments(tmp_path)
    df = build_e1_table(experiments)
    assert df.empty


def test_build_e1_table_ignores_other_experiment_ids(tmp_path):
    _write_experiment(tmp_path, "E-1-e3-a1-raw", "a1", "e3", "raw", {"eval": _eval_result()})
    _write_registry(tmp_path, [{"id": "E-1-e3-a1-raw", "status": "done"}])
    experiments = load_done_experiments(tmp_path)
    df = build_e1_table(experiments)
    assert df.empty


# ---------------------------------------------------------------------------
# E2 placeholder
# ---------------------------------------------------------------------------


def test_build_e2_table_is_always_not_run_placeholder(tmp_path):
    _write_experiment(tmp_path, "E-1-e1-a1-raw", "a1", "e1", "raw", {"eval": _eval_result()})
    _write_registry(tmp_path, [{"id": "E-1-e1-a1-raw", "status": "done"}])
    experiments = load_done_experiments(tmp_path)
    df = build_e2_table(experiments)
    assert list(df["status"]) == ["not run"]


# ---------------------------------------------------------------------------
# E3: itw + per-attack
# ---------------------------------------------------------------------------


def test_build_e3_itw_table(tmp_path):
    _write_experiment(
        tmp_path,
        "E-1-e3-a1-raw",
        "a1",
        "e3",
        "raw",
        {"itw": {"n": 50, "eer": 0.22, "min_dcf": 0.7}},
    )
    _write_registry(tmp_path, [{"id": "E-1-e3-a1-raw", "status": "done"}])
    experiments = load_done_experiments(tmp_path)
    df = build_e3_itw_table(experiments)
    assert len(df) == 1
    assert df.iloc[0]["eer"] == pytest.approx(0.22)


def test_build_e3_per_attack_table_covers_a07_to_a19(tmp_path):
    per_attack = {"A07": 0.1, "A19": 0.3, "A02": 0.9}  # A02 is a seen attack, not unseen range
    _write_experiment(
        tmp_path,
        "E-1-e3-a1-raw",
        "a1",
        "e3",
        "raw",
        {"eval": _eval_result(per_attack=per_attack)},
    )
    _write_registry(tmp_path, [{"id": "E-1-e3-a1-raw", "status": "done"}])
    experiments = load_done_experiments(tmp_path)
    df = build_e3_per_attack_table(experiments)
    assert "A07" in df.columns and "A19" in df.columns
    assert "A02" not in df.columns
    assert df.iloc[0]["A07"] == pytest.approx(0.1)
    assert pd.isna(df.iloc[0]["A08"])  # not present in per_attack -> NaN, not dropped


# ---------------------------------------------------------------------------
# E4
# ---------------------------------------------------------------------------


def test_build_e4_table_orders_clean_reference_first(tmp_path):
    _write_experiment(
        tmp_path,
        "E-1-e4-a1-raw",
        "a1",
        "e4",
        "raw",
        {
            "dev": _eval_result(),
            "eval_noise_10db": _eval_result(eer=0.30),
            "eval": _eval_result(eer=0.05),
        },
    )
    _write_registry(tmp_path, [{"id": "E-1-e4-a1-raw", "status": "done"}])
    experiments = load_done_experiments(tmp_path)
    df = build_e4_table(experiments)
    assert "dev" not in set(df["testset"])
    assert list(df["testset"])[0] == "eval"


# ---------------------------------------------------------------------------
# E5
# ---------------------------------------------------------------------------


def test_build_e5_table_a2_vs_a3(tmp_path):
    _write_experiment(
        tmp_path, "E-1-e5-a2-raw", "a2", "e5", "raw", {"jev_dev500": _eval_result(eer=0.08)}
    )
    _write_experiment(
        tmp_path, "E-2-e5-a3-raw", "a3", "e5", "raw", {"jev_dev500": _eval_result(eer=0.06)}
    )
    _write_registry(
        tmp_path,
        [
            {"id": "E-1-e5-a2-raw", "status": "done"},
            {"id": "E-2-e5-a3-raw", "status": "done"},
        ],
    )
    experiments = load_done_experiments(tmp_path)
    df = build_e5_table(experiments)
    assert list(df["system"]) == ["a2", "a3"]


# ---------------------------------------------------------------------------
# E6
# ---------------------------------------------------------------------------


def test_build_e6_table_reads_efficiency_block(tmp_path):
    _write_experiment(
        tmp_path,
        "E-1-e6-a1-raw",
        "a1",
        "e6",
        "raw",
        {"eff_eval1k": _eval_result()},
        efficiency={"params": 12345, "ms_per_clip": 3.2, "rtf": 0.01, "peak_memory_mb": 512},
    )
    _write_registry(tmp_path, [{"id": "E-1-e6-a1-raw", "status": "done"}])
    experiments = load_done_experiments(tmp_path)
    df = build_e6_table(experiments)
    assert df.iloc[0]["params"] == 12345
    assert df.iloc[0]["rtf"] == pytest.approx(0.01)


def test_build_e6_table_handles_missing_efficiency_block(tmp_path):
    _write_experiment(
        tmp_path, "E-1-e6-a1-raw", "a1", "e6", "raw", {"eff_eval1k": _eval_result()}
    )
    _write_registry(tmp_path, [{"id": "E-1-e6-a1-raw", "status": "done"}])
    experiments = load_done_experiments(tmp_path)
    df = build_e6_table(experiments)
    assert df.iloc[0]["params"] is None


# ---------------------------------------------------------------------------
# make_tables: end-to-end, file writing, empty-case
# ---------------------------------------------------------------------------


def test_make_tables_writes_all_six_and_no_runs_yet_when_empty(tmp_path):
    out_dir = tmp_path / "tables"
    tables = make_tables(experiments_root=tmp_path / "experiments", out_dir=out_dir)
    assert set(tables) == {"E1", "E2", "E3_itw", "E3_per_attack", "E4", "E5", "E6"}
    for name in ["E1", "E3_itw", "E3_per_attack", "E4", "E5"]:
        md = (out_dir / f"{name}.md").read_text()
        assert "no runs yet" in md
        assert (out_dir / f"{name}.csv").exists()
    # E2 is always the placeholder, never "no runs yet"
    assert "not run" in (out_dir / "E2.md").read_text()


def test_make_tables_end_to_end_with_real_run(tmp_path):
    experiments_root = tmp_path / "experiments"
    _write_experiment(
        experiments_root, "E-1-e1-a1-raw", "a1", "e1", "raw", {"eval": _eval_result(eer=0.07)}
    )
    _write_registry(experiments_root, [{"id": "E-1-e1-a1-raw", "status": "done"}])

    out_dir = tmp_path / "tables"
    tables = make_tables(experiments_root=experiments_root, out_dir=out_dir)
    assert len(tables["E1"]) == 1
    md = (out_dir / "E1.md").read_text()
    assert "a1" in md
    csv_df = pd.read_csv(out_dir / "E1.csv")
    assert csv_df.iloc[0]["eer"] == pytest.approx(0.07)


def test_make_tables_is_deterministic_across_reruns(tmp_path):
    experiments_root = tmp_path / "experiments"
    _write_experiment(
        experiments_root, "E-2-e1-a2-raw", "a2", "e1", "raw", {"eval": _eval_result(eer=0.07)}
    )
    _write_experiment(
        experiments_root, "E-1-e1-a1-raw", "a1", "e1", "raw", {"eval": _eval_result(eer=0.09)}
    )
    _write_registry(
        experiments_root,
        [
            {"id": "E-2-e1-a2-raw", "status": "done"},
            {"id": "E-1-e1-a1-raw", "status": "done"},
        ],
    )
    out_dir = tmp_path / "tables"
    make_tables(experiments_root=experiments_root, out_dir=out_dir)
    first = (out_dir / "E1.md").read_text()
    make_tables(experiments_root=experiments_root, out_dir=out_dir)
    second = (out_dir / "E1.md").read_text()
    assert first == second
