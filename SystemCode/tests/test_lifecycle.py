"""experiment_run: registry rows, run_summary.json, failure logging, pressure refusal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio_df.runtime.lifecycle import experiment_run, machine_cell
from audio_df.runtime.resources import MemoryPressureError, MemoryPressureExit

NORMAL = {
    "pressure_level": 1,
    "pressure_name": "normal",
    "swap_used_gb": 0.1,
    "swap_total_gb": 1.0,
    "rss_mb": 1.0,
}
WARN = {**NORMAL, "pressure_level": 2, "pressure_name": "warn"}


def _config() -> dict:
    return {
        "system": "a1",
        "experiment": "e1",
        "data": {"testsets": ["dev", "eval"]},
        "preprocessing": {"sample_rate": 16000, "crop_seconds": None, "silence_mode": "raw"},
        "seed": 1337,
        "features": {"name": "lfcc_b02"},
    }


def _run(tmp_path: Path, **kw):
    return experiment_run(
        "E-20261001-e1-a1-raw",
        _config(),
        job_name="test",
        root=tmp_path / "runs",
        lock_path=tmp_path / "lock",
        status_fn=kw.pop("status_fn", lambda: NORMAL),
        **kw,
    )


def _registry(tmp_path: Path) -> str:
    return (tmp_path / "runs" / "REGISTRY.md").read_text()


def test_success_writes_summary_and_marks_done(tmp_path):
    with _run(tmp_path) as run:
        assert run.exp_dir.is_dir()
        assert "running" in _registry(tmp_path)
    summary = json.loads((run.exp_dir / "run_summary.json").read_text())
    assert {"peak_rss_mb", "wall_time_s", "swap_start_gb", "pressure_max", "mps_peak_mb"} <= set(
        summary
    )
    reg = _registry(tmp_path)
    assert "| done |" in reg and "E-20261001-e1-a1-raw" in reg


def test_exception_is_logged_as_failed_and_reraised(tmp_path):
    with pytest.raises(ValueError, match="boom"), _run(tmp_path):
        raise ValueError("boom")
    reg = _registry(tmp_path)
    assert "| failed |" in reg and "ValueError: boom" in reg


def test_memory_pressure_exit_records_resume_hint(tmp_path):
    with pytest.raises(MemoryPressureExit), _run(tmp_path):
        raise MemoryPressureExit({})
    assert "--resume" in _registry(tmp_path)


def test_refuses_to_start_under_pressure_without_creating_anything(tmp_path):
    with pytest.raises(MemoryPressureError), _run(tmp_path, status_fn=lambda: WARN):
        pass
    assert not (tmp_path / "runs" / "E-20261001-e1-a1-raw").exists()


def test_resume_adds_the_registry_row_if_it_is_missing(tmp_path):
    # 2026-10-01: a run was interrupted after creating its folder but before its registry
    # row was written, and `--resume` then crashed on "no row".
    from audio_df.runtime.exp_io import create_experiment

    create_experiment("E-20261001-e1-a1-raw", _config(), root=tmp_path / "runs")
    with _run(tmp_path, resume=True) as run:
        assert "| running |" in _registry(tmp_path)
    assert "| done |" in _registry(tmp_path)
    assert run.exp_dir.name == "E-20261001-e1-a1-raw"


def test_note_goes_to_notes_md_and_registry(tmp_path):
    with _run(tmp_path) as run:
        run.note("A1 uses whole utterances (B02), not the 4 s crop")
    assert "whole utterances" in (run.exp_dir / "notes.md").read_text()
    assert "whole utterances" in _registry(tmp_path)


def test_machine_cell_is_compact():
    env = {
        "machine_model": "Mac16,13",
        "cpu_brand": "Apple M4",
        "ram_gb": 16.0,
        "gpu_mps_available": True,
        "os": "Darwin",
        "os_version": "26.5.2",
        "python": "3.11.13",
        "torch": "2.14.0",
    }
    assert machine_cell(env) == "Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0"
