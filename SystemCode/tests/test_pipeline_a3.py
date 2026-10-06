"""run_a3: dry-run end-to-end on synthetic audio + a fake A2 exp dir, no dataset needed."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from audio_df.pipelines.a3 import resolve_eer_threshold, run_a3

NORMAL_STATUS = {
    "pressure_level": 1,
    "pressure_name": "normal",
    "swap_used_gb": 0.1,
    "swap_total_gb": 1.0,
    "rss_mb": 1.0,
}


def _write_clip(path, freq=440.0, sr=16000, seconds=1.0):
    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(str(path), x, sr)


def _build_fixture(tmp_path, n_clips=6):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    rows = []
    scores = []
    for i in range(n_clips):
        utt_id = f"CLIP_{i:04d}"
        label = "bonafide" if i % 2 == 0 else "spoof"
        attack_id = "-" if label == "bonafide" else "A01"
        path = audio_dir / f"{utt_id}.wav"
        _write_clip(path, freq=300.0 + 50 * i)
        rows.append(
            {
                "utt_id": utt_id,
                "path": str(path),
                "label": label,
                "speaker_id": f"SPK_{i % 2}",
                "attack_id": attack_id,
                "sample_rate": 16000,
                "duration_s": 1.0,
                "codec": "WAV",
                "split": "dev",
                "language": "en",
            }
        )
        score = 0.5 if label == "bonafide" else -0.5
        scores.append({"clip_id": utt_id, "label": label, "score": score, "attack_id": attack_id})

    manifest_path = tmp_path / "dev.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    a2_dir = tmp_path / "a2_exp"
    (a2_dir / "scores").mkdir(parents=True)
    pd.DataFrame(scores).to_csv(a2_dir / "scores" / "dev.csv", index=False)

    return manifest_path, a2_dir


def _write_a2_metrics(a2_dir, eer_threshold=0.0, via="testsets_dev"):
    """A minimal, real-shaped `metrics.json` (as `audio_df.evaluation.scoring` writes it)."""
    if via == "testsets_dev":
        payload = {
            "exp_id": a2_dir.name,
            "threshold_source": {"exp_dir": str(a2_dir), "testset": "dev", "value": eer_threshold},
            "testsets": {"dev": {"n": 6, "eer": 0.1, "eer_threshold": eer_threshold}},
        }
    elif via == "threshold_source_only":
        payload = {
            "exp_id": a2_dir.name,
            "threshold_source": {"exp_dir": str(a2_dir), "testset": "dev", "value": eer_threshold},
            "testsets": {"eval": {"n": 6, "eer": 0.1}},
        }
    elif via == "neither":
        payload = {
            "exp_id": a2_dir.name,
            "threshold_source": {"exp_dir": None, "testset": "dev", "value": None},
            "testsets": {"eval": {"n": 6, "eer": 0.1}},
        }
    else:
        raise ValueError(via)
    (a2_dir / "metrics.json").write_text(json.dumps(payload))


def _cfg(manifest_path, a2_dir, cache_dir, n=6, eer_threshold=0.0):
    return {
        "a2_experiment": str(a2_dir),
        "sample_rate": 16000,
        "subset": {
            "manifest": str(manifest_path),
            "n": n,
            "seed": 1337,
            "strata": ["label", "attack_id"],
        },
        "variants": {"state": ["numbers", "words"], "question_set": ["noul_only", "all3"]},
        "client": {"kind": "dry-run", "route": "typesafe", "model_version": "unknown"},
        "thresholds": {
            "detector": {"eer_threshold": eer_threshold, "margins": [0.05, 0.2]},
            "acoustic": {
                "high_band_hz": 4000,
                "high_band": [0.05, 0.15],
                "silence_top_db": 30,
                "silence": [0.05, 0.2],
                "duration": [2.0, 6.0],
            },
        },
        "cache_dir": str(cache_dir),
    }


def test_run_a3_dry_run_end_to_end(tmp_path):
    manifest_path, a2_dir = _build_fixture(tmp_path)
    cfg = _cfg(manifest_path, a2_dir, tmp_path / "jev_cache")

    exp_dir = run_a3(
        cfg,
        state_variant="numbers",
        question_set="noul_only",
        client_kind="dry-run",
        root=tmp_path / "runs",
        lock_path=tmp_path / "lock",
        status_fn=lambda: NORMAL_STATUS,
    )

    assert (exp_dir / "scores" / "jev_dev500.csv").exists()
    assert (exp_dir / "scores" / "a2_dev500.csv").exists()
    assert (exp_dir / "answers.csv").exists()
    assert (exp_dir / "cost_summary.json").exists()
    assert (exp_dir / "subset.csv").exists()

    jev_scores = pd.read_csv(exp_dir / "scores" / "jev_dev500.csv")
    a2_scores = pd.read_csv(exp_dir / "scores" / "a2_dev500.csv")
    assert set(jev_scores["clip_id"]) == set(a2_scores["clip_id"])
    assert {"clip_id", "label", "score"} <= set(jev_scores.columns)

    summary = json.loads((exp_dir / "cost_summary.json").read_text())
    assert summary["cost_total_usd"] == 0.0
    assert summary["n_clips"] == len(jev_scores)

    notes = (exp_dir / "notes.md").read_text()
    assert "DRY RUN" in notes

    registry = (tmp_path / "runs" / "REGISTRY.md").read_text()
    assert "| done |" in registry


def test_run_a3_all3_writes_all_question_answers(tmp_path):
    manifest_path, a2_dir = _build_fixture(tmp_path)
    cfg = _cfg(manifest_path, a2_dir, tmp_path / "jev_cache")

    exp_dir = run_a3(
        cfg,
        state_variant="words",
        question_set="all3",
        client_kind="dry-run",
        root=tmp_path / "runs",
        lock_path=tmp_path / "lock",
        status_fn=lambda: NORMAL_STATUS,
    )
    answers = pd.read_csv(exp_dir / "answers.csv")
    assert set(answers["question_id"]) == {"noul", "choice", "score"}


def test_run_a3_resumable_cache_avoids_recompute(tmp_path):
    """A second dry-run with the same config reuses the on-disk cache (no crash, same score)."""
    manifest_path, a2_dir = _build_fixture(tmp_path)
    cache_dir = tmp_path / "jev_cache"
    cfg = _cfg(manifest_path, a2_dir, cache_dir)

    exp_dir_1 = run_a3(
        cfg,
        state_variant="numbers",
        question_set="noul_only",
        client_kind="dry-run",
        root=tmp_path / "runs",
        lock_path=tmp_path / "lock",
        status_fn=lambda: NORMAL_STATUS,
        date=__import__("datetime").date(2026, 1, 1),
    )
    exp_dir_2 = run_a3(
        cfg,
        state_variant="numbers",
        question_set="noul_only",
        client_kind="dry-run",
        root=tmp_path / "runs",
        lock_path=tmp_path / "lock",
        status_fn=lambda: NORMAL_STATUS,
        date=__import__("datetime").date(2026, 1, 2),
    )
    scores_1 = pd.read_csv(exp_dir_1 / "scores" / "jev_dev500.csv").sort_values("clip_id")
    scores_2 = pd.read_csv(exp_dir_2 / "scores" / "jev_dev500.csv").sort_values("clip_id")
    assert list(scores_1["score"]) == list(scores_2["score"])
    # Cached on the second run: cache_calls should be 0 (everything already on disk).
    summary_2 = json.loads((exp_dir_2 / "cost_summary.json").read_text())
    assert summary_2["cache_calls"] == 0
    assert summary_2["cache_hits"] == summary_2["n_questions"]


def test_run_a3_typesafe_requires_small_n_first(tmp_path):
    manifest_path, a2_dir = _build_fixture(tmp_path)
    cfg = _cfg(manifest_path, a2_dir, tmp_path / "jev_cache", n=21)
    cfg["client"]["kind"] = "typesafe"

    with pytest.raises(ValueError, match="cost per call"):
        run_a3(
            cfg,
            state_variant="numbers",
            question_set="noul_only",
            client_kind="typesafe",
            n=21,
            root=tmp_path / "runs",
            lock_path=tmp_path / "lock",
            status_fn=lambda: NORMAL_STATUS,
        )


def test_run_a3_budget_guard_aborts(tmp_path, monkeypatch):
    """A client with nonzero cost per call trips the --max-cost-usd guard."""
    manifest_path, a2_dir = _build_fixture(tmp_path, n_clips=10)
    cfg = _cfg(manifest_path, a2_dir, tmp_path / "jev_cache", n=10)

    import audio_df.pipelines.a3 as run_a3_mod
    from audio_df.models.jev.client import DryRunJevClient, JevRequest

    class _ExpensiveClient(DryRunJevClient):
        route = "typesafe"
        model_version = "expensive-test"

        def ask(self, request: JevRequest):
            resp = super().ask(request)
            return resp.__class__(
                question_id=resp.question_id,
                question_type=resp.question_type,
                result=resp.result,
                route=self.route,
                model_version=self.model_version,
                latency_s=resp.latency_s,
                cost_usd=1.0,
                raw=resp.raw,
            )

    monkeypatch.setattr(run_a3_mod, "build_client", lambda kind, cfg: _ExpensiveClient())

    with pytest.raises(RuntimeError, match="budget guard"):
        run_a3(
            cfg,
            state_variant="numbers",
            question_set="noul_only",
            client_kind="dry-run",
            max_cost_usd=0.5,
            root=tmp_path / "runs",
            lock_path=tmp_path / "lock",
            status_fn=lambda: NORMAL_STATUS,
        )


# --- EER threshold resolution (from <a2_experiment>/metrics.json, not a placeholder) -------


def test_resolve_eer_threshold_config_override_wins(tmp_path):
    cfg = {"thresholds": {"detector": {"eer_threshold": 0.42}}}
    thr, info = resolve_eer_threshold(cfg, tmp_path / "does-not-exist")
    assert thr == 0.42
    assert info["source"] == "config_override"


def test_resolve_eer_threshold_reads_a2_metrics_testsets_dev(tmp_path):
    a2_dir = tmp_path / "a2_exp"
    a2_dir.mkdir()
    _write_a2_metrics(a2_dir, eer_threshold=0.137, via="testsets_dev")
    cfg = {"thresholds": {"detector": {"eer_threshold": None}}}
    thr, info = resolve_eer_threshold(cfg, a2_dir)
    assert thr == pytest.approx(0.137)
    assert info["source"] == "a2_metrics.testsets.dev.eer_threshold"


def test_resolve_eer_threshold_falls_back_to_threshold_source_value(tmp_path):
    a2_dir = tmp_path / "a2_exp"
    a2_dir.mkdir()
    _write_a2_metrics(a2_dir, eer_threshold=-0.05, via="threshold_source_only")
    cfg = {"thresholds": {"detector": {"eer_threshold": None}}}
    thr, info = resolve_eer_threshold(cfg, a2_dir)
    assert thr == pytest.approx(-0.05)
    assert info["source"] == "a2_metrics.threshold_source.value"


def test_resolve_eer_threshold_missing_metrics_json_is_actionable(tmp_path):
    a2_dir = tmp_path / "a2_exp_never_scored"
    a2_dir.mkdir()
    cfg = {"thresholds": {"detector": {"eer_threshold": None}}}
    with pytest.raises(FileNotFoundError, match="score_experiment.py"):
        resolve_eer_threshold(cfg, a2_dir)


def test_resolve_eer_threshold_metrics_json_without_a_threshold_raises(tmp_path):
    a2_dir = tmp_path / "a2_exp"
    a2_dir.mkdir()
    _write_a2_metrics(a2_dir, via="neither")
    cfg = {"thresholds": {"detector": {"eer_threshold": None}}}
    with pytest.raises(ValueError, match="score_experiment.py"):
        resolve_eer_threshold(cfg, a2_dir)


def test_run_a3_reads_threshold_from_a2_metrics_when_no_override(tmp_path):
    manifest_path, a2_dir = _build_fixture(tmp_path)
    _write_a2_metrics(a2_dir, eer_threshold=0.0, via="testsets_dev")
    cfg = _cfg(manifest_path, a2_dir, tmp_path / "jev_cache", eer_threshold=None)

    exp_dir = run_a3(
        cfg,
        state_variant="numbers",
        question_set="noul_only",
        client_kind="dry-run",
        root=tmp_path / "runs",
        lock_path=tmp_path / "lock",
        status_fn=lambda: NORMAL_STATUS,
    )

    config = yaml.safe_load((exp_dir / "config.yaml").read_text())
    assert config["features"]["eer_threshold"] == pytest.approx(0.0)
    assert config["features"]["eer_threshold_source"] == "a2_metrics.testsets.dev.eer_threshold"
    notes = (exp_dir / "notes.md").read_text()
    assert "eer_threshold" in notes


def test_run_a3_refuses_to_start_without_a2_metrics_or_override(tmp_path):
    manifest_path, a2_dir = _build_fixture(tmp_path)
    # No metrics.json written for this A2 dir, and no config override.
    cfg = _cfg(manifest_path, a2_dir, tmp_path / "jev_cache", eer_threshold=None)

    with pytest.raises(FileNotFoundError, match="score_experiment.py"):
        run_a3(
            cfg,
            state_variant="numbers",
            question_set="noul_only",
            client_kind="dry-run",
            root=tmp_path / "runs",
            lock_path=tmp_path / "lock",
            status_fn=lambda: NORMAL_STATUS,
        )
    # Refusing to start must not create the experiment directory or a registry row.
    assert not (tmp_path / "runs").exists()
