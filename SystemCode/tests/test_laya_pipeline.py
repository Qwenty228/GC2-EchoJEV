"""Offline composition tests for Laya pipelines (a3_laya and fp2)."""
from pathlib import Path

import pytest

from audio_df.pipelines import a3_laya, fp2


class FakeRun:
    command = "evaluate"

    def __init__(self):
        self.selected = None
        self.called = None

    def model_from_for(self, name):
        self.selected = name
        return Path("/fake/artifacts") / name / "completed"

    def evaluate_policy(self, detector, policy, *, base, limit, jev_model):
        self.called = (detector, policy, base, limit, jev_model)
        return {"eval": {"n": limit}}


def test_a3_laya_evaluates_frozen_a2(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(a3_laya.SSLDetector, "load", staticmethod(lambda path: sentinel))
    run = FakeRun()
    assert a3_laya.main.__wrapped__(run, limit=5, laya_model="dry-run") == {"eval": {"n": 5}}
    assert run.selected == "a2"
    assert run.called == (sentinel, a3_laya.decide, "a2", 5, "dry-run")


@pytest.mark.parametrize("base", ["d1", "d2", "d3"])
def test_fp2_reuses_fusion_detector_and_laya_policy(monkeypatch, base):
    sentinel = object()
    monkeypatch.setattr(fp2, "BASE", base)
    monkeypatch.setitem(fp2.DETECTORS, base, type("FakeLoader", (), {
        "load": staticmethod(lambda path: sentinel)
    }))
    run = FakeRun()
    assert fp2.main.__wrapped__(run, limit=10, laya_model="dry-run") == {"eval": {"n": 10}}
    assert run.selected == base
    assert run.called == (sentinel, fp2.decide, base, 10, "dry-run")


def test_pipelines_reject_training():
    run = FakeRun()
    run.command = "train"
    with pytest.raises(ValueError, match="evaluation-only"):
        a3_laya.main.__wrapped__(run)
    with pytest.raises(ValueError, match="evaluation-only"):
        fp2.main.__wrapped__(run)
