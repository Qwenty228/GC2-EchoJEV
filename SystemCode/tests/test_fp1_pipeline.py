"""Offline composition tests: no CUDA, weights, or paid JEV requests."""
from pathlib import Path
import pytest

from audio_df.pipelines import fp1


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


@pytest.mark.parametrize("base", ["d1", "d2", "d3"])
def test_fp1_reuses_fusion_detector_and_policy(monkeypatch, base):
    sentinel = object()
    monkeypatch.setattr(fp1, "BASE", base)
    monkeypatch.setitem(fp1.DETECTORS, base, type("FakeLoader", (), {
        "load": staticmethod(lambda path: sentinel)
    }))
    run = FakeRun()
    assert fp1.main.__wrapped__(run, limit=2, jev_model="test") == {"eval": {"n": 2}}
    assert run.selected == base
    assert run.called == (sentinel, fp1.decide, base, 2, "test")


def test_fp1_never_trains():
    run = FakeRun()
    run.command = "train"
    with pytest.raises(ValueError, match="evaluation-only"):
        fp1.main.__wrapped__(run)
