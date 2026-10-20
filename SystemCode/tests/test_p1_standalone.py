"""Offline tests: no TypeSafe key or network access needed."""
from audio_df.pipelines.p1 import SCENARIOS, WARNING


def test_demo_is_explicitly_not_a_benchmark():
    assert "NOT a benchmarkable" in WARNING
    assert "security_review" in SCENARIOS["launch"]
    assert "independent_replication" in SCENARIOS["research"]
