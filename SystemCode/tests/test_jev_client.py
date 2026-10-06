"""JevClient interface: DryRunJevClient determinism/orientation, TypesafeJevClient stub."""

from __future__ import annotations

import pytest

from audio_df.models.jev.client import DryRunJevClient, TypesafeJevClient
from audio_df.models.jev.questions import (
    CHOICE_QUESTION,
    NOUL_QUESTION,
    SCORE_QUESTION,
    build_request,
    noul_to_a3_score,
)
from audio_df.models.jev.state import LEVEL_WORDS


def _state_with_level(level: str) -> dict:
    return {"variant": "words", "detector": {"level": level, "confidence": "low"}, "acoustic": {}}


def test_dry_run_labels_itself_unmistakably():
    client = DryRunJevClient()
    assert client.route == "dry-run"
    assert client.model_version == "dry-run-0"
    req = build_request(NOUL_QUESTION, _state_with_level("bona fide"))
    resp = client.ask(req)
    assert resp.route == "dry-run"
    assert resp.model_version == "dry-run-0"
    assert resp.cost_usd == 0.0
    assert resp.raw.get("dry_run") is True


def test_dry_run_deterministic():
    client = DryRunJevClient()
    req = build_request(NOUL_QUESTION, _state_with_level("spoof"))
    a = client.ask(req)
    b = client.ask(req)
    assert a.result == b.result
    assert a.route == b.route == "dry-run"


def test_dry_run_noul_result_sums_to_one_and_is_typed():
    client = DryRunJevClient()
    req = build_request(NOUL_QUESTION, _state_with_level("bona fide"))
    resp = client.ask(req)
    assert set(resp.result) == {"yes", "no"}
    assert resp.result["yes"] + resp.result["no"] == pytest.approx(1.0)


def test_dry_run_choice_result_typed():
    client = DryRunJevClient()
    req = build_request(CHOICE_QUESTION, _state_with_level("bona fide"))
    resp = client.ask(req)
    assert set(resp.result) == {"bonafide", "spoof", "insufficient_evidence"}
    assert sum(resp.result.values()) == pytest.approx(1.0)


def test_dry_run_score_result_typed():
    client = DryRunJevClient()
    req = build_request(SCORE_QUESTION, _state_with_level("bona fide"))
    resp = client.ask(req)
    for v in range(0, 4):
        assert str(v) in resp.result
    assert "expected_value" in resp.result


def test_dry_run_monotone_in_detector_level():
    """More bona-fide-leaning level words -> higher p(no) on the noul question."""
    client = DryRunJevClient()
    p_no_by_level = []
    for level in LEVEL_WORDS:  # ordered low (spoof) -> high (bona fide)
        req = build_request(NOUL_QUESTION, _state_with_level(level))
        resp = client.ask(req)
        p_no_by_level.append(resp.result["no"])
    assert p_no_by_level == sorted(p_no_by_level)
    assert p_no_by_level[0] < p_no_by_level[-1]


def test_dry_run_unknown_level_falls_back_without_crashing():
    client = DryRunJevClient()
    req = build_request(NOUL_QUESTION, {"detector": {"level": "not-a-real-level"}, "acoustic": {}})
    resp = client.ask(req)
    assert 0.0 < resp.result["no"] < 1.0


def test_noul_to_a3_score_orientation():
    """Higher p(no) (more bona fide) -> higher A3 score."""
    low = noul_to_a3_score({"yes": 0.9, "no": 0.1})
    high = noul_to_a3_score({"yes": 0.1, "no": 0.9})
    assert high > low


def test_noul_to_a3_score_never_infinite():
    assert noul_to_a3_score({"yes": 0.0, "no": 1.0}) < float("inf")
    assert noul_to_a3_score({"yes": 1.0, "no": 0.0}) > float("-inf")


def test_typesafe_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        TypesafeJevClient()


def test_typesafe_ask_is_not_implemented(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "test-key-not-real")
    client = TypesafeJevClient(model_version="v-test", route="typesafe")
    assert client.route == "typesafe"
    assert client.model_version == "v-test"
    req = build_request(NOUL_QUESTION, _state_with_level("bona fide"))
    with pytest.raises(NotImplementedError, match="JEV API docs"):
        client.ask(req)


def test_typesafe_client_never_stores_the_api_key(monkeypatch):
    secret = "sk-super-secret-value-do-not-leak"
    monkeypatch.setenv("JEV_API_KEY", secret)
    client = TypesafeJevClient()
    assert secret not in repr(vars(client))
    assert secret not in str(vars(client))
    assert not hasattr(client, "api_key")
    for value in vars(client).values():
        assert value != secret
