"""JevCache: never re-calls a cached key, key changes on any config element, no secrets."""

from __future__ import annotations

import json

import pytest

from audio_df.jev.cache import JevCache, cache_key, config_hash
from audio_df.jev.client import DryRunJevClient, JevClient, JevRequest, JevResponse
from audio_df.jev.questions import NOUL_QUESTION, QUESTION_TEXT_VERSION, build_request


class _CountingClient(JevClient):
    def __init__(self, inner: JevClient) -> None:
        self._inner = inner
        self.calls = 0

    @property
    def route(self) -> str:
        return self._inner.route

    @property
    def model_version(self) -> str:
        return self._inner.model_version

    def ask(self, request: JevRequest) -> JevResponse:
        self.calls += 1
        return self._inner.ask(request)


def _state(level="bona fide"):
    return {"variant": "words", "detector": {"level": level, "confidence": "low"}, "acoustic": {}}


def test_get_or_call_never_recalls_a_cached_key(tmp_path):
    cache = JevCache(root=tmp_path / "cache")
    counting = _CountingClient(DryRunJevClient())
    req = build_request(NOUL_QUESTION, _state())

    r1 = cache.get_or_call(
        "clip-1", req, counting, question_version=QUESTION_TEXT_VERSION, variant="words"
    )
    r2 = cache.get_or_call(
        "clip-1", req, counting, question_version=QUESTION_TEXT_VERSION, variant="words"
    )

    assert counting.calls == 1
    assert r1.result == r2.result


def test_cache_writes_a_file_at_the_documented_layout(tmp_path):
    cache = JevCache(root=tmp_path / "cache")
    counting = _CountingClient(DryRunJevClient())
    req = build_request(NOUL_QUESTION, _state())
    cache.get_or_call(
        "clip-1", req, counting, question_version=QUESTION_TEXT_VERSION, variant="words"
    )

    cfg_hash = config_hash(req, QUESTION_TEXT_VERSION, "words", "dry-run", "dry-run-0")
    expected = tmp_path / "cache" / cfg_hash[:12] / "clip-1__noul.json"
    assert expected.exists()
    payload = json.loads(expected.read_text())
    assert payload["route"] == "dry-run"
    assert payload["response"]["question_id"] == "noul"


@pytest.mark.parametrize(
    "mutate",
    ["question_id", "question_version", "variant", "route_and_model", "state"],
)
def test_key_changes_when_any_config_element_changes(tmp_path, mutate):
    cache = JevCache(root=tmp_path / "cache")
    counting = _CountingClient(DryRunJevClient())
    req = build_request(NOUL_QUESTION, _state())
    cache.get_or_call(
        "clip-1", req, counting, question_version=QUESTION_TEXT_VERSION, variant="words"
    )
    assert counting.calls == 1

    if mutate == "question_id":
        req2 = JevRequest("different-question", req.question_type, req.prompt_text, req.state_json)
        variant, question_version = "words", QUESTION_TEXT_VERSION
    elif mutate == "question_version":
        req2 = req
        variant, question_version = "words", "v2"
    elif mutate == "variant":
        req2 = req
        variant, question_version = "numbers", QUESTION_TEXT_VERSION
    elif mutate == "state":
        req2 = build_request(NOUL_QUESTION, _state(level="strongly spoof"))
        variant, question_version = "words", QUESTION_TEXT_VERSION
    else:  # route_and_model — same request, different client entirely
        req2 = req
        variant, question_version = "words", QUESTION_TEXT_VERSION

    if mutate == "route_and_model":
        other_counting = _CountingClient(DryRunJevClient())
        # Simulate a distinct route/model_version via a thin wrapper.
        class _OtherRoute(JevClient):
            route = "typesafe"
            model_version = "some-other-model"

            def ask(self, request: JevRequest) -> JevResponse:
                other_counting.calls += 1
                return DryRunJevClient().ask(request)

        client2 = _OtherRoute()
    else:
        client2 = counting

    cache.get_or_call("clip-1", req2, client2, question_version=question_version, variant=variant)

    if mutate == "route_and_model":
        assert other_counting.calls == 1
    else:
        assert counting.calls == 2


def test_no_secrets_in_cache_file(tmp_path, monkeypatch):
    secret = "sk-do-not-leak-this-anywhere"
    monkeypatch.setenv("JEV_API_KEY", secret)
    cache = JevCache(root=tmp_path / "cache")
    counting = _CountingClient(DryRunJevClient())
    req = build_request(NOUL_QUESTION, _state())
    cache.get_or_call(
        "clip-1", req, counting, question_version=QUESTION_TEXT_VERSION, variant="words"
    )

    files = list((tmp_path / "cache").rglob("*.json"))
    assert files
    for f in files:
        assert secret not in f.read_text()


def test_cache_key_is_order_independent_dict_wise(tmp_path):
    req_a = JevRequest("noul", "noul", "p", {"b": 1, "a": 2})
    req_b = JevRequest("noul", "noul", "p", {"a": 2, "b": 1})
    h_a = config_hash(req_a, "v1", "numbers", "dry-run", "dry-run-0")
    h_b = config_hash(req_b, "v1", "numbers", "dry-run", "dry-run-0")
    assert h_a == h_b


def test_cache_key_helper_changes_with_clip_id():
    cfg_hash = "abc123"
    assert cache_key("clip-1", cfg_hash) != cache_key("clip-2", cfg_hash)
