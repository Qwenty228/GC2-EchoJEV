"""Offline tests for Jev configuration and per-request overrides."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from audio_df.modules.jev import JEVConfig, JEVState, decide


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    sdk = ModuleType("typesafe_sdk")
    for name in ("Choice", "Score", "Noul"):
        setattr(sdk, name, type(name, (), {"__init__": lambda self, **kwargs: self.__dict__.update(kwargs)}))
    monkeypatch.setitem(sys.modules, "typesafe_sdk", sdk)


class Client:
    def system_one(self, **kwargs):
        return kwargs


def test_defaults_from_yaml():
    result = decide({"deadline": "tomorrow"}, client=Client())
    assert set(result["questions"]) == {"decision", "evidence_quality", "critical_blocker"}
    assert result["state"]["context"]["deadline"] == "tomorrow"
    assert result["state"]["decision_guidance"]


def test_override_does_not_change_defaults():
    case = JEVState(
        context={"decision": "release?"},
        prompt="Only use verified release evidence.",
        questions={"ready": {"type": "noul", "instructions": "Is the release approved?"}},
    )
    result = decide(case, client=Client())
    assert set(result["questions"]) == {"ready"}
    assert result["state"]["decision_guidance"] == case.prompt
    assert set(decide({"x": 1}, client=Client())["questions"]) == {
        "decision", "evidence_quality", "critical_blocker"
    }


def test_rejects_bad_score():
    with pytest.raises(ValueError, match="ordered criteria"):
        decide({"x": 1}, config=JEVConfig(questions={"bad": {
            "type": "score", "instructions": "Rate", "criteria": ["one"]
        }}), client=Client())
