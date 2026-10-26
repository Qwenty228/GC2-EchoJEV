"""Offline tests for Laya configuration and local decisions."""
from audio_df.modules.laya import DryRunLayaClient, LAYAState, decide


class MockLayaClient:
    def system_one(self, **kwargs):
        return kwargs


def test_laya_defaults_from_yaml():
    result = decide({"deadline": "tomorrow"}, client=MockLayaClient())
    assert "context" in result["state"]
    assert result["state"]["context"]["deadline"] == "tomorrow"
    assert "decision" in result["questions"]
    assert "evidence_quality" in result["questions"]


def test_laya_dry_run_deterministic_decision():
    client = DryRunLayaClient()
    
    # Near-threshold margin -> uncertain
    res_uncertain = client.system_one({"context": {"signed_margin": 0.05}}, {})
    assert res_uncertain["answers"]["decision"]["choice"] == "uncertain"
    
    # Strong positive margin -> authentic (bonafide)
    res_bonafide = client.system_one({"context": {"signed_margin": 1.2}}, {})
    assert res_bonafide["answers"]["decision"]["choice"] == "bonafide"
    
    # Strong negative margin -> synthetic (spoof)
    res_spoof = client.system_one({"context": {"signed_margin": -1.5}}, {})
    assert res_spoof["answers"]["decision"]["choice"] == "spoof"


def test_laya_decide_with_dry_run_model():
    state = LAYAState(context={"signed_margin": 0.8})
    result = decide(state, model="dry-run")
    assert result["answers"]["decision"]["choice"] == "bonafide"
