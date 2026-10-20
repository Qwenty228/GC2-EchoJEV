"""Offline regression: per-run records survive a JEV cache hit."""
import json

import pandas as pd

from audio_df.runtime import Run


def test_policy_preserves_original_and_raw_responses(tmp_path, monkeypatch):
    artifact = tmp_path / "artifacts" / "a2" / "v1"
    artifact.mkdir(parents=True)
    (artifact / "COMPLETE").write_text("ok")
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    for split in ("dev", "eval"):
        pd.DataFrame({"utt_id": ["real", "fake"],
                      "path": ["unused1.wav", "unused2.wav"],
                      "label": ["bonafide", "spoof"]}).to_csv(manifests / f"{split}.csv", index=False)

    class FakeDetector:
        cache_namespace = ("a2",)

        def score_manifest(self, df, destination, *, split, cache_dir):
            return pd.DataFrame({"clip_id": ["real", "fake"],
                                 "label": ["bonafide", "spoof"],
                                 "score": [2.0, -2.0], "diagnostic": ["x", "y"]})

    monkeypatch.setattr("audio_df.runtime.trimmed_manifest", lambda df, cache, split: df)
    calls = []

    def fake_policy(state, *, model=None, client=None):
        calls.append(state)
        return {"model": "jev-1.13.0", "usage": {"input_tokens": 12, "output_tokens": 3},
                "answers": {"decision": {"type": "choice", "choice": "bonafide",
                                         "probabilities": {"bonafide": 0.8, "spoof": 0.2}},
                            "ambiguity": {"type": "noul", "noul": 0.1}}}

    for _ in range(2):
        run = Run(name="a3", root=str(tmp_path), testsets="dev,eval", command="evaluate")
        run.evaluate_policy(FakeDetector(), fake_policy, base="a2", limit=1, jev_model="test-version")
        requests = [json.loads(line) for line in (run.output / "jev" / "requests.jsonl").read_text().splitlines()]
        responses = [json.loads(line) for line in (run.output / "jev" / "responses.jsonl").read_text().splitlines()]
        assert len(requests) == len(responses) == 1
        assert requests[0]["original_detector_response"]["diagnostic"] == "x"
        assert requests[0]["original_detector_response"]["label"] == "bonafide"
        assert "label" not in requests[0]["state"]["context"]
        assert responses[0]["raw_jev_response"]["answers"]["ambiguity"]["noul"] == 0.1
        assert responses[0]["raw_jev_response"]["usage"]["input_tokens"] == 12
    assert len(calls) == 1  # second run reused cache but retained complete records
