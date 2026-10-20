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
        pd.DataFrame({"utt_id": [f"{split}_real", f"{split}_fake"],
                      "path": [f"{split}_real.wav", f"{split}_fake.wav"],
                      "label": ["bonafide", "spoof"]}).to_csv(manifests / f"{split}.csv", index=False)

    class FakeDetector:
        cache_namespace = ("a2",)

        def score_manifest(self, df, destination, *, split, cache_dir):
            return pd.DataFrame({
                "clip_id": df["utt_id"].tolist(),
                "label": df["label"].tolist(),
                "score": [2.0 if label == "bonafide" else -2.0 for label in df["label"]],
                "diagnostic": ["x" if label == "bonafide" else "y" for label in df["label"]],
            })

    monkeypatch.setattr("audio_df.runtime.trimmed_manifest", lambda df, cache, split: df)
    calls = []

    def fake_policy(state, *, model=None, client=None):
        calls.append(state)
        return {"model": "jev-1.13.0", "usage": {"input_tokens": 12, "output_tokens": 3},
                "answers": {"decision": {"type": "choice", "choice": "bonafide",
                                         "probabilities": {"bonafide": 0.8, "spoof": 0.2}},
                            "ambiguity": {"type": "noul", "noul": 0.1}}}

    selected_ids = []
    for _ in range(2):
        run = Run(name="a3", root=str(tmp_path), testsets="eval", command="evaluate")
        run.evaluate_policy(FakeDetector(), fake_policy, base="a2", limit=1, jev_model="test-version")
        requests = [json.loads(line) for line in (run.output / "jev" / "requests.jsonl").read_text().splitlines()]
        responses = [json.loads(line) for line in (run.output / "jev" / "responses.jsonl").read_text().splitlines()]
        assert len(requests) == len(responses) == 1
        selected_ids.append(requests[0]["clip_id"])
        original = requests[0]["original_detector_response"]
        assert original["clip_id"] == requests[0]["clip_id"]
        assert original["label"] in {"bonafide", "spoof"}
        assert original["diagnostic"] == ("x" if original["label"] == "bonafide" else "y")
        assert "label" not in requests[0]["state"]["context"]
        assert responses[0]["raw_jev_response"]["answers"]["ambiguity"]["noul"] == 0.1
        assert responses[0]["raw_jev_response"]["usage"]["input_tokens"] == 12
    assert selected_ids[0] == selected_ids[1]  # Stable seeded sample.
    assert len(calls) == 1  # Second run reused cache but retained complete records.
