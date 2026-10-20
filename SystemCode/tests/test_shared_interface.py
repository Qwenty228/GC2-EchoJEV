import pandas as pd

from audio_df.runtime import Run
from audio_df.modules.detector import GMMDetector


def test_a1_has_shared_scoring_method():
    assert callable(GMMDetector.score_manifest)


def test_runtime_evaluate_has_no_architecture_special_cases(tmp_path, monkeypatch):
    root = tmp_path
    artifact = root / "artifacts" / "a2" / "version"
    artifact.mkdir(parents=True)
    (artifact / "COMPLETE").write_text("ok")

    manifest_dir = root / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    for split in ("dev", "eval"):
        pd.DataFrame({
            "utt_id": [f"{split}_x", f"{split}_y"],
            "path": [f"{split}_x.wav", f"{split}_y.wav"],
            "label": ["bonafide", "spoof"],
        }).to_csv(manifest_dir / f"{split}.csv", index=False)

    class FakeDetector:
        cache_namespace = ("a2",)

        def score_manifest(self, df, destination, *, split, cache_dir):
            assert cache_dir == root / "data" / "cache" / "a2"
            assert split in ("dev", "eval")
            assert len(df) == 2
            return pd.DataFrame({
                "clip_id": df["utt_id"].tolist(),
                "label": ["bonafide", "spoof"],
                "score": [2.0, -2.0],
            })

    run = Run(name="a2", root=str(root), testsets="eval", command="evaluate", dataset="2019LA")
    monkeypatch.setattr(
    "audio_df.runtime.trimmed_manifest",
    lambda df, cache_root, split: df)
    result = run.evaluate(FakeDetector())

    assert "eval" in result
    assert result["eval"]["eer"] == 0.0
    assert result["eval"]["min_dcf"] == 0.0
    assert result["eval"]["roc_auc"] == 1.0
    assert result["eval"]["f1"] == 1.0
    assert (run.output / "metrics.json").exists()
    import json
    import yaml
    assert set(json.loads((run.output / "metrics.json").read_text())) == {"eval"}
    meta = yaml.safe_load((run.output / "config.yaml").read_text())
    assert meta["split_roles"] == {"dev": "validation_threshold", "eval": "held_out_test"}
    assert meta["threshold_source"] == "dev"
    assert (run.output / "calibration.json").exists()
