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
            "utt_id": ["x", "y"],
            "path": ["unused_x.wav", "unused_y.wav"],
            "label": ["bonafide", "spoof"],
        }).to_csv(manifest_dir / f"{split}.csv", index=False)

    class FakeDetector:
        cache_namespace = ("a2",)

        def score_manifest(self, df, destination, *, split, cache_dir):
            assert cache_dir == root / "data" / "cache" / "a2"
            assert split in ("dev", "eval")
            assert len(df) == 2
            return pd.DataFrame({
                "clip_id": ["x", "y"],
                "label": ["bonafide", "spoof"],
                "score": [2.0, -2.0],
            })

    monkeypatch.setattr(
        "audio_df.runtime.evaluate_scores",
        lambda dev, test, path: {"eer": 0.0},
    )

    run = Run(name="a2", root=str(root), testsets="dev,eval", command="evaluate")
    result = run.evaluate(FakeDetector())

    assert result == {"eval": {"eer": 0.0}}
    assert (run.output / "metrics.json").exists()
