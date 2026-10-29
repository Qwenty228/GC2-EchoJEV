from __future__ import annotations

import pandas as pd

from audio_df.runtime import Run


def _phonemedf_manifests(root) -> None:
    manifest_dir = root / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "development", "evaluation"):
        rows = []
        for attack_id, label in (
            ("-", "bonafide"),
            ("tts-TestTTS", "spoof"),
            ("vc-TestVC", "spoof"),
        ):
            for index in range(5):
                utt_id = f"{split}-{attack_id}-{index}"
                rows.append(
                    {
                        "utt_id": utt_id,
                        "path": f"data/raw/phonemedf/{utt_id}.wav",
                        "label": label,
                        "attack_id": attack_id,
                        "speaker_id": f"speaker-{split}-{index}",
                    }
                )
        pd.DataFrame(rows).to_csv(
            manifest_dir / f"phonemedf_{split}.csv", index=False
        )


def test_phonemedf_training_uses_same_uniform_subsets_for_a1_a2_a4(tmp_path):
    _phonemedf_manifests(tmp_path)
    config_path = tmp_path / "configs" / "data.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "phonemedf:\n  training_subset:\n    max_per_stratum: 2\n"
        "    strata: [label, attack_id]\n",
        encoding="utf-8",
    )

    class CaptureA1:
        cache_namespace = ("test",)

        def fit(self, train, cache_dir, seed=None):
            self.train_ids = train["utt_id"].tolist()
            return self

    class CaptureDevFit:
        cache_namespace = ("test",)

        def fit(self, train, cache_dir, seed=None, dev=None):
            self.train_ids = train["utt_id"].tolist()
            self.dev_ids = dev["utt_id"].tolist()
            return self

    runs = [
        Run(name=name, dataset="phonemedf", root=str(tmp_path), command="train")
        for name in ("a1", "a2", "a4")
    ]
    models = [CaptureA1(), CaptureDevFit(), CaptureDevFit()]
    for run, model in zip(runs, models, strict=True):
        run.train(model)

    assert models[0].train_ids == models[1].train_ids == models[2].train_ids
    assert models[1].dev_ids == models[2].dev_ids
    selected = runs[0]._dataset_training_manifest("train")
    assert selected.groupby(["label", "attack_id"]).size().to_dict() == {
        ("bonafide", "-"): 2,
        ("spoof", "tts-TestTTS"): 2,
        ("spoof", "vc-TestVC"): 2,
    }
    assert len(runs[0].manifest("eval")) == 15
    assert (tmp_path / "data" / "cache" / "phonemedf" / "training_subsets" / "train_per-stratum-2_seed-1337.csv").is_file()


def test_phonemedf_routes_dev_and_eval_manifests_without_overlap(tmp_path):
    _phonemedf_manifests(tmp_path)
    run = Run(
        name="a1",
        dataset="phonemedf",
        root=str(tmp_path),
        threshold_from="development",
        testsets="evaluation",
        command="evaluate",
    )

    assert len(run.manifest("dev")) == 15
    assert len(run.manifest("development")) == 15
    assert len(run.manifest("eval")) == 15
    assert len(run.manifest("evaluation")) == 15
    run.validate_testsets()
