from __future__ import annotations

import json

import pandas as pd
import pytest

from audio_df.runtime import Run


def _write_manifest(path, utt_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "utt_id": [utt_id],
            "path": [f"data/raw/spoofceleb/{utt_id}.flac"],
            "label": ["bonafide"],
            "attack_id": ["-"],
        }
    ).to_csv(path, index=False)


def _spoofceleb_manifests(root) -> None:
    manifest_dir = root / "data" / "manifests"
    _write_manifest(manifest_dir / "spoofceleb_train.csv", "train-id")
    _write_manifest(manifest_dir / "spoofceleb_development.csv", "dev-id")
    _write_manifest(manifest_dir / "spoofceleb_evaluation.csv", "eval-id")


def test_spoofceleb_training_loads_train_manifest(tmp_path):
    _spoofceleb_manifests(tmp_path)
    run = Run(
        name="a1",
        dataset="spoofceleb",
        root=str(tmp_path),
        command="train",
    )

    class CaptureFit:
        cache_namespace = ("test",)

        def fit(self, train, cache_dir, seed=None):
            self.train_ids = train["utt_id"].tolist()
            self.seed = seed
            return self

        def save(self, directory):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "model.bin").write_bytes(b"fixture")

    model = CaptureFit()
    run.train(model)
    artifact = run.save(model)

    assert model.train_ids == ["train-id"]
    assert model.seed == 1337
    assert run._train_preprocessing == "raw"
    assert json.loads((artifact / "preprocessing.json").read_text())["dataset"] == "spoofceleb"
    assert json.loads((artifact / "preprocessing.json").read_text())["training_subset"] == {
        "max_per_stratum": 100,
        "strata": ["label", "attack_id"],
        "seed": 1337,
    }


def test_spoofceleb_training_uses_same_uniform_subset_and_keeps_eval_full(tmp_path):
    _spoofceleb_manifests(tmp_path)
    config_path = tmp_path / "configs" / "data.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "spoofceleb:\n  training_subset:\n    max_per_stratum: 2\n"
        "    strata: [label, attack_id]\n",
        encoding="utf-8",
    )

    def write_split(split: str) -> None:
        rows = []
        for attack_id, label in (("-", "bonafide"), ("a01", "spoof"), ("a02", "spoof")):
            for index in range(5):
                utt_id = f"{split}-{attack_id}-{index}"
                rows.append(
                    {
                        "utt_id": utt_id,
                        "path": f"data/raw/spoofceleb/{utt_id}.flac",
                        "label": label,
                        "attack_id": attack_id,
                    }
                )
        pd.DataFrame(rows).to_csv(
            tmp_path / "data" / "manifests" / f"spoofceleb_{split}.csv",
            index=False,
        )

    write_split("train")
    write_split("development")

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

    a1_run = Run(name="a1", dataset="spoofceleb", root=str(tmp_path), command="train")
    a2_run = Run(name="a2", dataset="spoofceleb", root=str(tmp_path), command="train")
    a4_run = Run(name="a4", dataset="spoofceleb", root=str(tmp_path), command="train")
    a1_model = CaptureA1()
    a2_model = CaptureDevFit()
    a4_model = CaptureDevFit()

    a1_run.train(a1_model)
    a2_run.train(a2_model)
    a4_run.train(a4_model)

    assert a1_model.train_ids == a2_model.train_ids == a4_model.train_ids
    assert a2_model.dev_ids == a4_model.dev_ids
    selected = a1_run._spoofceleb_training_manifest("train")
    counts = selected.groupby(["label", "attack_id"]).size().to_dict()
    assert counts == {("bonafide", "-"): 2, ("spoof", "a01"): 2, ("spoof", "a02"): 2}
    assert len(a1_run.manifest("eval")) == 1


def test_spoofceleb_run_routes_development_and_evaluation_manifests(tmp_path):
    _spoofceleb_manifests(tmp_path)
    run = Run(
        name="a1",
        dataset="spoofceleb",
        root=str(tmp_path),
        threshold_from="dev",
        testsets="eval",
        command="evaluate",
    )

    assert run.manifest("train").iloc[0]["utt_id"] == "train-id"
    assert run.manifest("dev").iloc[0]["utt_id"] == "dev-id"
    assert run.manifest("development").iloc[0]["utt_id"] == "dev-id"
    assert run.manifest("eval").iloc[0]["utt_id"] == "eval-id"
    assert run.manifest("evaluation").iloc[0]["utt_id"] == "eval-id"
    run.validate_testsets()
    assert not run._is_trimmed("dev")


def test_spoofceleb_rejects_same_development_manifest_for_calibration_and_test(tmp_path):
    _spoofceleb_manifests(tmp_path)
    run = Run(
        name="a1",
        dataset="spoofceleb",
        root=str(tmp_path),
        threshold_from="dev",
        testsets="development",
    )

    with pytest.raises(ValueError, match="cannot appear"):
        run.validate_testsets()


def test_spoofceleb_selects_its_own_artifact_not_a_newer_2019_model(tmp_path):
    artifact_root = tmp_path / "artifacts" / "a1"
    spoof_artifact = artifact_root / "20261025-000000"
    la_artifact = artifact_root / "20261026-000000"
    for artifact, dataset in ((spoof_artifact, "spoofceleb"), (la_artifact, "2019LA")):
        artifact.mkdir(parents=True)
        (artifact / "COMPLETE").write_text("ok\n", encoding="utf-8")
        (artifact / "preprocessing.json").write_text(
            json.dumps({"dataset": dataset}), encoding="utf-8"
        )

    run = Run(name="a1", dataset="spoofceleb", root=str(tmp_path), command="evaluate")

    assert run.model_from == spoof_artifact


def test_spoofceleb_rejects_explicit_artifact_from_another_dataset(tmp_path):
    artifact = tmp_path / "artifacts" / "a1" / "20261026-000000"
    artifact.mkdir(parents=True)
    (artifact / "COMPLETE").write_text("ok\n", encoding="utf-8")
    (artifact / "preprocessing.json").write_text(
        json.dumps({"dataset": "2019LA"}), encoding="utf-8"
    )
    run = Run(
        name="a1",
        dataset="spoofceleb",
        root=str(tmp_path),
        model_from="artifacts/a1/20261026-000000",
    )

    with pytest.raises(ValueError, match="not 'spoofceleb'"):
        _ = run.model_from
