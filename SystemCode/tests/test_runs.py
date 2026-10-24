"""collect(): newest artifact per (model, training audio), runs of one artifact merged."""
from __future__ import annotations

import json

import pandas as pd
import yaml

from audio_df.helpers.research.runs import collect, eer_table


def _run(root, name, model, artifact, sets, train=None, score=1.0):
    run = root / "runs" / name
    (run / "scores").mkdir(parents=True)
    cfg = {"architecture": model, "model_artifact": str(root / "artifacts" / model / artifact)}
    if train:
        cfg["train_preprocessing"] = train
    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    for s in sets:
        pd.DataFrame({"label": ["bonafide", "spoof"], "score": [score, -score]}).to_csv(
            run / "scores" / f"{s}.csv", index=False)


def test_collect_merges_runs_of_one_artifact_and_keeps_newest(tmp_path):
    _run(tmp_path, "E-20261020-000000-a2", "a2", "20261020-000000", ["dev", "eval"])
    _run(tmp_path, "E-20261023-000000-a2", "a2", "20261023-000000", ["dev", "eval", "itw"], train="trimmed")
    _run(tmp_path, "E-20261024-000000-a2", "a2", "20261023-000000", ["dev", "asvspoof2021_eval"], train="trimmed")
    _run(tmp_path, "E-20261024-000001-a3", "a3", "x", ["dev"])  # not requested
    res = collect(tmp_path / "runs", ["a2"])

    assert set(res) == {("a2", "raw"), ("a2", "trimmed")}
    assert set(res[("a2", "trimmed")]["scores"]) == {"dev", "eval", "itw", "asvspoof2021_eval"}
    assert res[("a2", "trimmed")]["runs"] == ["E-20261023-000000-a2", "E-20261024-000000-a2"]


def test_collect_reads_training_audio_from_artifact_when_config_lacks_it(tmp_path):
    art = tmp_path / "artifacts" / "d1" / "20261023-000000"
    art.mkdir(parents=True)
    (art / "preprocessing.json").write_text(json.dumps({"train": "trimmed"}))
    _run(tmp_path, "E-20261023-000000-d1", "d1", "20261023-000000", ["dev"])
    table = eer_table(collect(tmp_path / "runs", ["d1"]))
    assert list(table.index) == [("d1", "trimmed")]
    assert table.loc[("d1", "trimmed"), "dev"] == 0.0
