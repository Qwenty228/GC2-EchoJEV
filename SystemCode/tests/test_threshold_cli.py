"""Calibration and test sets must be distinct."""
import pandas as pd
import pytest
from audio_df.runtime import Run


def _manifest(root, name, ids):
    directory = root / "data" / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"utt_id": ids, "path": [f"{i}.wav" for i in ids],
                  "label": ["bonafide" if n % 2 else "spoof" for n, _ in enumerate(ids)]}).to_csv(directory / f"{name}.csv", index=False)


def test_default_calibration_and_test(tmp_path):
    _manifest(tmp_path, "dev", ["d0", "d1"])
    _manifest(tmp_path, "eval", ["e0", "e1"])
    run = Run(name="a1", root=str(tmp_path), command="evaluate")
    assert run.threshold_from == "dev"
    assert run.testsets == ["eval"]
    run.validate_testsets()


def test_reject_same_calibration_and_test(tmp_path):
    _manifest(tmp_path, "dev", ["d0", "d1"])
    with pytest.raises(ValueError, match="cannot appear"):
        Run(name="a1", root=str(tmp_path), testsets="dev").validate_testsets()


def test_reject_overlapping_eval_family(tmp_path):
    _manifest(tmp_path, "eval_clean10k", ["e0", "e1"])
    _manifest(tmp_path, "eval", ["e0", "e1", "e2"])
    with pytest.raises(ValueError, match="may overlap"):
        Run(name="a1", root=str(tmp_path), threshold_from="eval_clean10k", testsets="eval").validate_testsets()


def test_custom_disjoint_calibration(tmp_path):
    _manifest(tmp_path, "itw", ["i0", "i1"])
    _manifest(tmp_path, "eval", ["e0", "e1"])
    Run(name="a1", root=str(tmp_path), threshold_from="itw", testsets="eval").validate_testsets()
