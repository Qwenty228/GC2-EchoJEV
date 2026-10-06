"""Tests for `scripts/make_degraded.py` against synthetic clips in `tmp_path`.

No real ASVspoof/MUSAN data is used: source clips are synthesised tones, and the "noise
bank" is synthesised white noise written to tiny WAV files. `scripts/` is not a package, so
the module is loaded by file path via `importlib`, matching the other script tests in this
repo (see `tests/test_make_tables.py`).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from audio_df.runtime.resources import load_resources

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "make_degraded.py"
_spec = importlib.util.spec_from_file_location("make_degraded", _SCRIPT_PATH)
make_degraded_mod = importlib.util.module_from_spec(_spec)
sys.modules["make_degraded"] = make_degraded_mod
_spec.loader.exec_module(make_degraded_mod)

compute_subset = make_degraded_mod.compute_subset
rebuild_config_text = make_degraded_mod.rebuild_config_text
write_clean_subset_manifest = make_degraded_mod.write_clean_subset_manifest
load_noise_bank = make_degraded_mod.load_noise_bank
generate_condition = make_degraded_mod.generate_condition
run_generate = make_degraded_mod.run_generate
verify_outputs = make_degraded_mod.verify_outputs

SR = 16000

NORMAL_STATUS = {
    "pressure_level": 1,
    "pressure_name": "normal",
    "swap_used_gb": 0.1,
    "swap_total_gb": 1.0,
    "rss_mb": 10.0,
}


def _resources_cfg():
    return load_resources("configs/resources.yaml")


def _write_tone(path: Path, freq: float, seconds: float = 0.5, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, x, sr)


def _full_manifest(tmp_path: Path, n_bonafide: int = 40, n_spoof_a: int = 30, n_spoof_b: int = 30):
    rows = []
    audio_dir = tmp_path / "audio"
    for i in range(n_bonafide):
        p = audio_dir / f"bona_{i:03d}.wav"
        _write_tone(p, 300 + i)
        rows.append(
            dict(
                utt_id=f"bona_{i:03d}",
                path=str(p),
                label="bonafide",
                speaker_id=f"spk{i % 5}",
                attack_id="-",
                source_corpus="synthetic",
                sample_rate=SR,
                duration_s=0.5,
                codec="WAV",
                split="eval",
                language="en",
            )
        )
    for i in range(n_spoof_a):
        p = audio_dir / f"a01_{i:03d}.wav"
        _write_tone(p, 500 + i)
        rows.append(
            dict(
                utt_id=f"a01_{i:03d}",
                path=str(p),
                label="spoof",
                speaker_id=f"spk{i % 5}",
                attack_id="A01",
                source_corpus="synthetic",
                sample_rate=SR,
                duration_s=0.5,
                codec="WAV",
                split="eval",
                language="en",
            )
        )
    for i in range(n_spoof_b):
        p = audio_dir / f"a02_{i:03d}.wav"
        _write_tone(p, 700 + i)
        rows.append(
            dict(
                utt_id=f"a02_{i:03d}",
                path=str(p),
                label="spoof",
                speaker_id=f"spk{i % 5}",
                attack_id="A02",
                source_corpus="synthetic",
                sample_rate=SR,
                duration_s=0.5,
                codec="WAV",
                split="eval",
                language="en",
            )
        )
    return pd.DataFrame.from_records(rows)


def _base_cfg(tmp_path: Path, manifest_path: Path) -> dict:
    return {
        "seed": 1337,
        "subset": {
            "source_manifest": str(manifest_path),
            "n": 30,
            "strata": ["label", "attack_id"],
            "clip_ids": [],
        },
        "conditions": {
            "eval_mp3_32k": {"type": "mp3", "bitrate_kbps": 32},
            "eval_resample_8k": {"type": "resample", "mid_sr": 8000},
            "eval_noise_10db": {
                "type": "noise",
                "snr_db": 10,
                "noise_dir": str(tmp_path / "noise"),
            },
        },
        "output_dir": str(tmp_path / "degraded"),
        "manifests": {
            "out_dir": str(tmp_path / "manifests"),
            "clean_subset": str(tmp_path / "manifests" / "eval_clean10k.csv"),
        },
    }


def _write_noise_bank(noise_dir: Path, n: int = 3) -> None:
    for i in range(n):
        _write_tone(noise_dir / f"noise_{i}.wav", freq=50 + i, seconds=2.0)


# ---------------------------------------------------------------------------
# subset proportions
# ---------------------------------------------------------------------------


def test_subset_preserves_proportions(tmp_path):
    manifest_path = tmp_path / "eval.csv"
    full_df = _full_manifest(tmp_path)
    full_df.to_csv(manifest_path, index=False)
    cfg = _base_cfg(tmp_path, manifest_path)

    sub_df = compute_subset(cfg, full_df)
    assert len(sub_df) == 30

    full_counts = full_df.groupby(["label", "attack_id"]).size()
    sub_counts = sub_df.groupby(["label", "attack_id"]).size()
    for key in full_counts.index:
        full_prop = full_counts[key] / len(full_df)
        sub_prop = sub_counts.get(key, 0) / len(sub_df)
        assert abs(full_prop - sub_prop) < 0.05


def test_rebuild_config_text_contains_clip_ids_and_stats(tmp_path):
    manifest_path = tmp_path / "eval.csv"
    full_df = _full_manifest(tmp_path)
    full_df.to_csv(manifest_path, index=False)
    cfg = _base_cfg(tmp_path, manifest_path)
    sub_df = compute_subset(cfg, full_df)
    cfg["subset"]["clip_ids"] = sorted(sub_df["utt_id"].tolist())

    text = rebuild_config_text(cfg, full_df, sub_df)
    assert "clip_ids" in text
    parsed = yaml.safe_load(text)
    assert len(parsed["subset"]["clip_ids"]) == 30
    assert "#" in text  # per-stratum comment block present


def test_write_clean_subset_manifest_columns_and_paths(tmp_path):
    manifest_path = tmp_path / "eval.csv"
    full_df = _full_manifest(tmp_path)
    full_df.to_csv(manifest_path, index=False)
    cfg = _base_cfg(tmp_path, manifest_path)
    sub_df = compute_subset(cfg, full_df)

    out_path = tmp_path / "manifests" / "eval_clean10k.csv"
    manifest = write_clean_subset_manifest(sub_df, out_path)

    assert out_path.exists()
    written = pd.read_csv(out_path)
    assert list(written.columns) == list(manifest.columns)
    assert (written["condition"] == "clean").all()
    assert set(written["path"]) == set(sub_df["path"])


# ---------------------------------------------------------------------------
# noise bank loading
# ---------------------------------------------------------------------------


def test_load_noise_bank_missing_dir_raises_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="get_musan"):
        load_noise_bank(tmp_path / "does_not_exist")


def test_load_noise_bank_loads_files(tmp_path):
    noise_dir = tmp_path / "noise"
    _write_noise_bank(noise_dir, n=3)
    bank = load_noise_bank(noise_dir)
    assert len(bank) == 3
    assert all(b.dtype == np.float32 for b in bank)


# ---------------------------------------------------------------------------
# generate_condition / run_generate end-to-end on synthetic clips
# ---------------------------------------------------------------------------


@pytest.fixture
def small_job(tmp_path):
    manifest_path = tmp_path / "eval.csv"
    full_df = _full_manifest(tmp_path, n_bonafide=2, n_spoof_a=2, n_spoof_b=2)
    full_df.to_csv(manifest_path, index=False)
    cfg = _base_cfg(tmp_path, manifest_path)
    cfg["subset"]["n"] = 6
    sub_df = compute_subset(cfg, full_df)
    cfg["subset"]["clip_ids"] = sorted(sub_df["utt_id"].tolist())
    _write_noise_bank(tmp_path / "noise", n=2)
    return tmp_path, cfg, full_df, sub_df


def test_generate_condition_writes_flac_and_manifest(small_job):
    tmp_path, cfg, full_df, sub_df = small_job
    output_dir = tmp_path / "degraded" / "eval_resample_8k"
    job_dir = output_dir / "_job"

    manifest = generate_condition(
        "eval_resample_8k",
        cfg["conditions"]["eval_resample_8k"],
        sub_df,
        output_dir,
        job_dir,
        seed=cfg["seed"],
        chunk_size=2,
    )

    assert len(manifest) == 6
    expected_cols = [
        "utt_id",
        "path",
        "label",
        "speaker_id",
        "attack_id",
        "source_corpus",
        "sample_rate",
        "duration_s",
        "codec",
        "split",
        "language",
        "condition",
    ]
    assert list(manifest.columns) == expected_cols
    assert (manifest["condition"] == "eval_resample_8k").all()
    for _, row in manifest.iterrows():
        p = Path(row["path"])
        assert p.exists()
        assert p.parent == output_dir
        info = sf.info(str(p))
        assert info.samplerate == SR


def test_generate_condition_resume_skips_done_clips(small_job, monkeypatch):
    tmp_path, cfg, full_df, sub_df = small_job
    output_dir = tmp_path / "degraded" / "eval_mp3_32k"
    job_dir = output_dir / "_job"

    calls = []
    real_process = make_degraded_mod.process_one_clip

    def _spy(row, *args, **kwargs):
        calls.append(row["utt_id"])
        return real_process(row, *args, **kwargs)

    monkeypatch.setattr(make_degraded_mod, "process_one_clip", _spy)

    # First run: only process the first 2 clips' chunk (simulate interruption by using
    # chunk_size=2 and stopping after the first pending chunk manually is awkward via the
    # public API, so instead we run the whole job once, then assert a *second* full run
    # under the same job_dir reprocesses nothing).
    generate_condition(
        "eval_mp3_32k",
        cfg["conditions"]["eval_mp3_32k"],
        sub_df,
        output_dir,
        job_dir,
        seed=cfg["seed"],
        chunk_size=2,
    )
    first_run_calls = len(calls)
    assert first_run_calls == 6

    calls.clear()
    generate_condition(
        "eval_mp3_32k",
        cfg["conditions"]["eval_mp3_32k"],
        sub_df,
        output_dir,
        job_dir,
        seed=cfg["seed"],
        chunk_size=2,
    )
    assert calls == []  # every chunk already done -> nothing reprocessed


def test_generate_condition_noise_requires_bank(small_job):
    tmp_path, cfg, full_df, sub_df = small_job
    output_dir = tmp_path / "degraded" / "eval_noise_10db"
    job_dir = output_dir / "_job"

    with pytest.raises(ValueError, match="noise_bank"):
        generate_condition(
            "eval_noise_10db",
            cfg["conditions"]["eval_noise_10db"],
            sub_df,
            output_dir,
            job_dir,
            seed=cfg["seed"],
            chunk_size=2,
            noise_bank=None,
        )


def test_run_generate_writes_per_condition_manifests(small_job):
    tmp_path, cfg, full_df, sub_df = small_job
    resources_cfg = _resources_cfg()
    output_root = Path(cfg["output_dir"])
    manifests_dir = Path(cfg["manifests"]["out_dir"])
    lock_path = tmp_path / "lock"

    results = run_generate(
        cfg,
        resources_cfg,
        ["eval_resample_8k", "eval_noise_10db"],
        output_root,
        manifests_dir,
        resume=False,
        full_df=full_df,
        status_fn=lambda: NORMAL_STATUS,
        lock_path=lock_path,
    )

    assert set(results) == {"eval_resample_8k", "eval_noise_10db"}
    for name in results:
        manifest_path = manifests_dir / f"{name}.csv"
        assert manifest_path.exists()
        written = pd.read_csv(manifest_path)
        assert len(written) == 6
        assert (written["condition"] == name).all()
        for p in written["path"]:
            assert Path(p).exists()


def test_run_generate_noise_missing_dir_raises_actionable(small_job):
    tmp_path, cfg, full_df, sub_df = small_job
    cfg["conditions"]["eval_noise_10db"]["noise_dir"] = str(tmp_path / "no_such_noise_dir")
    resources_cfg = _resources_cfg()

    with pytest.raises(FileNotFoundError, match="get_musan"):
        run_generate(
            cfg,
            resources_cfg,
            ["eval_noise_10db"],
            Path(cfg["output_dir"]),
            Path(cfg["manifests"]["out_dir"]),
            resume=False,
            full_df=full_df,
            status_fn=lambda: NORMAL_STATUS,
            lock_path=tmp_path / "lock",
        )


# ---------------------------------------------------------------------------
# verify_outputs
# ---------------------------------------------------------------------------


def test_verify_outputs_reports_missing_file(small_job):
    tmp_path, cfg, full_df, sub_df = small_job
    output_root = Path(cfg["output_dir"])
    manifests_dir = Path(cfg["manifests"]["out_dir"])
    resources_cfg = _resources_cfg()

    run_generate(
        cfg,
        resources_cfg,
        ["eval_resample_8k"],
        output_root,
        manifests_dir,
        resume=False,
        full_df=full_df,
        status_fn=lambda: NORMAL_STATUS,
        lock_path=tmp_path / "lock",
    )
    manifest_path = manifests_dir / "eval_resample_8k.csv"
    problems = verify_outputs([manifest_path], sample=6)
    assert problems == []

    # Now delete one output file and confirm verify catches it.
    written = pd.read_csv(manifest_path)
    Path(written.iloc[0]["path"]).unlink()
    problems = verify_outputs([manifest_path], sample=6)
    assert any("missing" in p for p in problems)
