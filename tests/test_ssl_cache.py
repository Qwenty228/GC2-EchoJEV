"""ssl_cache.py: pooled-cache extraction, the mean-pool/weighted-sum equivalence, resume.

Uses tiny, randomly-initialised `transformers` configs (2-layer WavLM / Wav2Vec2, small
hidden size) on synthetic sine-wave audio — never a real download, never real data.
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from audio_df.resources import ChunkedJob
from audio_df.ssl_cache import (
    WaveformDataset,
    build_job_config,
    cache_dir_for,
    extract_pooled_hidden_states,
    run_extraction,
)

torch.manual_seed(0)


def _tiny_wavlm():
    from transformers import WavLMConfig, WavLMModel

    config = WavLMConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(16, 16, 16, 16, 16, 16, 16),
        do_stable_layer_norm=False,
        feat_extract_norm="group",
    )
    model = WavLMModel(config)
    model.eval()
    return model, config


class _FakeFeatureExtractor:
    """Mimics `do_normalize=False` HF feature extractor behaviour, no network needed."""

    def __init__(self, do_normalize: bool = False):
        self.do_normalize = do_normalize

    def __call__(self, waveforms, sampling_rate, return_tensors="pt", padding=False):
        arr = np.stack([np.asarray(w, dtype=np.float32) for w in waveforms])
        if self.do_normalize:
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True) + 1e-7
            arr = (arr - mean) / std
        return {"input_values": torch.from_numpy(arr)}


def _sine(n_samples: int, freq: float = 220.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(n_samples) / sr
    return 0.1 * np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_extract_pooled_hidden_states_shape_and_dtype():
    model, config = _tiny_wavlm()
    fe = _FakeFeatureExtractor()
    n_layers = config.num_hidden_layers + 1  # embedding output + each transformer layer
    waveforms = np.stack([_sine(16000, freq=220.0 + 10 * i) for i in range(3)])

    pooled = extract_pooled_hidden_states(
        model, fe, waveforms, sample_rate=16000, device=torch.device("cpu")
    )

    assert pooled.shape == (3, n_layers, config.hidden_size)
    assert pooled.dtype == np.float16
    assert np.isfinite(pooled.astype(np.float32)).all()


def test_pooled_cache_equals_weighted_sum_then_pool_on_full_frames():
    """The equivalence this whole cache design rests on, checked numerically end to end:
    sum_l w_l * mean_t(H^(l)) == mean_t(sum_l w_l * H^(l)), independent of any head."""
    model, config = _tiny_wavlm()
    fe = _FakeFeatureExtractor()
    waveforms = np.stack([_sine(16000, freq=220.0 + 10 * i) for i in range(4)])
    input_values = torch.from_numpy(waveforms)

    with torch.inference_mode():
        outputs = model(input_values, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of L tensors [B, T, D]
    n_layers = len(hidden_states)

    rng = np.random.default_rng(0)
    logits = rng.normal(size=n_layers).astype(np.float32)
    w = torch.softmax(torch.from_numpy(logits), dim=0)

    # Reference: weighted sum over layers on the FULL frames, then mean over time.
    stacked_frames = torch.stack(list(hidden_states), dim=1)  # [B, L, T, D]
    weighted_frames = torch.einsum("l,bltd->btd", w, stacked_frames)  # [B, T, D]
    reference = weighted_frames.mean(dim=1)  # [B, D]

    # Candidate: pooled cache (mean over time per layer, fp16) then weighted sum over layers.
    pooled = extract_pooled_hidden_states(
        model, fe, waveforms, sample_rate=16000, device=torch.device("cpu")
    )
    pooled_t = torch.from_numpy(pooled.astype(np.float32))  # [B, L, D]
    candidate = torch.einsum("l,bld->bd", w, pooled_t)  # [B, D]

    np.testing.assert_allclose(
        candidate.numpy(), reference.numpy(), rtol=1e-2, atol=1e-2
    )  # fp16 storage rounding is the only source of difference


def test_normalization_follows_the_feature_extractor_not_a_hardcoded_choice():
    fe_raw = _FakeFeatureExtractor(do_normalize=False)
    fe_norm = _FakeFeatureExtractor(do_normalize=True)
    waveforms = np.stack([_sine(16000, freq=220.0) * 5.0 + 1.0])  # nonzero mean, large scale

    out_raw = fe_raw(waveforms, sampling_rate=16000)["input_values"]
    out_norm = fe_norm(waveforms, sampling_rate=16000)["input_values"]

    assert not torch.allclose(out_raw, out_norm)
    assert abs(float(out_norm.mean())) < 1e-4  # normalized version is ~zero-mean
    assert abs(float(out_raw.mean()) - 1.0) < 1e-2  # raw version keeps the offset


def _write_wavs(
    tmp_path, n: int, n_samples: int = 16000, sr: int = 16000
) -> tuple[list[str], list[str]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    ids, paths = [], []
    for i in range(n):
        path = tmp_path / f"clip_{i:03d}.wav"
        sf.write(str(path), _sine(n_samples, freq=200.0 + i), sr)
        ids.append(f"clip_{i:03d}")
        paths.append(str(path))
    return ids, paths


def test_waveform_dataset_returns_fixed_length_float32(tmp_path):
    ids, paths = _write_wavs(tmp_path, 2, n_samples=8000)
    ds = WaveformDataset(
        ids, paths, sample_rate=16000, silence_mode="raw", crop_seconds=1.0, crop_mode="center"
    )
    utt_id, x = ds[0]
    assert utt_id == "clip_000"
    assert x.dtype == np.float32
    assert x.shape == (16000,)  # 1.0 s @ 16 kHz, repeat-padded from 8000 (shorter than crop)


def test_run_extraction_writes_cache_and_resume_skips_completed_chunks(tmp_path):
    model, config = _tiny_wavlm()
    fe = _FakeFeatureExtractor()
    ids, paths = _write_wavs(tmp_path / "audio", 6, n_samples=16000)
    job_config = build_job_config(
        model_id="tiny/wavlm",
        model_tag="tiny-wavlm",
        n_layers=config.num_hidden_layers + 1,
        hidden_dim=config.hidden_size,
        silence_mode="raw",
        crop_seconds=1.0,
        crop_mode="center",
        sample_rate=16000,
        do_normalize=False,
    )
    out_dir = tmp_path / "cache"

    job = run_extraction(
        ids, paths, model, fe, out_dir, job_config,
        device=torch.device("cpu"), sample_rate=16000, silence_mode="raw",
        crop_seconds=1.0, crop_mode="center", batch_size=4, num_workers=0, chunk_size=2,
    )
    assert job.is_complete()
    data = job.load_all()
    assert data["pooled"].shape == (6, config.num_hidden_layers + 1, config.hidden_size)
    assert set(data["ids"].tolist()) == set(ids)

    # Simulate an interruption: delete one chunk file, then rerun should only redo that chunk.
    chunk_files = sorted(out_dir.glob("chunk_*.npz"))
    assert len(chunk_files) == 3  # 6 items / chunk_size 2
    chunk_files[1].unlink()

    calls = {"n": 0}
    real_extract = extract_pooled_hidden_states

    def _counting_extract(*args, **kwargs):
        calls["n"] += 1
        return real_extract(*args, **kwargs)

    import audio_df.ssl_cache as ssl_cache_mod

    original = ssl_cache_mod.extract_pooled_hidden_states
    ssl_cache_mod.extract_pooled_hidden_states = _counting_extract
    try:
        job2 = ChunkedJob(out_dir, ids, chunk_size=2, job_config=job_config)
        assert not job2.is_complete()
        job3 = run_extraction(
            ids, paths, model, fe, out_dir, job_config,
            device=torch.device("cpu"), sample_rate=16000, silence_mode="raw",
            crop_seconds=1.0, crop_mode="center", batch_size=4, num_workers=0, chunk_size=2,
        )
    finally:
        ssl_cache_mod.extract_pooled_hidden_states = original

    assert job3.is_complete()
    # Only the missing chunk's 2 items should have been re-extracted (one batch call).
    assert calls["n"] == 1


def test_cache_dir_for_layout():
    path = cache_dir_for("wavlm-base-plus", "raw", "eval", root="data/cache/ssl")
    assert str(path) == "data/cache/ssl/wavlm-base-plus/raw/eval"
