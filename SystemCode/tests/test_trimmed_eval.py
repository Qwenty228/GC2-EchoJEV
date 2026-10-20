"""Ensure test-only trimming and shared feature-cache identity."""
import numpy as np
import pandas as pd
import soundfile as sf

from audio_df.helpers.eval_audio import trimmed_manifest, trim_silence


def test_trim_preserves_internal_pause():
    x = np.r_[np.zeros(16000), np.ones(16000) * .2,
              np.zeros(16000), np.ones(16000) * .2, np.zeros(16000)].astype('float32')
    y = trim_silence(x)
    assert len(y) < len(x)
    assert np.count_nonzero(y == 0) >= 16000
    assert np.array_equal(trim_silence(np.zeros(800, dtype='float32')),
                          np.zeros(800, dtype='float32'))


def test_shared_trim_cache_and_raw_manifest_unchanged(tmp_path):
    original = tmp_path / 'raw.wav'
    sf.write(original, np.r_[np.zeros(16000), np.ones(16000) * .2,
                                 np.zeros(16000)].astype('float32'), 16000)
    df = pd.DataFrame({'utt_id': ['a'], 'path': [str(original)], 'label': ['spoof']})
    first = trimmed_manifest(df, tmp_path / 'cache', 'eval')
    second = trimmed_manifest(df, tmp_path / 'cache', 'eval')
    assert first.path.iloc[0] == second.path.iloc[0]
    assert first.path.iloc[0] != df.path.iloc[0]
    assert df.path.iloc[0] == str(original)
    assert sf.info(first.path.iloc[0]).frames < sf.info(original).frames
