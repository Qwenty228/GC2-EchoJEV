from pathlib import Path
import pandas as pd
from audio_df.runtime import Run
from audio_df.modules.detector import GMMDetector
from audio_df.modules.lfcc import LFCC


def test_a1_has_shared_scoring_method():
    assert callable(GMMDetector.score_manifest)


def test_runtime_evaluate_has_no_architecture_special_cases(tmp_path, monkeypatch):
    root = tmp_path
    (root / 'artifacts' / 'a2' / 'version').mkdir(parents=True)
    (root / 'artifacts' / 'a2' / 'version' / 'COMPLETE').write_text('ok')
    class FakeDetector:
        def score_manifest(self, df, destination, *, split, cache_dir):
            assert cache_dir == root / 'data' / 'cache' / 'a2'
            return pd.DataFrame({'clip_id': ['x','y'], 'label': ['bonafide','spoof'], 'score': [2.0,-2.0]})
    monkeypatch.setattr(Run, 'manifest', lambda self, split: pd.DataFrame())
    monkeypatch.setattr('audio_df.runtime.evaluate_scores', lambda dev, test, path: {'eer':0.0})
    run = Run(name='a2', root=str(root), testsets='dev,eval', command='evaluate')
    result = run.evaluate(FakeDetector())
    assert result == {'eval': {'eer':0.0}}
    assert (run.output / 'metrics.json').exists()
