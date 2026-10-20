"""A3 samples evaluation audio before extraction; dev calibration remains complete."""
import json

import pandas as pd

from audio_df.runtime import Run


def test_seeded_policy_sampling_and_persistence(tmp_path, monkeypatch):
    root = tmp_path
    artifact = root / 'artifacts' / 'a1' / 'complete'
    artifact.mkdir(parents=True)
    (artifact / 'COMPLETE').write_text('ok')
    manifests = root / 'data' / 'manifests'
    manifests.mkdir(parents=True)
    for split, size in [('dev', 20), ('eval', 50)]:
        pd.DataFrame({
            'utt_id': [f'{split}_{i}' for i in range(size)],
            'path': [f'unused_{i}.wav' for i in range(size)],
            'label': ['bonafide' if i % 2 else 'spoof' for i in range(size)],
        }).to_csv(manifests / f'{split}.csv', index=False)

    observed = {}
    class FakeDetector:
        cache_namespace = ('a1',)
        def score_manifest(self, df, destination, *, split, cache_dir):
            observed[split] = df.utt_id.tolist()
            return pd.DataFrame({
                'clip_id': df.utt_id.tolist(),
                'label': df.label.tolist(),
                'score': [1.0 if x == 'bonafide' else -1.0 for x in df.label],
            })

    monkeypatch.setattr('audio_df.runtime.trimmed_manifest', lambda df, cache, split: df)
    calls = []
    def fake_policy(state, *, model, **kwargs):
        calls.append(state)
        return {'model': 'jev-test', 'usage': {'input_tokens': 3},
                'answers': {'decision': {'choice': state.context['base_verdict']}}}

    def run_once():
        run = Run(name='a3', root=str(root), command='evaluate', seed=123,
                  testsets='dev,eval')
        result = run.evaluate_policy(FakeDetector(), fake_policy, base='a1', limit=3)
        return run, result

    first, report = run_once()
    assert len(observed['dev']) == 20
    assert len(observed['eval']) == 3
    assert len(calls) == 3
    selected = observed['eval'][:]
    assert report['eval']['n'] == 3
    request_records = [json.loads(x) for x in (first.output / 'jev' / 'requests.jsonl').read_text().splitlines()]
    response_records = [json.loads(x) for x in (first.output / 'jev' / 'responses.jsonl').read_text().splitlines()]
    assert len(request_records) == len(response_records) == 3
    assert request_records[0]['original_detector_response']['label'] in {'spoof', 'bonafide'}
    assert response_records[0]['raw_jev_response']['usage']['input_tokens'] == 3
    second, _ = run_once()
    assert observed['eval'] == selected
    assert len(calls) == 3  # Persistent JEV cache hit; no extra paid calls.
    assert all(json.loads(line)['cached'] for line in (second.output / 'jev' / 'responses.jsonl').read_text().splitlines())
