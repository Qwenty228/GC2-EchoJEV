from dataclasses import replace
import numpy as np
import pandas as pd
import soundfile as sf
from audio_df2.modules.wavlm import WavLMConfig
from audio_df2.modules.neural import NeuralConfig, NeuralHead
from audio_df2.modules.ssl_detector import SSLDetector

class FakeEncoder:
    def __init__(self):
        import torch
        self.device = torch.device('cpu')
        self.config = WavLMConfig('mock', None, 16000, 0.1, 3, 4, 2, 0, 2)
    @property
    def settings(self):
        from dataclasses import asdict
        return asdict(self.config)
    def __call__(self, waveforms):
        x = np.asarray(waveforms)
        m = x.mean(axis=1)
        s = x.std(axis=1)
        return np.stack([np.stack([m,s,m+s,m-s],axis=-1)+i for i in range(3)],axis=1).astype('float16')

def test_end_to_end(tmp_path):
    rows=[]
    for i in range(8):
        path=tmp_path/f'{i}.wav'
        sf.write(path,np.full(1600, (i+1)*0.02,dtype='float32'),16000)
        rows.append({'utt_id':f'clip{i}', 'path':str(path), 'label':'bonafide' if i%2 else 'spoof', 'attack_id':'A01'})
    train=pd.DataFrame(rows[:6]); dev=pd.DataFrame(rows[6:])
    head=NeuralHead(3,4,NeuralConfig(4,4,0.0,0.001,2,2,2))
    detector=SSLDetector(FakeEncoder(),head)
    detector.fit(train,tmp_path/'cache'/'a2',seed=1,dev=dev)
    out=tmp_path/'artifact'; detector.save(out)
    assert (out/'head.pt').exists()
    scored=detector.score_manifest(dev,tmp_path/'scores.csv',split='dev',cache_dir=tmp_path/'cache'/'a2')
    assert len(scored)==2 and np.isfinite(scored.score).all()
    assert len(__import__('json').loads((out/'layer_weights.json').read_text()))==3

def test_checkpoint_load_without_download(tmp_path):
    from audio_df2.modules.wavlm import WavLM
    from audio_df2.modules.ssl_detector import SSLDetector
    from audio_df2.modules.neural import NeuralConfig, NeuralHead
    encoder = WavLM(WavLMConfig('mock', None, 16000, 0.1, 3, 4, 2, 0, 2), device='cpu')
    model = SSLDetector(encoder, NeuralHead(3, 4, NeuralConfig(4, 4, 0.0, 0.001, 2, 2, 2)))
    model.save(tmp_path)
    loaded = SSLDetector.load(tmp_path, device='cpu')
    import torch
    sample = torch.randn(2, 3, 4)
    with torch.no_grad():
        assert torch.allclose(model.head(sample), loaded.head(sample))
    assert loaded.encoder.model is None
