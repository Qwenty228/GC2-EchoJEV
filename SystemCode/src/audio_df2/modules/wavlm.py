"""Frozen WavLM representations. No dataset or pipeline dependencies."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from audio_df2.helpers.config import load_config
from audio_df2.helpers.device import get_device

@dataclass(frozen=True)
class WavLMConfig:
    model_id: str
    revision: str | None
    sample_rate: int
    crop_seconds: float
    n_layers: int
    hidden_dim: int
    batch_size: int
    num_workers: int
    chunk_size: int

class WavLM:
    def __init__(self, config: WavLMConfig | None = None, device: str | torch.device = 'auto'):
        self.config = config if config is not None else load_config(WavLMConfig)
        self.device = get_device(str(device))
        self.model = None
        self.processor = None

    def _load(self):
        if self.model is None or self.processor is None:
            from transformers import AutoModel, AutoFeatureExtractor
            kwargs = {'revision': self.config.revision} if self.config.revision else {}
            model = AutoModel.from_pretrained(self.config.model_id, **kwargs)
            processor = AutoFeatureExtractor.from_pretrained(self.config.model_id, **kwargs)
            shape = model.config
            if (shape.num_hidden_layers + 1, shape.hidden_size) != (self.config.n_layers, self.config.hidden_dim):
                raise ValueError('Backbone dimensions differ from configs/wavlm.yaml')
            model.requires_grad_(False).eval().to(self.device)
            self.model, self.processor = model, processor
        return self.model, self.processor

    def __call__(self, waveforms: np.ndarray) -> np.ndarray:
        """Batch of fixed-length waveforms -> fp16 [batch, layers, hidden]."""
        model, processor = self._load()
        inputs = processor(list(waveforms), sampling_rate=self.config.sample_rate,
                                padding=False, return_tensors='pt')
        kwargs = {'output_hidden_states': True}
        if 'attention_mask' in inputs:
            kwargs['attention_mask'] = inputs['attention_mask'].to(self.device)
        with torch.inference_mode():
            out = model(inputs['input_values'].to(self.device), **kwargs)
            pooled = torch.stack([h.mean(dim=1) for h in out.hidden_states], dim=1)
        return pooled.to(torch.float16).cpu().numpy()

    @property
    def settings(self):
        return asdict(self.config)
