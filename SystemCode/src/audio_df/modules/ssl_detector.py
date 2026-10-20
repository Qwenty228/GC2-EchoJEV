"""A2: frozen WavLM features and one trainable layer-weighted head."""
import copy
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from audio_df.helpers.config import load_config
from audio_df.helpers.ssl_cache import cached_pooled, iter_pooled_chunks, center_crop_repeat
from audio_df.helpers.audio import load_audio
from audio_df.modules.wavlm import WavLM, WavLMConfig
from audio_df.modules.neural import NeuralHead, NeuralConfig
from audio_df.helpers.training import TrainingConfig
from numpy.typing import NDArray

class SSLDetector:
    def __init__(
        self,
        encoder: WavLM | None = None,
        head: NeuralHead | None = None,
        head_config: NeuralConfig | None = None,
        train_config: TrainingConfig | None = None,
    ):
        if head is not None and head_config is not None:
            raise ValueError("Specify either head or head_config, not both")
        self.encoder = encoder if encoder is not None else WavLM()

        self.head = head if head is not None else NeuralHead(
            n_layers=self.encoder.config.n_layers,
            hidden_dim=self.encoder.config.hidden_dim,
            config=head_config,
        )

        self.train_config = train_config if train_config is not None else load_config(TrainingConfig)
        self.history = {}

    @property
    def cache_namespace(self) -> tuple[str, ...]:
        # Name by encoder and representation, never by A2/D1/D2/D3.
        return (self.encoder.config.model_id.rsplit("/", 1)[-1], "pooled")

    def _features(self, df, split, cache_dir):
        return cached_pooled(df, self.encoder, cache_dir, split)

    def fit(self, train, cache_dir, seed=1337, dev=None):
        if dev is None:
            raise ValueError('A2 needs a development manifest for early stopping')
        torch.manual_seed(seed)
        np.random.seed(seed)
        x_train = torch.from_numpy(self._features(train, 'train', cache_dir).astype('float32'))
        x_dev = torch.from_numpy(self._features(dev, 'dev', cache_dir).astype('float32'))
        y_train = torch.tensor((train.label == 'bonafide').to_numpy(), dtype=torch.float32)
        y_dev = torch.tensor((dev.label == 'bonafide').to_numpy(), dtype=torch.float32)
        cfg = self.train_config
        device = self.encoder.device
        head = self.head.to(device)
        weight = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
        optimizer = torch.optim.Adam(head.parameters(), lr=cfg.lr)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size,
                            shuffle=True, generator=torch.Generator().manual_seed(seed))
        best_loss, best_epoch, stale = float('inf'), 0, 0
        best_state = copy.deepcopy(head.state_dict())
        losses, dev_losses = [], []
        for epoch in tqdm(range(cfg.max_epochs), desc='Training A2 head', unit='epoch'):
            head.train()
            running, steps = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = loss_fn(head(xb), yb)
                loss.backward()
                optimizer.step()
                running += float(loss.detach())
                steps += 1
            losses.append(running / max(steps, 1))
            head.eval()
            with torch.no_grad():
                # Dev loss in batches avoids OOM for larger heads.
                total = 0.0
                for start in range(0, len(x_dev), cfg.batch_size):
                    xb = x_dev[start:start+cfg.batch_size].to(device)
                    yb = y_dev[start:start+cfg.batch_size].to(device)
                    total += float(loss_fn(head(xb), yb)) * len(xb)
                dev_loss = total / len(x_dev)
            dev_losses.append(dev_loss)
            if dev_loss < best_loss:
                best_loss, best_epoch, stale = dev_loss, epoch + 1, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                stale += 1
            tqdm.write(f'A2 epoch {epoch+1}: train={losses[-1]:.4f} dev={dev_loss:.4f}')
            if stale >= cfg.patience:
                break
        head.load_state_dict(best_state)
        head.eval()
        self.history = {'train_loss': losses, 'dev_loss': dev_losses,
                        'best_epoch': best_epoch, 'pos_weight': weight}
        return self

    def score_audio(self, waveform: NDArray[np.float32]) -> float:
        """Score one 16 kHz waveform using the frozen encoder and trained head."""
        cfg = self.encoder.config
        x = center_crop_repeat(np.asarray(waveform, dtype=np.float32), round(cfg.sample_rate * cfg.crop_seconds))
        pooled = self.encoder(x[np.newaxis, :]).astype(np.float32)
        self.head.to(self.encoder.device).eval()
        with torch.inference_mode():
            return float(self.head(torch.from_numpy(pooled).to(self.encoder.device))[0])

    def score_file(self, path) -> float:
        return self.score_audio(load_audio(path, self.encoder.config.sample_rate))

    def score_manifest(self, df, output, *, split: str, cache_dir: Path):
        import pandas as pd
        # Split-specific cache keys must match the requested manifest.
        # Cache identity includes split and manifest order.
        chunks = iter_pooled_chunks(df, self.encoder, cache_dir, split)
        scores = []
        self.head.to(self.encoder.device).eval()
        with torch.inference_mode():
            for pooled in chunks:
                for start in range(0, len(pooled), 512):
                    batch = torch.from_numpy(pooled[start:start+512].astype('float32')).to(self.encoder.device)
                    scores.extend(self.head(batch).cpu().numpy().tolist())
        result = pd.DataFrame({'clip_id': df.utt_id.astype(str), 'label': df.label,
                               'score': scores})
        if 'attack_id' in df:
            result['attack_id'] = df.attack_id.to_numpy()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.cpu().state_dict(), directory / 'head.pt')
        (directory / 'model.json').write_text(json.dumps({
            'encoder': asdict(self.encoder.config), 'head': asdict(self.head.config),
            'training': asdict(self.train_config)}, indent=2))
        (directory / 'training_curve.json').write_text(json.dumps(self.history, indent=2))
        (directory / 'layer_weights.json').write_text(json.dumps(
            self.head.layer_weights().detach().cpu().tolist(), indent=2))
        self.head.to(self.encoder.device)

    @classmethod
    def load(cls, directory, device='auto'):
        directory = Path(directory)
        meta = json.loads((directory / 'model.json').read_text())
        encoder = WavLM(WavLMConfig(**meta['encoder']), device=device)
        head = NeuralHead(encoder.config.n_layers, encoder.config.hidden_dim,
                          NeuralConfig(**meta['head']))
        head.load_state_dict(torch.load(directory / 'head.pt', map_location='cpu', weights_only=True))
        head.eval()
        training_cfg = (TrainingConfig(**meta['training'])
                        if 'training' in meta else load_config(TrainingConfig))
        return cls(encoder=encoder, head=head, train_config=training_cfg)
