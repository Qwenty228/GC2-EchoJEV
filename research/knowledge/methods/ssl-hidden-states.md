---
id: ssl-hidden-states
type: method
sources:
  - https://huggingface.co/microsoft/wavlm-base-plus (`config.json`, `preprocessor_config.json` fetched directly via `huggingface.co/.../raw/main/...`)
  - https://huggingface.co/facebook/wav2vec2-xls-r-300m (`config.json`, `preprocessor_config.json` fetched directly)
  - https://huggingface.co/api/models/microsoft/wavlm-base-plus and .../facebook/wav2vec2-xls-r-300m (model API metadata — used for license tag check)
  - https://github.com/microsoft/UniSpeech/blob/main/LICENSE (root LICENSE of the repo that hosts WavLM; fetched directly, confirmed CC-BY-SA-3.0 text)
confidence: high
last_verified: 2026-09-30
related: [multi-layer-ssl-aggregation, ssl-layer-probing]
---

# WavLM Base+ and XLS-R 300M — hidden-state counts, frame rate, and input normalization

## Claim in one sentence
`microsoft/wavlm-base-plus` (12 transformer layers, hidden=768) and `facebook/wav2vec2-xls-r-300m` (24 transformer layers, hidden=1024) both use the identical 7-layer strided-conv feature encoder (total stride 320, ≈50 Hz frame rate), so a 64,000-sample (4 s @ 16 kHz) input produces **199 frames** from either model; `output_hidden_states=True` returns **13** tensors for WavLM Base+ (embedding output + 12 layers) and **25** for XLS-R 300M (embedding output + 24 layers); the two models' input-normalization convention **differs** — WavLM Base+'s feature extractor has `do_normalize=false`, XLS-R 300M's has `do_normalize=true` — and their licenses differ (WavLM: CC-BY-SA-3.0 via the UniSpeech repo LICENSE, no explicit HF `license` tag; XLS-R 300M: `apache-2.0`, explicit HF tag).

## Findings

### Model identifiers and architecture (from each model's `config.json`, fetched directly)
| field | `microsoft/wavlm-base-plus` | `facebook/wav2vec2-xls-r-300m` |
|---|---|---|
| `model_type` | `wavlm` | `wav2vec2` |
| `num_hidden_layers` | 12 | 24 |
| `hidden_size` | 768 | 1024 |
| `num_attention_heads` | 12 | 16 |
| `conv_dim` | [512]×7 | [512]×7 |
| `conv_kernel` | [10,3,3,3,3,2,2] | [10,3,3,3,3,2,2] |
| `conv_stride` | [5,2,2,2,2,2,2] | [5,2,2,2,2,2,2] |
| `feat_extract_norm` | `group` | `layer` |
| `do_stable_layer_norm` | `false` | `true` |

Both models' feature encoders use the identical kernel/stride schedule (this is the standard wav2vec2-family CNN encoder that WavLM also adopted), so the frame-rate/frame-count arithmetic below is the same for both.

### `output_hidden_states=True` tensor count
`output_hidden_states=True` in a HF transformer returns one tensor for the embedding/feature-encoder output plus one tensor per transformer layer, i.e. `num_hidden_layers + 1` tensors total:
- **WavLM Base+: 12 + 1 = 13 hidden-state tensors**, each `[batch, T, 768]`.
- **XLS-R 300M: 24 + 1 = 25 hidden-state tensors**, each `[batch, T, 1024]`.
(This is a direct arithmetic reading of the standard HF convention applied to each model's own `num_hidden_layers`, not a separate fetched claim — flagging as derived, not quoted, per the note's own discipline.)

### Frame rate and frame count for a 64,000-sample (4 s @ 16 kHz) input
Total conv stride = 5×2×2×2×2×2×2 = 320 → **frame rate ≈ 16000/320 = 50 Hz (20 ms/frame)**, same for both models (identical `conv_stride`).
Exact frame count via the standard no-padding Conv1d length rule `L_{i+1} = floor((L_i − kernel_i)/stride_i) + 1`, chained through all 7 layers starting from `L_0 = 64000`:

| stage | kernel | stride | L_out |
|---|---|---|---|
| 0 | — | — | 64000 |
| 1 | 10 | 5 | 12799 |
| 2 | 3 | 2 | 6399 |
| 3 | 3 | 2 | 3199 |
| 4 | 3 | 2 | 1599 |
| 5 | 3 | 2 | 799 |
| 6 | 2 | 2 | 399 |
| 7 | 2 | 2 | **199** |

**Both models produce 199 frames for a 64,000-sample input.** (Derived directly from each model's own published `conv_kernel`/`conv_stride`, not looked up from a separate "expected frame count" source — this is arithmetic on primary-source config values, flagged as `mechanism/derivation` rather than `findings`.)

### Input normalization — `preprocessor_config.json`, fetched directly
- **WavLM Base+:** `"do_normalize": false`. Raw waveform amplitude is fed to the model as-is (aside from the feature extractor's padding/attention-mask handling); no zero-mean/unit-variance rescaling.
- **XLS-R 300M:** `"do_normalize": true`. Each input waveform is zero-mean/unit-variance normalized before being passed to the model.
- **This is a real, easy-to-miss difference between the two SSL front-ends this project benchmarks** — using the wrong `do_normalize` setting (e.g. copy-pasting the WavLM preprocessing config for XLS-R, or vice-versa) silently changes the input distribution each model sees and was pretrained on.

### Licenses
- **XLS-R 300M:** HF model API reports an explicit `license: apache-2.0` tag.
- **WavLM Base+:** the HF model API returns **no `license` field in `cardData`** for `microsoft/wavlm-base-plus`. The model card links to `https://github.com/microsoft/UniSpeech/blob/main/LICENSE`, which (fetched directly) is the **Creative Commons Attribution-ShareAlike 3.0 Unported** license text. This is the only license artifact found for WavLM; it is a repo-level LICENSE file covering the UniSpeech project as a whole (code + released checkpoints), not a HF-native `license:` tag, and is share-alike (copyleft-like) rather than permissive — a materially different obligation than XLS-R 300M's Apache-2.0.

## Mechanism / derivation
See the frame-count table above; the underlying rule is the standard PyTorch `Conv1d` output-length formula with no padding, no dilation: `L_out = floor((L_in − kernel)/stride) + 1`, applied once per of the 7 feature-encoder conv layers, using each model's own published `conv_kernel`/`conv_stride` arrays (identical for both models here).

## Implications for us
- **A2's learnable layer-weighted head must be sized for 13 layers (WavLM) or 25 layers (XLS-R)** respectively — these counts differ, so the layer-weighting module cannot share a fixed-size weight vector across both backbones without either padding or a backbone-specific weight count.
- **`do_normalize` must be set per-model, not globally.** Any shared preprocessing utility in `src/` that hardcodes one `do_normalize` value for both SSL backbones will silently mismatch XLS-R 300M's pretraining input distribution (if left `false`) or add an unwanted normalization step to WavLM Base+ (if left `true`).
- **199 frames for a 4 s crop is a fixed, derivable constant** — useful for sizing any fixed-length pooling/attention module without needing to run a forward pass just to discover `T`.
- **WavLM's CC-BY-SA-3.0 obligation is share-alike** — if this project's own report/model artifacts incorporate or are considered derivative of the WavLm checkpoint under a strict reading, that license's share-alike clause is a materially different constraint from XLS-R's Apache-2.0, and should be flagged to whoever writes the final report's licensing/compliance section (out of this note's scope to resolve, but the asymmetry itself is a fact worth carrying forward).

## Open questions / what would change this
- Whether `microsoft/wavlm-base-plus`'s omission of an HF `license:` tag is an oversight (and the real intended license is something more permissive, e.g. MIT for the model weights specifically) versus a deliberate signal that the repo-level CC-BY-SA-3.0 applies to the checkpoint too, was not resolved — this project should not assume permissive use of WavLM weights beyond what CC-BY-SA-3.0 allows without checking further (e.g. contacting Microsoft or finding a more specific statement).
- The 199-frame count assumes no padding/dilation in the conv feature encoder, which matches the standard HF `Wav2Vec2FeatureEncoder`/`WavLMFeatureEncoder` implementation as documented in each model's own `config.json`; this was not independently confirmed by tracing HF's actual `modeling_wavlm.py`/`modeling_wav2vec2.py` source in this pass (medium confidence on the *implementation* matching the config-only derivation, though this is the standard, well-known behaviour of this model family).
