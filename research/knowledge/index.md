# Knowledge index

Status: `seed` = URL known, note not yet written. `drafted` = written from source.
`verified` = written, sources re-read, numbers spot-checked.

| id | type | status | confidence | note |
|---|---|---|---|---|
| asvspoof-2021 | dataset | seed | — | LA/PA/DF databases, official protocols and baselines (LFCC-GMM, CQCC-GMM, LFCC-LCNN, RawNet2) — https://www.asvspoof.org/index2021.html |
| asvspoof-2019-la | dataset | seed | — | Entry-point protocol for this project; official train/dev/eval split |
| asvspoof-5 | dataset | seed | — | Evaluation plan phase 2; names AASIST + RawNet2 as CM baselines — https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf |
| audio-df-survey-2023 | paper | seed | — | Yi et al., open problems: unseen-attack generalization, in-the-wild scarcity, interpretability — https://arxiv.org/abs/2308.14970 |
| rawnet2-antispoofing | paper | seed | — | Tak et al., raw-waveform CM baseline — https://arxiv.org/abs/2011.01108 |
| wav2vec2-antispoofing | paper | seed | — | Tak et al., SSL front-end + augmentation, generalization gains — https://arxiv.org/abs/2202.12233 |
| wavlm-ensemble-asvspoof5 | paper | seed | — | Ten pretrained representations benchmarked; wav2vec2/WavLM strongest — https://arxiv.org/abs/2408.07414 |
| aasist | paper | seed | — | Spectro-temporal graph attention CM; official ASVspoof-family baseline |
| asvspoof-workshop-2024 | benchmark | seed | — | Map of contemporary systems — https://www.asvspoof.org/workshop2024_program |
| lfcc-mfcc-dct | method | seed | — | STFT → power → filterbank → log → DCT; what each stage discards |
| eer-and-metrics | method | seed | — | EER as primary anti-spoofing metric; ROC/PR-AUC, calibration, per-attack breakdown |
| ssl-layer-probing | method | seed | — | Layer-wise linear probes; learned softmax layer weighting |
| pooling-methods | method | seed | — | mean / statistics / attentive pooling over time |
| multitask-hard-sharing | method | seed | — | Shared-trunk gradients, negative transfer, gradient cosine similarity diagnostic |
| leakage-audit | method | seed | — | Metadata-only classifiers as a shortcut detector |
