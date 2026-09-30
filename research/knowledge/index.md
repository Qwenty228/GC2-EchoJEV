# Knowledge index

Status: `seed` = URL known, note not yet written. `drafted` = written from source.
`verified` = written, sources re-read, numbers spot-checked.

| id | type | status | confidence | note |
|---|---|---|---|---|
| asvspoof-2021 | dataset | verified | high | LA/PA/DF databases, official protocols and baselines (LFCC-GMM, CQCC-GMM, LFCC-LCNN, RawNet2) — https://www.asvspoof.org/index2021.html; per-track train/dev/eval utterance counts not found, open question in note |
| asvspoof-2019-la | dataset | verified | high | Entry-point protocol for this project; official train/dev/eval split; exact eval split-size unconfirmed (medium) |
| asvspoof-5 | dataset | seed | — | Evaluation plan phase 2; names AASIST + RawNet2 as CM baselines — https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf |
| audio-df-survey-2023 | paper | seed | — | Yi et al., open problems: unseen-attack generalization, in-the-wild scarcity, interpretability — https://arxiv.org/abs/2308.14970 |
| rawnet2-antispoofing | paper | seed | — | Tak et al., raw-waveform CM baseline — https://arxiv.org/abs/2011.01108 |
| wav2vec2-antispoofing | paper | seed | — | Tak et al., SSL front-end + augmentation, generalization gains — https://arxiv.org/abs/2202.12233 |
| wavlm-ensemble-asvspoof5 | paper | seed | — | Ten pretrained representations benchmarked; wav2vec2/WavLM strongest — https://arxiv.org/abs/2408.07414 |
| aasist | paper | seed | — | Spectro-temporal graph attention CM; official ASVspoof-family baseline |
| asvspoof-workshop-2024 | benchmark | seed | — | Map of contemporary systems — https://www.asvspoof.org/workshop2024_program |
| lfcc-mfcc-dct | method | verified | high | STFT → power → filterbank → log → DCT; what each stage discards; orthonormal DCT-II scaling derived |
| eer-and-metrics | method | verified | high | Official EER = non-interpolated nearest-neighbour crossing; higher score = bonafide; per-attack pools shared bonafide set; t-DCF vs EER |
| ssl-layer-probing | method | seed | — | Layer-wise linear probes; learned softmax layer weighting |
| pooling-methods | method | seed | — | mean / statistics / attentive pooling over time |
| multitask-hard-sharing | method | seed | — | Shared-trunk gradients, negative transfer, gradient cosine similarity diagnostic |
| leakage-audit | method | verified | high | Metadata-only classifiers as a shortcut detector; ASVspoof LA silence-duration shortcut quantified (Müller et al. 2106.12914); repo's 0.65 AUC red-flag default is a project-local convention, not sourced from literature; survey 2308.14970 PDF unreachable as text this pass |
| in-the-wild | dataset | verified | high | 37.9h (17.2 fake/20.7 bona), 58 speakers, no attack labels, no official split, manually curated pairing — https://arxiv.org/pdf/2203.16263; EER inflates ~30x zero-shot from ASVspoof19; single HF download (8.16GB zip, no visible meta.csv, clip count unverified); **license conflict**: authors' own page says Apache-2.0, HF card says CC-BY-SA-4.0 |
| b02-lfcc-gmm | method | verified | high | Official ASVspoof2019 baseline code (asvspoof.org zip) read directly: 20ms Hamming/10ms hop/512pt FFT/20 linear filters/**0-8000Hz** (not 30Hz-8kHz)/pre-emph 0.97/no CMVN/19+0th+Δ+ΔΔ=60dim/512-comp diag-cov GMM×2 (train-only, random-init EM, 10 iter)/score=mean-frame LLR whole-utterance; LA eval pooled B02: t-DCF=0.2116, EER=8.09% (arXiv:1904.05441 Table 1, eval-only, no dev column) |
| min-dcf | method | verified | high | ASVspoof5 Track1: DCF(τ)=β·Pmiss+Pfa, β=Cmiss(1-π)/(Cfa·π), Cmiss=1/Cfa=10/π_spf=0.05→β≈1.90 (arXiv:2601.03944); paper's normalizer is fixed Cfa·π_spf, not general min(...) — coincide only because β≥1 for these params |
| ssl-hidden-states | method | verified | high | WavLM Base+ (12 layers/768dim/do_normalize=false) vs XLS-R 300M (24 layers/1024dim/do_normalize=true); output_hidden_states → 13 vs 25 tensors; identical conv stride-320 encoder → 199 frames for 64000-sample input both; WavLM license unclear (CC-BY-SA-3.0 via repo LICENSE, no HF tag) vs XLS-R apache-2.0 |
| musan | dataset | verified | high | OpenSLR SLR17, 11GB tar.gz, page-level CC BY 4.0; noise/ subset = 929 files/~6h (arXiv:1510.08484); underlying per-file licensing is mixed (public domain + various CC), not uniformly CC BY 4.0 |
| mlaad | dataset | verified | high | Versioned, continuously expanded (v1: 63h/21 models/22 langs -> v11: 1153.6h/205 models/54 langs); bonafide pairing is M-AILABS, distributed separately; CC-BY-NC-4.0 — https://arxiv.org/pdf/2401.09512 |
| multi-layer-ssl-aggregation | method | drafted | low | SLS (Zhang et al. ACM MM 2024) and MSLA-XLS-R (Meng et al. 2026) exist and are citable, but primary texts paywalled/403-blocked; mechanism not independently confirmed — do not schedule implementation on this note alone |
