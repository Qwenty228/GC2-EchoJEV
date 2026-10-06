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
| stourbe-2024-wavlm-backends | paper | drafted | medium | Handoff [3]. Fine-tuned WavLM **Base** on ASVspoof 5; WA back end = layer-weighted avg → time mean → linear (≈1.5k params), same structure as A2's head; frozen MHFA 6.78 % vs fine-tuned + noise/RIR + codec 2.18 %; frozen + aug worse (8.78 %) |
| tak-2022-wav2vec2-data-augmentation | paper | drafted | high | Handoff [7]. Fine-tuned XLS-R 0.3B + AASIST, last layer, 2021 LA 11.47→0.82 % with RawBoost; no ITW; seed gaps up to 1.7 pt |
| wang-2024-moe-fusion-frozen-wav2vec2 | paper | drafted | high | Handoff [22]. Frozen wav2vec 2.0, train LA19: 0.74 % LA19 eval / 12.48 % ITW (MoE), 1.93 / 23.22 % (simple head); fine-tuned ITW 9.17 %; hidden-track degradation 2–4× |
| muller-2022-does-audio-deepfake-detection-generalize | paper | drafted | high | Handoff [13]. 56 configs; means 60.10 % (full) / 67.25 % (4 s) ITW; lowest ITW RawNet2 4 s 33.94 %; RawGAT-ST best in-domain 1.23 % / 37.15 % ITW; no LFCC/GMM, no silence analysis |
| pascu-2024-generalisable-calibrated-ssl | paper | drafted | high | Handoff [8]. Frozen last layer + LR: WavLM Base+ 38.6 % ITW EER, XLS-R 2B 7.2 %; ECE; prints 84k h for Base+ (WavLM paper says 94k) |
| rimon-2025-unmasking-deepfakes | paper | drafted | medium | Handoff [4]. Fine-tuned SSL + ResNet34; augmentation gains on ASVspoof 5 only (8.34→4.53 %); no frozen/ITW; internal number inconsistencies |
| el-kheir-2025-two-views-spectral-ssl-fusion | paper | drafted | medium | Handoff [6]. Spectral + XLS-R fusion helps DF21/ITW/ASV5 (ITW 14.03→6.87 %), not LA19; no overlap or silence analysis; baseline copied from Tak table |
| mo-2022-multi-task-learning-synthetic-speech-detection | paper | drafted | medium | Handoff [17]. LFCC ResNet34 + OC-softmax 2.19→1.47 % LA19 eval with auxiliary tasks (no attack IDs); shows A1's gap is the GMM back end |
| sutharya-2026-half-truth-cafnet-cross-corpus | paper | drafted | low | Handoff [18]. Preprint; handcrafted model near-chance zero-shot on PartialSpoof; VAD-trim AUC 0.55→0.29, cause unexplained |
| delgado-2025-deepfake-voice-presentation | paper | drafted | medium | Handoff [1]. Playback (not phone injection) is the harsh condition; 8 kHz + VAD; E4 = mild case |
| stan-2025-tada-attribution-ood | paper | drafted | medium | Handoff [19]. kNN on time-averaged frozen SSL states attributes generator (F1 0.93), fake-only; does not transfer cross-corpus; kNN OOD cue = A3 design option |
| zhu-2022-source-tracing-voice-spoofing | paper | drafted | medium | Handoff [20]. Attribute heads need labels for every attack; reprints LA attack table (A16 = A04, A19 = A06) |
| cheng-2025-xai-forensic-speech-authentication | paper | drafted | low | Handoff [21]. XAI technique pointer only; single-example saliency, internal inconsistencies; ranked XAI analysis list for A1/A2 |
| linear-vs-mel-cepstral-speaker-recognition | paper | drafted | medium | Handoff [23]. Zhou et al. 2011, speaker recognition at 300–3400 Hz; filter-count geometry only, not spoofing evidence |
| lfcc-vs-mfcc-explained | method | drafted | medium | Maths of LFCC vs MFCC from our code (filter table: 10 vs 5 filters above 4 kHz), coefficient meanings, literature (Sahidullah, WaveFake, SAS counter), our EDA (weak, silence-confounded tilt), A1 per-attack trimming mechanisms M1–M3, XAI specs X1–X6 |
