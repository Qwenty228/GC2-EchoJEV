# Glossary

Abbreviations and names used in the report and the technical reference. Each is also spelled
out where it first appears in a chapter.

| term | meaning |
|---|---|
| A1, A2, A3 | our three systems: LFCC + GMM baseline; frozen WavLM + small head; A2 + JEV |
| E1–E6 | our experiments: benchmark, ablation, generalisation, robustness, JEV, efficiency |
| X1–X8 | explainability analyses; X4 and X6 are done, the rest proposed ([technical reference §13](../technical-reference.md#13-what-is-left)) |
| AASIST | Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention networks: a strong neural spoof detector, an official ASVspoof baseline |
| ASVspoof | Automatic Speaker Verification Spoofing and Countermeasures Challenge; its 2019 logical access (LA) set is our main data. LA19 = 2019 LA, DF21 = 2021 deepfake (DF) track |
| AUC | area under the receiver operating characteristic (ROC) curve; 1.0 is perfect ranking, 0.5 is chance |
| B02 | the official ASVspoof 2019 baseline number 2, LFCC + GMM, which A1 reproduces |
| CI | confidence interval: the range a number would plausibly fall in if the data were resampled; we use 95 % bootstrap intervals |
| CMVN | cepstral mean and variance normalisation: subtracting each recording's average feature, which removes the channel |
| DCT | discrete cosine transform: turns log band energies into cepstral coefficients |
| EER | equal error rate: the error at the threshold where missed bona fide equals accepted spoof; lower is better |
| ECE | expected calibration error: how far predicted probabilities are from observed frequencies |
| EM | expectation-maximisation, the algorithm that fits a GMM |
| GMM | Gaussian mixture model: a weighted sum of Gaussian bell curves used as a probability density |
| In-the-Wild (ITW) | 31,779 real-world clips of celebrities and politicians, real and deepfake; our zero-training test set |
| JEV | a generalised classifier API (typed text or JSON in, typed probabilities out); not a large language model |
| LFCC / MFCC | linear- / mel-frequency cepstral coefficients: hand-made audio features that differ only in where the frequency filters sit ([chapter 5](05_sound_into_numbers.md)) |
| LLR | log-likelihood ratio: log p(x given bona fide) − log p(x given spoof) |
| minDCF | minimum detection cost function: a weighted miss and false-accept cost, ASVspoof 5 definition |
| MPS | Metal Performance Shaders: Apple's GPU backend for PyTorch |
| MUSAN | Music, Speech and Noise corpus; we use its noise clips for E4 |
| RawBoost | a raw-waveform data-augmentation method (adds filtered noise and distortions) |
| RawNet2, RawGAT-ST | neural detectors that read the raw waveform; RawGAT-ST adds graph attention over spectral and temporal features |
| RMS | root-mean-square: the square root of the mean squared sample value, i.e. overall loudness |
| RSS | resident set size: how much memory a process is actually using |
| SSL | self-supervised learning: models pre-trained on unlabelled audio |
| TTS / VC | text-to-speech / voice conversion: the two ways spoofs are made |
| VCTK | Voice Cloning Toolkit corpus: the studio recordings of 107 English speakers that all ASVspoof 2019 bona fide speech comes from |
| wav2vec 2.0, XLS-R, WavLM | self-supervised speech models. XLS-R (cross-lingual speech representations) is a multilingual wav2vec 2.0; WavLM (Microsoft) adds denoising during pre-training |
| WORLD, STRAIGHT, WaveNet, WaveRNN, Griffin-Lim, Vocaine | waveform generators (vocoders) used to build the ASVspoof attacks ([chapter 1](01_what_is_a_voice.md)) |
