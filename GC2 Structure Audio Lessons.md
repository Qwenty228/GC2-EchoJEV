# Handoff: Audio Deepfake Detection Workshop Project

## Purpose
Build a reproducible, implementation-first research workshop in VS Code that turns the theory from audio signal processing through modern deepfake detection into a compact experimental project. Use **uv** for environment/dependency management. Keep notebooks short and exploratory; reusable logic belongs in Python modules under `src/`.

This workshop begins after the learner already understands waveform sampling, STFT, LFCC mechanics, and basic neural-network training. It should therefore start from the question: **what statistical evidence actually separates bona fide and synthetic speech, and how do increasingly learned representations expose it?**

The project should progress from pure EDA -> hand-engineered features -> simple classifiers -> CNN/ResNet backends -> self-supervised speech representations -> wav2vec 2.0/XLS-R -> fusion/multi-task architecture -> rigorous benchmarking/generalization.

---

## Core research question
Given an utterance x and label y in {bonafide, spoof}, can we learn a detector that captures synthesis-related evidence rather than dataset/generator shortcuts?

The conceptual assumption is that for at least some observable representation Z,

P(Z | real) != P(Z | fake).

The project must investigate **where** this difference is visible and whether it survives unseen attacks/domains.

Do not frame deepfake detection as looking for one magic artifact. Treat it as distributional classification with overlapping class distributions and potentially weak evidence spread across spectrum, temporal dynamics, source/excitation behavior, learned SSL representations, and recording conditions.

---

## Recommended repository layout

```text
audio-deepfake-workshop/
├── pyproject.toml
├── uv.lock
├── README.md
├── handoff.md
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── lfcc_resnet.yaml
│   ├── ssl_probe.yaml
│   └── fusion_multitask.yaml
├── data/
│   ├── raw/                 # gitignored
│   ├── metadata/
│   ├── manifests/
│   └── cache/               # extracted LFCC/MFCC/SSL embeddings
├── notebooks/
│   ├── 01_dataset_audit.ipynb
│   ├── 02_spectral_eda.ipynb
│   ├── 03_cepstral_eda.ipynb
│   ├── 04_simple_detection.ipynb
│   ├── 05_cnn_resnet_experiments.ipynb
│   ├── 06_ssl_representation_probe.ipynb
│   └── 07_fusion_error_analysis.ipynb
├── src/audio_df/
│   ├── __init__.py
│   ├── data.py
│   ├── audio.py
│   ├── features.py
│   ├── stats.py
│   ├── visualization.py
│   ├── metrics.py
│   ├── evaluation.py
│   ├── cache.py
│   ├── training.py
│   ├── models/
│   │   ├── linear.py
│   │   ├── mlp.py
│   │   ├── cnn.py
│   │   ├── resnet.py
│   │   ├── ssl.py
│   │   ├── pooling.py
│   │   ├── fusion.py
│   │   └── multitask.py
│   └── experiments/
│       ├── run_baseline.py
│       ├── run_ssl.py
│       └── run_fusion.py
├── tests/
├── reports/
│   ├── figures/
│   ├── tables/
│   ├── experiment_registry.csv
│   └── final_report.md
└── scripts/
    ├── prepare_data.py
    ├── extract_features.py
    └── evaluate.py
```

### Notebook rule
Notebooks are for questions, plots, short experiments, and interpretation. If a function is used twice, move it to `src/audio_df/`. Do not bury feature extraction, data splitting, metrics, model definitions, or training loops in notebooks.

---

## Environment: use uv

Initialize with something like:

```bash
uv init
uv add numpy scipy pandas matplotlib scikit-learn librosa soundfile torch torchaudio transformers datasets accelerate pyyaml tqdm rich
uv add --dev pytest ruff mypy ipykernel
```

Pin versions through `uv.lock`. Use `uv run ...` for scripts/tests/notebooks. Keep heavyweight optional packages in dependency groups if appropriate.

Set deterministic seeds and record package versions, device, git commit, config, dataset split hash, and random seed for every experiment.

---

# Workshop implementation path

## Stage 0 - Dataset and leakage audit

Before modeling, build a manifest with at least:

- file path / utterance ID
- label: bonafide/spoof
- speaker ID if available
- attack/generator ID if available
- source corpus
- sample rate
- duration
- codec/container
- split
- language/domain metadata if available

Start with ASVspoof because it provides reproducible protocols and baselines. ASVspoof 2021 provides LA/PA/DF databases and reproducible LFCC-GMM, CQCC-GMM, LFCC-LCNN, and RawNet2 baselines. ASVspoof 5 moved explicitly toward speech deepfake detection and spoofing-robust ASV; its official evaluation plan names AASIST and RawNet2 as CM baselines.

References for Claude to inspect:
- https://www.asvspoof.org/index2021.html
- https://www.asvspoof.org/
- https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf
- https://www.asvspoof.org/workshop2024_program

### Leakage tests
Before training, test whether labels are predictable from nuisance metadata:

- duration alone
- sample rate
- codec/container
- RMS/loudness
- leading/trailing silence
- source corpus
- speaker identity

Fit tiny classifiers on metadata-only features. If these perform suspiciously well, flag the dataset/split before proceeding.

Also visualize class distributions. A detector that learns `microphone noise -> real` or `clean WAV -> fake` is not solving the intended problem.

---

## Stage 1 - Pure waveform/spectral EDA: where do classes differ?

Use matched examples whenever possible. Do not cherry-pick one impressive real/fake pair; aggregate distributions.

For each class and attack/generator:

1. waveform amplitude and RMS distributions
2. duration and silence ratios
3. STFT magnitude/power spectra
4. log-power spectrograms
5. average spectrum with confidence intervals
6. spectral centroid, bandwidth, rolloff, flatness
7. low/mid/high-band energy ratios
8. zero-crossing rate
9. F0 estimates and voiced/unvoiced proportions if robust enough
10. temporal changes of these features

Important interpretation: these are candidate statistical cues, not universal properties of fake speech.

Implement reusable functions in `features.py` and `stats.py`.

### Statistical comparison
For scalar features, record:

- class mean/median/std/IQR
- effect size (e.g. Cohen's d where appropriate)
- non-parametric alternative when distributions are poor
- bootstrap confidence intervals
- ROC-AUC of each feature used alone

For high-dimensional spectra, visualize mean differences and standardized effect sizes per frequency bin, but avoid treating thousands of uncorrected pointwise significance tests as scientific evidence.

Add generator-conditioned plots: a feature that separates one generator but reverses for another is likely generator-specific.

---

## Stage 2 - LFCC, MFCC, and DCT as controlled feature engineering

Implement or wrap a feature extractor with explicit configuration. Keep notation clear:

- R = STFT hop
- J = number of filterbank channels
- Psi_r[k] = filterbank weights

Pipeline:

x[n] -> STFT X[m,k] -> power P[m,k] -> filterbank E[m,r] -> log L[m,r] -> DCT c[m,q]

where

E[m,r] = sum_k P[m,k] Psi_r[k]

and

c[m,q] = sum_r L[m,r] cos(pi*q/J*(r+1/2)).

### Critical implementation experiments
Do not merely compare LFCC vs MFCC. Decompose the representation choices:

A. linear log-filterbank without DCT
B. LFCC with all DCT coefficients
C. LFCC with truncated DCT coefficients
D. mel log-filterbank without DCT
E. MFCC
F. optional static + delta + delta-delta

This isolates whether gains come from:

- frequency warping (linear vs mel)
- filterbank smoothing
- logarithmic compression
- DCT/decorrelation
- coefficient truncation
- temporal derivatives

Remember:
- power discards phase;
- filterbank pooling discards fine FFT detail;
- log is mostly a dynamic-range/geometry transform for positive values;
- a full orthonormal DCT is invertible;
- DCT truncation discards information;
- LFCC high-order coefficients describe finer variation of the **filterbank spectral shape**, not necessarily the original harmonic comb because the filterbank has already smoothed the spectrum.

### Source-filter caveat
The exact identity log|X(f)| = log|G(f)| + log|H(f)| holds at spectral frequencies for the simplified source-filter model. It does **not** exactly survive `log(sum_k P[k] Psi[k])`; log of a sum is not a sum of logs. Only use the source/filter intuition approximately under suitable within-band smoothness.

### DCT historical context
The DCT was useful partly because correlated filterbank channels were inconvenient for diagonal-covariance GMM backends. Modern neural networks can model correlated filterbanks directly, so DCT vs no-DCT is an empirical representation choice.

### EDA in feature space
For each representation:

- per-dimension distributions
- class mean difference / effect size
- correlation matrices
- PCA (visualization only, not proof)
- UMAP/t-SNE only as exploratory visualization, never as evidence of separability
- attack-conditioned plots
- temporal trajectories

Then fit exactly the same simple backend to each representation so representation quality is compared fairly.

---

## Stage 3 - Simple detection: establish intuition and baselines

Before deep models, reduce variable-length sequences with simple statistics such as per-feature mean/std or statistics pooling. Then train:

1. logistic regression
2. linear SVM (optional)
3. small MLP

The core detector is simply:

z = w^T x + b
p(fake|x) = sigmoid(z).

Use this to show that detection means finding a boundary in feature space, not recognizing a hard-coded artifact.

Record:

- EER (primary anti-spoofing metric)
- ROC-AUC
- PR-AUC if imbalance matters
- F1/precision/recall at a stated threshold
- calibration/Brier score if probability interpretation matters
- per-attack/generator EER

Never tune the decision threshold on the evaluation set.

---

## Stage 4 - Sequence-aware LFCC/MFCC backends: CNN -> ResNet

The feature tensor is T x D, not just D. Build a small CNN first to learn local time/cepstral patterns. Then introduce residual blocks.

A residual block learns:

y = x + F(x)

which gives deep networks a direct identity path while learning residual refinements.

Suggested progression:

- 2D CNN over time x cepstral dimension
- ResNet18-style small backbone adapted to one-channel feature maps
- temporal/statistics pooling
- binary classification head

Do not blindly use ImageNet conventions. Audio axes are not interchangeable image axes. Document kernel shapes/strides and what they mean temporally and spectrally.

A useful architecture:

LFCC [B,1,D,T]
 -> stem conv
 -> residual stages
 -> feature map
 -> temporal/statistics pooling
 -> embedding h_L
 -> linear head

Run the same backbone with LFCC, MFCC, linear log-filterbank, and mel log-filterbank where dimensions permit. This separates backend strength from representation strength.

---

## Stage 5 - Raw waveform baseline: RawNet2

Include RawNet2 as an important conceptual and contemporary baseline because it learns directly from raw audio rather than requiring LFCC/MFCC. The anti-spoofing RawNet2 paper explicitly motivates raw waveform learning as a way to learn cues missed by traditional countermeasure features.

Reference:
- https://arxiv.org/abs/2011.01108

Do not necessarily reimplement it from scratch if an official/reliable implementation is available. Prefer reproducing official configuration and reporting compute/parameter differences.

---

## Stage 6 - Self-supervised representations: wav2vec 2.0

The next question is whether a representation learned from massive unlabeled speech is better than manually engineered spectral features.

Conceptual pipeline:

raw waveform
 -> convolutional feature encoder
 -> latent sequence
 -> masking during pretraining
 -> Transformer context network
 -> contextual hidden states

wav2vec 2.0 learns speech representations through self-supervised pretraining rather than real/fake supervision. The anti-spoofing literature shows that a wav2vec 2.0 front-end with fine-tuning can be very strong and improve generalization compared with conventional baselines.

Reference:
- https://arxiv.org/abs/2202.12233

### Workshop experiments
Start with frozen representations to keep the experiment feasible:

1. load pretrained model
2. standardize input to expected sample rate
3. extract hidden states from each Transformer layer
4. pool each layer to utterance vectors
5. train the same linear probe on every layer
6. plot EER vs layer index

This is extremely informative: it tests **where in the SSL hierarchy deepfake-relevant information becomes linearly accessible**.

Then compare:

- final layer only
- selected best validation layer
- mean of layers
- learned weighted layer sum

For learned layer weighting:

alpha_l = softmax(a)_l
H = sum_l alpha_l H^(l)

The SSL encoder can remain frozen while `a_l` is trainable.

---

## Stage 7 - Transformer/self-attention workshop

Do not turn this into a generic Transformer tutorial. Implement one tiny attention example to make the operation concrete, then use the library implementation for actual models.

For hidden sequence H:

Q = H W_Q
K = H W_K
V = H W_V

Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V.

Interpretation for audio: each time position forms a query and assigns data-dependent weights to information at other positions. Unlike a fixed local convolution, attention can directly combine distant positions.

Multi-head attention means multiple Q/K/V projections learn different relational subspaces. Do not confuse Transformer attention heads with the **task heads** in the later multi-task model.

Experiments:

- compare local CNN representation vs SSL Transformer representation
- inspect representation similarity / layer probing rather than treating raw attention maps as explanations
- optionally compare simple mean pooling vs statistics pooling vs attentive pooling

---

# Stage 8 - XLS-R and the proposed representation branch

This is the bridge into the final model. Use a pretrained wav2vec 2.0 XLS-R checkpoint suitable for 16 kHz speech. Start frozen for feasibility and reproducibility.

Branch:

waveform -> frozen XLS-R -> hidden states H^(1)...H^(L)
 -> learned softmax layer weighting
 -> temporal/statistics pooling
 -> projection + LayerNorm
 -> 256-d vector z_X

Why learned layer weighting instead of blindly using the final layer? Different SSL layers can expose different acoustic/phonetic/contextual information, and the linear-probe experiment in Stage 6 should empirically justify this design.

If compute allows, add a controlled fine-tuning experiment later. Do not mix frozen and fine-tuned results without clearly reporting trainable parameter count and compute.

---

# Stage 9 - Proposed LFCC branch

Do not feed flattened LFCC directly to a huge MLP. Use a compact sequence-aware backbone.

Recommended:

LFCC/log-linear-filterbank -> small ResNet/CNN -> statistics/attentive pooling -> projection + LayerNorm -> 256-d z_L

The exact representation should be chosen from Stage 2/4 validation results. If log-linear filterbanks outperform LFCC under the same backend, the final architecture should use the empirically stronger representation rather than preserving LFCC for ideological reasons. Still retain LFCC as a named baseline.

---

# Stage 10 - Proposed fusion + shared multi-task trunk

Concatenate aligned branch embeddings:

z = [z_L ; z_X] in R^512

Then use a compact shared trunk, e.g.:

512 -> Linear(512,256) -> LayerNorm -> GELU -> Dropout -> shared h

Attach task-specific heads:

Head 1: bonafide/spoof binary logit
Head 2: auxiliary target **only if the dataset provides a scientifically defensible label available during training** (e.g. attack/generator family for spoof examples, or another justified auxiliary task).

Do not invent pseudo-ground-truth merely to make the architecture multi-task.

Loss:

L_total = lambda_det L_det + lambda_aux L_aux.

For shared parameters theta_s:

grad_theta_s L_total = lambda_det grad L_det + lambda_aux grad L_aux.

This is the reason hard parameter sharing can alter the learned representation. It can help when tasks provide compatible constraints, and hurt through negative transfer when gradients conflict.

Optionally log gradient cosine similarity:

cos(g1,g2) = g1^T g2 / (||g1|| ||g2||)

on selected shared layers during diagnostic runs.

### Required ablations

- LFCC/selected handcrafted feature only
- XLS-R only
- concatenation without shared nonlinear trunk
- fusion + shared trunk, detection head only
- fusion + shared trunk + auxiliary head
- final layer XLS-R vs learned layer weighting
- mean pooling vs selected pooling method

The project should not claim that LFCC and XLS-R are complementary until the ablations show it.

---

# Benchmark suite

The final project should compare **families**, not just one model against ours.

## Classical / interpretable
- LFCC + logistic regression/GMM-style baseline where feasible
- LFCC + small CNN/LCNN or ResNet
- MFCC/log-mel equivalent backend

## Raw waveform
- RawNet2

## Graph/spectro-temporal
- AASIST

ASVspoof 5 names RawNet2 and AASIST as official countermeasure baselines.
Reference: https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf

## SSL
- wav2vec 2.0 + simple backend
- WavLM + simple backend
- XLS-R + simple backend

A 2024 ASVspoof 5 study benchmarked ten pretrained representation types and reported wav2vec2/WavLM families as strongest among those studied; WavLM fine-tuning and model fusion were used in its final system.
Reference: https://arxiv.org/abs/2408.07414

The ASVspoof 5 workshop program itself shows the contemporary landscape: WavLM backends, WavLM-ResNet18 fusion, advanced ResNets, SSL features with AASIST variants, feature aggregation, augmentation, and generalization studies.
Reference: https://www.asvspoof.org/workshop2024_program

## Proposed
- best handcrafted branch + frozen XLS-R learned-layer branch + shared trunk
- multi-task variant if auxiliary labels are justified

---

# Datasets and evaluation protocol

Use at least one standardized benchmark first, preferably ASVspoof 2019/2021 or ASVspoof 5 depending data access/compute. For a feasible student project, start with ASVspoof 2019 LA or ASVspoof 2021 DF/LA, then add a cross-dataset or unseen-generator evaluation if resources permit.

ASVspoof 2021 explicitly provides real bona fide and spoofed speech generated with diverse attacks plus evaluation metadata and official metric scripts.
Reference: https://www.asvspoof.org/index2021.html

For a stronger generalization claim, add one external/in-the-wild dataset only after the in-domain pipeline is stable. A major survey identifies unknown-attack generalization, in-the-wild data scarcity, and interpretability as unresolved problems.
Reference: https://arxiv.org/abs/2308.14970

### Splitting rules
- respect official train/dev/eval protocols
- never randomly reshuffle official evaluation data into training
- prevent speaker overlap when designing custom splits
- for unseen-generator experiments, hold out complete attack/generator families, not random utterances from each generator
- preserve a final untouched evaluation set

### Record for every model

Performance:
- EER
- ROC-AUC
- PR-AUC where relevant
- F1/precision/recall at declared operating point
- per-attack/generator EER
- macro summaries across attacks
- cross-dataset/unseen-attack result

Efficiency:
- total parameters
- trainable parameters
- model checkpoint size
- peak GPU memory if available
- training wall time
- inference latency / real-time factor
- feature extraction/cache time

Robustness:
- codec/compression conditions
- additive noise
- resampling
- optional reverberation
- short-duration slices

Reliability:
- at least 3 seeds for models small enough to repeat
- mean +/- std
- bootstrap confidence intervals for key evaluation comparisons where practical

Do not compare published EER numbers as if they are directly comparable when datasets/protocols/training data differ. Reproduce models under the same local protocol wherever possible; otherwise label literature numbers as context only.

---

# Research questions that can become the project contribution

The project should be organized around explicit questions rather than "we built a bigger network":

### RQ1 - Where is fake-vs-real information visible?
Compare raw spectral statistics, LFCC/MFCC/log-filterbank features, and SSL layer probes.

### RQ2 - Is LFCC itself useful, or are its constituent choices responsible?
Ablate linear vs mel spacing, DCT vs no DCT, coefficient truncation, and deltas.

### RQ3 - Does learned SSL representation make handcrafted features redundant?
Compare best handcrafted-only, XLS-R-only, and fusion under the same split.

### RQ4 - Does fusion improve unseen-generator/cross-domain generalization rather than only in-domain EER?
This should be the key test of "complementarity."

### RQ5 - Does hard parameter sharing help?
Compare detection-only shared trunk against multi-task shared trunk; inspect gradient alignment where useful.

### RQ6 - Which XLS-R layers contain the strongest linearly accessible spoof information?
Layer-wise linear probing can itself produce a strong analysis chapter.

These questions create a defensible project even if the final fusion model does not win.

---

# Reporting and experiment discipline

Every run should append one row to `reports/experiment_registry.csv` containing:

- run ID
- timestamp
- git commit
- dataset/split
- preprocessing config
- feature representation
- backbone
- SSL checkpoint/layers/frozen status
- pooling
- loss weights
- augmentation
- seed
- trainable/total params
- EER and other metrics
- artifact/checkpoint path

Save machine-readable JSON metrics beside human-readable tables.

Use a final report structure:

1. Problem and research questions
2. Dataset/protocol and leakage audit
3. Spectral/cepstral EDA
4. Feature-engineering experiments
5. Classical/simple baselines
6. CNN/ResNet baselines
7. SSL representation probing
8. Proposed fusion/multi-task architecture
9. Ablations
10. Generalization/robustness
11. Efficiency comparison
12. Error analysis
13. Limitations and conclusions

---

# Contemporary references / knowledge seed

Claude should independently read and structure these rather than copying conclusions blindly:

1. **ASVspoof 2021 resources and baselines**
   https://www.asvspoof.org/index2021.html

2. **ASVspoof 5 official site / evaluation plan**
   https://www.asvspoof.org/
   https://www.asvspoof.org/file/ASVspoof5___Evaluation_Plan_Phase2.pdf

3. **ASVspoof Workshop 2024 program** - useful map of contemporary systems (WavLM, ResNet, SSL+AASIST, fusion, augmentation, unseen-data generalization)
   https://www.asvspoof.org/workshop2024_program

4. **Audio Deepfake Detection: A Survey** (Yi et al.)
   https://arxiv.org/abs/2308.14970

5. **RawNet2 anti-spoofing** (Tak et al.)
   https://arxiv.org/abs/2011.01108

6. **wav2vec 2.0 + data augmentation for spoof/deepfake detection** (Tak et al.)
   https://arxiv.org/abs/2202.12233

7. **WavLM model ensemble for audio deepfake detection / ASVspoof 5**
   https://arxiv.org/abs/2408.07414

8. **AASIST** - follow from ASVspoof references / official implementation; compare its integrated spectro-temporal graph-attention approach with CNN/ResNet and SSL approaches.

Use papers' official repositories where available. Record exact commit/tag/checkpoint used for reproduced baselines.

---

# Expected end-state architecture

The intended candidate model is:

```text
                       HANDCRAFTED / SPECTRAL BRANCH
waveform -> LFCC or best Stage-2 representation -> compact CNN/ResNet
                                             -> pooling -> projection -> LN -> z_L (256)

                       SELF-SUPERVISED BRANCH
waveform 16 kHz -> frozen XLS-R -> hidden states from all layers
                               -> learned softmax layer weighting
                               -> pooling -> projection -> LN -> z_X (256)

z = concat(z_L, z_X)  # 512
        |
        v
shared trunk: Linear -> LayerNorm -> GELU -> Dropout
        |
        +--------------------+
        |                    |
        v                    v
 detection head         auxiliary head
 real / fake            justified task label
        |                    |
        +---------+----------+
                  |
L_total = lambda_det L_det + lambda_aux L_aux
```

The architecture is intentionally modular. The workshop's earlier experiments should be allowed to change the handcrafted representation, pooling, or auxiliary task if evidence says so.

---

# Final benchmark table concept

The final report should contain a table roughly like:

| Family | Model | Input | Pretraining | Trainable params | In-domain EER | Unseen-generator EER | Cross-dataset EER | RTF/latency | Notes |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| Classical | LFCC + simple backend | LFCC | none | ... | ... | ... | ... | ... | interpretable baseline |
| CNN | LFCC + ResNet | LFCC | none | ... | ... | ... | ... | ... | learned local patterns |
| Raw | RawNet2 | waveform | none | ... | ... | ... | ... | ... | end-to-end waveform |
| Graph | AASIST | spectral/raw as implementation | none | ... | ... | ... | ... | ... | official ASVspoof-family baseline |
| SSL | wav2vec2 backend | waveform | SSL | ... | ... | ... | ... | ... | contextual speech repr. |
| SSL | WavLM backend | waveform | SSL | ... | ... | ... | ... | ... | contemporary SSL comparison |
| SSL | XLS-R probe | waveform | SSL multilingual | ... | ... | ... | ... | ... | proposed branch baseline |
| Proposed | handcrafted + XLS-R fusion | both | SSL | ... | ... | ... | ... | ... | fusion ablation |
| Proposed | fusion + MTL | both | SSL | ... | ... | ... | ... | ... | full model |

The key result is **not simply lowest in-domain EER**. The report should explicitly judge whether any improvement survives unseen-generator/cross-domain evaluation and whether the extra compute is justified.

---

# Definition of done

The project is complete when it has:

1. reproducible uv environment;
2. dataset manifest and leakage audit;
3. compact EDA notebooks with reusable modules outside notebooks;
4. controlled LFCC/MFCC/log-filterbank/DCT experiments;
5. simple linear baseline;
6. CNN/ResNet handcrafted-feature baseline;
7. RawNet2 and AASIST comparison where feasible;
8. wav2vec2/WavLM/XLS-R representation experiments;
9. layer-wise SSL probing;
10. proposed fusion architecture;
11. multi-task ablation if a defensible auxiliary label exists;
12. in-domain + unseen-generator/cross-domain evaluation;
13. efficiency measurements;
14. experiment registry;
15. final tables/figures/report with limitations.

Do not optimize the final architecture before the diagnostic stages. The point of the workshop is that each architectural choice should be motivated by an observed limitation or experimental result.
