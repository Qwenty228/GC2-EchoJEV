# References

Every citation in the report and the [technical reference](../technical-reference.md) is a
number in square brackets, optionally followed by a location in the paper: `[n]`,
`[n, Table 2, p.4]`, `[n, §3]`. Each entry gives the authors, title, venue, identifier, what the
project uses it for, and a link to the public paper. Page-referenced reading notes for most
papers are kept under [`Miscellaneous/research/knowledge/`](../../Miscellaneous/research/knowledge/index.md).

**Core reading list**

- [1] Delgado et al. *On Deepfake Voice Detection – It's All in the Presentation.* ICASSP 2026. arXiv:2509.26471. Used for: E4 motivation, playback vs phone injection. [paper](https://arxiv.org/abs/2509.26471)
- [2] Wang et al. *ASVspoof 5: Evaluation of Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech.* 2026. arXiv:2601.03944. Used for: minDCF definition (Eq. 1, β = 1.90). [paper](https://arxiv.org/abs/2601.03944)
- [3] Stourbe et al. *Exploring WavLM Back-ends for Speech Spoofing and Deepfake Detection.* ASVspoof 2024 Workshop, pp. 72–78. DOI 10.21437/ASVspoof.2024-11. Used for: WavLM Base (not Base+) fine-tuned on ASVspoof 5; layer-weighted back end. [paper](https://arxiv.org/abs/2409.05032)
- [4] Rimon et al. *Unmasking Deepfakes: Leveraging Augmentations and Features Variability for Deepfake Speech Detection.* arXiv preprint, 2025 (no venue stated). arXiv:2501.05545. Used for: augmentation with end-to-end fine-tuning on ASVspoof 5; no frozen or ITW experiment. [paper](https://arxiv.org/abs/2501.05545)
- [5] Wang et al. *ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech.* 2025. arXiv:2502.08857. Used for: codec conditions in the next-generation benchmark. [paper](https://arxiv.org/abs/2502.08857)
- [6] El Kheir et al. *Two Views, One Truth: Spectral and Self-Supervised Features Fusion for Robust Speech Deepfake Detection.* WASPAA 2025. arXiv:2507.20417. Used for: spectral + SSL fusion (helps out of domain, not on LA19). [paper](https://arxiv.org/abs/2507.20417)
- [7] Tak et al. *Automatic Speaker Verification Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation.* Odyssey 2022. arXiv:2202.12233. Used for: fine-tuned XLS-R + AASIST, last layer only, tested on 2021 LA/DF. [paper](https://arxiv.org/abs/2202.12233)
- [8] Pascu et al. *Towards Generalisable and Calibrated Audio Deepfake Detection with Self-Supervised Representations.* Interspeech 2024. arXiv:2309.05384. Used for: frozen last layer + logistic regression; ECE. [paper](https://arxiv.org/abs/2309.05384)
- [9] TypeSafe. *Jev — Project Knowledge Resource.* Community summary, 16 Sep 2026. Used for: JEV interface. To be checked against official documentation.
- [10] *JEV-as-a-Judge: Accept When Confident, Escalate When Unsure.* Used for: A3 decision rule.
- [11] Jev AI. *Model page.* Used for: Jev is text-only.
- [12] Jev AI. *About Jev AI* and evaluation and quality use cases. Used for: JEV scope.
- [13] Müller et al. *Does Audio Deepfake Detection Generalize?* Interspeech 2022. arXiv:2203.16263. Used for: In-the-Wild dataset and the collapse of ASVspoof-trained models. [paper](https://arxiv.org/abs/2203.16263)
- [14] Wang et al. *ASVspoof 2019: A Large-Scale Public Database of Synthesized, Converted and Replayed Speech.* Computer Speech & Language 64, 2020. arXiv:1911.01601. Used for: attack generators (Table 1, p.6); A16 = A04, A19 = A06. [paper](https://arxiv.org/abs/1911.01601)
- [15] Todisco et al. *ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection.* Interspeech 2019. arXiv:1904.05441. Used for: B02 8.09 % EER / t-DCF 0.2116, B01 9.57 % (Table 1). [paper](https://arxiv.org/abs/1904.05441)
- [16] Chen F. et al. *Deepfake Speech Detection Method Based on Wav2Vec2.0 Feature Merging and Joint Loss.* Journal of Signal Processing 41(9):1547–1557, 2025. DOI 10.12466/xhcl.2025.09.008. Used for: merging multiple SSL layers.

**Further sources**

- [17] Mo and Wang. *Multi-task Learning Improves Synthetic Speech Detection.* ICASSP 2022, pp. 6392–6396. DOI 10.1109/ICASSP43922.2022.9746059. Used for: LFCC ResNet on LA19 eval; auxiliary tasks. [paper](https://doi.org/10.1109/ICASSP43922.2022.9746059)
- [18] Sutharya and Sasi. *Half-Truth Audio Detection and Localisation: A Lightweight Cross-Attentive Architecture and a Cross-Corpus Diagnostic Study.* arXiv preprint, 2026 (AI-assisted writing disclosed). arXiv:2605.29531. Used for: cross-corpus collapse of a handcrafted-feature model. [paper](https://arxiv.org/abs/2605.29531)
- [19] Stan et al. *TADA: Training-free Attribution and Out-of-Domain Detection of Audio Deepfakes.* Interspeech 2025. DOI 10.21437/Interspeech.2025-472. Used for: generator attribution from time-averaged frozen SSL states (fake clips only). [paper](https://arxiv.org/abs/2506.05802)
- [20] Zhu et al. *Source Tracing: Detecting Voice Spoofing.* APSIPA ASC 2022. arXiv:2212.08601. Used for: attack attribute table; attribute heads as an auxiliary technique. [paper](https://arxiv.org/abs/2212.08601)
- [21] Cheng et al. *Explainable AI for Forensic Speech Authentication within Cognitive and Computational Neuroscience.* Frontiers in Neuroscience 19:1692122, 2025. DOI 10.3389/fnins.2025.1692122. Used for: pointer to XAI methods only (**low confidence**). [paper](https://doi.org/10.3389/fnins.2025.1692122)
- [22] Wang Z. et al. *Mixture of Experts Fusion for Fake Audio Detection Using Frozen wav2vec 2.0.* arXiv preprint, 2024 (submitted to ICASSP 2025). arXiv:2409.11909. Used for: the closest protocol to ours (frozen SSL, train 2019 LA, test LA19 eval and ITW). [paper](https://arxiv.org/abs/2409.11909)
- [23] Zhou et al. *Linear versus Mel Frequency Cepstral Coefficients for Speaker Recognition.* IEEE ASRU 2011, pp. 559–564. DOI 10.1109/ASRU.2011.6163888. Used for: filter-count geometry only (speaker recognition, not spoofing). [paper](https://doi.org/10.1109/ASRU.2011.6163888)
- [24] Frank and Schönherr. *WaveFake: A Data Set to Facilitate Audio Deepfake Detection.* NeurIPS 2021 Datasets and Benchmarks. arXiv:2111.02813. Used for: LFCC vs MFCC with the same GMM (Tables 2 and 6). [paper](https://arxiv.org/abs/2111.02813) · discussed in [`Miscellaneous/research/knowledge/methods/lfcc-vs-mfcc-explained.md`](../../Miscellaneous/research/knowledge/methods/lfcc-vs-mfcc-explained.md)
- [25] Sriskandaraja et al. *Investigation of Sub-Band Discriminative Information Between Spoofed and Genuine Speech.* Interspeech 2016. Used for: counter-evidence, mel beat linear on SAS (Table 1). [paper](https://www.isca-archive.org/interspeech_2016/sriskandaraja16_interspeech.pdf) · discussed in [`Miscellaneous/research/knowledge/methods/lfcc-vs-mfcc-explained.md`](../../Miscellaneous/research/knowledge/methods/lfcc-vs-mfcc-explained.md)
- [26] Chen S. et al. *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing.* IEEE Journal of Selected Topics in Signal Processing 16(6), 2022. arXiv:2110.13900. Used for: 94k h pre-training data (p.5); layer roles (Fig. 2, p.7). [paper](https://arxiv.org/abs/2110.13900)
- [27] Müller, Dieckmann, Czempin, Canals, Böttinger and Williams. *Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?* ASVspoof 2021 Workshop. arXiv:2106.12914. Used for: the silence shortcut (silence-only classifier 15.12 % EER; RawNet2 3.61 → 15.50 % when trained on trimmed audio). [paper](https://arxiv.org/abs/2106.12914) · discussed in [`Miscellaneous/research/knowledge/methods/leakage-audit.md`](../../Miscellaneous/research/knowledge/methods/leakage-audit.md)
- [28] Sahidullah, Kinnunen and Hanilçi. *A Comparison of Features for Synthetic Speech Detection.* Interspeech 2015. Used for: the LFCC recipe B02 follows; LFCC vs MFCC vs IMFCC with the same GMM (Tables 3–4). [paper](https://www.isca-archive.org/interspeech_2015/sahidullah15_interspeech.pdf)
- [29] Zhang, Wen and Hu. *Audio Deepfake Detection with Self-Supervised XLS-R and SLS Classifier.* ACM Multimedia 2024. DOI 10.1145/3664647.3681345. Used for: the 7.46 % ITW EER from its official repository README (paper text not reachable). [paper](https://dl.acm.org/doi/10.1145/3664647.3681345) · discussed in [`Miscellaneous/research/knowledge/methods/multi-layer-ssl-aggregation.md`](../../Miscellaneous/research/knowledge/methods/multi-layer-ssl-aggregation.md)
- [30] Meng et al. *MSLA-XLS-R: Enhancing Hierarchical SSL Representations for Audio Deepfake Detection.* Computer Speech & Language, 2026 (in press). DOI 10.1016/j.csl.2026.102044. Used for: existence of richer layer aggregation only (paper text not reachable). [paper](https://doi.org/10.1016/j.csl.2026.102044)

**Code**

- Official ASVspoof 2019 baseline code, `ASVspoof_2019_baseline_CM_v1.zip`, read line by line for A1 ([`Miscellaneous/research/knowledge/methods/b02-lfcc-gmm.md`](../../Miscellaneous/research/knowledge/methods/b02-lfcc-gmm.md)).

## Verification status

Numbers from this project's runs come from the run outputs named by run ID (see the
[report index](README.md#where-the-numbers-come-from)). Numbers marked *recomputed* come from a
light re-analysis described in
[`lfcc-vs-mfcc-explained.md`](../../Miscellaneous/research/knowledge/methods/lfcc-vs-mfcc-explained.md) and are
not run outputs. The descriptions of [1], [3], [4], [6]–[8], [13], [14] and [17]–[28] have been
checked against the papers, with headline numbers spot-checked against the PDF text. Not yet
checked against the full text: [2], [5], [9]–[12], [16], [29] and [30] (the full texts of [29]
and [30] were not reachable). [9]–[12] are JEV sources to be confirmed against official
documentation.
