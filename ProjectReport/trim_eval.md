# Silence trimming eval

Models overfit on silence. ASVspoof 2019 LA bona fide clips have more leading silence than spoof (median 1.02s vs 0.74s, 45% vs 28% of clip removed by trim). We trained on raw audio and tested on trimmed audio, so the silence cue disappears at test time.

Same model, dev EER (%):

| model | raw dev | trimmed dev |
|---|---|---|
| A2 | 0.47 | 5.49 |
| D1 | 0.43 | 7.38 |
| D2 | 3.81 | 28.84 |
| D3 | 2.53 | 31.47 |

- every model drops, D2/D3 worst. Temporal tokens + time encoding let them see where the silence is
- D2/D3 are not broken: caches match re-extraction, live = cached score, sign not flipped. On trimmed audio they just call almost everything spoof (80% of bona fide score < 0)
- D2/D3 only use WavLM layer 12. Linear probe: layer 12 gets 3.9 raw / 16.5 trimmed, A2 layer mix gets 0.22 / 5.68. So layer 12 is weaker, but it's not the main cause
- even a linear probe on layer 12 (16%) beats D3 (31%) on trimmed, so the attention head makes it worse
- early stopping used raw dev, so it picked the checkpoint that used silence most

Fix: trim train + early stopping dev too (same recipe as eval), and make the WavLM layer for D2/D3 configurable. Retrain everything for a fair comparison.

Note: commit b5ab793 run config says `dev: raw` but the code trimmed dev.
