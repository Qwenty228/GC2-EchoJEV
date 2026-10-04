# E-20261004-e2-a2-raw-layers

created: 2026-10-04T10:00:39.432364+00:00
- Lock: data/cache/xai_light.lock, NOT the heavy-job lock. CPU-only (torch threads 4, no MPS), seconds per head, run alongside the queued A1 heavy job.
- Thresholds: each variant's own raw dev EER threshold (score_experiment default), applied unchanged to eval raw, eval trimmed and ITW. LR probes are trained on the train split only (Pascu et al. trained on train+dev).
- reproduction check vs experiments/E-20261001-e1-a2-raw (CPU retrain, seed 1337): {"dev": {"max_abs_diff": 11.369684219360352, "mean_abs_diff": 2.7040272171202298, "pearson": 0.990463441214091}, "eval": {"max_abs_diff": 12.754886627197266, "mean_abs_diff": 2.5877409820573827, "pearson": 0.9867690427462783}, "eer_e1": {"dev": 0.003935771543029732, "eval": 0.009640211044279444}, "eer_retrained_seed1337": {"dev": 0.005017830598331906, "eval": 0.015621261124058802}}
