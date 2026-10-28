"""Arbitration test: the judge in configs/arbitrator.yaml decides where A1, A2 and A4 disagree.

Fixed test case, no audio is loaded: each detector's saved In-the-Wild scores and the dev EER
threshold of its own artifact. CLIPS are In-the-Wild file numbers (release_in_the_wild/<n>.wav)
from notebook 05, "Arbitration test case". The runtime only provides the CLI and the output run
directory.

Writes runs/E-<time>-arbitration/cases.csv: each detector's verdict and P(bona fide), then the
judge's answer per request variant and listing order; judge.json records the judge. Accuracy and
confusion matrices are in notebook 06.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from audio_df.helpers.config import load_config
from audio_df.modules.arbitrator import VARIANTS, ArbitratorConfig, decide, request
from audio_df.runtime import Run

RUNS = {  # latest trimmed-training run per detector
    "a1": ("A1 LFCC-GMM", "runs/E-20261023-172139-766981-a1"),
    "a2": ("A2 WavLM", "runs/E-20261023-200932-976565-a2"),
    "a4": ("A4 LFCC-MLP", "runs/E-20261023-194755-458184-a4"),
}
ORDERS = {"A1 first": ("a1", "a2", "a4"), "A1 last": ("a2", "a4", "a1")}
CLIPS = [  # 11249.wav (bona fide; A2 alone says spoof) + 200 clips where the detectors split
    11249, 3190, 19614, 25652, 6867, 487, 6350, 10494, 310, 23213, 29329, 8560, 29362, 26539,
    19760, 19501, 8100, 26653, 22426, 4650, 19557, 16874, 2495, 2862, 22092, 15569, 23927,
    27981, 22541, 9453, 2051, 10864, 23972, 1023, 11391, 12921, 12977, 4349, 18916, 17352,
    13752, 27877, 14528, 21290, 22784, 18434, 19613, 24075, 23608, 31022, 19208, 159, 28245,
    22800, 4718, 17464, 19314, 30069, 26673, 15430, 27871, 20732, 10960, 27152, 14069, 26558,
    17435, 31662, 11551, 3410, 15728, 29784, 2281, 12747, 24714, 15380, 17001, 12629, 8516,
    23257, 10027, 2876, 29148, 349, 30308, 9828, 20909, 22494, 13778, 3021, 4379, 9369, 17972,
    25957, 18436, 28295, 14566, 7039, 30267, 13262, 18797, 2987, 29487, 183, 26337, 11281,
    21746, 10942, 12027, 11375, 8650, 28393, 14187, 26880, 5614, 2187, 13036, 13267, 3389,
    28113, 24879, 18337, 25064, 27555, 30345, 10113, 15089, 20789, 2126, 15545, 10842, 12881,
    20183, 19271, 31743, 24908, 4128, 28866, 3227, 587, 1563, 16046, 6021, 7093, 29092, 21012,
    24220, 26589, 14106, 10415, 17260, 16197, 12163, 30744, 6920, 26304, 20308, 24093, 16945,
    20644, 30628, 11519, 17600, 25286, 27296, 3407, 2118, 15178, 1459, 31154, 24904, 26611,
    6005, 10, 8875, 21607, 23101, 7506, 14686, 13207, 22902, 743, 28541, 29155, 8658, 23109,
    30824, 24819, 5992, 8998, 24831, 7654, 12042, 10368, 24737, 29131, 18293, 28799, 10847,
    14819, 22072,
]


def load_cases(root: Path) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """CLIPS with each detector's verdict and P(bona fide); per detector its name and error rate.

    Threshold: the run's calibration.json (dev EER). P(bona fide): Platt scaling on dev scores.
    Error rate: the run's 2021 LA eval EER (%), a different set from the one tested here.
    """
    ids = [f"ITW_{n}" for n in CLIPS]
    cases, det = pd.DataFrame({"clip_id": ids}), {}
    for m, (name, run) in RUNS.items():
        d = root / run
        dev = pd.read_csv(d / "scores" / "dev.csv")
        platt = LogisticRegression().fit(dev[["score"]], dev.label == "bonafide")
        threshold = json.loads((d / "calibration.json").read_text())["eer_threshold"]
        eer_2021 = json.loads((d / "metrics_asvspoof2021_eval.json").read_text())["eer"]
        test = pd.read_csv(d / "scores" / "itw.csv").set_index("clip_id").loc[ids]
        cases["label"] = test.label.to_numpy()
        cases[f"v_{m}"] = np.where(test.score >= threshold, "bonafide", "spoof")
        cases[f"p_{m}"] = platt.predict_proba(test[["score"]])[:, 1]
        det[m] = {"name": name, "threshold": threshold, "error_rate_pct": round(100 * eer_2021, 1)}
    return cases, det


def evidence(row: dict[str, Any], det: dict[str, dict[str, Any]], order) -> list[dict[str, Any]]:
    """One clip's evidence for the arbitrator, detectors in the given listing order."""
    return [{"name": det[m]["name"], "verdict": row[f"v_{m}"], "p_bonafide": row[f"p_{m}"],
             "error_rate_pct": det[m]["error_rate_pct"]} for m in order]


@Run.cli(dataset="2019LA")
def main(run: Run) -> None:
    """Arbitration test on fixed In-the-Wild clips; the judge is set in configs/arbitrator.yaml."""
    cases, det = load_cases(run.root)
    for variant in VARIANTS:
        for name, order in ORDERS.items():
            requests = [request(evidence(r, det, order), variant) for r in cases.to_dict("records")] # type: ignore
            cases[f"{variant} | {name}"] = [decide(req)["choice"] for req in requests]
            print(f"  {variant} | {name}: done", flush=True)
    cases.to_csv(run.output / "cases.csv", index=False)
    meta = {"judge": load_config(ArbitratorConfig).judge,
            "detectors": {m: d["name"] for m, d in det.items()}}
    (run.output / "judge.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved: {run.output / 'cases.csv'}")


if __name__ == "__main__":
    main()
