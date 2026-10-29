"""Arbitration test: the judge in configs/arbitrator.yaml decides where the six detectors disagree.

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

RUNS = {  # latest trimmed-training run per detector; D2/D3 use WavLM layer 6 (round 3)
    "a1": ("A1 LFCC-GMM", "runs/E-20261023-172139-766981-a1"),
    "a2": ("A2 WavLM", "runs/E-20261023-200932-976565-a2"),
    "a4": ("A4 LFCC-MLP", "runs/E-20261023-194755-458184-a4"),
    "d1": ("D1 LFCC+WavLM concat", "runs/E-20261023-201728-785995-d1"),
    "d2": ("D2 LFCC->WavLM attention", "runs/E-20261024-214524-879357-d2"),
    "d3": ("D3 LFCC<->WavLM attention", "runs/E-20261024-171010-159987-d3"),
}
# The same detectors listed in reverse: a judge that reads the evidence gives the same answer.
ORDERS = {"A1 first": ("a1", "a2", "a4", "d1", "d2", "d3"), "A1 last": ("d3", "d2", "d1", "a4", "a2", "a1")}
CLIPS = [  # 11249.wav (bona fide) + 200 clips where the six detectors split (notebook 05)
    11249, 5707, 3589, 7423, 8056, 12496, 24621, 23201, 3313, 7082, 30199, 28559, 28997, 24805,
    7216, 7435, 14433, 30336, 3610, 27864, 12291, 30593, 9440, 21797, 1421, 544, 14930, 18847,
    29905, 18251, 27237, 10337, 12063, 2647, 12326, 9102, 8475, 31727, 22447, 6881, 29553, 14497,
    28378, 26262, 8598, 7469, 30489, 11126, 11507, 10197, 9617, 25472, 27730, 10117, 13760, 2574,
    1610, 22491, 27584, 2026, 12732, 27112, 31543, 13892, 26388, 3710, 14328, 24773, 2781, 6081,
    25822, 25652, 6768, 26035, 16806, 908, 12746, 26366, 3626, 6766, 22196, 2251, 8177, 939,
    13745, 20045, 8561, 9332, 17177, 3227, 15959, 18976, 11010, 12188, 1380, 20338, 20794, 11783,
    17292, 2106, 18700, 17199, 13909, 16292, 5463, 12515, 29974, 28982, 26102, 23584, 18003,
    11543, 15918, 25206, 19516, 1520, 23167, 29388, 21810, 7105, 6116, 13544, 12730, 26260,
    24583, 26234, 22373, 4342, 22824, 4838, 6386, 14087, 22645, 4602, 31616, 1909, 5236, 21864,
    11621, 3664, 12816, 9978, 31189, 31746, 21922, 18562, 4401, 7001, 9022, 26986, 4007, 20155,
    8370, 21535, 31703, 24519, 7067, 17253, 13222, 19401, 188, 11902, 22157, 29598, 12831, 165,
    3830, 9398, 8647, 1418, 15693, 26440, 12658, 819, 4813, 10662, 23738, 20355, 15029, 9288,
    1526, 4608, 23952, 25794, 2637, 15707, 4951, 11329, 20237, 2855, 14330, 19420, 27851, 24167,
    4302, 12236, 17166, 28108, 13954, 14177, 24856,
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
    """Arbitration test on fixed In-the-Wild clips (six detectors); the judge is set in configs/arbitrator.yaml."""
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
