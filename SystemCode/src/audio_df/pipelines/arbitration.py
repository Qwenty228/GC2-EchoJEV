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
CLIPS = [  # 200 clips where the six detectors split, 100 bona fide (11249.wav among them) + 100 spoof (notebook 05)
    11249, 23944, 4635, 7387, 23400, 19612, 18522, 12526, 3928, 6257, 26916, 6291, 2219, 14486,
    14618, 4521, 4227, 3441, 17783, 29322, 24218, 13511, 12825, 15738, 27145, 5052, 1740, 10787,
    25354, 22368, 31469, 7168, 29609, 24969, 23187, 18429, 8272, 19039, 16979, 18051, 16246,
    27570, 10453, 9360, 16603, 14457, 24906, 4042, 8727, 28833, 8771, 543, 28762, 11374, 5857,
    23493, 29564, 17962, 7971, 18863, 28757, 29684, 6672, 7350, 2128, 30938, 3261, 20521, 23977,
    16237, 2509, 4008, 16595, 22402, 25329, 13600, 862, 12365, 15202, 14387, 19325, 255, 23768,
    28438, 18701, 25346, 21603, 21730, 11264, 19589, 30150, 1143, 14011, 19056, 24665, 24759,
    12464, 25935, 23275, 26006, 15019, 24880, 18295, 22438, 17506, 14772, 7462, 30843, 9851,
    7320, 374, 22221, 19396, 16792, 28581, 20566, 7179, 10763, 19720, 28746, 8751, 7444, 4233,
    7262, 6812, 24316, 6566, 19012, 29351, 22845, 19305, 14953, 7965, 8118, 23023, 10066, 9537,
    28519, 1811, 24476, 11186, 1639, 16461, 1335, 13222, 19104, 21040, 11745, 29652, 29768,
    24208, 2076, 3415, 30887, 22575, 30536, 10000, 28790, 531, 8444, 23990, 4878, 26508, 2429,
    26469, 7841, 7767, 5975, 26159, 24364, 7014, 19322, 27970, 5650, 440, 30601, 4160, 22644,
    21799, 11785, 1246, 31579, 1353, 19254, 17634, 17074, 28427, 28292, 8744, 21721, 13018,
    9765, 17794, 6175, 21442, 23736, 4150, 10583, 16318, 14031,
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
    config = load_config(ArbitratorConfig)  # read once: editing the yaml mid-run changes nothing
    cases, det = load_cases(run.root)
    meta = {"judge": config.judge, "detectors": {m: d["name"] for m, d in det.items()}}
    (run.output / "judge.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for variant in VARIANTS:
        for name, order in ORDERS.items():
            requests = [request(evidence(r, det, order), variant) for r in cases.to_dict("records")] # type: ignore
            cases[f"{variant} | {name}"] = [decide(req, config)["choice"] for req in requests]
            cases.to_csv(run.output / "cases.csv", index=False)  # after each column: a crash keeps the rest
            print(f"  {variant} | {name}: done", flush=True)
    print(f"Saved: {run.output / 'cases.csv'}")


if __name__ == "__main__":
    main()
