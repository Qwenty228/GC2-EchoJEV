"""Build the web app's leaderboard file (rankings.json) from runs/.

Run on the research machine after new evaluations, then copy the file into the deployed volume:

    uv run make-rankings        # writes <ADF_ARTIFACTS or artifacts>/rankings.json

Every number comes from a run's score files, and each row records its run and clip count.
Detectors: the newest trimmed-trained artifact per architecture, scored at its own dev EER
threshold (bona fide when score >= threshold, as in the app). Arbitration: every
runs/E-*-arbitration result.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from audio_df.helpers.artifacts import artifacts_dir
from audio_df.helpers.metrics import compute_eer
from audio_df.helpers.research.runs import collect

DETECTORS = {
    "a1": "A1 · LFCC-GMM",
    "a2": "A2 · WavLM Base+",
    "a4": "A4 · LFCC-MLP",
    "d1": "D1 · LFCC + WavLM, concatenated",
    "d2": "D2 · LFCC → WavLM cross-attention",
    "d3": "D3 · LFCC ↔ WavLM cross-attention",
}
FINAL = "d2"
TESTSETS = {  # id: (label, short tab label, group); "overall" is the mean EER over all of them
    "eval": ("ASVspoof 2019 LA eval", "2019 eval", "In-domain"),
    "eval_noise_10db": ("2019 eval + MUSAN noise 10 dB", "Noise", "Degraded"),
    "eval_mp3_32k": ("2019 eval as MP3 32 kbps", "MP3", "Degraded"),
    "eval_opus_16k": ("2019 eval as Opus 16 kbps", "Opus", "Degraded"),
    "eval_resample_8k": ("2019 eval resampled to 8 kHz", "8 kHz", "Degraded"),
    "itw": ("In-the-Wild", "In-the-Wild", "Real-world"),
    "asvspoof2021_eval": ("ASVspoof 2021 LA eval", "2021 LA", "Unseen attacks"),
    "asvspoof2021_hidden": ("ASVspoof 2021 LA hidden", "2021 LA hidden", "Unseen attacks"),
    "asvspoof2021df_eval": ("ASVspoof 2021 DF eval", "2021 DF", "Unseen attacks"),
    "asvspoof2021df_hidden": ("ASVspoof 2021 DF hidden", "2021 DF hidden", "Unseen attacks"),
}


def score_at_threshold(scores: pd.DataFrame, threshold: float) -> dict[str, float]:
    """EER plus what the app would decide at `threshold` (percentages, rounded)."""
    spoof = scores.label.eq("spoof").to_numpy()
    called_spoof = scores.score.to_numpy() < threshold
    eer = compute_eer(scores.score, scores.label, higher_is_spoof=False)[0]
    return {
        "clips": int(len(scores)),
        "eer": round(100 * eer, 2),
        "missed_spoofs": round(100 * (spoof & ~called_spoof).sum() / max(spoof.sum(), 1), 2),
        "false_alarms": round(100 * (~spoof & called_spoof).sum() / max((~spoof).sum(), 1), 2),
        "accuracy": round(100 * (called_spoof == spoof).mean(), 2),
    }


def detector_rows(runs_dir: Path) -> list[dict]:
    rows = []
    results = collect(runs_dir, list(DETECTORS))
    for system, name in DETECTORS.items():
        mine = {k: v for k, v in results.items() if k[0].split()[0] == system and k[1] == "trimmed"}
        if not mine:
            continue
        entry = max(mine.values(), key=lambda v: v["artifact"])  # artifact names are timestamps
        scores = entry["scores"]
        dev = scores["dev"]
        threshold = compute_eer(dev.score, dev.label, higher_is_spoof=False)[1]
        metrics = {t: score_at_threshold(scores[t], threshold) for t in TESTSETS if t in scores}
        complete = len(metrics) == len(TESTSETS)
        rows.append({
            "id": system, "name": name, "final": system == FINAL, "artifact": entry["artifact"],
            "runs": entry["runs"], "threshold": round(float(threshold), 6), "metrics": metrics,
            "overall_eer": round(sum(m["eer"] for m in metrics.values()) / len(metrics), 2)
            if complete else None,
        })
    return rows


def arbitration_rows(runs_dir: Path) -> list[dict]:
    rows = []
    for run in sorted(runs_dir.glob("E-*-arbitration")):
        cases = pd.read_csv(run / "cases.csv")
        meta_path = run / "judge.json"
        # Runs made before judge.json was written all used LAYA.
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"judge": "laya"}
        detectors = [c[2:] for c in cases.columns if c.startswith("v_")]
        label = cases.label.to_numpy()
        best = max(detectors, key=lambda m: (cases[f"v_{m}"] == cases.label).mean())
        best_acc = round(100 * (cases[f"v_{best}"] == cases.label).mean(), 2)
        for column in [c for c in cases.columns if " | " in c]:
            request, order = column.split(" | ")
            pred = cases[column].to_numpy()
            decided = pred != "uncertain"
            rows.append({
                "run": run.name, "judge": meta["judge"], "testset": "itw", "clips": int(len(cases)),
                "detectors": detectors, "request": request, "order": order,
                "decided": round(100 * decided.mean(), 2),
                "accuracy_on_decided": round(100 * ((pred == label) & decided).sum()
                                             / max(decided.sum(), 1), 2),
                "best_single": {"id": best, "accuracy": best_acc},
            })
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".", help="project root containing runs/ (default: .)")
    parser.add_argument("--out", default=None,
                        help="output file (default: <artifacts>/rankings.json)")
    args = parser.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out) if args.out else artifacts_dir(root) / "rankings.json"
    data = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "final": FINAL,
        "testsets": [{"id": k, "label": v[0], "short": v[1], "group": v[2]}
                     for k, v in TESTSETS.items()],
        "detectors": detector_rows(root / "runs"),
        "arbitration": arbitration_rows(root / "runs"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}: {len(data['detectors'])} detectors, "
          f"{len(data['arbitration'])} arbitration rows")


if __name__ == "__main__":
    main()
