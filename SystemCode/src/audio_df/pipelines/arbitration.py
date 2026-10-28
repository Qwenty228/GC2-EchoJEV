"""Arbitration: LAYA (or JEV) picks a verdict when detectors disagree on a clip.

Evaluation only, no audio is loaded: it reads each detector's saved In-the-Wild scores. The test case is fixed
here: the detector runs (each with the dev EER threshold of its own artifact), the clips (ITW_11249 plus 200
In-the-Wild clips where the detectors do not all agree) and the request variants. The runtime only supplies the
CLI and the run directory; --testsets, --threshold-from and --model-from are ignored.

Writes runs/E-<time>-arbitration/cases.csv (one answer column per variant and order) and metrics.json.
`uv run arbitration` runs LAYA locally for free; `--arbiter jev` calls the paid TypeSafe API.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from audio_df.modules import jev, laya
from audio_df.modules.jev import JEVState
from audio_df.runtime import Run

RUNS = {  # latest trimmed-training run per detector: scores, dev threshold (calibration.json), 2021 LA metrics
    "a1": ("A1 LFCC-GMM", "runs/E-20261023-172139-766981-a1"),
    "a2": ("A2 WavLM", "runs/E-20261023-200932-976565-a2"),
    "a4": ("A4 LFCC-MLP", "runs/E-20261023-194755-458184-a4"),
}
TESTSET, KEEP, N_CASES, SEED = "itw", "ITW_11249", 200, 0
VARIANTS = ["verdicts", "probabilities", "reliability", "reliability order", "combined"]
ORDERS = {"A1 first": ("a1", "a2", "a4"), "A1 last": ("a2", "a4", "a1")}
QUESTION = {"decision": {"type": "choice", "instructions": "Which verdict should be trusted for this clip?",
                         "criteria": {"bonafide": "The trustworthy evidence says authentic speech.",
                                      "spoof": "The trustworthy evidence says spoofed speech.",
                                      "uncertain": "The evidence conflicts too much to decide."}}}
BASE = "Several spoofing detectors judged the same clip and disagree. Decide which verdict to trust."


def load_detectors(root: Path) -> dict[str, dict]:
    """Per detector: name, dev threshold, 2021 LA EER %, and dev/test scores."""
    out = {}
    for m, (name, run) in RUNS.items():
        d = root / run
        dev = pd.read_csv(d / "scores" / "dev.csv")
        out[m] = {"name": name, "run": run,
                  "threshold": json.loads((d / "calibration.json").read_text())["eer_threshold"],
                  "error_rate_pct": round(100 * json.loads((d / "metrics_asvspoof2021_eval.json").read_text())["eer"], 1),
                  "platt": LogisticRegression().fit(dev[["score"]], dev.label == "bonafide"),
                  "test": pd.read_csv(d / "scores" / f"{TESTSET}.csv")}
    return out


def build_cases(det: dict[str, dict]) -> pd.DataFrame:
    """Fixed test case: clips where the detectors do not all agree, with verdicts and calibrated P(bona fide)."""
    df = None
    for m, d in det.items():
        part = d["test"][["clip_id", "label", "score"]].rename(columns={"score": m})
        df = part if df is None else df.merge(part, on=["clip_id", "label"])
    for m, d in det.items():
        df[f"v_{m}"] = np.where(df[m] >= d["threshold"], "bonafide", "spoof")
        df[f"p_{m}"] = d["platt"].predict_proba(df[[m]].rename(columns={m: "score"}))[:, 1]
    split = df[df[[f"v_{m}" for m in det]].nunique(axis=1) > 1]
    cases = pd.concat([split[split.clip_id == KEEP], split[split.clip_id != KEEP].sample(N_CASES, random_state=SEED)],
                      ignore_index=True)
    weight = {m: 1 / d["error_rate_pct"] for m, d in det.items()}
    cases["p_weighted"] = sum(weight[m] * cases[f"p_{m}"] for m in det) / sum(weight.values())
    return cases


def request(r: dict, det: dict[str, dict], variant: str, order=ORDERS["A1 first"]) -> JEVState:
    """The evidence LAYA/JEV sees for one clip. Never includes the clip id or label."""
    if variant == "reliability order":
        ranked = sorted(det, key=lambda m: det[m]["error_rate_pct"])
        ctx = {f"detector_{i + 1}": {"verdict": r[f"v_{m}"], "p_bonafide": round(float(r[f"p_{m}"]), 3)}
               for i, m in enumerate(ranked)}
        return JEVState(context=ctx, prompt=BASE + " Detectors are listed from most to least reliable.", questions=QUESTION)
    if variant == "combined":
        ctx = {"weighted_p_bonafide": round(float(r["p_weighted"]), 3),
               "detectors_saying_bonafide": f"{sum(r[f'v_{m}'] == 'bonafide' for m in det)} of {len(det)}"}
        return JEVState(context=ctx, questions=QUESTION, prompt="Detector evidence for one clip was combined, weighting "
                        "each detector by its reliability. Answer uncertain when weighted_p_bonafide is near 0.5.")
    ctx = {}
    for m in order:
        e = {"verdict": r[f"v_{m}"]}
        if variant != "verdicts":
            e["p_bonafide"] = round(float(r[f"p_{m}"]), 3)
        if variant == "reliability":
            e["error_rate_pct"] = det[m]["error_rate_pct"]
        ctx[det[m]["name"]] = e
    prompt = BASE + (" Trust detectors with a lower error_rate_pct more." if variant == "reliability" else "")
    return JEVState(context=ctx, prompt=prompt, questions=QUESTION)


def choice(response) -> str:
    """The decision choice from a LAYA dict or a JEV SDK response."""
    response = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
    return response["answers"]["decision"]["choice"]


def judge(pred, label) -> dict[str, float]:
    pred, label = np.asarray(pred), np.asarray(label)
    decided = pred != "uncertain"
    return {"coverage": float(decided.mean()),
            "accuracy_on_decided": float(((pred == label) & decided).sum() / max(decided.sum(), 1)),
            "errors": int(((pred != label) & decided).sum())}


@Run.cli(dataset="2019LA")
def main(run: Run, arbiter: str = "laya", limit: int = 0):
    """Arbitration: LAYA (free, local) or JEV (paid) picks a verdict on fixed In-the-Wild disagreement clips."""
    decide = {"laya": laya.decide, "jev": jev.decide}[arbiter]
    det = load_detectors(run.root)
    cases = build_cases(det)
    if limit:
        cases = cases.head(limit)
    for v in VARIANTS:
        for o, order in ORDERS.items():
            cases[f"{v} | {o}"] = [choice(decide(request(r, det, v, order))) for r in cases.to_dict("records")]
            print(f"  {v} | {o}: done", flush=True)
    rules = {**{f"{d['name']} alone": cases[f"v_{m}"] for m, d in det.items()},
             "majority vote": cases[[f"v_{m}" for m in det]].mode(axis=1)[0],
             "reliability-weighted P >= 0.5": np.where(cases.p_weighted >= 0.5, "bonafide", "spoof")}
    rows = {**rules, **{f"{arbiter.upper()} {v} | {o}": cases[f"{v} | {o}"] for v in VARIANTS for o in ORDERS}}
    metrics = {"arbiter": arbiter, "testset": TESTSET, "n_cases": len(cases),
               "detectors": {m: {k: d[k] for k in ("name", "run", "threshold", "error_rate_pct")} for m, d in det.items()},
               "results": {k: judge(v, cases.label) for k, v in rows.items()}}
    cases.to_csv(run.output / "cases.csv", index=False)
    (run.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(pd.DataFrame(metrics["results"]).T.round(3))
    print(f"Saved: {run.output}")
    return run.output


if __name__ == "__main__":
    main()
