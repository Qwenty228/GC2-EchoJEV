"""Evaluate local LAYA decision policy on pre-computed detector scores.

Allows running comparative studies immediately on existing A2 and D2 evaluation runs
without needing raw audio files present.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

from audio_df.modules.jev import JEVState
from audio_df.modules.laya import decide


def find_latest_run(runs_dir: Path, arch: str) -> Path:
    matches = sorted(
        [p for p in runs_dir.glob(f"*-{arch}") if p.is_dir() and (p / "scores").is_dir()]
    )
    if not matches:
        raise FileNotFoundError(f"No completed run directory found for architecture {arch}")
    return matches[-1]


def run_laya_on_score_file(
    scores_path: Path,
    threshold: float,
    split_name: str,
    output_dir: Path,
    limit: int = 100,
    model: str = "convaiinnovations/laya",
) -> dict[str, object]:
    df = pd.read_csv(scores_path)
    total_len = len(df)
    if limit > 0 and limit < total_len:
        # Sample deterministically across labels
        df = df.sample(n=limit, random_state=1337).reset_index(drop=True)

    records = []
    total_latency = 0.0

    questions = {
        "decision": {
            "type": "choice",
            "instructions": "Choose the audio verdict supported by the detector evidence.",
            "criteria": {
                "bonafide": "Evidence favors authentic speech.",
                "spoof": "Evidence favors spoofed speech.",
                "uncertain": "Evidence is insufficient for a reliable verdict.",
            },
        },
        "evidence_quality": {
            "type": "score",
            "instructions": "How sufficient is the supplied evidence?",
            "criteria": ["Insufficient", "Limited", "Strong"],
        },
        "ambiguity": {
            "type": "noul",
            "instructions": "Is the supplied evidence ambiguous?",
        },
    }

    print(f"Evaluating LAYA on {split_name} ({len(df)} clips)...", flush=True)

    from audio_df.modules.laya import get_laya_agent
    agent = get_laya_agent(model)

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=split_name):
        score = float(row.score)
        signed_margin = score - threshold
        base_label = "bonafide" if score >= threshold else "spoof"

        state_payload = {
            "context": {
                "base_score": score,
                "dev_threshold": threshold,
                "base_verdict": base_label,
                "signed_margin": signed_margin,
                "score_direction": "higher means bonafide",
            },
            "decision_guidance": (
                "Decide whether the supplied detector evidence supports bona fide or spoof. "
                "Only one detector score is provided; do not invent acoustic evidence. "
                "Abstain when insufficient."
            ),
        }

        start = time.monotonic()
        resp = agent.system_one(state_payload, questions)
        latency = time.monotonic() - start
        total_latency += latency

        answer = resp.get("answers", {}).get("decision", {})
        verdict = answer.get("choice", "uncertain")

        records.append({
            "clip_id": str(row.clip_id),
            "label": str(row.label),
            "score": score,
            "base": base_label,
            "jev": verdict,
            "latency_s": latency,
        })

    paired = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / f"paired_{split_name}.csv", index=False)

    decided = paired.jev != "uncertain"
    base_ok = paired.base == paired.label
    policy_ok = paired.jev == paired.label

    return {
        "n": len(paired),
        "base_accuracy": float(base_ok.mean()),
        "coverage": float(decided.mean()),
        "policy_accuracy_on_decided": float(policy_ok[decided].mean()) if decided.any() else None,
        "corrected_errors": int(((~base_ok) & policy_ok & decided).sum()),
        "introduced_errors": int((base_ok & (~policy_ok) & decided).sum()),
        "abstentions": int((~decided).sum()),
        "request_latency_s": total_latency,
        "note": "Evaluated using local Laya System 1 decision engine.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local LAYA policy on pre-scored runs.")
    parser.add_argument(
        "--base",
        "--arch",
        dest="base",
        choices=["a2", "d2"],
        default="a2",
        help="Base detector (a2 or d2)",
    )
    parser.add_argument(
        "--splits", default="eval,itw", help="Comma-separated split names to evaluate"
    )
    parser.add_argument(
        "--limit", type=int, default=50, help="Clips per split to evaluate (0 for full)"
    )
    parser.add_argument("--model", default="convaiinnovations/laya", help="Laya model or 'dry-run'")
    args = parser.parse_args()

    runs_dir = Path("runs")
    source_run = find_latest_run(runs_dir, args.base)
    print(f"Loading scores from latest {args.base.upper()} run: {source_run.name}")

    with (source_run / "metrics.json").open(encoding="utf-8") as f:
        source_metrics = json.load(f)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    policy_arch = f"{args.base}-laya" if args.base == "a2" else "fp2"
    output_dir = runs_dir / f"E-{timestamp}-{policy_arch}"

    all_metrics = {}
    for split in splits:
        score_file = source_run / "scores" / f"{split}.csv"
        if not score_file.exists():
            print(f"Warning: score file {score_file} not found; skipping.")
            continue

        split_info = source_metrics.get(split, {})
        threshold = float(split_info.get("threshold", 0.0))

        report = run_laya_on_score_file(
            scores_path=score_file,
            threshold=threshold,
            split_name=split,
            output_dir=output_dir,
            limit=args.limit,
            model=args.model,
        )
        all_metrics[split] = report

    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "architecture": policy_arch,
                "base": args.base,
                "source_run": source_run.name,
                "jev_model_requested": args.model,
                "testsets": splits,
                "limit": args.limit,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(f"\n[DONE] Laya evaluation complete. Output saved to: {output_dir}")
    print("Run 'uv run compare' to view comparative study tables and figures.")


if __name__ == "__main__":
    main()
