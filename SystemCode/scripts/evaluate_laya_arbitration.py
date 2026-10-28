"""Multi-detector arbitration and selective abstention evaluation using JEV/LAYA.

Demonstrates:
1. Option 1: Selective Abstention (Risk-Coverage trade-off).
2. Option 2: Multi-Detector Arbitration (A2 WavLM vs D2 Cross-Attention Fusion).
3. Combined Option 1+2: Joint Arbitration with Selective Abstention.
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def find_latest_run(runs_dir: Path, arch_substring: str) -> Path:
    candidates = []
    for p in runs_dir.glob("E-*"):
        if not p.is_dir() or not (p / "scores").is_dir():
            continue
        cfg = p / "config.yaml"
        if cfg.exists():
            try:
                with cfg.open(encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    arch = str(data.get("architecture", "")).lower()
                    if arch == arch_substring.lower() or f"-{arch_substring.lower()}" in p.name.lower():
                        candidates.append((p.stat().st_mtime, p))
            except Exception:
                pass
        elif p.name.lower().endswith(f"-{arch_substring.lower()}"):
            candidates.append((p.stat().st_mtime, p))

    if not candidates:
        raise FileNotFoundError(f"No existing run with scores/ found for architecture: {arch_substring}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def run_arbitration_on_split(
    a2_scores_path: Path,
    d2_scores_path: Path,
    th_a2: float,
    th_d2: float,
    split_name: str,
    output_dir: Path,
    limit: int = 100,
    model: str = "convaiinnovations/laya",
    abstain_margin: float = 0.25,
) -> dict[str, Any]:
    df_a2 = pd.read_csv(a2_scores_path)
    df_d2 = pd.read_csv(d2_scores_path)

    merged = pd.merge(df_a2, df_d2, on=["clip_id", "label"], suffixes=("_a2", "_d2"))
    if limit > 0 and limit < len(merged):
        merged = merged.sample(n=limit, random_state=1337).reset_index(drop=True)

    merged["margin_a2"] = merged["score_a2"] - th_a2
    merged["margin_d2"] = merged["score_d2"] - th_d2

    std_a2 = float(merged["margin_a2"].std()) or 1.0
    std_d2 = float(merged["margin_d2"].std()) or 1.0

    merged["norm_m_a2"] = merged["margin_a2"] / std_a2
    merged["norm_m_d2"] = merged["margin_d2"] / std_d2

    merged["pred_a2"] = np.where(merged["margin_a2"] >= 0, "bonafide", "spoof")
    merged["pred_d2"] = np.where(merged["margin_d2"] >= 0, "bonafide", "spoof")

    print(f"\nEvaluating JEV/LAYA Multi-Detector Arbitration on {split_name} ({len(merged)} clips)...", flush=True)

    from audio_df.modules.laya import get_laya_agent
    agent = get_laya_agent(model)

    records = []
    total_latency = 0.0

    questions = {
        "decision": {
            "type": "choice",
            "instructions": "Arbitrate between the two speech deepfake detectors.",
            "criteria": {
                "bonafide": "Evidence favors authentic, human-produced speech.",
                "spoof": "Evidence favors synthetic, cloned, or spoofed speech.",
                "uncertain": "Detectors conflict and margin evidence is insufficient or ambiguous.",
            },
        },
        "dominant_detector": {
            "type": "choice",
            "instructions": "Which detector provides stronger margin evidence?",
            "criteria": {
                "a2": "WavLM Base+ evidence is stronger.",
                "d2": "Cross-Attention Fusion evidence is stronger.",
                "tie": "Neither detector has a decisive margin.",
            },
        },
    }

    for row in tqdm(merged.itertuples(index=False), total=len(merged), desc=split_name):
        agree = (row.pred_a2 == row.pred_d2)
        norm_diff = abs(abs(row.norm_m_a2) - abs(row.norm_m_d2))

        # Check if ambiguous / abstention applies
        is_ambiguous = (abstain_margin > 0.0) and (
            (not agree and norm_diff < abstain_margin) or (
                agree and abs(row.norm_m_a2) < 0.10 and abs(row.norm_m_d2) < 0.10
            )
        )

        state_payload = {
            "context": {
                "detector_1": "A2 (WavLM Base+)",
                "d1_verdict": row.pred_a2,
                "d1_normalized_margin": float(row.norm_m_a2),
                "detector_2": "D2 (Cross-Attention Fusion)",
                "d2_verdict": row.pred_d2,
                "d2_normalized_margin": float(row.norm_m_d2),
                "unanimous_agreement": bool(agree),
                "margin_difference": float(norm_diff),
                "abstention_threshold": abstain_margin,
            },
            "decision_guidance": (
                "You are the JEV/LAYA System-1 Decision Arbitrator. "
                "When both detectors agree, affirm their unanimous finding. "
                "When detectors disagree, favor the detector with the decisively higher normalized margin. "
                "If the margin difference is narrower than the abstention threshold, output 'uncertain' to escalate."
            ),
        }

        start = time.monotonic()
        if is_ambiguous:
            latency = 0.001
            verdict = "uncertain"
            dominant = "tie"
        elif agree:
            latency = 0.001
            verdict = row.pred_a2
            dominant = "a2" if abs(row.norm_m_a2) >= abs(row.norm_m_d2) else "d2"
        else:
            # Send disagreement to LAYA System-1 engine
            resp = agent.system_one(state_payload, questions)
            latency = time.monotonic() - start
            answer = resp.get("answers", {}).get("decision", {})
            verdict = answer.get("choice", "uncertain")
            if verdict == "uncertain":
                # Fallback to stronger margin if Laya defaulted
                verdict = row.pred_a2 if abs(row.norm_m_a2) >= abs(row.norm_m_d2) else row.pred_d2
            dominant = resp.get("answers", {}).get("dominant_detector", {}).get("choice", "tie")

        total_latency += latency

        records.append({
            "clip_id": str(row.clip_id),
            "label": str(row.label),
            "score": float(row.score_a2),
            "score_a2": float(row.score_a2),
            "score_d2": float(row.score_d2),
            "base": row.pred_a2,
            "pred_a2": row.pred_a2,
            "pred_d2": row.pred_d2,
            "jev": verdict,
            "agree": agree,
            "dominant": dominant,
            "latency_s": latency,
        })

    paired = pd.DataFrame(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(output_dir / f"paired_{split_name}.csv", index=False)

    decided = paired.jev != "uncertain"
    a2_ok = paired.pred_a2 == paired.label
    d2_ok = paired.pred_d2 == paired.label
    policy_ok = paired.jev == paired.label

    # Corrected vs Introduced relative to A2 baseline
    corrected_a2 = int(((~a2_ok) & policy_ok & decided).sum())
    introduced_a2 = int((a2_ok & (~policy_ok) & decided).sum())

    # Corrected vs Introduced relative to D2 baseline
    corrected_d2 = int(((~d2_ok) & policy_ok & decided).sum())
    introduced_d2 = int((d2_ok & (~policy_ok) & decided).sum())

    disagreed_count = int((~paired.agree).sum())

    return {
        "n": len(paired),
        "disagreed_clips": disagreed_count,
        "agreement_rate": float(paired.agree.mean()),
        "base_accuracy_a2": float(a2_ok.mean()),
        "base_accuracy_d2": float(d2_ok.mean()),
        "base_accuracy": float(max(a2_ok.mean(), d2_ok.mean())),
        "coverage": float(decided.mean()),
        "policy_accuracy_on_decided": float(policy_ok[decided].mean()) if decided.any() else None,
        "corrected_errors": corrected_a2,
        "introduced_errors": introduced_a2,
        "corrected_vs_d2": corrected_d2,
        "introduced_vs_d2": introduced_d2,
        "net_gain_vs_a2": corrected_a2 - introduced_a2,
        "net_gain_vs_d2": corrected_d2 - introduced_d2,
        "abstentions": int((~decided).sum()),
        "request_latency_s": total_latency,
        "note": "Evaluated using JEV/LAYA Multi-Detector Arbitration & Selective Abstention.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JEV/LAYA Multi-Detector Arbitration & Selective Abstention.")
    parser.add_argument("--splits", default="eval,itw", help="Comma-separated splits to evaluate")
    parser.add_argument("--limit", type=int, default=100, help="Clips per split (0 for full)")
    parser.add_argument("--abstain-margin", type=float, default=0.25, help="Normalized margin threshold for selective abstention")
    parser.add_argument("--model", default="convaiinnovations/laya", help="Laya model or 'dry-run'")
    args = parser.parse_args()

    runs_dir = Path("runs")
    a2_run = find_latest_run(runs_dir, "a2")
    d2_run = find_latest_run(runs_dir, "d2")

    print(f"Loaded A2 Source Run: {a2_run.name}")
    print(f"Loaded D2 Source Run: {d2_run.name}")

    with (a2_run / "metrics.json").open(encoding="utf-8") as f:
        a2_metrics = json.load(f)
    with (d2_run / "metrics.json").open(encoding="utf-8") as f:
        d2_metrics = json.load(f)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    policy_arch = f"arbitration-m{args.abstain_margin}"
    output_dir = runs_dir / f"E-{timestamp}-{policy_arch}"

    all_metrics = {}
    for split in splits:
        score_a2 = a2_run / "scores" / f"{split}.csv"
        score_d2 = d2_run / "scores" / f"{split}.csv"

        if not score_a2.exists() or not score_d2.exists():
            print(f"Skipping {split}: score files not found in both runs.")
            continue

        th_a2 = float(a2_metrics.get(split, {}).get("threshold", -0.0027))
        th_d2 = float(d2_metrics.get(split, {}).get("threshold", -0.9905))

        report = run_arbitration_on_split(
            a2_scores_path=score_a2,
            d2_scores_path=score_d2,
            th_a2=th_a2,
            th_d2=th_d2,
            split_name=split,
            output_dir=output_dir,
            limit=args.limit,
            model=args.model,
            abstain_margin=args.abstain_margin,
        )
        all_metrics[split] = report

    (output_dir / "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (output_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "architecture": policy_arch,
                "base": "A2+D2 Joint",
                "source_runs": [a2_run.name, d2_run.name],
                "jev_model_requested": args.model,
                "abstain_margin": args.abstain_margin,
                "testsets": splits,
                "limit": args.limit,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("JEV/LAYA MULTI-DETECTOR ARBITRATION & SELECTIVE ABSTENTION REPORT")
    print("=" * 80)
    for split, m in all_metrics.items():
        print(f"Split: {split.upper()} (N={m['n']})")
        print(f"  • Base A2 Accuracy:             {m['base_accuracy_a2']*100:.2f}%")
        print(f"  • Base D2 Accuracy:             {m['base_accuracy_d2']*100:.2f}%")
        print(f"  • Detectors Agreement Rate:     {m['agreement_rate']*100:.2f}% ({m['disagreed_clips']} disagreements)")
        print(f"  • JEV/LAYA Decision Coverage:   {m['coverage']*100:.2f}% ({m['abstentions']} abstentions escalated)")
        print(f"  • JEV/LAYA Policy Accuracy:     {m['policy_accuracy_on_decided']*100:.2f}%")
        gain_a2 = (m['policy_accuracy_on_decided'] - m['base_accuracy_a2']) * 100
        gain_d2 = (m['policy_accuracy_on_decided'] - m['base_accuracy_d2']) * 100
        print(f"  • Accuracy Gain over A2:        {gain_a2:+.2f} percentage points")
        print(f"  • Accuracy Gain over D2:        {gain_d2:+.2f} percentage points")
        print("-" * 80)
    print(f"\n[DONE] Arbitration run saved to: {output_dir}")


if __name__ == "__main__":
    main()
