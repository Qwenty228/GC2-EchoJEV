"""Compare Base Detectors against Decision Policies (JEV vs LAYA).

Parses runs/ directory and compiles accuracy, coverage, and error-transition metrics.
Outputs a markdown report and saves comparison figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def collect_policy_runs(runs_dir: Path) -> pd.DataFrame:
    records = []
    for metrics_file in runs_dir.glob("*/metrics.json"):
        run_dir = metrics_file.parent
        config_file = run_dir / "config.yaml"
        if not config_file.exists():
            continue

        try:
            with metrics_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for split, split_data in data.items():
            if not isinstance(split_data, dict) or "base_accuracy" not in split_data:
                continue

            # Determine architecture & policy
            arch = "unknown"
            jev_model = "none"
            try:
                import yaml

                with config_file.open(encoding="utf-8") as cf:
                    cfg = yaml.safe_load(cf) or {}
                    arch = cfg.get("architecture", run_dir.name.split("-")[-1])
                    jev_model = cfg.get("jev_model_requested", "unknown")
            except Exception:
                pass

            policy_type = (
                "LAYA" if "laya" in str(jev_model).lower() or "laya" in arch.lower() else "JEV"
            )
            base_detector = "a2" if "a2" in arch.lower() or "a3" in arch.lower() else "d2"

            records.append(
                {
                    "run_id": run_dir.name,
                    "architecture": arch,
                    "base_detector": base_detector.upper(),
                    "policy": policy_type,
                    "model_engine": jev_model,
                    "split": split,
                    "n_samples": split_data.get("n", 0),
                    "base_accuracy": round(split_data.get("base_accuracy", 0.0) * 100, 2),
                    "policy_accuracy": round(
                        (split_data.get("policy_accuracy_on_decided") or 0.0) * 100, 2
                    ),
                    "coverage": round(split_data.get("coverage", 0.0) * 100, 2),
                    "corrected_errors": split_data.get("corrected_errors", 0),
                    "introduced_errors": split_data.get("introduced_errors", 0),
                    "abstentions": split_data.get("abstentions", 0),
                    "net_error_reduction": split_data.get("corrected_errors", 0)
                    - split_data.get("introduced_errors", 0),
                }
            )

    return pd.DataFrame(records)


def plot_comparisons(df: pd.DataFrame, output_path: Path) -> None:
    if df.empty:
        print("No policy runs found to plot.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Accuracy & Coverage
    summary = (
        df.groupby(["base_detector", "policy", "split"])[
            ["base_accuracy", "policy_accuracy", "coverage"]
        ]
        .mean()
        .reset_index()
    )

    labels = [f"{r.base_detector}+{r.policy} ({r.split})" for _, r in summary.iterrows()]
    x = range(len(labels))
    width = 0.28

    axes[0].bar(
        [i - width for i in x],
        summary["base_accuracy"],
        width=width,
        label="Base Detector Acc %",
        color="#7046c5",
    )
    axes[0].bar(
        x, summary["policy_accuracy"], width=width, label="Policy Acc (Decided) %", color="#17654a"
    )
    axes[0].bar(
        [i + width for i in x],
        summary["coverage"],
        width=width,
        label="Coverage %",
        color="#e67e22",
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_title("Accuracy & Coverage Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Net Error Correction
    err_summary = (
        df.groupby(["base_detector", "policy", "split"])[
            ["corrected_errors", "introduced_errors", "net_error_reduction"]
        ]
        .sum()
        .reset_index()
    )

    err_labels = [f"{r.base_detector}+{r.policy} ({r.split})" for _, r in err_summary.iterrows()]
    x_err = range(len(err_labels))
    axes[1].bar(
        [i - width / 2 for i in x_err],
        err_summary["corrected_errors"],
        width=width,
        label="Corrected Errors",
        color="#27ae60",
    )
    axes[1].bar(
        [i + width / 2 for i in x_err],
        err_summary["introduced_errors"],
        width=width,
        label="Introduced Errors",
        color="#c0392b",
    )
    axes[1].plot(
        x_err,
        err_summary["net_error_reduction"],
        "o--",
        color="#2980b9",
        label="Net Error Reduction",
        linewidth=2,
    )
    axes[1].set_xticks(list(x_err))
    axes[1].set_xticklabels(err_labels, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Utterance Count")
    axes[1].set_title("Decision Transitions (Net Gain)")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Comparison figure saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Base Detectors against Decision Policies (JEV vs LAYA)."
    )
    parser.add_argument("--runs", default="runs", help="Directory containing run results")
    parser.add_argument(
        "--plot", default="reports/figures/policy_comparison.png", help="Figure output path"
    )
    args = parser.parse_args()

    df = collect_policy_runs(Path(args.runs))
    if df.empty:
        print("No completed decision policy runs found under runs/ directory yet.")
        print("Run 'uv run a3-laya evaluate' or 'uv run fp2 evaluate' first.")
        return

    print("\n" + "=" * 80)
    print("DECISION POLICY COMPARATIVE STUDY REPORT (Base vs JEV vs LAYA)")
    print("=" * 80)
    display_cols = [
        "run_id",
        "base_detector",
        "policy",
        "split",
        "n_samples",
        "base_accuracy",
        "policy_accuracy",
        "coverage",
        "net_error_reduction",
    ]
    try:
        print(df[display_cols].to_markdown(index=False))
    except Exception:
        print(df[display_cols].to_string(index=False))
    print("=" * 80)

    plot_comparisons(df, Path(args.plot))


if __name__ == "__main__":
    main()
