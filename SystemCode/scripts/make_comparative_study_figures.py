"""Generate publication-ready figures for Option 1 and Option 2 comparative study.

Ensures all axes and titles explicitly use 'JEV/LAYA' throughout.
Produces:
1. jev_laya_comparative_benchmark.png: Performance of Base vs Option 1 vs Option 2 vs Combined.
2. jev_laya_accuracy_coverage_tradeoff.png: Risk-coverage curve across tau thresholds.
3. jev_laya_arbitration_analysis.png: Multi-detector disagreement resolution & margin dominance.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set high-quality styling
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_comparative_benchmark(out_dir: Path) -> None:
    """Benchmark comparison across all 5 configurations on Eval and ITW."""
    # Data gathered from runs
    data = [
        {"config": "Base A2 (Naive)", "split": "Eval", "base_acc": 85.0, "pol_acc": 85.0, "cov": 100.0, "gain": 0},
        {"config": "Base A2 (Naive)", "split": "ITW", "base_acc": 68.0, "pol_acc": 68.0, "cov": 100.0, "gain": 0},
        {"config": "Base D2 (Naive)", "split": "Eval", "base_acc": 87.0, "pol_acc": 87.0, "cov": 100.0, "gain": 0},
        {"config": "Base D2 (Naive)", "split": "ITW", "base_acc": 56.0, "pol_acc": 56.0, "cov": 100.0, "gain": 0},
        {"config": "Opt 1: Abstention (tau=3.0)", "split": "Eval", "base_acc": 85.0, "pol_acc": 90.3, "cov": 93.0, "gain": 5.3},
        {"config": "Opt 1: Abstention (tau=3.0)", "split": "ITW", "base_acc": 68.0, "pol_acc": 73.0, "cov": 89.0, "gain": 5.0},
        {"config": "Opt 2: Pure Arbitration", "split": "Eval", "base_acc": 85.0, "pol_acc": 90.0, "cov": 100.0, "gain": 5.0},
        {"config": "Opt 2: Pure Arbitration", "split": "ITW", "base_acc": 68.0, "pol_acc": 67.0, "cov": 100.0, "gain": -1.0},
        {"config": "Combined: Opt 1+2 (tau=0.25)", "split": "Eval", "base_acc": 85.0, "pol_acc": 90.6, "cov": 96.0, "gain": 5.6},
        {"config": "Combined: Opt 1+2 (tau=0.25)", "split": "ITW", "base_acc": 68.0, "pol_acc": 64.9, "cov": 94.0, "gain": -3.1},
    ]
    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Panel A: ASVspoof 2019 Eval (In-Domain)
    df_eval = df[df.split == "Eval"].reset_index(drop=True)
    x = np.arange(len(df_eval))
    width = 0.28

    axes[0].bar(x - width, df_eval["base_acc"], width=width, label="Base Detector Accuracy (%)", color="#6c5ce7", alpha=0.9, edgecolor="#4834d4")
    axes[0].bar(x, df_eval["pol_acc"], width=width, label="JEV/LAYA Policy Accuracy (%)", color="#00b894", alpha=0.9, edgecolor="#009472")
    axes[0].bar(x + width, df_eval["cov"], width=width, label="JEV/LAYA Decision Coverage (%)", color="#f39c12", alpha=0.9, edgecolor="#d35400")

    axes[0].set_title("(A) ASVspoof 2019 Evaluation (In-Domain Benchmark)", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xlabel("Decision Policy Configuration (Evaluated by JEV/LAYA)", fontsize=11, fontweight="bold", labelpad=8)
    axes[0].set_ylabel("JEV/LAYA Metric Score (%) [Accuracy & Coverage]", fontsize=11, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_eval["config"], rotation=25, ha="right", fontsize=9.5)
    axes[0].set_ylim(0, 108)
    axes[0].legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Values on bars
    for i in x:
        axes[0].text(i - width, df_eval["base_acc"][i] + 1.2, f"{df_eval['base_acc'][i]:.1f}%", ha="center", va="bottom", fontsize=8)
        axes[0].text(i, df_eval["pol_acc"][i] + 1.2, f"{df_eval['pol_acc'][i]:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#007958")
        axes[0].text(i + width, df_eval["cov"][i] + 1.2, f"{df_eval['cov'][i]:.1f}%", ha="center", va="bottom", fontsize=8)

    # Panel B: In-The-Wild (Out-of-Domain)
    df_itw = df[df.split == "ITW"].reset_index(drop=True)
    axes[1].bar(x - width, df_itw["base_acc"], width=width, label="Base Detector Accuracy (%)", color="#6c5ce7", alpha=0.9, edgecolor="#4834d4")
    axes[1].bar(x, df_itw["pol_acc"], width=width, label="JEV/LAYA Policy Accuracy (%)", color="#00b894", alpha=0.9, edgecolor="#009472")
    axes[1].bar(x + width, df_itw["cov"], width=width, label="JEV/LAYA Decision Coverage (%)", color="#f39c12", alpha=0.9, edgecolor="#d35400")

    axes[1].set_title("(B) In-The-Wild (Out-of-Domain Generalization)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xlabel("Decision Policy Configuration (Evaluated by JEV/LAYA)", fontsize=11, fontweight="bold", labelpad=8)
    axes[1].set_ylabel("JEV/LAYA Metric Score (%) [Accuracy & Coverage]", fontsize=11, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_itw["config"], rotation=25, ha="right", fontsize=9.5)
    axes[1].set_ylim(0, 108)
    axes[1].legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for i in x:
        axes[1].text(i - width, df_itw["base_acc"][i] + 1.2, f"{df_itw['base_acc'][i]:.1f}%", ha="center", va="bottom", fontsize=8)
        axes[1].text(i, df_itw["pol_acc"][i] + 1.2, f"{df_itw['pol_acc'][i]:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#007958")
        axes[1].text(i + width, df_itw["cov"][i] + 1.2, f"{df_itw['cov'][i]:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("EchoJEV Policy Comparative Study: Base Detectors vs Option 1 & Option 2 (JEV/LAYA)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "jev_laya_comparative_benchmark.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: jev_laya_comparative_benchmark.png")


def plot_accuracy_coverage_tradeoff(out_dir: Path) -> None:
    """Option 1: Risk-Coverage curve and error clustering in threshold neighborhood."""
    # Compute on full ASVspoof 2019 Eval (71,237 clips)
    a2_path = Path("runs/E-20261023-200932-976565-a2/scores/eval.csv")
    if not a2_path.exists():
        print("A2 eval scores not found, skipping tradeoff plot.")
        return

    df = pd.read_csv(a2_path)
    th = -0.0027
    margin = (df["score"] - th).abs().values
    correct = (np.where(df["score"] >= th, "bonafide", "spoof") == df["label"]).values

    taus = np.linspace(0.0, 12.0, 50)
    accuracies = []
    coverages = []

    for t in taus:
        mask = margin >= t
        if mask.sum() > 0:
            cov = mask.mean() * 100
            acc = correct[mask].mean() * 100
            coverages.append(cov)
            accuracies.append(acc)
        else:
            coverages.append(0)
            accuracies.append(100)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: Accuracy vs Coverage Curve
    axes[0].plot(coverages, accuracies, "o-", color="#00b894", linewidth=2.5, markersize=5, label="JEV/LAYA Selective Accuracy")
    axes[0].axhline(84.98, color="#6c5ce7", linestyle="--", linewidth=1.8, label="Base A2 Detector Accuracy (84.98%)")
    axes[0].scatter([93.0, 85.5, 75.2], [90.3, 91.96, 93.45], color="#e74c3c", s=80, zorder=5, label="Evaluated Policy Operating Points (tau=3, 5, 8)")
    axes[0].annotate("tau=3.0\n(Acc=90.3%, Cov=93%)", xy=(93.0, 90.3), xytext=(80, 88),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="#e74c3c"), fontsize=9, fontweight="bold")
    axes[0].annotate("tau=5.0\n(Acc=92.0%, Cov=85.5%)", xy=(85.5, 91.96), xytext=(70, 93),
                     arrowprops=dict(arrowstyle="->", lw=1.5, color="#e74c3c"), fontsize=9, fontweight="bold")

    axes[0].set_xlabel("JEV/LAYA Decision Coverage (%) [1 - Abstention Rate]", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("JEV/LAYA Selective Accuracy (%) on Decided Clips", fontsize=11, fontweight="bold")
    axes[0].set_title("(A) Option 1: Risk-Coverage Trade-off Curve (ASVspoof 2019 Eval, N=71,237)", fontsize=11, fontweight="bold")
    axes[0].invert_xaxis()  # Standard risk-coverage convention: high coverage to low coverage
    axes[0].legend(loc="lower left", fontsize=9.5, framealpha=0.95)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Panel B: Error Density vs Distance from Operating Threshold
    errors_margin = margin[~correct]
    correct_margin = margin[correct]

    axes[1].hist(errors_margin, bins=35, range=(0, 25), density=True, alpha=0.6, color="#e74c3c", label="Detection Errors (Misclassifications)", edgecolor="#c0392b")
    axes[1].hist(correct_margin, bins=35, range=(0, 25), density=True, alpha=0.5, color="#2ecc71", label="Correct Predictions", edgecolor="#27ae60")
    axes[1].axvline(3.0, color="#f39c12", linestyle="-", linewidth=2.5, label="JEV/LAYA Abstention Cutoff (tau=3.0)")
    axes[1].axvspan(0.0, 3.0, color="#f39c12", alpha=0.18, label="Abstained Uncertainty Zone (Escalated)")

    axes[1].set_xlabel("Absolute Margin |s - theta_dev| (Evaluated by JEV/LAYA)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Probability Density (Evaluated by JEV/LAYA)", fontsize=11, fontweight="bold")
    axes[1].set_title("(B) Error Density Clustering in Threshold Neighborhood", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.95)
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Option 1 Selective Abstention Analysis (Risk Control via Chow's Rule)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "jev_laya_accuracy_coverage_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: jev_laya_accuracy_coverage_tradeoff.png")


def plot_arbitration_analysis(out_dir: Path) -> None:
    """Option 2: Multi-detector disagreement breakdown and margin-based resolution."""
    a2_path = Path("runs/E-20261023-200932-976565-a2/scores/eval.csv")
    d2_path = Path("runs/E-20261024-214524-879357-d2/scores/eval.csv")
    if not a2_path.exists() or not d2_path.exists():
        print("Score files missing for arbitration plot.")
        return

    df_a2 = pd.read_csv(a2_path)
    df_d2 = pd.read_csv(d2_path)
    merged = pd.merge(df_a2, df_d2, on=["clip_id", "label"], suffixes=("_a2", "_d2"))

    th_a2 = -0.0027
    th_d2 = -0.9905
    pred_a2 = np.where(merged["score_a2"] >= th_a2, "bonafide", "spoof")
    pred_d2 = np.where(merged["score_d2"] >= th_d2, "bonafide", "spoof")
    agree = (pred_a2 == pred_d2)

    a2_correct = (pred_a2 == merged["label"])
    d2_correct = (pred_d2 == merged["label"])

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # Panel A: Disagreement Breakdown Bar Chart
    both_correct = int((a2_correct & d2_correct).sum())
    both_wrong = int(((~a2_correct) & (~d2_correct)).sum())
    a2_only = int((a2_correct & (~d2_correct)).sum())
    d2_only = int(((~a2_correct) & d2_correct).sum())

    categories = [
        "Unanimous Agreement\n(Both Correct)",
        "Unanimous Agreement\n(Both Error)",
        "Disagreement\n(Only A2 Correct)",
        "Disagreement\n(Only D2 Correct)",
    ]
    counts = [both_correct, both_wrong, a2_only, d2_only]
    pcts = [c / len(merged) * 100 for c in counts]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

    bars = axes[0].bar(categories, counts, color=colors, alpha=0.9, edgecolor="#2c3e50")
    axes[0].set_ylabel("Utterance Count (ASVspoof 2019 Eval, N=71,237)", fontsize=11, fontweight="bold")
    axes[0].set_title("(A) Inductive Bias Complementarity: A2 (WavLM) vs D2 (LFCC-Fusion)", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    for bar, pct in zip(bars, pcts):
        h = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, h + 800, f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")

    axes[0].set_ylim(0, max(counts) * 1.18)

    # Panel B: Disagreement Resolution Accuracy vs Normalized Margin Gap
    norm_m_a2 = (merged["score_a2"] - th_a2) / merged["score_a2"].std()
    norm_m_d2 = (merged["score_d2"] - th_d2) / merged["score_d2"].std()
    norm_diff = (norm_m_a2.abs() - norm_m_d2.abs()).abs()

    disagreements = merged[~agree].copy()
    disagreements["norm_diff"] = norm_diff[~agree]
    disagreements["arb_correct"] = np.where(
        norm_m_a2[~agree].abs() >= norm_m_d2[~agree].abs(),
        a2_correct[~agree],
        d2_correct[~agree],
    )

    diff_bins = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0]
    bin_labels = [f"[{diff_bins[i]},{diff_bins[i+1]})" for i in range(len(diff_bins)-1)]
    disagreements["bin"] = pd.cut(disagreements["norm_diff"], bins=diff_bins, labels=bin_labels, right=False)

    bin_acc = disagreements.groupby("bin", observed=False)["arb_correct"].mean() * 100
    bin_counts = disagreements.groupby("bin", observed=False)["arb_correct"].count()

    ax2_twin = axes[1].twinx()
    axes[1].bar(bin_labels, bin_acc, color="#00b894", alpha=0.8, edgecolor="#009472", label="JEV/LAYA Arbitration Accuracy (%)")
    ax2_twin.plot(bin_labels, bin_counts, "D--", color="#e74c3c", linewidth=2, markersize=6, label="Utterance Count in Bin")

    axes[1].axhline(50.0, color="#7f8c8d", linestyle=":", linewidth=1.5, label="Random Guess (50%)")
    axes[1].set_xlabel("Normalized Acoustic Margin Gap |Δz| = ||z_1| - |z_2||", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Arbitration Accuracy on Disagreements (%)", fontsize=11, fontweight="bold", color="#007958")
    ax2_twin.set_ylabel("Disagreement Clip Count", fontsize=11, fontweight="bold", color="#c0392b")
    axes[1].set_title("(B) Arbitration Accuracy vs Margin Dominance |Δz|", fontsize=11, fontweight="bold")
    axes[1].set_ylim(40, 75)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    # Combine legends
    lines_1, labels_1 = axes[1].get_legend_handles_labels()
    lines_2, labels_2 = ax2_twin.get_legend_handles_labels()
    axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle("Option 2 Multi-Detector Arbitration Analysis (WavLM vs LFCC Cross-Attention Fusion)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / "jev_laya_arbitration_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: jev_laya_arbitration_analysis.png")


def main() -> None:
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_comparative_benchmark(out_dir)
    plot_accuracy_coverage_tradeoff(out_dir)
    plot_arbitration_analysis(out_dir)
    print("\n[DONE] High-resolution comparative study figures generated in reports/figures/")


if __name__ == "__main__":
    main()
