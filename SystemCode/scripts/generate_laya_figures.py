"""Generate comprehensive publication-grade graphs and diagrams for JEV/LAYA comparative study.

Axes and labels explicitly use 'JEV/LAYA' throughout to reflect the identical
underlying decision policy architecture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set overall styling
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def load_runs_and_paired_data(runs_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    records = []
    paired_dict = {}

    for metrics_file in sorted(runs_dir.glob("*/metrics.json")):
        run_dir = metrics_file.parent
        config_file = run_dir / "config.yaml"
        if not config_file.exists():
            continue

        try:
            with metrics_file.open(encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            continue

        for split, split_data in metrics_data.items():
            if not isinstance(split_data, dict) or "base_accuracy" not in split_data:
                continue

            arch = "unknown"
            jev_model = "laya"
            try:
                import yaml

                with config_file.open(encoding="utf-8") as cf:
                    cfg = yaml.safe_load(cf) or {}
                    arch = cfg.get("architecture", run_dir.name.split("-")[-1])
                    jev_model = cfg.get("jev_model_requested", "laya")
            except Exception:
                pass

            if "arbitration" in arch.lower():
                base_det = "A2+D2 (Opt 2 Arb)"
            elif "m3" in arch.lower() or "abstain" in arch.lower():
                base_det = "A2 (Opt 1 Abstain)"
            elif "a2" in arch.lower() or "a3" in arch.lower():
                base_det = "A2 (Naive)"
            else:
                base_det = "D2 (Naive)"

            records.append({
                "run_id": run_dir.name,
                "architecture": arch,
                "base_detector": base_det,
                "split": split,
                "n_samples": split_data.get("n", 0),
                "base_accuracy": round(split_data.get("base_accuracy", 0.0) * 100, 2),
                "policy_accuracy": round((split_data.get("policy_accuracy_on_decided") or 0.0) * 100, 2),
                "coverage": round(split_data.get("coverage", 0.0) * 100, 2),
                "corrected_errors": split_data.get("corrected_errors", 0),
                "introduced_errors": split_data.get("introduced_errors", 0),
                "abstentions": split_data.get("abstentions", 0),
                "net_gain": split_data.get("corrected_errors", 0) - split_data.get("introduced_errors", 0),
            })

            # Check for paired CSV
            paired_file = run_dir / f"paired_{split}.csv"
            if paired_file.exists():
                try:
                    p_df = pd.read_csv(paired_file)
                    paired_dict[f"{base_det}_{split}"] = p_df
                except Exception:
                    pass

    return pd.DataFrame(records), paired_dict


def plot_performance_comparison(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Clean label format (latest run per configuration)
    summary = df.sort_values("run_id").groupby(["base_detector", "split"], as_index=False).last()

    labels = [f"{r.base_detector} + JEV/LAYA\n({r.split})" for _, r in summary.iterrows()]
    x = np.arange(len(labels))
    width = 0.26

    # 1. Accuracy & Coverage
    axes[0].bar(x - width, summary["base_accuracy"], width=width, label="Base Detector Accuracy (%)", color="#6c5ce7", alpha=0.9, edgecolor="#4834d4")
    axes[0].bar(x, summary["policy_accuracy"], width=width, label="JEV/LAYA Policy Accuracy (%)", color="#00b894", alpha=0.9, edgecolor="#009472")
    axes[0].bar(x + width, summary["coverage"], width=width, label="JEV/LAYA Decision Coverage (%)", color="#fdcb6e", alpha=0.9, edgecolor="#e17055")

    axes[0].set_xlabel("Architecture Configuration (with JEV/LAYA Policy)", fontsize=11, fontweight="bold", labelpad=8)
    axes[0].set_ylabel("JEV/LAYA Metric Score (%) [Accuracy & Coverage]", fontsize=11, fontweight="bold")
    axes[0].set_title("(A) Base Detector vs JEV/LAYA Policy: Accuracy & Coverage", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylim(0, 105)
    axes[0].legend(loc="lower right", framealpha=0.95, fontsize=10)
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    # Add data values on bars
    for i in x:
        axes[0].text(i - width, summary["base_accuracy"].iloc[i] + 1.2, f"{summary['base_accuracy'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=8)
        axes[0].text(i, summary["policy_accuracy"].iloc[i] + 1.2, f"{summary['policy_accuracy'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#007958")
        axes[0].text(i + width, summary["coverage"].iloc[i] + 1.2, f"{summary['coverage'].iloc[i]:.1f}%", ha="center", va="bottom", fontsize=8)

    # 2. Decision Transitions & Error Corrections
    axes[1].bar(x - width/2, summary["corrected_errors"], width=width, label="Errors Corrected by JEV/LAYA", color="#2ed573", alpha=0.9, edgecolor="#20bf6b")
    axes[1].bar(x + width/2, summary["introduced_errors"], width=width, label="Errors Introduced by JEV/LAYA", color="#ff4757", alpha=0.9, edgecolor="#ee5253")
    axes[1].plot(x, summary["net_gain"], "D--", color="#0984e3", label="JEV/LAYA Net Error Reduction (Gain)", linewidth=2.5, markersize=8)

    axes[1].set_xlabel("Architecture Configuration (with JEV/LAYA Policy)", fontsize=11, fontweight="bold", labelpad=8)
    axes[1].set_ylabel("Utterance Count (Evaluated by JEV/LAYA)", fontsize=11, fontweight="bold")
    axes[1].set_title("(B) JEV/LAYA Decision Transitions & Net Error Gain", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylim(-1, 5)
    axes[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axes[1].legend(loc="upper left", framealpha=0.95, fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    for i in x:
        net = summary["net_gain"].iloc[i]
        sign = "+" if net > 0 else ""
        axes[1].text(i, net + 0.35, f"Net: {sign}{int(net)}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#0984e3")

    fig.suptitle("EchoJEV: Speech Deepfake Detection Performance with JEV/LAYA Decision Policy", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_transition_matrices(paired_dict: dict[str, pd.DataFrame], out_path: Path) -> None:
    if not paired_dict:
        return

    n_keys = len(paired_dict)
    if n_keys <= 2:
        nrows, ncols = 1, n_keys
        figsize = (7.5 * ncols, 5.5)
    else:
        nrows = 2
        ncols = (n_keys + 1) // 2
        figsize = (15, 11)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, (name, df_p) in enumerate(paired_dict.items()):
        ax = axes_flat[idx]

        # Categorize base detector state: Correct vs Error for each ground truth
        def categorize(row):
            gt = "Real" if row["label"] == "bonafide" else "Fake"
            corr = "Correct" if row["base"] == row["label"] else "Error"
            return f"{gt} Speech ({corr} by Base)"

        df_p = df_p.copy()
        df_p["Base_Category"] = df_p.apply(categorize, axis=1)

        # Cross tabulate with JEV/LAYA verdict
        ct = pd.crosstab(
            df_p["Base_Category"],
            df_p["jev"],
            margins=False,
        )

        for col in ["bonafide", "spoof", "uncertain"]:
            if col not in ct.columns:
                ct[col] = 0
        ct = ct[["bonafide", "spoof", "uncertain"]]

        # Display heatmap
        im = ax.imshow(ct.values, cmap="Purples", aspect="auto")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Utterance Count (Evaluated by JEV/LAYA)", fontsize=9)

        # Labels
        ax.set_xticks(range(3))
        ax.set_xticklabels(["Bona fide\n(Authentic)", "Spoof\n(Synthetic)", "Uncertain\n(Abstained)"], fontsize=10, fontweight="bold")
        ax.set_yticks(range(len(ct.index)))
        ax.set_yticklabels(ct.index, fontsize=10)

        ax.set_xlabel("JEV/LAYA Decision Policy Verdict (Predicted Class)", fontsize=10, fontweight="bold", labelpad=8)
        ax.set_ylabel("Ground Truth & Base Detector State (Input to JEV/LAYA)", fontsize=10, fontweight="bold")
        ax.set_title(f"{name.replace('_', ' ')}: JEV/LAYA Decision Flow Matrix", fontsize=11, fontweight="bold", pad=8)

        # Annotate numbers in heatmap
        for r_i in range(len(ct.index)):
            for c_i in range(3):
                val = ct.values[r_i, c_i]
                pct = (val / len(df_p)) * 100
                color = "white" if val > ct.values.max() / 2 else "#2d3436"
                ax.text(c_i, r_i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    # Hide unused subplots if any
    for j in range(n_keys, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Utterance Decision Flow: Base Detector to JEV/LAYA Policy Verdict", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_margin_abstention_distribution(paired_dict: dict[str, pd.DataFrame], out_path: Path) -> None:
    if not paired_dict:
        return

    n_keys = len(paired_dict)
    if n_keys <= 2:
        nrows, ncols = 1, n_keys
        figsize = (8 * ncols, 5.2)
    else:
        nrows = 2
        ncols = (n_keys + 1) // 2
        figsize = (15, 10)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, (name, df_p) in enumerate(paired_dict.items()):
        ax = axes_flat[idx]

        # Extract real vs fake scores
        scores = df_p["score"].values
        labels = df_p["label"].values
        jev_verdicts = df_p["jev"].values

        # Separate by class
        real_scores = scores[labels == "bonafide"]
        fake_scores = scores[labels == "spoof"]

        # Histograms
        ax.hist(fake_scores, bins=20, alpha=0.55, color="#e74c3c", label="Spoofed Speech (Fake)", density=True)
        ax.hist(real_scores, bins=20, alpha=0.55, color="#2ecc71", label="Bona fide Speech (Real)", density=True)

        # Highlight uncertainty region where JEV/LAYA abstains
        uncert_scores = scores[jev_verdicts == "uncertain"]
        if len(uncert_scores) > 0:
            u_min, u_max = uncert_scores.min(), uncert_scores.max()
            ax.axvspan(u_min, u_max, color="#f39c12", alpha=0.25, label=f"JEV/LAYA Abstention Band\n[{u_min:.2f}, {u_max:.2f}]")
        else:
            ax.axvline(0.0, color="#2c3e50", linestyle="--", linewidth=1.5, label="Operating Threshold (dev)")

        ax.set_xlabel("Detector Output Score / Margin (JEV/LAYA Domain)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Empirical Probability Density (Evaluated by JEV/LAYA)", fontsize=10, fontweight="bold")
        ax.set_title(f"{name.replace('_', ' ')}: Margin Distribution & JEV/LAYA Escalation Zone", fontsize=11, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
        ax.grid(True, linestyle="--", alpha=0.3)

    for j in range(n_keys, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Score Distribution & JEV/LAYA Decision Operating Regimes", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def plot_architecture_diagram(out_path: Path) -> None:
    """Generate high-resolution diagram of JEV/LAYA System-1 Decision Architecture."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # Define boxes in clean left-to-right processing order
    boxes = [
        {"x": 0.03, "y": 0.33, "w": 0.18, "h": 0.34, "title": "1. Audio Input", "desc": "Raw Speech Utterance\n(16kHz Mono WAV)\n• ASVspoof 2019 LA\n• In-The-Wild (ITW)", "color": "#74b9ff"},
        {"x": 0.26, "y": 0.55, "w": 0.21, "h": 0.35, "title": "2a. Base Detector A2", "desc": "WavLM Base+\nFrozen Transformer\n+ Linear Classifier Head\nLogit score s", "color": "#a29bfe"},
        {"x": 0.26, "y": 0.10, "w": 0.21, "h": 0.35, "title": "2b. Base Detector D2", "desc": "LFCC Spectral Tokens\nCross-Attention Fusion\nwith WavLM Layer-6 Tokens\nFused logit score s", "color": "#81ecec"},
        {"x": 0.51, "y": 0.30, "w": 0.21, "h": 0.40, "title": "3. State Formatter", "desc": "Calculates Margin Δ = s - θ_dev\nExtracts Pitch, Energy, Duration\nConstructs JSON State Schema\nDefines Typed Question", "color": "#ffeaa7"},
        {"x": 0.76, "y": 0.55, "w": 0.21, "h": 0.35, "title": "4. JEV/LAYA Policy", "desc": "System-1 ModernBERT-large\nNon-Autoregressive Decision Policy\nEvaluates: margin, entropy, context\nDirect Fast Latency Forward Pass", "color": "#55efc4"},
        {"x": 0.76, "y": 0.10, "w": 0.21, "h": 0.35, "title": "5. Calibrated Verdicts", "desc": "• Bona fide (Authentic Speech)\n• Spoof (Synthetic Deepfake)\n• Uncertain (Abstain / Escalate\nto human forensic inspection)", "color": "#fab1a0"},
    ]

    for b in boxes:
        rect = mpatches.FancyBboxPatch(
            (b["x"], b["y"]),
            b["w"],
            b["h"],
            boxstyle="round,pad=0.015",
            facecolor=b["color"],
            edgecolor="#2d3436",
            linewidth=1.5,
            alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"] - 0.04, b["title"], ha="center", va="top", fontsize=11, fontweight="bold", color="#2d3436")
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]/2 - 0.02, b["desc"], ha="center", va="center", fontsize=8.5, color="#2d3436", multialignment="center")

    # Connect forward arrows without crossing boxes
    arrow_props = dict(arrowstyle="->,head_width=0.4,head_length=0.7", lw=2, color="#2d3436")
    # 1 -> 2a
    ax.annotate("", xy=(0.26, 0.72), xytext=(0.21, 0.55), arrowprops=arrow_props)
    # 1 -> 2b
    ax.annotate("", xy=(0.26, 0.28), xytext=(0.21, 0.45), arrowprops=arrow_props)
    # 2a -> 3
    ax.annotate("", xy=(0.51, 0.55), xytext=(0.47, 0.68), arrowprops=arrow_props)
    # 2b -> 3
    ax.annotate("", xy=(0.51, 0.45), xytext=(0.47, 0.32), arrowprops=arrow_props)
    # 3 -> 4
    ax.annotate("", xy=(0.76, 0.70), xytext=(0.72, 0.55), arrowprops=arrow_props)
    # 4 -> 5
    ax.annotate("", xy=(0.865, 0.46), xytext=(0.865, 0.55), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("EchoJEV Architecture: Acoustic Base Detectors + JEV/LAYA System-1 Decision Policy", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate JEV/LAYA comparative study graphs.")
    parser.add_argument("--runs", default="runs", help="Runs directory")
    parser.add_argument("--outdir", default="reports/figures", help="Output directory for figures")
    args = parser.parse_args()

    runs_dir = Path(args.runs)
    out_dir = Path(args.outdir)

    df, paired_dict = load_runs_and_paired_data(runs_dir)
    if df.empty:
        print("No JEV/LAYA policy runs found to plot.")
        return

    print(f"Generating JEV/LAYA figures from {len(df)} run splits...")
    plot_performance_comparison(df, out_dir / "jev_laya_performance_comparison.png")
    plot_transition_matrices(paired_dict, out_dir / "jev_laya_transition_matrices.png")
    plot_margin_abstention_distribution(paired_dict, out_dir / "jev_laya_margin_abstention_distribution.png")
    plot_architecture_diagram(out_dir / "jev_laya_architecture_diagram.png")
    print("[DONE] All figures successfully generated.")


if __name__ == "__main__":
    main()
