"""CLI: score one experiment directory and update its registry row.

    uv run python scripts/score_experiment.py runs/<id> \\
        [--threshold-from runs/<id2>] \\
        [--manifest data/manifests/eval.csv --manifest data/manifests/dev.csv] \\
        [--scoring-config configs/scoring.yaml] \\
        [--registry-root experiments]

Reads every `scores/*.csv` under the given experiment directory, writes
`metrics.json` next to it (via `audio_df.scoring.score_experiment`), prints a
compact per-testset table, and updates that experiment's row in
`<registry-root>/REGISTRY.md` with a short "key EERs" summary. Never computes a
metric itself — all numbers come from `audio_df.scoring`/`audio_df.metrics`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_df.exp_io import registry_update
from audio_df.scoring import score_experiment


def _build_key_eers_summary(result: dict) -> str:
    parts = []
    for name in sorted(result["testsets"]):
        eer = result["testsets"][name]["eer"]
        parts.append(f"{name}={eer * 100:.2f}")
    return "; ".join(parts)


def _print_table(result: dict) -> None:
    cols = ["testset", "n", "eer", "min_dcf", "roc_auc", "f1"]
    rows = []
    for name in sorted(result["testsets"]):
        t = result["testsets"][name]
        rows.append(
            [
                name,
                str(t["n"]),
                f"{t['eer']:.4f}",
                f"{t['min_dcf']:.4f}",
                f"{t['roc_auc']:.4f}",
                "n/a" if t["f1"] is None else f"{t['f1']:.4f}",
            ]
        )
    widths = [max(len(c), *(len(r[i]) for r in rows)) for i, c in enumerate(cols)]
    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths, strict=True))
    print(header)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp_dir", type=Path, help="path to runs/<id>")
    parser.add_argument("--threshold-from", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=None,
        help="manifest CSV to join attack_id from (repeatable)",
    )
    parser.add_argument("--scoring-config", type=Path, default=Path("configs/scoring.yaml"))
    parser.add_argument("--registry-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    result = score_experiment(
        args.exp_dir,
        scoring_cfg_path=args.scoring_config,
        threshold_from=args.threshold_from,
        manifest_paths=args.manifest,
    )

    print(f"scored {args.exp_dir.name}")
    _print_table(result)

    exp_id = args.exp_dir.name
    registry_path = args.registry_root / "REGISTRY.md"
    if registry_path.exists():
        update_fields = {"key_eers": _build_key_eers_summary(result)}
        if result.get("resources") is not None:
            update_fields["resources"] = result["resources"]
        registry_update(exp_id, root=args.registry_root, **update_fields)
        print(f"updated key EERs in {registry_path}")
        if "resources" in update_fields:
            print(f"updated resources in {registry_path}")
    else:
        print(f"note: {registry_path} does not exist; skipped registry update")


if __name__ == "__main__":
    main()
