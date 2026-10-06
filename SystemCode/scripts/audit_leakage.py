#!/usr/bin/env python
"""Run the Stage 0 leakage audit end-to-end and fail loudly if a shortcut is found.

Usage:
    uv run python scripts/audit_leakage.py --config configs/data.yaml

Builds the manifest, extracts nuisance features (duration, sample rate, codec, RMS,
leading/trailing/total silence, source corpus, speaker identity), audits each alone and all
combined, writes `reports/tables/leakage_audit.csv`, and prints the report.

Exit behaviour: this script exits **non-zero (2)** when any feature is flagged, in addition
to printing an unmissable banner. A non-zero exit is chosen (rather than banner-only) because
this is meant to run in CI/pre-training automation, per the handoff's framing that Stage 0
"can invalidate the whole project if it fires" — a warning that only humans reading scrollback
would notice is not adequate for a gate that other scripts/pipelines should be able to check
programmatically (`if audit_leakage.py; then train; fi`). The banner is printed regardless of
whether the process is run interactively or from a script, so both audiences are covered.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import yaml

from audio_df.leakage import run_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data.yaml"),
        help="Path to the dataset config (default: configs/data.yaml)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        report = run_audit(cfg)

    print("Stage 0 leakage audit report:")
    print(report.to_string(index=False))

    leak_warnings = [str(w.message) for w in caught if "LEAKAGE WARNING" in str(w.message)]
    for msg in leak_warnings:
        print(msg)

    flagged = report.loc[report["flagged"], "feature"].tolist()
    if flagged:
        print()
        print("=" * 78)
        print("LEAKAGE WARNING: Stage 0 audit flagged nuisance feature(s):", ", ".join(flagged))
        print("Do not proceed to modelling until this is investigated and fixed.")
        print("=" * 78)
        return 2

    print()
    print("No nuisance feature exceeded the leakage threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
