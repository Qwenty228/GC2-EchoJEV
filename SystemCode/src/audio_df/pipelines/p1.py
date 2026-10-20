"""P1 standalone TypeSafe/JEV API demo. NOT an audio benchmark pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from audio_df.modules.jev import decide

WARNING = (
    "WARNING: P1 standalone JEV demonstration. This is an API integration test, "
    "NOT a benchmarkable audio deepfake detection pipeline. "
    "No detector, training, or evaluation metrics are involved."
)

SCENARIOS: dict[str, dict[str, Any]] = {
    "launch": {
        "decision": "Launch a public-facing service today?",
        "deadline": "today",
        "security_review": "not completed",
        "known_vulnerability": "high severity, unresolved",
        "customer_commitment": "launch announced",
        "alternatives": ["proceed", "delay", "reject"],
    },
    "research": {
        "decision": "Publish an experimental result as definitive?",
        "observations": "promising result on one dataset",
        "independent_replication": "not performed",
        "external_validation": "pending",
        "alternatives": ["proceed", "delay", "reject"],
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone JEV decision demo (not benchmarkable)")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="launch")
    parser.add_argument("--dry-run", action="store_true", help="Print payload without calling API")
    args = parser.parse_args()
    print(WARNING, file=sys.stderr, flush=True)
    state = SCENARIOS[args.scenario]
    print("State:\n" + json.dumps(state, indent=2), flush=True)
    if args.dry_run:
        return
    try:
        response = decide(state)
    except Exception as exc:
        print(f"JEV request failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if hasattr(response, "model_dump"):
        result = response.model_dump(mode="json")
    elif hasattr(response, "to_dict"):
        result = response.to_dict()
    else:
        result = str(response)
    print("JEV response:\n" + json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
