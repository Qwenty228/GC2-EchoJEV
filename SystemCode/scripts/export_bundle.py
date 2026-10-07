"""Turn a trained, scored run into a model bundle the application can load.

    uv run python scripts/export_bundle.py runs/E-20261001-e1-a2-raw artifacts/a2-wavlm-base-plus
    uv run python scripts/export_bundle.py runs/E-20261001-e1-a2-raw artifacts/a2-jev \\
        --jev configs/a3_jev.yaml [--jev-margin 2.0] [--state numbers] [--questions noul_only]

The run's `config.yaml` names the system (`a1`, `a2` or `fusion`); its dev EER threshold comes from
its `metrics.json`. `--jev` wraps the detector with `JevEscalation`. Load the result with
`audio_df.inference.load_detector(<bundle dir>)`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml

from audio_df.models.a1_lfcc_gmm import LfccGmmDetector
from audio_df.models.a2_ssl_head import SslHeadDetector
from audio_df.models.base import Detector
from audio_df.models.fusion_detector import FusionDetector
from audio_df.models.jev.wrapper import JevEscalation

_FROM_RUN = {
    "a1": LfccGmmDetector.from_run,
    "a2": SslHeadDetector.from_run,
    "fusion": FusionDetector.from_run,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--jev", type=Path, default=None, help="A3/JEV config to wrap with")
    parser.add_argument("--jev-margin", type=float, default=None)
    parser.add_argument("--state", default="numbers", choices=["numbers", "words"])
    parser.add_argument("--questions", default="noul_only")
    args = parser.parse_args()

    system = yaml.safe_load((args.run_dir / "config.yaml").read_text())["system"]
    if system not in _FROM_RUN:
        raise SystemExit(f"cannot export system {system!r}; known: {sorted(_FROM_RUN)}")
    detector: Detector = _FROM_RUN[system](args.run_dir)
    if args.jev is not None:
        detector = JevEscalation(
            detector,
            yaml.safe_load(args.jev.read_text()),
            state_variant=args.state,
            question_set=args.questions,
            escalate_margin=args.jev_margin,
        )
    if not math.isfinite(detector.threshold):
        raise SystemExit(
            f"{args.run_dir} has no dev threshold yet; score it first: "
            f"uv run python scripts/score_experiment.py {args.run_dir}"
        )
    detector.save(args.out_dir)
    print(f"wrote {args.out_dir} ({detector.system}, threshold {detector.threshold:.4g})")


if __name__ == "__main__":
    main()
