"""One command line for every system: the config's `system` key picks the pipeline.

    uv run adf --config configs/a1_lfcc_gmm.yaml train --exp e1
    uv run adf --config configs/a2_wavlm_base_plus.yaml prepare --testsets train,dev,eval
    uv run adf --config configs/a2_wavlm_base_plus.yaml train --exp e1
    uv run adf --config configs/a1_lfcc_gmm.yaml score --model-from runs/<raw run id> \\
        --exp e3 --silence-mode raw --testsets itw
    uv run adf --config configs/a3_jev.yaml jev --state numbers --questions noul_only

Commands, and the systems that have them:

| command   | a1 | a2 | a3 | does |
|-----------|----|----|----|------|
| `timing`  | x  | x  |    | time a few clips, extrapolate; builds nothing |
| `prepare` |    | x  |    | build the pooled SSL feature cache |
| `train`   | x  | x  |    | train, then score dev and eval into `runs/<id>/` |
| `score`   | x  | x  |    | score more test sets with a trained run's model |
| `jev`     |    |    | x  | the E5 JEV evaluation on a stratified dev subset |

`uv run adf` is `python -m audio_df.pipelines.run`. Metrics always come afterwards from
`scripts/score_experiment.py`. A new system adds a pipeline module with a `COMMANDS` dict
and one line in `PIPELINES`.
"""

from __future__ import annotations

import argparse
from typing import Any

from audio_df.config import load_config
from audio_df.models.jev.questions import QUESTION_SETS
from audio_df.pipelines import Command, a1, a2, a3

__all__ = ["PIPELINES", "build_parser", "main"]

PIPELINES: dict[str, dict[str, Command]] = {
    "a1": a1.COMMANDS,
    "a2": a2.COMMANDS,
    "a3": a3.COMMANDS,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="adf", description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, help="configs/<system>.yaml")
    parser.add_argument("--resources", default="configs/resources.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("timing", help="time a few clips; builds nothing")
    p.add_argument("--n", type=int, default=None)

    p = sub.add_parser("prepare", help="build a feature cache (a2)")
    p.add_argument("--testsets", required=True)
    p.add_argument("--silence-mode", default="raw", choices=["raw", "trimmed"])
    p.add_argument("--resume", action="store_true")

    p = sub.add_parser("train", help="train, then score dev and eval")
    p.add_argument("--exp", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tag", default=None, help="suffix for the run id (a1)")
    p.add_argument("--seed", type=int, default=None, help="override the config seed (a1)")

    p = sub.add_parser("score", help="score test sets with a trained run's model")
    p.add_argument("--model-from", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--silence-mode", required=True, choices=["raw", "trimmed"])
    p.add_argument("--testsets", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--tag", default=None, help="suffix for the run id (a1)")
    p.add_argument("--seed", type=int, default=None, help="override the config seed (a1)")

    p = sub.add_parser("jev", help="the E5 JEV evaluation (a3)")
    p.add_argument("--state", choices=["numbers", "words"], required=True)
    p.add_argument("--questions", choices=sorted(QUESTION_SETS), required=True)
    p.add_argument("--client", choices=["dry-run", "typesafe"], default=None)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-cost-usd", type=float, default=0.50)

    return parser


def resolve(args: argparse.Namespace) -> Command:
    """The command function for `args.command` under the config's system, or a clear error."""
    system = load_config(args.config).get("system")
    if system not in PIPELINES:
        raise SystemExit(f"{args.config}: unknown system {system!r}; known: {sorted(PIPELINES)}")
    commands = PIPELINES[system]
    if args.command not in commands:
        raise SystemExit(
            f"system {system!r} has no {args.command!r} command; it has: {sorted(commands)}"
        )
    if system != "a1" and (getattr(args, "tag", None) or getattr(args, "seed", None)):
        raise SystemExit("--tag and --seed are only supported for a1 runs")
    return commands[args.command]


def main(argv: list[str] | None = None) -> Any:
    args = build_parser().parse_args(argv)
    return resolve(args)(args)


if __name__ == "__main__":
    main()
