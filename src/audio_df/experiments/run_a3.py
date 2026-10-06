"""A3 runner: frozen A2 dev scores + the JEV API over a stratified dev subset (E5).

```
uv run python -m audio_df.experiments.run_a3 run \\
    --state numbers|words --questions noul_only|all3 \\
    [--client dry-run|typesafe] [--n 500] [--max-cost-usd 0.5] [--resume]
```

For each clip in the subset: load its A2 dev score, load its audio, build the evidence
state (`audio_df.jev.state.build_state`), ask every question in the chosen set through the
cache (`audio_df.jev.cache.JevCache`), and derive the A3 score from the noul answer
(`audio_df.jev.questions.noul_to_a3_score`, higher = more bona fide). Writes
`scores/jev_dev500.csv`, `scores/a2_dev500.csv` (the matching A2 scores on the same clips,
for the A2-vs-A3 comparison), `answers.csv` (every typed answer), and `cost_summary.json`.

Every run goes through `audio_df.experiments.lifecycle.experiment_run` (resource limits,
seeding, the heavy-job lock, the registry row, `run_summary.json`) — see that module. A
`--client dry-run` run additionally calls `run.note("DRY RUN — stub client, not a result")`
so its registry row is unmistakable.

**Budget guard.** The real per-call cost of the JEV API is unknown until `TypesafeJevClient`
is implemented (it currently raises `NotImplementedError`, see `audio_df.jev.client`). Until
then: `--client typesafe` refuses to run with `--n` above 20, and the loop aborts with a
`RuntimeError` if the running cost total reaches `--max-cost-usd` before finishing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from audio_df.audio import load_audio
from audio_df.data import stratified_subset
from audio_df.exp_io import make_exp_id, write_scores
from audio_df.experiments.lifecycle import RunContext, experiment_run
from audio_df.jev.cache import JevCache
from audio_df.jev.client import (
    DryRunJevClient,
    JevClient,
    JevRequest,
    JevResponse,
    TypesafeJevClient,
)
from audio_df.jev.questions import (
    QUESTION_SETS,
    QUESTION_TEXT_VERSION,
    build_request,
    noul_to_a3_score,
)
from audio_df.jev.state import build_state
from audio_df.resources import memory_status

__all__ = ["run_a3", "build_arg_parser", "main", "resolve_eer_threshold"]

_TYPESAFE_FIRST_RUN_MAX_N = 20


def resolve_eer_threshold(cfg: dict[str, Any], a2_experiment: str | Path) -> tuple[float, dict]:
    """Where the detector EER threshold used to build A3's evidence state comes from.

    `cfg["thresholds"]["detector"]["eer_threshold"]` is an **explicit override only** —
    its documented default is `None`/absent, meaning "read it from the A2 experiment
    instead". When a non-null override is given it wins outright (e.g. for a dry-run/dev
    smoke test that has no real A2 metrics.json yet); otherwise this reads
    `<a2_experiment>/metrics.json` (written by `scripts/score_experiment.py` /
    `audio_df.scoring.score_experiment`), preferring `testsets.dev.eer_threshold` (the A2
    experiment's own dev EER threshold) and falling back to `threshold_source.value` (in
    case, unusually, that metrics.json itself borrowed its threshold from yet another
    experiment via `--threshold-from`).

    Raises `FileNotFoundError`/`ValueError` with an actionable "score the A2 experiment
    first" message rather than silently defaulting to some placeholder threshold — the
    detector-level word buckets (`audio_df.jev.state`) are meaningless relative to the
    wrong threshold, and a silent 0.0 fallback would look like a valid (if unlucky) run.

    Returns `(threshold, source_info)`, where `source_info` is JSON-serialisable and
    meant to be recorded in the experiment's config/notes (never silently discarded).
    """
    override = (cfg.get("thresholds") or {}).get("detector", {}).get("eer_threshold")
    if override is not None:
        return float(override), {"source": "config_override", "value": float(override)}

    metrics_path = Path(a2_experiment) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} does not exist, and configs/a3_jev.yaml has no "
            "thresholds.detector.eer_threshold override. Score the A2 experiment first: "
            f"uv run python scripts/score_experiment.py {a2_experiment}"
        )
    metrics = json.loads(metrics_path.read_text())

    dev_thr = ((metrics.get("testsets") or {}).get("dev") or {}).get("eer_threshold")
    if dev_thr is not None:
        return float(dev_thr), {
            "source": "a2_metrics.testsets.dev.eer_threshold",
            "value": float(dev_thr),
            "metrics_path": str(metrics_path),
        }

    ts_value = (metrics.get("threshold_source") or {}).get("value")
    if ts_value is not None:
        return float(ts_value), {
            "source": "a2_metrics.threshold_source.value",
            "value": float(ts_value),
            "metrics_path": str(metrics_path),
        }

    raise ValueError(
        f"{metrics_path} has neither testsets.dev.eer_threshold nor threshold_source.value "
        "set (both null/absent) — re-score the A2 experiment: "
        f"uv run python scripts/score_experiment.py {a2_experiment}"
    )


class _CountingClient(JevClient):
    """Wraps a real `JevClient`, counting `ask()` calls so the runner can report cache-hit
    vs real-call counts without the cache needing to expose hit/miss itself."""

    def __init__(self, inner: JevClient) -> None:
        self._inner = inner
        self.route = inner.route
        self.model_version = inner.model_version
        self.calls = 0

    def ask(self, request: JevRequest) -> JevResponse:
        self.calls += 1
        return self._inner.ask(request)


def _build_client(kind: str, cfg: dict[str, Any]) -> JevClient:
    if kind == "dry-run":
        return DryRunJevClient()
    if kind == "typesafe":
        client_cfg = cfg.get("client", {})
        return TypesafeJevClient(
            model_version=client_cfg.get("model_version", "unknown"),
            route=client_cfg.get("route", "typesafe"),
        )
    raise ValueError(f"unknown client kind {kind!r}; expected 'dry-run' or 'typesafe'")


def _question_set_tag(question_set: str) -> str:
    return "noul" if question_set == "noul_only" else question_set


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def run_a3(
    cfg: dict[str, Any],
    *,
    state_variant: str,
    question_set: str,
    client_kind: str | None = None,
    n: int | None = None,
    resume: bool = False,
    max_cost_usd: float = 0.50,
    root: str | Path = "experiments",
    lock_path: str | Path = "data/cache/.heavy_job.lock",
    status_fn=memory_status,
    date: _dt.date | None = None,
) -> Path:
    """Run one A3 (state variant x question set x client) configuration. Returns `exp_dir`."""
    if state_variant not in ("numbers", "words"):
        raise ValueError(f"state_variant must be 'numbers' or 'words', got {state_variant!r}")
    if question_set not in QUESTION_SETS:
        raise ValueError(
            f"question_set must be one of {sorted(QUESTION_SETS)}, got {question_set!r}"
        )

    client_kind = str(client_kind or cfg["client"]["kind"])
    n_eff = n if n is not None else int(cfg["subset"].get("n", 500))

    if client_kind == "typesafe" and n_eff > _TYPESAFE_FIRST_RUN_MAX_N:
        raise ValueError(
            "TypesafeJevClient's cost per call is not yet known (ask() is not implemented "
            f"yet — see audio_df.jev.client). Pass --n <= {_TYPESAFE_FIRST_RUN_MAX_N} for the "
            "first real run so the measured cost per clip can be printed before scaling up."
        )

    questions = QUESTION_SETS[question_set]

    tag = f"{state_variant}-{_question_set_tag(question_set)}"
    if client_kind == "dry-run":
        tag += "-dryrun"

    exp_date = date or _dt.date.today()
    exp_id = make_exp_id(exp_date, "e5", "a3", "raw", tag=tag)

    manifest_path = str(cfg["subset"]["manifest"])
    a2_experiment = str(cfg["a2_experiment"])

    eer_threshold, threshold_info = resolve_eer_threshold(cfg, a2_experiment)
    effective_cfg = dict(cfg)
    effective_cfg["thresholds"] = dict(cfg["thresholds"])
    effective_cfg["thresholds"]["detector"] = {
        **cfg["thresholds"]["detector"],
        "eer_threshold": eer_threshold,
    }

    run_config = {
        "system": "a3",
        "experiment": "e5",
        "data": {"a2_experiment": a2_experiment, "manifest": manifest_path},
        "preprocessing": {
            "sample_rate": int(cfg.get("sample_rate", 16000)),
            "silence_mode": "raw",
        },
        "seed": int(cfg["subset"].get("seed", 1337)),
        "features": {
            "state_variant": state_variant,
            "question_set": question_set,
            "client_kind": client_kind,
            "eer_threshold": eer_threshold,
            "eer_threshold_source": threshold_info["source"],
        },
    }

    with experiment_run(
        exp_id,
        run_config,
        job_name="a3-jev",
        root=root,
        lock_path=lock_path,
        status_fn=status_fn,
        resume=resume,
    ) as run:
        run.note(
            f"detector eer_threshold={eer_threshold!r} (source: {threshold_info['source']})"
        )
        if client_kind == "dry-run":
            run.note("DRY RUN — stub client, not a result")

        _run_a3_body(
            run,
            effective_cfg,
            state_variant=state_variant,
            questions=questions,
            client_kind=client_kind,
            n_eff=n_eff,
            max_cost_usd=max_cost_usd,
            manifest_path=manifest_path,
            a2_experiment=a2_experiment,
        )

    return run.exp_dir


def _run_a3_body(
    run: RunContext,
    cfg: dict[str, Any],
    *,
    state_variant: str,
    questions: tuple[Any, ...],
    client_kind: str,
    n_eff: int,
    max_cost_usd: float,
    manifest_path: str,
    a2_experiment: str,
) -> None:
    a2_dir = Path(a2_experiment)
    a2_scores = pd.read_csv(a2_dir / "scores" / "dev.csv")
    manifest = pd.read_csv(manifest_path)

    subset = stratified_subset(
        manifest,
        n_eff,
        strata=tuple(cfg["subset"].get("strata", ["label", "attack_id"])),
        seed=int(cfg["subset"].get("seed", 1337)),
    )
    subset.to_csv(run.exp_dir / "subset.csv", index=False)

    merged = subset.merge(
        a2_scores.rename(columns={"clip_id": "utt_id"}),
        on="utt_id",
        how="inner",
        suffixes=("", "_a2"),
    )
    if merged.empty:
        raise ValueError(
            "no overlap between the subset drawn from the manifest and the A2 dev scores "
            f"in {a2_experiment}/scores/dev.csv — check that clip ids (utt_id) match"
        )

    a2_out_cols = {"utt_id": "clip_id", "label": "label", "score": "score"}
    a2_out = merged.rename(columns=a2_out_cols)[["clip_id", "label", "score"]]
    if "attack_id" in merged.columns:
        a2_out["attack_id"] = merged["attack_id"].values
    write_scores(run.exp_dir, "a2_dev500", a2_out)

    cache = JevCache(cfg.get("cache_dir", "data/jev_cache"))
    client = _build_client(client_kind, cfg)
    counting = _CountingClient(client)

    sample_rate = int(cfg.get("sample_rate", 16000))

    answers_rows: list[dict[str, Any]] = []
    a3_rows: list[dict[str, Any]] = []
    latencies: list[float] = []
    costs: list[float] = []
    running_cost = 0.0

    row: Any  # itertuples() rows are untyped namedtuples
    for row in merged.itertuples(index=False):
        clip_id = str(row.utt_id)
        audio, _sr = load_audio(row.path, target_sr=sample_rate)
        state = build_state(
            clip_id, float(row.score), np.asarray(audio), cfg, variant=state_variant
        )

        noul_score: float | None = None
        for question in questions:
            if running_cost >= max_cost_usd:
                raise RuntimeError(
                    f"budget guard: cumulative cost {running_cost:.4f} USD reached "
                    f"--max-cost-usd={max_cost_usd} before finishing all clips; rerun with "
                    "--resume after raising the budget or reducing --n"
                )
            request = build_request(question, state)
            response = cache.get_or_call(
                clip_id,
                request,
                counting,
                question_version=QUESTION_TEXT_VERSION,
                variant=state_variant,
            )
            cost = response.cost_usd or 0.0
            running_cost += cost
            latencies.append(response.latency_s)
            costs.append(cost)
            answers_rows.append(
                {
                    "clip_id": clip_id,
                    "question_id": response.question_id,
                    "question_type": response.question_type,
                    "result": json.dumps(response.result, sort_keys=True),
                    "route": response.route,
                    "model_version": response.model_version,
                    "latency_s": response.latency_s,
                    "cost_usd": response.cost_usd,
                }
            )
            if question.id == "noul":
                noul_score = noul_to_a3_score(response.result)

        if noul_score is None:
            # Should not happen: "noul" is in every configured question set today, but guard
            # against a future question set that omits it rather than writing a bad score.
            raise ValueError(
                "question_set produced no 'noul' answer; the A3 score is defined only from "
                "the noul question (see audio_df.jev.questions module docstring)"
            )

        a3_rows.append(
            {
                "clip_id": clip_id,
                "label": row.label,
                "score": noul_score,
                "attack_id": getattr(row, "attack_id", "-"),
            }
        )

    answers_df = pd.DataFrame(answers_rows)
    answers_df.to_csv(run.exp_dir / "answers.csv", index=False)

    a3_df = pd.DataFrame(a3_rows)
    write_scores(run.exp_dir, "jev_dev500", a3_df)

    n_questions = len(answers_rows)
    hits = max(n_questions - counting.calls, 0)
    summary = {
        "n_clips": len(a3_rows),
        "n_questions": n_questions,
        "cache_hits": hits,
        "cache_calls": counting.calls,
        "latency_median_s": float(np.median(latencies)) if latencies else None,
        "latency_p90_s": float(np.percentile(latencies, 90)) if latencies else None,
        "cost_mean_usd": float(np.mean(costs)) if costs else None,
        "cost_total_usd": float(np.sum(costs)) if costs else 0.0,
    }
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (run.exp_dir / "cost_summary.json").write_text(summary_text)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"Next: uv run python scripts/score_experiment.py {run.exp_dir} "
        "--threshold-from <a2 dev exp dir, if a3 has no fresh dev threshold of its own>"
    )
    print("Then: uv run python scripts/make_tables.py")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="run one A3 configuration")
    run_p.add_argument("--config", default="configs/a3_jev.yaml")
    run_p.add_argument("--state", choices=["numbers", "words"], required=True)
    run_p.add_argument("--questions", choices=sorted(QUESTION_SETS), required=True)
    run_p.add_argument("--client", choices=["dry-run", "typesafe"], default=None)
    run_p.add_argument("--n", type=int, default=None)
    run_p.add_argument("--resume", action="store_true")
    run_p.add_argument("--max-cost-usd", type=float, default=0.50)

    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    return run_a3(
        cfg,
        state_variant=args.state,
        question_set=args.questions,
        client_kind=args.client,
        n=args.n,
        resume=args.resume,
        max_cost_usd=args.max_cost_usd,
    )


if __name__ == "__main__":
    main()
