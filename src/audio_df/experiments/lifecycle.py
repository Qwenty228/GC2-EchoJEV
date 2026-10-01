"""One run's lifecycle, shared by every runner (A1, A2, A3, degradations).

Every heavy run on this machine goes through the same steps, in the same order, so that no
runner can forget one:

1. apply the resource limits (threads, MPS watermark) — before anything touches MPS;
2. seed everything from `config["seed"]`;
3. refuse to start if the disk floor would be breached or memory pressure is not normal;
4. take the heavy-job lock, so two heavy jobs never run at once (a second one queues);
5. create `experiments/<exp_id>/` and append a `running` row to `experiments/REGISTRY.md`;
6. on exit, write `run_summary.json` (wall time, peak RSS, MPS peak, swap, pressure) and set
   the row to `done` or `failed`, with the reason. A memory-pressure exit is recorded as
   `failed` with a "resume with --resume" note: the chunked job already checkpointed its
   last chunk, so nothing completed is lost.

Metrics are deliberately absent: runners only write score files, and
`scripts/score_experiment.py` is the only place metrics are computed.

Usage, inside `if __name__ == "__main__":`::

    with experiment_run(exp_id, config, job_name="a1-train") as run:
        job = ChunkedJob(..., memory_guard=run.guard)
        ...
        write_scores(run.exp_dir, "eval", df)
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from audio_df.exp_io import capture_env, create_experiment, registry_append, registry_update
from audio_df.resources import (
    MemoryGuard,
    MemoryPressureExit,
    _mps_peak_mb,
    apply_resource_limits,
    check_can_start,
    check_free_disk,
    heavy_job_lock,
    load_resources,
    memory_status,
    run_resource_summary,
)
from audio_df.seeding import seed_everything

__all__ = ["RunContext", "experiment_run", "machine_cell"]


def machine_cell(env: dict[str, Any]) -> str:
    """Short machine description for the registry, e.g.
    "Mac16,13 Apple M4 16GB MPS; macOS 26.5.2; Py 3.11.13; torch 2.14.0"."""
    ram = env.get("ram_gb")
    ram_s = f"{round(ram)}GB" if isinstance(ram, (int, float)) else "?GB"
    accel = (
        "MPS" if env.get("gpu_mps_available") else ("CUDA" if env.get("cuda_available") else "CPU")
    )
    os_name = "macOS" if env.get("os") == "Darwin" else env.get("os", "?")
    return (
        f"{env.get('machine_model') or '?'} {env.get('cpu_brand') or '?'} {ram_s} {accel}; "
        f"{os_name} {env.get('os_version') or '?'}; "
        f"Py {env.get('python')}; torch {env.get('torch')}"
    )


@dataclass
class RunContext:
    """What a runner gets inside `experiment_run`."""

    exp_id: str
    exp_dir: Path
    resources: dict[str, Any]
    guard: MemoryGuard
    env: dict[str, Any]
    seeds: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        """Record a line for the registry `notes` cell and `notes.md` (e.g. a deviation)."""
        self.notes.append(text)
        with open(self.exp_dir / "notes.md", "a") as f:
            f.write(f"- {text}\n")


@contextmanager
def experiment_run(
    exp_id: str,
    config: dict[str, Any],
    *,
    job_name: str,
    need_disk_gb: float = 0.0,
    config_summary: str = "",
    root: str | Path = "experiments",
    resources_path: str | Path = "configs/resources.yaml",
    resume: bool = False,
    lock_path: str | Path = "data/cache/.heavy_job.lock",
    status_fn: Callable[[], dict[str, Any]] = memory_status,
) -> Generator[RunContext]:
    """Run one experiment under the project's resource and logging discipline (module doc)."""
    res = load_resources(resources_path)
    apply_resource_limits(res)
    seeds = seed_everything(int(config["seed"]))
    check_free_disk(".", need_disk_gb, res)
    check_can_start(res, status_fn=status_fn)

    with heavy_job_lock(job_name, lock_path=lock_path, wait=True):
        exp_dir = create_experiment(exp_id, config, root=root, resume=resume)
        env = capture_env()
        guard = MemoryGuard(res, status_fn=status_fn)
        has_row = False
        if resume:
            # A run interrupted between creating its folder and writing its row has no row
            # yet; resuming it must add one rather than crash.
            try:
                registry_update(exp_id, status="running", root=root)
                has_row = True
            except (ValueError, FileNotFoundError):
                has_row = False
        if not has_row:
            registry_append(
                exp_id,
                date=_dt.date.today().isoformat(),
                system=str(config["system"]),
                experiment=str(config["experiment"]),
                config_summary=config_summary,
                status="running",
                machine=machine_cell(env),
                root=root,
            )
        ctx = RunContext(exp_id, exp_dir, res, guard, env, seeds)
        t0 = time.monotonic()

        def finish(status: str, extra_note: str | None) -> None:
            summary = run_resource_summary(guard)
            summary["wall_time_s"] = time.monotonic() - t0
            summary["mps_peak_mb"] = _mps_peak_mb()
            (exp_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            if extra_note:
                ctx.note(extra_note)
            registry_update(
                exp_id,
                status=status,
                resources=summary,
                notes="; ".join(ctx.notes),
                root=root,
            )

        try:
            yield ctx
        except MemoryPressureExit:
            finish(
                "failed",
                "memory pressure reached critical; last chunk checkpointed — rerun with --resume",
            )
            raise
        except BaseException as exc:  # noqa: BLE001 — record every failure, then re-raise
            finish("failed", f"failed: {type(exc).__name__}: {exc}")
            raise
        else:
            finish("done", None)
