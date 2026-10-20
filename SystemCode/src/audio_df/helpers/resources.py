"""Resource limits and resumable-job helpers for a fanless 16 GB Apple Silicon machine.

This module owns nothing about *what* is computed — only how much CPU/MPS/disk a job is
allowed to touch, and how a long job can be interrupted and resumed without losing more
than one chunk of work. Every script that does real work (SSL feature extraction, GMM
training, audio degradation, ...) should call `apply_resource_limits` at the very top of
its `if __name__ == "__main__":` block, before any heavy import (torch, torchaudio,
transformers) has a chance to touch MPS — see the docstring on `apply_resource_limits`.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import time
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sys

import numpy as np
import psutil
import torch
import yaml
from tqdm import tqdm

_HAS_FCNTL = sys.platform != "win32"

__all__ = [
    "DiskBudgetError",
    "ChunkConfigMismatchError",
    "MemoryPressureError",
    "MemoryPressureExit",
    "HeavyJobBusy",
    "load_resources",
    "apply_resource_limits",
    "loader_workers",
    "ffmpeg_workers",
    "check_free_disk",
    "ChunkedJob",
    "timing_run",
    "format_timing",
    "memory_status",
    "check_can_start",
    "MemoryGuard",
    "run_resource_summary",
    "heavy_job_lock",
]

_REQUIRED_KEYS: dict[str, type | tuple[type, ...]] = {
    "mps_high_watermark_ratio": float,
    "cpu_threads": int,
    "max_loader_workers": int,
    "max_loader_workers_ssl": int,
    "ssl_extract_batch_size": int,
    "max_ffmpeg_workers": int,
    "min_free_disk_gb": (int, float),
    "timing_run_clips": int,
    "chunk_size": int,
}

_REQUIRED_MEMORY_GUARD_KEYS = {"poll", "swap_growth_limit_gb", "pause_on", "exit_on"}

_PRESSURE_NAME_TO_LEVEL = {"normal": 1, "warn": 2, "critical": 4}
# Swap counts as "stopped growing" if it rose by at most this much between two checks.
_SWAP_STABLE_GB = 0.05
_PRESSURE_LEVEL_TO_NAME = {v: k for k, v in _PRESSURE_NAME_TO_LEVEL.items()}

_DEFAULT_CONFIG_PATH = "configs/resources.yaml"
_DEFAULT_HEAVY_JOB_LOCK_PATH = "data/cache/.heavy_job.lock"


def _rss_mb() -> float:
    """Current process RSS in MB, cross-platform."""
    return psutil.Process().memory_info().rss / (1024**2)


class DiskBudgetError(RuntimeError):
    """Raised when a job would push free disk below the configured floor."""


class ChunkConfigMismatchError(RuntimeError):
    """Raised on resume when the job's item ids / config no longer match the saved job."""


class MemoryPressureError(RuntimeError):
    """Raised by `check_can_start` when the machine is already under memory pressure."""


class MemoryPressureExit(RuntimeError):
    """Raised by `ChunkedJob` between chunks when pressure reaches the configured
    `exit_on` level. Never raised mid-chunk — the triggering chunk has already been
    checkpointed by the time this is raised, so resuming the job later is safe. Runner
    scripts should catch this, log `exc.args[0]` (a resource-summary dict), and exit with
    a "resume later" message (e.g. `sys.exit(3)`) rather than treating it as a crash.
    """


class HeavyJobBusy(RuntimeError):
    """Raised by `heavy_job_lock(..., wait=False)` when another heavy job holds the lock."""


def load_resources(path: str | Path = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate `configs/resources.yaml`.

    Returns the full parsed dict (with the top-level `"resources"` key intact) so callers
    can do `cfg["resources"]["cpu_threads"]`. Raises `ValueError` with a clear message if a
    required key is missing or has the wrong type.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"resources config not found at {path}. Expected a YAML file with a top-level "
            f"'resources:' block (see configs/resources.yaml in the repo)."
        )
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "resources" not in raw:
        raise ValueError(f"{path}: expected a top-level 'resources:' key, got {raw!r}")

    section = raw["resources"]
    for key, expected_type in _REQUIRED_KEYS.items():
        if key not in section:
            raise ValueError(f"{path}: 'resources.{key}' is missing")
        value = section[key]
        if not isinstance(value, expected_type) or isinstance(value, bool):
            raise ValueError(
                f"{path}: 'resources.{key}' must be {expected_type}, got {type(value)} ({value!r})"
            )

    if "memory_guard" not in section or not isinstance(section["memory_guard"], dict):
        raise ValueError(f"{path}: 'resources.memory_guard' must be a mapping")
    guard_section = section["memory_guard"]
    missing_guard_keys = _REQUIRED_MEMORY_GUARD_KEYS - guard_section.keys()
    if missing_guard_keys:
        raise ValueError(f"{path}: 'resources.memory_guard' is missing keys {missing_guard_keys}")
    for name_key in ("pause_on", "exit_on"):
        name_value = guard_section[name_key]
        if name_value not in _PRESSURE_NAME_TO_LEVEL:
            raise ValueError(
                f"{path}: 'resources.memory_guard.{name_key}' must be one of "
                f"{sorted(_PRESSURE_NAME_TO_LEVEL)}, got {name_value!r}"
            )

    if "close_before_heavy_jobs" not in section or not isinstance(
        section["close_before_heavy_jobs"], list
    ):
        raise ValueError(f"{path}: 'resources.close_before_heavy_jobs' must be a list")

    return raw


def apply_resource_limits(cfg: dict[str, Any]) -> dict[str, Any]:
    """Apply CPU-thread and MPS-memory limits from `cfg["resources"]`, best-effort.

    Must be called at the very top of a script's `if __name__ == "__main__":` block,
    before any heavy import (torch, torchaudio, transformers) touches MPS. The MPS
    allocator reads `PYTORCH_MPS_HIGH_WATERMARK_RATIO` from the environment at *first
    use*, not continuously — setting it after MPS has already allocated memory is too
    late to have full effect. This function makes a best-effort detection of that case
    (torch.mps already reporting nonzero allocated memory) and emits a `warnings.warn`
    if so; it still sets the variable, since it is harmless and may still help.

    Returns a dict describing exactly what was applied (for tests and logging).
    """
    section = cfg["resources"]
    ratio = str(section["mps_high_watermark_ratio"])
    threads = str(section["cpu_threads"])

    applied: dict[str, Any] = {}

    if _mps_appears_initialised():
        warnings.warn(
            "apply_resource_limits: torch.mps already appears to have allocated memory; "
            "setting PYTORCH_MPS_HIGH_WATERMARK_RATIO now may be too late to take full "
            "effect. Call apply_resource_limits() before any heavy import touches MPS.",
            stacklevel=2,
        )

    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = ratio
    applied["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = ratio

    blas_env_vars = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    )
    for var in blas_env_vars:
        os.environ[var] = threads
        applied[var] = threads

    torch.set_num_threads(int(threads))
    applied["torch_num_threads"] = torch.get_num_threads()

    return applied


def _mps_appears_initialised() -> bool:
    """Best-effort check for whether torch's MPS backend has already allocated memory."""
    try:
        if not torch.backends.mps.is_available():
            return False
        allocated = torch.mps.current_allocated_memory()
        return allocated > 0
    except Exception:
        return False


def loader_workers(requested: int, cfg: dict[str, Any], kind: str = "default") -> int:
    """Clamp a requested DataLoader worker count.

    `kind="default"` clamps to `cfg["resources"]["max_loader_workers"]` (handcrafted
    feature / audio loaders). `kind="ssl"` clamps to the tighter
    `cfg["resources"]["max_loader_workers_ssl"]`, since SSL extraction loaders hold much
    larger per-item tensors and this machine's real headroom is only ~6-8 GB.
    """
    section = cfg["resources"]
    if kind == "default":
        cap = section["max_loader_workers"]
    elif kind == "ssl":
        cap = section["max_loader_workers_ssl"]
    else:
        raise ValueError(f"loader_workers: kind must be 'default' or 'ssl', got {kind!r}")
    return max(0, min(requested, cap))


def ffmpeg_workers(requested: int, cfg: dict[str, Any]) -> int:
    """Clamp a requested ffmpeg subprocess count to `cfg["resources"]["max_ffmpeg_workers"]`."""
    cap = cfg["resources"]["max_ffmpeg_workers"]
    return max(0, min(requested, cap))


def check_free_disk(path: str | Path, need_gb: float, cfg: dict[str, Any]) -> None:
    """Raise `DiskBudgetError` if writing `need_gb` more would breach the configured floor.

    `path` only needs to exist on the same filesystem/volume as the intended write
    (e.g. the parent directory of a cache); `shutil.disk_usage` resolves the volume.
    """
    floor_gb = cfg["resources"]["min_free_disk_gb"]
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    remaining_gb = free_gb - need_gb
    if remaining_gb < floor_gb:
        raise DiskBudgetError(
            f"disk budget breached at {path}: {free_gb:.1f} GB free, job needs {need_gb:.1f} GB, "
            f"which would leave {remaining_gb:.1f} GB < the {floor_gb:.1f} GB floor"
        )


_warned_no_sysctl = False


def _warn_no_sysctl_once(reason: str) -> None:
    global _warned_no_sysctl
    if not _warned_no_sysctl:
        warnings.warn(
            f"memory_status: could not read memory pressure via sysctl ({reason}); "
            f"pressure_level will be None and callers will not block on it.",
            stacklevel=2,
        )
        _warned_no_sysctl = True


def _parse_swapusage(text: str) -> tuple[float | None, float | None]:
    """Parse `sysctl vm.swapusage` output into `(used_gb, total_gb)`.

    Format on this machine: `vm.swapusage: total = 1024.00M  used = 592.25M  free =
    431.75M  (encrypted)`. Handles both `M` and `G` unit suffixes.
    """

    def find(field: str) -> float | None:
        match = re.search(rf"{field}\s*=\s*([\d.]+)\s*([MG])", text)
        if not match:
            return None
        value, unit = float(match.group(1)), match.group(2)
        return value / 1024.0 if unit == "M" else value

    return find("used"), find("total")


def memory_status() -> dict[str, Any]:
    """Best-effort snapshot of current memory pressure, swap, and this process's RSS.

    On macOS, `pressure_level` is read from `sysctl -n kern.memorystatus_vm_pressure_level`
    (1 normal, 2 warn, 4 critical) and swap from `sysctl vm.swapusage`. On non-macOS, or if
    either `sysctl` call fails for any reason, `pressure_level`/`pressure_name` and the
    swap fields are `None` and a `warnings.warn` fires once (not on every call) — callers
    must treat `None` as "unknown, do not block on this", never as "normal".
    """
    rss_mb = _rss_mb()
    pressure_level: int | None = None
    pressure_name = "unknown"
    swap_used_gb: float | None = None
    swap_total_gb: float | None = None

    if platform.system() != "Darwin":
        _warn_no_sysctl_once(f"platform is {platform.system()!r}, not Darwin")
    else:
        try:
            level_out = subprocess.run(
                ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            pressure_level = int(level_out)
            pressure_name = _PRESSURE_LEVEL_TO_NAME.get(pressure_level, "unknown")
        except Exception as exc:  # noqa: BLE001 - deliberately broad, this must never raise
            _warn_no_sysctl_once(f"pressure level query failed: {exc}")

        try:
            swap_out = subprocess.run(
                ["sysctl", "vm.swapusage"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout
            swap_used_gb, swap_total_gb = _parse_swapusage(swap_out)
        except Exception as exc:  # noqa: BLE001
            _warn_no_sysctl_once(f"swapusage query failed: {exc}")

    return {
        "pressure_level": pressure_level,
        "pressure_name": pressure_name,
        "swap_used_gb": swap_used_gb,
        "swap_total_gb": swap_total_gb,
        "rss_mb": rss_mb,
    }


def check_can_start(
    cfg: dict[str, Any], status_fn: Callable[[], dict[str, Any]] = memory_status
) -> None:
    """Raise `MemoryPressureError` at job start unless memory pressure is normal.

    `pressure_level is None` (sysctl unavailable / non-macOS) does not block — see
    `memory_status`. The error message lists current pressure, swap, and the apps to
    close from `cfg["resources"]["close_before_heavy_jobs"]`.
    """
    status = status_fn()
    level = status["pressure_level"]
    if level is not None and level != _PRESSURE_NAME_TO_LEVEL["normal"]:
        close_apps = cfg["resources"].get("close_before_heavy_jobs", [])
        raise MemoryPressureError(
            f"memory pressure is {status['pressure_name']} (level {level}), not normal; "
            f"swap used {status['swap_used_gb']} / {status['swap_total_gb']} GB. "
            f"Close before starting a heavy job: {close_apps}"
        )


class MemoryGuard:
    """Tracks memory pressure/swap growth across a long job's chunk boundaries.

    Constructed once at job start (records the starting swap usage); `check()` is called
    between chunks (never mid-chunk) and returns `"ok"`, `"pause"`, or `"exit"`:

    - `"exit"` if pressure has reached `cfg["resources"]["memory_guard"]["exit_on"]`
      (default `"critical"`, level 4).
    - `"pause"` if pressure has reached `["pause_on"]` (default `"warn"`, level 2) **or**
      swap has grown by more than `swap_growth_limit_gb` since the swap baseline.
    - `"ok"` otherwise.

    The swap baseline starts at the job's starting swap. macOS rarely gives swap back, so
    "swap grew since the start" alone would pause forever once tripped (it did, on
    2026-10-01, with pressure back to normal). So a swap pause ends when pressure is
    normal and swap has stopped growing between two checks (within `_SWAP_STABLE_GB`);
    the current swap then becomes the new baseline, and a further growth of more than
    `swap_growth_limit_gb` pauses again. `swap_start_gb` in the summary stays the job start.

    `status_fn` is injectable so tests never depend on this machine's real state.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        status_fn: Callable[[], dict[str, Any]] = memory_status,
    ) -> None:
        section = cfg["resources"]["memory_guard"]
        self.swap_growth_limit_gb: float = section["swap_growth_limit_gb"]
        self.pause_level = _PRESSURE_NAME_TO_LEVEL[section["pause_on"]]
        self.exit_level = _PRESSURE_NAME_TO_LEVEL[section["exit_on"]]
        self.status_fn = status_fn

        start_status = status_fn()
        self.start_swap_gb: float = start_status["swap_used_gb"] or 0.0
        self.baseline_swap_gb: float = self.start_swap_gb
        self.last_swap_gb: float | None = start_status["swap_used_gb"]
        self.paused_on_swap = False
        self.swap_peak_gb: float = self.start_swap_gb
        self.pressure_max: int = start_status["pressure_level"] or _PRESSURE_NAME_TO_LEVEL["normal"]

    def check(self) -> str:
        status = self.status_fn()
        level = status["pressure_level"]
        swap = status["swap_used_gb"]
        previous_swap, self.last_swap_gb = self.last_swap_gb, swap

        if swap is not None:
            self.swap_peak_gb = max(self.swap_peak_gb, swap)
        if level is not None:
            self.pressure_max = max(self.pressure_max, level)

        if level is not None and level >= self.exit_level:
            return "exit"
        if level is not None and level >= self.pause_level:
            return "pause"

        if swap is not None and (swap - self.baseline_swap_gb) > self.swap_growth_limit_gb:
            stable = previous_swap is not None and swap - previous_swap <= _SWAP_STABLE_GB
            if self.paused_on_swap and stable:
                self.baseline_swap_gb = swap
                self.paused_on_swap = False
                return "ok"
            self.paused_on_swap = True
            return "pause"

        self.paused_on_swap = False
        return "ok"

    def run_resource_summary(self) -> dict[str, Any]:
        current = self.status_fn()
        return {
            "swap_start_gb": self.start_swap_gb,
            "swap_end_gb": current["swap_used_gb"],
            "swap_peak_gb": self.swap_peak_gb,
            "pressure_max": self.pressure_max,
        }


def run_resource_summary(guard: MemoryGuard | None = None) -> dict[str, Any]:
    """Peak RSS plus (if `guard` is given) swap/pressure accumulated by a `MemoryGuard`.

    Intended for runner scripts to log alongside a training/extraction job's registry row.
    """
    peak_rss_mb = _rss_mb()
    summary: dict[str, Any] = {
        "peak_rss_mb": peak_rss_mb,
        "swap_start_gb": None,
        "swap_end_gb": None,
        "swap_peak_gb": None,
        "pressure_max": None,
    }
    if guard is not None:
        summary.update(guard.run_resource_summary())
    return summary


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for a pid recorded in a lock file."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission to signal it - still alive.
        return True
    except OSError:
        return False
    return True


def _read_lock_holder(lock_path: Path) -> dict[str, Any] | None:
    try:
        text = lock_path.read_text().strip()
        if not text:
            return None
        return json.loads(text)
    except Exception:
        return None


def _describe_holder(holder: dict[str, Any] | None) -> str:
    if holder is None:
        return "an unknown process (lock file unreadable or empty)"
    return (
        f"{holder.get('name', '?')} (pid={holder.get('pid', '?')}, "
        f"started={holder.get('started', '?')})"
    )


@contextmanager
def heavy_job_lock(
    name: str,
    lock_path: str | Path = _DEFAULT_HEAVY_JOB_LOCK_PATH,
    wait: bool = True,
) -> Iterator[None]:
    """Process-wide mutex so only one heavy job runs at a time on this machine.

    On POSIX uses `fcntl.flock` (kernel-released on crash). On Windows uses atomic
    O_EXCL file creation with PID liveness checking — same semantics, no busy-poll.
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if _HAS_FCNTL:
        yield from _heavy_job_lock_posix(name, lock_path, wait)
    else:
        yield from _heavy_job_lock_pidfile(name, lock_path, wait)


def _heavy_job_lock_posix(name: str, lock_path: Path, wait: bool) -> Iterator[None]:
    import fcntl  # local: only called on POSIX where this always exists

    # Extract constants once; `# type: ignore` because Pylance lacks fcntl stubs on Windows.
    _flock = fcntl.flock  # type: ignore[attr-defined]
    LOCK_EX: int = fcntl.LOCK_EX  # type: ignore[attr-defined]
    LOCK_NB: int = fcntl.LOCK_NB  # type: ignore[attr-defined]
    LOCK_UN: int = fcntl.LOCK_UN  # type: ignore[attr-defined]

    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    try:
        acquired = False
        try:
            _flock(fd, LOCK_EX | LOCK_NB)
            acquired = True
        except BlockingIOError:
            holder = _read_lock_holder(lock_path)
            holder_alive = holder is not None and _pid_alive(holder.get("pid", -1))
            if holder is not None and not holder_alive:
                try:
                    _flock(fd, LOCK_EX | LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    pass

            if not acquired:
                if not wait:
                    raise HeavyJobBusy(
                        f"heavy job lock at {lock_path} is held by {_describe_holder(holder)}; "
                        f"refusing to start {name!r} (wait=False)"
                    ) from None
                print(
                    f"[heavy_job_lock] {name!r} waiting: lock at {lock_path} held by "
                    f"{_describe_holder(holder)}. Blocking until released."
                )
                _flock(fd, LOCK_EX)
                acquired = True

        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        payload = json.dumps(
            {"name": name, "pid": os.getpid(), "started": datetime.now(UTC).isoformat()}
        ).encode("utf-8")
        os.write(fd, payload)
        os.fsync(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            _flock(fd, LOCK_UN)
        os.close(fd)


def _heavy_job_lock_pidfile(name: str, lock_path: Path, wait: bool) -> Iterator[None]:
    """Windows fallback: write-our-PID + liveness check polling.

    On release we write pid=0 (rather than unlink) so the file persists with the last
    holder's name — matches the POSIX behaviour where the file outlives the lock.
    """
    my_pid = os.getpid()
    payload = json.dumps(
        {"name": name, "pid": my_pid, "started": datetime.now(UTC).isoformat()}
    ).encode("utf-8")

    while True:
        holder = _read_lock_holder(lock_path)
        holder_pid = holder.get("pid", 0) if holder else 0

        if holder_pid == 0 or not _pid_alive(holder_pid):
            lock_path.write_bytes(payload)
            verify = _read_lock_holder(lock_path)
            if verify and verify.get("pid") == my_pid:
                break
            # Race: retry immediately.
            continue

        if not wait:
            raise HeavyJobBusy(
                f"heavy job lock at {lock_path} is held by {_describe_holder(holder)}; "
                f"refusing to start {name!r} (wait=False)"
            ) from None
        print(
            f"[heavy_job_lock] {name!r} waiting: lock at {lock_path} held by "
            f"{_describe_holder(holder)}. Polling every 5 s."
        )
        time.sleep(5)

    try:
        yield
    finally:
        released = json.dumps({"name": name, "pid": 0, "started": ""}).encode("utf-8")
        with contextlib.suppress(OSError):
            lock_path.write_bytes(released)


def _config_hash(job_config: dict[str, Any], item_ids: list[str]) -> str:
    payload = json.dumps(
        {"job_config": job_config, "item_ids": list(item_ids)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ChunkedJob:
    """Resumable chunked processing over an ordered list of item ids.

    Splits `item_ids` into ordered chunks of `chunk_size`. On first construction it
    writes `out_dir/job.json` recording a hash of `(job_config, item_ids)`. On a later
    construction against the same `out_dir`, if that hash differs, construction raises
    `ChunkConfigMismatchError` rather than silently mixing incompatible runs.

    Each chunk is written atomically: results are first written to a `.tmp` file, then
    `os.replace`d into place. A chunk only counts as done if its final `.npz` file exists
    and loads successfully — a `KeyboardInterrupt` or crash mid-`np.savez` leaves only an
    orphaned `.tmp` file, which is never counted as complete and is safely overwritten by
    the next attempt at that chunk.

    If `memory_guard` (a `MemoryGuard`) is given, `save_chunk` calls `guard.check()`
    immediately after the chunk is checkpointed (never mid-chunk, by construction — the
    check only happens after the atomic rename). On `"pause"`, this prints a message and,
    if `blocking_pause` (default), sleeps `pause_poll_interval_s` and re-checks in a loop
    until `"ok"`; pass `blocking_pause=False` (as tests do) to check once, print, and
    return without looping. On `"exit"`, raises `MemoryPressureExit` — the just-saved
    chunk stays valid on disk, so resuming later replays no work.
    """

    def __init__(
        self,
        out_dir: str | Path,
        item_ids: Iterable[str],
        chunk_size: int,
        job_config: dict[str, Any],
        memory_guard: MemoryGuard | None = None,
        pause_poll_interval_s: float = 30.0,
        blocking_pause: bool = True,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.item_ids = list(item_ids)
        self.chunk_size = chunk_size
        self.job_config = job_config
        self.memory_guard = memory_guard
        self.pause_poll_interval_s = pause_poll_interval_s
        self.blocking_pause = blocking_pause
        self.sleep_fn = sleep_fn

        self._chunks: list[list[str]] = [
            self.item_ids[i : i + chunk_size] for i in range(0, len(self.item_ids), chunk_size)
        ]
        self.n_chunks = len(self._chunks)

        job_json_path = self.out_dir / "job.json"
        new_hash = _config_hash(job_config, self.item_ids)
        if job_json_path.exists():
            with open(job_json_path) as f:
                existing = json.load(f)
            if existing.get("config_hash") != new_hash:
                raise ChunkConfigMismatchError(
                    f"{job_json_path}: existing job was built from a different "
                    f"(job_config, item_ids) than this run. Existing hash="
                    f"{existing.get('config_hash')!r}, new hash={new_hash!r}. "
                    f"Use a fresh out_dir for a different job."
                )
        else:
            with open(job_json_path, "w") as f:
                json.dump(
                    {
                        "config_hash": new_hash,
                        "n_items": len(self.item_ids),
                        "chunk_size": chunk_size,
                        "n_chunks": self.n_chunks,
                    },
                    f,
                    indent=2,
                )

    def _chunk_path(self, chunk_idx: int) -> Path:
        return self.out_dir / f"chunk_{chunk_idx:05d}.npz"

    def _tmp_path(self, chunk_idx: int) -> Path:
        # Ends in .npz (not .npz.tmp) so np.savez does not append a second .npz suffix.
        return self.out_dir / f"chunk_{chunk_idx:05d}.tmp.npz"

    def _chunk_done(self, chunk_idx: int) -> bool:
        path = self._chunk_path(chunk_idx)
        if not path.exists():
            return False
        try:
            with np.load(path) as data:
                _ = data["ids"]
            return True
        except Exception:
            return False

    def is_complete(self) -> bool:
        return all(self._chunk_done(i) for i in range(self.n_chunks))

    def pending(self) -> Iterator[tuple[int, list[str]]]:
        """Yield `(chunk_idx, ids)` for chunks not yet completed, with a tqdm progress bar."""
        todo = [i for i in range(self.n_chunks) if not self._chunk_done(i)]
        total_items = sum(len(self._chunks[i]) for i in todo)
        with tqdm(total=total_items, desc="ChunkedJob", unit="item") as bar:
            for chunk_idx in todo:
                ids = self._chunks[chunk_idx]
                yield chunk_idx, ids
                bar.update(len(ids))

    def save_chunk(self, chunk_idx: int, ids: list[str], arrays: dict[str, np.ndarray]) -> None:
        """Atomically write one chunk's results: tmp file first, then `os.replace`.

        If a `memory_guard` was supplied, this then calls `_handle_memory_guard` — always
        *after* the chunk is durably on disk, never before or during, so a pause/exit
        decision never costs already-completed work.
        """
        tmp_path = self._tmp_path(chunk_idx)
        final_path = self._chunk_path(chunk_idx)
        payload: dict[str, Any] = dict(arrays)
        payload["ids"] = np.asarray(ids)
        np.savez(tmp_path, **payload)
        os.replace(tmp_path, final_path)

        if self.memory_guard is not None:
            self._handle_memory_guard()

    def _handle_memory_guard(self) -> None:
        assert self.memory_guard is not None
        while True:
            state = self.memory_guard.check()
            if state == "ok":
                return
            if state == "exit":
                raise MemoryPressureExit(self.memory_guard.run_resource_summary())
            # state == "pause"
            print(
                f"[ChunkedJob] memory pressure elevated at {self.out_dir}; pausing before "
                f"the next chunk (swap peak {self.memory_guard.swap_peak_gb:.2f} GB, "
                f"pressure_max {self.memory_guard.pressure_max}). "
                f"Re-checking in {self.pause_poll_interval_s}s."
            )
            if not self.blocking_pause:
                return
            self.sleep_fn(self.pause_poll_interval_s)

    def load_all(self) -> dict[str, np.ndarray]:
        """Load and concatenate all chunks, in item order, keyed by array name plus 'ids'."""
        if not self.is_complete():
            missing = [i for i in range(self.n_chunks) if not self._chunk_done(i)]
            raise RuntimeError(
                f"ChunkedJob at {self.out_dir} is incomplete; missing chunks {missing}"
            )

        collected: dict[str, list[np.ndarray]] = {}
        for chunk_idx in range(self.n_chunks):
            with np.load(self._chunk_path(chunk_idx)) as data:
                for key in data.files:
                    collected.setdefault(key, []).append(data[key])

        return {key: np.concatenate(arrs, axis=0) for key, arrs in collected.items()}


def timing_run(
    fn: Callable[[Any], Any],
    items: list[Any],
    n: int | None = None,
    total_items: int | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Time `fn` over the first `n` items to sanity-check a job before running it in full.

    `n` defaults to `cfg["resources"]["timing_run_clips"]` (loading the default config if
    `cfg` is not given). Returns per-item median/p90/mean wall time, items/sec, peak RSS in
    MB (via psutil, cross-platform), peak MPS memory in MB
    (`mps_peak_mb`, via `torch.mps.driver_allocated_memory`/`current_allocated_memory`,
    `None` if MPS is unavailable), swap used at start/end (`swap_start_gb`/`swap_end_gb`,
    via `memory_status`, `None` if unavailable), memory pressure level at start/end
    (`pressure_start`/`pressure_end`), and, if `total_items` is given, an **extrapolated**
    total wall time in hours.

    That extrapolation is a lower bound, not a promise: this is a fanless machine that
    throttles under sustained thermal load, so a real full run will typically take longer
    than `n`-item timing suggests.
    """
    if n is None:
        if cfg is None:
            cfg = load_resources()
        n = cfg["resources"]["timing_run_clips"]

    start_status = memory_status()

    sample = items[:n]
    durations: list[float] = []
    for item in sample:
        start = time.perf_counter()
        fn(item)
        durations.append(time.perf_counter() - start)

    end_status = memory_status()

    n_run = len(durations)
    mean_s = statistics.fmean(durations) if durations else float("nan")
    median_s = statistics.median(durations) if durations else float("nan")
    if len(durations) >= 2:
        sorted_d = sorted(durations)
        p90_idx = min(len(sorted_d) - 1, int(round(0.9 * (len(sorted_d) - 1))))
        p90_s = sorted_d[p90_idx]
    else:
        p90_s = mean_s

    items_per_sec = (1.0 / mean_s) if mean_s > 0 else float("nan")
    peak_rss_mb = _rss_mb()

    result: dict[str, Any] = {
        "n_items_timed": n_run,
        "mean_s_per_item": mean_s,
        "median_s_per_item": median_s,
        "p90_s_per_item": p90_s,
        "items_per_sec": items_per_sec,
        "peak_rss_mb": peak_rss_mb,
        "mps_peak_mb": _mps_peak_mb(),
        "swap_start_gb": start_status["swap_used_gb"],
        "swap_end_gb": end_status["swap_used_gb"],
        "pressure_start": start_status["pressure_level"],
        "pressure_end": end_status["pressure_level"],
        "extrapolated_total_items": total_items,
        "extrapolated_total_hours": None,
    }
    if total_items is not None and mean_s > 0:
        result["extrapolated_total_hours"] = (total_items * mean_s) / 3600.0

    return result


def _mps_peak_mb() -> float | None:
    """Best-effort peak MPS allocation in MB, `None` if MPS is unavailable."""
    try:
        if not torch.backends.mps.is_available():
            return None
        driver_bytes = torch.mps.driver_allocated_memory()
        current_bytes = torch.mps.current_allocated_memory()
        return max(driver_bytes, current_bytes) / (1024**2)
    except Exception:
        return None


def format_timing(result: dict[str, Any]) -> str:
    """Human-readable one-block summary of a `timing_run` result."""
    lines = [
        f"timed {result['n_items_timed']} items",
        f"  median {result['median_s_per_item'] * 1000:.2f} ms/item, "
        f"mean {result['mean_s_per_item'] * 1000:.2f} ms/item, "
        f"p90 {result['p90_s_per_item'] * 1000:.2f} ms/item",
        f"  {result['items_per_sec']:.2f} items/sec, peak RSS {result['peak_rss_mb']:.1f} MB",
    ]
    if result.get("mps_peak_mb") is not None:
        lines.append(f"  peak MPS allocation {result['mps_peak_mb']:.1f} MB")
    if result.get("swap_start_gb") is not None or result.get("swap_end_gb") is not None:
        lines.append(
            f"  swap {result.get('swap_start_gb')} GB -> {result.get('swap_end_gb')} GB, "
            f"pressure {result.get('pressure_start')} -> {result.get('pressure_end')}"
        )
    if result.get("extrapolated_total_hours") is not None:
        lines.append(
            f"  EXTRAPOLATED (not measured) for {result['extrapolated_total_items']} items: "
            f"{result['extrapolated_total_hours']:.2f} h — a fanless machine throttles under "
            f"sustained load, so a real full run will likely take longer than this."
        )
    return "\n".join(lines)
