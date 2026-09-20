"""Device resolution and environment capture.

Every experiment records the environment it ran in: without the torch version, the device
and the git commit, a number in `reports/experiment_registry.csv` cannot be reproduced or
fairly compared against another row.

Target hardware for this project is Apple Silicon (MPS) or CPU. Code must never assume CUDA.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from functools import lru_cache

import numpy as np
import torch

__all__ = ["resolve_device", "describe_env", "git_commit"]


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a config device string to a real `torch.device`.

    `"auto"` prefers cuda, then mps, then cpu. An explicit request for an unavailable
    backend raises rather than silently falling back: a run that quietly dropped to CPU
    would make the wall-time and peak-memory columns of the registry meaningless.
    """
    spec = spec.lower().strip()

    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but CUDA is not available")
        return torch.device(spec)

    if spec == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "device 'mps' requested but MPS is not available "
                f"(built={torch.backends.mps.is_built()})"
            )
        return torch.device("mps")

    if spec == "cpu":
        return torch.device("cpu")

    raise ValueError(f"unknown device spec {spec!r}; expected auto|cuda|mps|cpu")


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Current commit, suffixed `-dirty` when the tree has uncommitted changes.

    Returns `"unknown"` outside a git repo rather than raising: an experiment should still
    run, it just records that its provenance is weaker.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def describe_env(device_spec: str = "auto") -> dict[str, str]:
    """Everything needed to reproduce a run, as flat strings for the CSV registry."""
    device = resolve_device(device_spec)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "mps_available": str(torch.backends.mps.is_available()),
        "cuda_available": str(torch.cuda.is_available()),
        "git_commit": git_commit(),
    }
