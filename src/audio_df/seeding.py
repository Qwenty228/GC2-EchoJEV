"""General-purpose seeding utilities shared across scripts that are not `training.py`.

`training.py` already owns `set_seed` for the train/eval loop (python/numpy/torch/mps) —
this module does not duplicate or replace it. `seed_everything` here is for scripts that
need a *return value* describing exactly what was set (a numpy `Generator`, applied flags,
etc.) plus DataLoader-worker seeding helpers (`worker_init_fn`, `make_generator`), which
`training.py` does not provide.

Execution note: on macOS (and Windows), `torch.utils.data.DataLoader` workers are spawned
with the `spawn` start method, not `fork`. Any script using multiple DataLoader workers
must therefore guard its top-level execution under `if __name__ == "__main__":` — code at
module scope would otherwise re-run in every spawned worker process.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

__all__ = ["seed_everything", "worker_init_fn", "make_generator"]


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    """Seed python `random`, numpy (legacy global + a fresh `Generator`), and torch.

    Sets, in order: `random.seed`, `np.random.seed` (legacy global state, for code that
    still calls `np.random.*` directly), `torch.manual_seed`, `torch.mps.manual_seed` (if
    the MPS backend is available), `torch.cuda.manual_seed_all` (if CUDA is available),
    and `os.environ["PYTHONHASHSEED"]` (note: this only affects hashing in *subprocesses*
    started after this call — it cannot retroactively change hash randomisation in the
    current process).

    If `deterministic`, also sets `torch.backends.cudnn.deterministic = True`,
    `torch.backends.cudnn.benchmark = False`, and
    `torch.use_deterministic_algorithms(True, warn_only=True)` (warn-only because some ops
    have no deterministic MPS/CPU implementation yet; failing loudly there would block
    otherwise-fine runs).

    Returns a dict of what was set, including a `"numpy_generator"` key holding a fresh
    `np.random.Generator(np.random.PCG64(seed))` for code that prefers the modern numpy
    RNG API over the legacy global one.
    """
    random.seed(seed)
    np.random.seed(seed)
    numpy_generator = np.random.Generator(np.random.PCG64(seed))

    torch.manual_seed(seed)

    mps_seeded = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        mps_seeded = True

    cuda_seeded = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cuda_seeded = True

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    return {
        "seed": seed,
        "deterministic": deterministic,
        "numpy_generator": numpy_generator,
        "mps_seeded": mps_seeded,
        "cuda_seeded": cuda_seeded,
        "pythonhashseed": str(seed),
    }


def worker_init_fn(worker_id: int) -> None:
    """`DataLoader(worker_init_fn=...)` hook: derive a per-worker seed and apply it.

    Each worker's `torch.initial_seed()` is already distinct (torch derives it from the
    main process's base seed plus the worker id) and reproducible given the same base
    seed, so deriving numpy/python seeds from it gives different-but-reproducible RNG
    state per worker without needing to know `worker_id`'s relationship to the base seed
    ourselves.
    """
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed)
    random.seed(base_seed)


def make_generator(seed: int) -> torch.Generator:
    """A `torch.Generator` seeded with `seed`, for `DataLoader(generator=...)`."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
