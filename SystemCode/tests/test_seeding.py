import random

import numpy as np
import torch

from audio_df.runtime.seeding import make_generator, seed_everything, worker_init_fn


def test_seed_everything_reproducible_numpy_torch_random():
    result1 = seed_everything(123)
    a_random = random.random()
    a_numpy_legacy = np.random.rand(3)
    a_numpy_gen = result1["numpy_generator"].random(3)
    a_torch = torch.rand(3)

    result2 = seed_everything(123)
    b_random = random.random()
    b_numpy_legacy = np.random.rand(3)
    b_numpy_gen = result2["numpy_generator"].random(3)
    b_torch = torch.rand(3)

    assert a_random == b_random
    assert np.allclose(a_numpy_legacy, b_numpy_legacy)
    assert np.allclose(a_numpy_gen, b_numpy_gen)
    assert torch.allclose(a_torch, b_torch)


def test_seed_everything_returns_expected_keys():
    result = seed_everything(7, deterministic=True)
    assert result["seed"] == 7
    assert result["deterministic"] is True
    assert isinstance(result["numpy_generator"], np.random.Generator)
    assert "mps_seeded" in result
    assert "cuda_seeded" in result
    import os

    assert os.environ["PYTHONHASHSEED"] == "7"


def test_seed_everything_different_seeds_differ():
    seed_everything(1)
    a = torch.rand(3)
    seed_everything(2)
    b = torch.rand(3)
    assert not torch.allclose(a, b)


def test_worker_init_fn_reproducible_and_distinct_per_worker():
    def seeded_state(base_seed, worker_id):
        make_generator(base_seed)  # exercise the import path; unused by design here
        # Simulate what torch.initial_seed() would report inside a worker:
        # different worker ids get different initial seeds under torch's real
        # DataLoader machinery; here we fake that distinctness directly by seeding
        # torch's default generator per "worker" and calling worker_init_fn.
        torch.manual_seed(base_seed + worker_id)
        worker_init_fn(worker_id)
        return random.random(), np.random.rand(1)[0]

    r0a, n0a = seeded_state(1000, 0)
    r0b, n0b = seeded_state(1000, 0)
    r1, n1 = seeded_state(1000, 1)

    # Same base seed + worker id -> reproducible.
    assert r0a == r0b
    assert n0a == n0b
    # Different worker id -> different seed derived, almost certainly different draw.
    assert (r0a, n0a) != (r1, n1)


def test_make_generator_is_seeded_torch_generator():
    gen1 = make_generator(42)
    gen2 = make_generator(42)
    t1 = torch.rand(4, generator=gen1)
    t2 = torch.rand(4, generator=gen2)
    assert torch.allclose(t1, t2)
