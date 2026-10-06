"""Smoke tests: the environment is usable and self-describing.

These run with no dataset and no model downloads.
"""

import pytest
import torch

import audio_df
from audio_df.runtime.device import describe_env, git_commit, resolve_device


def test_package_imports_with_version():
    assert audio_df.__version__


def test_resolve_device_auto_returns_real_device():
    device = resolve_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
    # the resolved device must actually accept a tensor
    assert torch.zeros(2, device=device).sum().item() == 0.0


def test_resolve_device_cpu_always_available():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_nonsense():
    with pytest.raises(ValueError, match="unknown device spec"):
        resolve_device("tpu")


def test_unavailable_backend_raises_rather_than_falling_back():
    """A silent CPU fallback would corrupt the timing columns of the registry."""
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            resolve_device("cuda")
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS is not available"):
            resolve_device("mps")


def test_describe_env_is_fully_populated():
    env = describe_env()
    expected = {
        "python",
        "platform",
        "machine",
        "torch",
        "numpy",
        "device",
        "mps_available",
        "cuda_available",
        "git_commit",
    }
    assert expected == set(env)
    assert all(isinstance(v, str) and v for v in env.values())


def test_git_commit_is_a_sha_or_unknown():
    commit = git_commit()
    assert commit == "unknown" or len(commit.split("-")[0]) == 40
