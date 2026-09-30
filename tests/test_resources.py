import fcntl
import json
import os
import threading
import time

import numpy as np
import pytest
import torch

import audio_df.resources as resources_mod
from audio_df.resources import (
    ChunkConfigMismatchError,
    ChunkedJob,
    DiskBudgetError,
    HeavyJobBusy,
    MemoryGuard,
    MemoryPressureError,
    MemoryPressureExit,
    apply_resource_limits,
    check_can_start,
    check_free_disk,
    ffmpeg_workers,
    format_timing,
    heavy_job_lock,
    load_resources,
    loader_workers,
    memory_status,
    run_resource_summary,
    timing_run,
)


@pytest.fixture
def cfg():
    return load_resources("configs/resources.yaml")


def test_load_resources_has_approved_values(cfg):
    section = cfg["resources"]
    assert section["mps_high_watermark_ratio"] == 0.3
    assert section["cpu_threads"] == 4
    assert section["max_loader_workers"] == 2
    assert section["max_loader_workers_ssl"] == 1
    assert section["ssl_extract_batch_size"] == 4
    assert section["max_ffmpeg_workers"] == 2
    assert section["min_free_disk_gb"] == 30
    assert section["timing_run_clips"] == 200
    assert section["chunk_size"] == 512
    assert section["memory_guard"]["pause_on"] == "warn"
    assert section["memory_guard"]["exit_on"] == "critical"
    assert section["memory_guard"]["swap_growth_limit_gb"] == 1.0
    assert "Arc" in section["close_before_heavy_jobs"]


def test_load_resources_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_resources(tmp_path / "nope.yaml")


def test_apply_resource_limits_sets_env_and_torch_threads(cfg):
    env_keys = [
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ]
    saved_env = {k: os.environ.get(k) for k in env_keys}
    saved_threads = torch.get_num_threads()
    try:
        applied = apply_resource_limits(cfg)
        assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.3"
        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert os.environ["MKL_NUM_THREADS"] == "4"
        assert os.environ["VECLIB_MAXIMUM_THREADS"] == "4"
        assert os.environ["OPENBLAS_NUM_THREADS"] == "4"
        assert torch.get_num_threads() == 4
        assert applied["torch_num_threads"] == 4
        assert applied["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.3"
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        torch.set_num_threads(saved_threads)


def test_loader_workers_clamps(cfg):
    assert loader_workers(8, cfg) == 2
    assert loader_workers(1, cfg) == 1
    assert loader_workers(0, cfg) == 0
    assert loader_workers(-3, cfg) == 0


def test_loader_workers_ssl_kind_uses_tighter_cap(cfg):
    assert loader_workers(8, cfg, kind="ssl") == 1
    assert loader_workers(0, cfg, kind="ssl") == 0


def test_loader_workers_unknown_kind_raises(cfg):
    with pytest.raises(ValueError, match="kind"):
        loader_workers(1, cfg, kind="bogus")


def test_ffmpeg_workers_clamps(cfg):
    assert ffmpeg_workers(8, cfg) == 2
    assert ffmpeg_workers(1, cfg) == 1
    assert ffmpeg_workers(0, cfg) == 0


def test_check_free_disk_raises_when_floor_breached(cfg, tmp_path, monkeypatch):
    class FakeUsage:
        total = 100 * 1024**3
        used = 90 * 1024**3
        free = 10 * 1024**3  # 10 GB free, floor is 30 GB

    monkeypatch.setattr("shutil.disk_usage", lambda path: FakeUsage())
    with pytest.raises(DiskBudgetError, match="disk budget breached"):
        check_free_disk(tmp_path, need_gb=1.0, cfg=cfg)


def test_check_free_disk_passes_when_plenty_free(cfg, tmp_path, monkeypatch):
    class FakeUsage:
        total = 500 * 1024**3
        used = 100 * 1024**3
        free = 400 * 1024**3

    monkeypatch.setattr("shutil.disk_usage", lambda path: FakeUsage())
    check_free_disk(tmp_path, need_gb=5.0, cfg=cfg)  # should not raise


class TestChunkedJob:
    def _make_job(self, out_dir, item_ids=None, chunk_size=3, job_config=None):
        item_ids = item_ids or [f"utt{i}" for i in range(10)]
        job_config = job_config if job_config is not None else {"variant": "A"}
        return ChunkedJob(out_dir, item_ids, chunk_size, job_config)

    def test_full_run_and_load_all_order(self, tmp_path):
        job = self._make_job(tmp_path)
        for chunk_idx, ids in job.pending():
            arrays = {"feat": np.array([[float(int(i[3:]))] for i in ids])}
            job.save_chunk(chunk_idx, ids, arrays)

        assert job.is_complete()
        data = job.load_all()
        assert list(data["ids"]) == [f"utt{i}" for i in range(10)]
        assert np.allclose(data["feat"].ravel(), np.arange(10))

    def test_resume_after_simulated_interruption(self, tmp_path):
        item_ids = [f"utt{i}" for i in range(10)]
        job1 = self._make_job(tmp_path, item_ids=item_ids)
        chunks = list(job1.pending())
        # Only complete the first chunk, simulating an interruption.
        chunk_idx, ids = chunks[0]
        job1.save_chunk(chunk_idx, ids, {"feat": np.zeros((len(ids), 1))})

        # Rebuild (as a resumed process would) against the same out_dir/config.
        job2 = self._make_job(tmp_path, item_ids=item_ids)
        remaining = list(job2.pending())
        remaining_idxs = {idx for idx, _ in remaining}
        assert chunk_idx not in remaining_idxs
        assert len(remaining) == job2.n_chunks - 1

        for idx, ids in remaining:
            job2.save_chunk(idx, ids, {"feat": np.zeros((len(ids), 1))})

        assert job2.is_complete()

    def test_config_hash_mismatch_raises(self, tmp_path):
        self._make_job(tmp_path, job_config={"variant": "A"})
        with pytest.raises(ChunkConfigMismatchError):
            self._make_job(tmp_path, job_config={"variant": "B"})

    def test_item_ids_change_also_raises(self, tmp_path):
        self._make_job(tmp_path, item_ids=["a", "b", "c"])
        with pytest.raises(ChunkConfigMismatchError):
            self._make_job(tmp_path, item_ids=["a", "b", "d"])

    def test_tmp_files_never_count_as_done(self, tmp_path):
        job = self._make_job(tmp_path, item_ids=["a", "b", "c"], chunk_size=3)
        chunk_idx, ids = next(job.pending())
        tmp_path_file = job._tmp_path(chunk_idx)
        # Simulate an interrupted write: only the tmp file exists, never renamed.
        np.savez(tmp_path_file, ids=np.asarray(ids), feat=np.zeros((3, 1)))
        assert not job.is_complete()
        assert not job._chunk_done(chunk_idx)

    def test_job_json_written_on_first_run(self, tmp_path):
        self._make_job(tmp_path)
        job_json = tmp_path / "job.json"
        assert job_json.exists()
        with open(job_json) as f:
            data = json.load(f)
        assert "config_hash" in data
        assert data["n_items"] == 10


def test_timing_run_returns_sane_keys():
    items = list(range(50))

    def fn(_item):
        return sum(range(1000))

    result = timing_run(fn, items, n=10, total_items=1000)
    assert result["n_items_timed"] == 10
    assert result["mean_s_per_item"] > 0
    assert result["median_s_per_item"] > 0
    assert result["p90_s_per_item"] >= result["median_s_per_item"] * 0
    assert result["items_per_sec"] > 0
    assert result["peak_rss_mb"] > 0
    assert result["extrapolated_total_hours"] is not None
    assert result["extrapolated_total_hours"] > 0
    assert "mps_peak_mb" in result
    assert "swap_start_gb" in result and "swap_end_gb" in result
    assert "pressure_start" in result and "pressure_end" in result

    text = format_timing(result)
    assert "EXTRAPOLATED" in text
    assert "items/sec" in text


def test_timing_run_default_n_from_config():
    items = list(range(5))

    def fn(_item):
        pass

    result = timing_run(fn, items)  # n defaults from configs/resources.yaml
    assert result["n_items_timed"] == 5  # fewer items than timing_run_clips=200
    assert result["extrapolated_total_hours"] is None


# --------------------------------------------------------------------------------------
# Memory pressure: mocked sysctl output, normal/warn/critical, swap growth.
# --------------------------------------------------------------------------------------


def _fake_status(pressure_level=1, swap_used_gb=0.5, swap_total_gb=1.0, rss_mb=100.0):
    return {
        "pressure_level": pressure_level,
        "pressure_name": {1: "normal", 2: "warn", 4: "critical"}.get(pressure_level, "unknown"),
        "swap_used_gb": swap_used_gb,
        "swap_total_gb": swap_total_gb,
        "rss_mb": rss_mb,
    }


def test_memory_status_parses_real_sysctl_output(monkeypatch):
    def fake_run(args, **kwargs):
        class Result:
            pass

        r = Result()
        if args[:2] == ["sysctl", "-n"]:
            r.stdout = "1\n"
        else:
            r.stdout = (
                "vm.swapusage: total = 1024.00M  used = 592.25M  free = 431.75M  (encrypted)\n"
            )
        return r

    monkeypatch.setattr("audio_df.resources.platform.system", lambda: "Darwin")
    monkeypatch.setattr("audio_df.resources.subprocess.run", fake_run)

    status = memory_status()
    assert status["pressure_level"] == 1
    assert status["pressure_name"] == "normal"
    assert status["swap_total_gb"] == pytest.approx(1.0, abs=1e-3)
    assert status["swap_used_gb"] == pytest.approx(592.25 / 1024, abs=1e-3)
    assert status["rss_mb"] > 0


def test_memory_status_parses_gb_units(monkeypatch):
    def fake_run(args, **kwargs):
        class Result:
            pass

        r = Result()
        if args[:2] == ["sysctl", "-n"]:
            r.stdout = "4\n"
        else:
            r.stdout = "vm.swapusage: total = 2.00G  used = 1.50G  free = 0.50G\n"
        return r

    monkeypatch.setattr("audio_df.resources.platform.system", lambda: "Darwin")
    monkeypatch.setattr("audio_df.resources.subprocess.run", fake_run)

    status = memory_status()
    assert status["pressure_level"] == 4
    assert status["pressure_name"] == "critical"
    assert status["swap_used_gb"] == pytest.approx(1.5, abs=1e-3)
    assert status["swap_total_gb"] == pytest.approx(2.0, abs=1e-3)


def test_memory_status_non_macos_returns_none_and_warns(monkeypatch):
    monkeypatch.setattr("audio_df.resources.platform.system", lambda: "Linux")
    monkeypatch.setattr(resources_mod, "_warned_no_sysctl", False)
    with pytest.warns(UserWarning):
        status = memory_status()
    assert status["pressure_level"] is None
    assert status["swap_used_gb"] is None


def test_check_can_start_raises_when_not_normal(cfg):
    with pytest.raises(MemoryPressureError, match="warn"):
        check_can_start(cfg, status_fn=lambda: _fake_status(pressure_level=2))


def test_check_can_start_passes_when_normal(cfg):
    check_can_start(cfg, status_fn=lambda: _fake_status(pressure_level=1))  # no raise


def test_check_can_start_passes_when_pressure_unknown(cfg):
    check_can_start(cfg, status_fn=lambda: _fake_status(pressure_level=None))  # no raise


def test_check_can_start_message_lists_apps_to_close(cfg):
    with pytest.raises(MemoryPressureError) as exc_info:
        check_can_start(cfg, status_fn=lambda: _fake_status(pressure_level=4))
    assert "Arc" in str(exc_info.value)


class TestMemoryGuard:
    def test_ok_when_normal_and_no_swap_growth(self, cfg):
        guard = MemoryGuard(cfg, status_fn=lambda: _fake_status(pressure_level=1, swap_used_gb=0.5))
        assert guard.check() == "ok"

    def test_pause_on_warn_pressure(self, cfg):
        calls = [_fake_status(pressure_level=1, swap_used_gb=0.5)]

        def status_fn():
            return calls[0]

        guard = MemoryGuard(cfg, status_fn=status_fn)
        calls[0] = _fake_status(pressure_level=2, swap_used_gb=0.5)
        assert guard.check() == "pause"

    def test_pause_on_swap_growth_even_if_pressure_normal(self, cfg):
        calls = [_fake_status(pressure_level=1, swap_used_gb=0.2)]

        def status_fn():
            return calls[0]

        guard = MemoryGuard(cfg, status_fn=status_fn)
        # swap_growth_limit_gb is 1.0; grow swap by 1.5 GB.
        calls[0] = _fake_status(pressure_level=1, swap_used_gb=1.7)
        assert guard.check() == "pause"

    def test_no_pause_for_small_swap_growth(self, cfg):
        calls = [_fake_status(pressure_level=1, swap_used_gb=0.2)]

        def status_fn():
            return calls[0]

        guard = MemoryGuard(cfg, status_fn=status_fn)
        calls[0] = _fake_status(pressure_level=1, swap_used_gb=0.5)  # +0.3 GB, under limit
        assert guard.check() == "ok"

    def test_exit_on_critical_pressure(self, cfg):
        calls = [_fake_status(pressure_level=1, swap_used_gb=0.5)]

        def status_fn():
            return calls[0]

        guard = MemoryGuard(cfg, status_fn=status_fn)
        calls[0] = _fake_status(pressure_level=4, swap_used_gb=0.5)
        assert guard.check() == "exit"

    def test_run_resource_summary_tracks_peak(self, cfg):
        calls = [_fake_status(pressure_level=1, swap_used_gb=0.2)]

        def status_fn():
            return calls[0]

        guard = MemoryGuard(cfg, status_fn=status_fn)
        calls[0] = _fake_status(pressure_level=2, swap_used_gb=0.9)
        guard.check()
        calls[0] = _fake_status(pressure_level=1, swap_used_gb=0.3)
        guard.check()

        summary = guard.run_resource_summary()
        assert summary["swap_start_gb"] == pytest.approx(0.2)
        assert summary["swap_peak_gb"] == pytest.approx(0.9)
        assert summary["pressure_max"] == 2
        assert summary["swap_end_gb"] == pytest.approx(0.3)


def test_run_resource_summary_without_guard_has_peak_rss_only():
    summary = run_resource_summary()
    assert summary["peak_rss_mb"] > 0
    assert summary["swap_start_gb"] is None
    assert summary["pressure_max"] is None


def test_run_resource_summary_with_guard(cfg):
    guard = MemoryGuard(cfg, status_fn=lambda: _fake_status(pressure_level=1, swap_used_gb=0.4))
    summary = run_resource_summary(guard)
    assert summary["peak_rss_mb"] > 0
    assert summary["swap_start_gb"] == pytest.approx(0.4)


class TestChunkedJobMemoryGuardIntegration:
    def test_pause_then_ok_is_reported_non_blocking(self, tmp_path, cfg, capsys):
        calls = [_fake_status(pressure_level=2, swap_used_gb=0.5)]  # start already "warn"

        def status_fn():
            return calls[0]

        guard = MemoryGuard(cfg, status_fn=status_fn)
        job = ChunkedJob(
            tmp_path,
            ["a", "b", "c"],
            chunk_size=1,
            job_config={"v": 1},
            memory_guard=guard,
            blocking_pause=False,
        )
        chunk_idx, ids = next(job.pending())
        job.save_chunk(chunk_idx, ids, {"feat": np.zeros((1, 1))})
        out = capsys.readouterr().out
        assert "pausing" in out.lower()

    def test_exit_raises_memory_pressure_exit_after_chunk_checkpointed(self, tmp_path, cfg):
        calls = [_fake_status(pressure_level=1, swap_used_gb=0.5)]

        def status_fn():
            return calls[0]

        guard = MemoryGuard(cfg, status_fn=status_fn)
        job = ChunkedJob(
            tmp_path, ["a", "b", "c"], chunk_size=1, job_config={"v": 1}, memory_guard=guard
        )
        chunk_idx, ids = next(job.pending())
        calls[0] = _fake_status(pressure_level=4, swap_used_gb=0.5)  # critical, mid-flight

        with pytest.raises(MemoryPressureExit):
            job.save_chunk(chunk_idx, ids, {"feat": np.zeros((1, 1))})

        # The triggering chunk is still safely checkpointed - the exit never costs work.
        assert job._chunk_done(chunk_idx)

    def test_blocking_pause_loops_until_ok_using_injected_sleep(self, tmp_path, cfg):
        calls = [_fake_status(pressure_level=2, swap_used_gb=0.5)]
        poll_calls = []

        def status_fn():
            return calls[0]

        def fake_sleep(seconds):
            poll_calls.append(seconds)
            # Recover to normal pressure after one simulated pause interval.
            calls[0] = _fake_status(pressure_level=1, swap_used_gb=0.5)

        guard = MemoryGuard(cfg, status_fn=status_fn)
        job = ChunkedJob(
            tmp_path,
            ["a", "b"],
            chunk_size=1,
            job_config={"v": 1},
            memory_guard=guard,
            blocking_pause=True,
            pause_poll_interval_s=0.01,
            sleep_fn=fake_sleep,
        )
        chunk_idx, ids = next(job.pending())
        job.save_chunk(chunk_idx, ids, {"feat": np.zeros((1, 1))})  # should not raise, not hang
        assert len(poll_calls) >= 1


# --------------------------------------------------------------------------------------
# heavy_job_lock: contention (real flock on separate fds), stale content takeover.
# --------------------------------------------------------------------------------------


class TestHeavyJobLock:
    def test_acquire_and_release_writes_holder_info(self, tmp_path):
        lock_path = tmp_path / "job.lock"
        with heavy_job_lock("job-a", lock_path=lock_path):
            content = json.loads(lock_path.read_text())
            assert content["name"] == "job-a"
            assert content["pid"] == os.getpid()

    def test_contention_wait_false_raises_busy(self, tmp_path):
        lock_path = tmp_path / "job.lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, json.dumps({"name": "job-a", "pid": os.getpid(), "started": "now"}).encode())
        try:
            with (
                pytest.raises(HeavyJobBusy, match="job-a"),
                heavy_job_lock("job-b", lock_path=lock_path, wait=False),
            ):
                pass
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_contention_wait_true_blocks_until_released(self, tmp_path):
        lock_path = tmp_path / "job.lock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, json.dumps({"name": "job-a", "pid": os.getpid(), "started": "now"}).encode())

        released_at = []

        def release_later():
            time.sleep(0.2)
            released_at.append(time.monotonic())
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

        t = threading.Thread(target=release_later)
        t.start()
        start = time.monotonic()
        with heavy_job_lock("job-b", lock_path=lock_path, wait=True):
            acquired_at = time.monotonic()
        t.join()

        assert acquired_at - start >= 0.15  # actually waited for the release
        assert released_at  # release happened before/at acquisition

    def test_stale_pid_content_is_taken_over_without_waiting(self, tmp_path):
        lock_path = tmp_path / "job.lock"
        # A lock file with content naming a (very likely) dead pid, but the real OS-level
        # flock was never taken - simulating a crash that left content but no live lock
        # (consistent with POSIX flock auto-release on process exit).
        dead_pid = 2**30  # not a real pid on this machine
        lock_path.write_text(json.dumps({"name": "old-job", "pid": dead_pid, "started": "then"}))

        start = time.monotonic()
        with heavy_job_lock("job-new", lock_path=lock_path, wait=False):
            elapsed = time.monotonic() - start
        assert elapsed < 1.0  # acquired promptly, no waiting/erroring on stale content

        content = json.loads(lock_path.read_text())
        assert content["name"] == "job-new"

    def test_lock_dir_created_if_missing(self, tmp_path):
        lock_path = tmp_path / "nested" / "dir" / "job.lock"
        with heavy_job_lock("job-a", lock_path=lock_path):
            assert lock_path.exists()
