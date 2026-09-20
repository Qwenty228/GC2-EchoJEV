"""Tests for the content-hashed feature cache (src/audio_df/cache.py).

No dataset is required: arrays are synthesised with numpy. These tests exist to catch the
single most dangerous failure mode in the project — a cache that silently returns stale or
corrupt features — so several tests deliberately try to trick `get()` into doing that.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from audio_df.cache import FeatureCache, cache_key

# --------------------------------------------------------------------------- cache_key


def test_same_content_different_insertion_order_same_key():
    a = {"n_fft": 512, "hop": 160, "spacing": "linear"}
    b = {"spacing": "linear", "hop": 160, "n_fft": 512}
    assert cache_key("utt1", a) == cache_key("utt1", b)


def test_nested_dict_and_list_order_canonicalised():
    a = {"outer": {"x": 1, "y": 2}, "list_field": {"a": 1, "b": 2}}
    b = {"list_field": {"b": 2, "a": 1}, "outer": {"y": 2, "x": 1}}
    assert cache_key("utt1", a) == cache_key("utt1", b)


def test_list_order_is_significant():
    # Lists (as opposed to dict keys) are NOT reordered: [1, 2] != [2, 1].
    a = {"seq": [1, 2, 3]}
    b = {"seq": [3, 2, 1]}
    assert cache_key("utt1", a) != cache_key("utt1", b)


def test_changing_top_level_field_changes_key():
    base = {"n_fft": 512, "hop": 160}
    changed = {"n_fft": 1024, "hop": 160}
    assert cache_key("utt1", base) != cache_key("utt1", changed)


def test_changing_nested_field_changes_key():
    base = {"stft": {"n_fft": 512, "hop": 160}, "filterbank": {"n_filters": 20}}
    changed = {"stft": {"n_fft": 512, "hop": 160}, "filterbank": {"n_filters": 40}}
    assert cache_key("utt1", base) != cache_key("utt1", changed)


def test_different_utt_id_changes_key():
    config = {"n_fft": 512}
    assert cache_key("utt1", config) != cache_key("utt2", config)


def test_int_and_float_with_same_numeric_value_produce_same_key():
    # 1 vs 1.0: we normalise numeric type so a YAML loader emitting either does not
    # silently split the cache in two for the "same" config.
    a = {"n_coeff": 20}
    b = {"n_coeff": 20.0}
    assert cache_key("utt1", a) == cache_key("utt1", b)


def test_numeric_value_that_actually_differs_changes_key():
    a = {"n_coeff": 20}
    b = {"n_coeff": 20.5}
    assert cache_key("utt1", a) != cache_key("utt1", b)


def test_bool_is_not_conflated_with_int():
    # In Python, True == 1 and hash(True) == hash(1); a naive normaliser could conflate
    # `deltas: true` with `deltas: 1`. We keep them distinct in the canonical form only
    # insofar as it matters for real configs (bools stay bools).
    a = {"deltas": True}
    b = {"deltas": 1}
    # This is a documented decision: bools are kept as a distinct JSON type from ints.
    assert cache_key("utt1", a) != cache_key("utt1", b)


def test_none_value_handled():
    a = {"n_coeff": None}
    b = {"n_coeff": None}
    assert cache_key("utt1", a) == cache_key("utt1", b)
    c = {"n_coeff": 13}
    assert cache_key("utt1", a) != cache_key("utt1", c)


def test_key_is_stable_across_calls():
    config = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}
    k1 = cache_key("utt1", config)
    k2 = cache_key("utt1", config)
    assert k1 == k2


def test_key_is_a_hex_string():
    key = cache_key("utt1", {"n_fft": 512})
    assert isinstance(key, str)
    int(key, 16)  # must be valid hex


def test_numpy_scalar_config_values_normalised():
    # A config accidentally holding a numpy scalar (e.g. from yaml + numpy post-processing)
    # must hash the same as the plain-python equivalent.
    a = {"n_fft": np.int64(512)}
    b = {"n_fft": 512}
    assert cache_key("utt1", a) == cache_key("utt1", b)


def test_type_confusion_matrix():
    # A fractional float and the string of that float must NOT collide, even though the
    # int/float unification rule intentionally makes int 20 and float 20.0 collide. Every
    # pair here is a distinct JSON type carrying visually/textually similar content; only
    # the genuinely-equal-value int/float pairs are allowed to match.
    def k(v):
        return cache_key("utt1", {"x": v})

    # float vs. its own string representation: must differ.
    assert k(20.5) != k("20.5")
    # int vs. its own string representation: must differ.
    assert k(20) != k("20")
    # int vs. float with the same value: must still be unified (documented exception).
    assert k(20) == k(20.0)
    # bool vs. int, vs. their string forms: all four must be pairwise distinct.
    assert k(True) != k(1)
    assert k(True) != k("True")
    assert k(1) != k("1")
    assert k(False) != k(0)
    assert k(False) != k("False")
    assert k(0) != k("0")
    # None vs. the strings "None" / "null": all three must be pairwise distinct.
    assert k(None) != k("None")
    assert k(None) != k("null")
    assert k("None") != k("null")
    # sanity: identical values still collide with themselves.
    assert k(20.5) == k(20.5)
    assert k("20.5") == k("20.5")
    assert k(None) == k(None)


# --------------------------------------------------------------------------- FeatureCache


def test_put_get_roundtrip_exact(tmp_path):
    cache = FeatureCache(tmp_path)
    key = cache_key("utt1", {"n_fft": 512})
    array = np.random.default_rng(0).standard_normal((20, 100)).astype(np.float32)
    cache.put(key, array, meta={"n_fft": 512})

    result = cache.get(key)
    assert result is not None
    np.testing.assert_array_equal(result, array)
    assert result.dtype == array.dtype


def test_get_miss_returns_none(tmp_path):
    cache = FeatureCache(tmp_path)
    assert cache.get("nonexistent_key") is None


def test_sidecar_json_contains_full_config(tmp_path):
    cache = FeatureCache(tmp_path)
    config = {"n_fft": 512, "hop": 160, "spacing": "linear", "nested": {"x": 1}}
    key = cache_key("utt1", config)
    array = np.zeros((5, 5), dtype=np.float32)
    cache.put(key, array, meta={"config": config, "utt_id": "utt1"})

    sidecar_path = tmp_path / f"{key}.json"
    assert sidecar_path.exists()
    sidecar = json.loads(sidecar_path.read_text())
    assert sidecar["meta"]["config"] == config
    assert sidecar["shape"] == [5, 5]
    assert sidecar["dtype"] == "float32"
    assert "timestamp" in sidecar


def test_atomicity_no_partial_write_visible(tmp_path):
    # Simulate a process crashing mid-write: only a .tmp file exists, never a finished
    # <key>.npy. A concurrent/subsequent get() must see a miss, not a partial array.
    cache = FeatureCache(tmp_path)
    key = "some_key"
    (tmp_path / f"{key}.npy.tmp").write_bytes(b"not a real npy file, truncated")
    assert cache.get(key) is None


def test_corrupt_npy_does_not_return_garbage(tmp_path):
    # A finished-looking <key>.npy that is actually corrupt/truncated must not be silently
    # loaded and handed back as if it were valid data. We choose: get() raises.
    cache = FeatureCache(tmp_path)
    key = "corrupt_key"
    (tmp_path / f"{key}.npy").write_bytes(b"garbage, not a valid npy file at all")

    with pytest.raises((ValueError, OSError, EOFError)):
        cache.get(key)


def test_stats_reports_entry_count(tmp_path):
    cache = FeatureCache(tmp_path)
    assert cache.stats()["n_entries"] == 0

    for i in range(3):
        key = cache_key(f"utt{i}", {"n_fft": 512})
        cache.put(key, np.zeros((2, 2), dtype=np.float32), meta={})

    stats = cache.stats()
    assert stats["n_entries"] == 3
    assert stats["bytes_on_disk"] > 0


def test_put_returns_path(tmp_path):
    cache = FeatureCache(tmp_path)
    key = cache_key("utt1", {"n_fft": 512})
    path = cache.put(key, np.zeros((2, 2), dtype=np.float32), meta={})
    assert path.exists()
    assert path.name == f"{key}.npy"


def test_put_leaves_no_stray_files(tmp_path):
    # Regression test: np.save() silently appends ".npy" to a target filename that
    # doesn't already end in ".npy". Our temp file is named "<key>.<rand>.npy.tmp", which
    # does NOT end in ".npy" as a string, so a naive `np.save(tmp_path, array)` call would
    # have written to "<key>.<rand>.npy.tmp.npy" instead of the intended temp path, and
    # the later `os.replace(tmp_path, npy_path)` would then either fail (tmp_path never
    # created) or silently move nothing. put() must produce exactly one file per key, at
    # exactly the documented path — nothing double-suffixed left behind.
    cache = FeatureCache(tmp_path)
    key = cache_key("utt1", {"n_fft": 512})
    cache.put(key, np.zeros((3, 3), dtype=np.float32), meta={})

    all_files = {p.name for p in tmp_path.iterdir()}
    assert all_files == {f"{key}.npy", f"{key}.json"}
    assert cache.get(key) is not None


def test_creates_root_dir_if_missing(tmp_path):
    root = tmp_path / "does" / "not" / "exist"
    cache = FeatureCache(root)
    key = cache_key("utt1", {"n_fft": 512})
    cache.put(key, np.zeros((2, 2), dtype=np.float32), meta={})
    assert cache.get(key) is not None
