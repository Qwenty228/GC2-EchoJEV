"""Content-hashed on-disk cache for extracted feature arrays.

LFCC/SSL extraction is the slowest step of every experiment, so every variant × utterance
pair is cached under a key derived from its full extraction config. A **stale cache silently
returning the wrong array is the most dangerous failure mode in this project** — it would
corrupt every downstream number without raising anything. This module is designed around
that risk:

- The key is a hash of the *entire* config plus the utterance id, canonicalised so that
  dict key order, list-vs-tuple, and int-vs-float representation of the same number never
  cause a false cache miss, while any real change to the config always changes the key.
- Writes are atomic (write to a temp file, then `os.replace`), so a process killed mid-write
  can never leave a `.npy` that looks finished but is truncated.
- A `get()` that finds a `.npy` which fails to load (truncated, corrupted, foreign format)
  raises rather than returning `None` or partial data — a raised exception is loud and stops
  a pipeline; a silent `None` would look identical to "never computed" and a badly-written
  caller could paper over it by just recomputing and moving on, but a corrupt *existing*
  file is a different, more alarming situation (bit rot, disk fault, non-atomic write from
  outside this module) that deserves to fail the run, not be quietly repaired.

This module knows nothing about what a feature array *means* — no filterbank, no DCT, no
sample rate handling. It only stores and retrieves arrays by key. See the module boundary
table in `CONVENTIONS.md`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["cache_key", "FeatureCache"]


def _canonicalise(value: Any) -> Any:
    """Recursively normalise a config value into a JSON-stable, **type-tagged** form.

    Every scalar leaf is encoded as a 2-element `[tag, payload]` list, where `tag` is one
    of `"n"` (number), `"s"` (string), `"b"` (bool), `"z"` (null). This exists to close a
    real collision: two *different* JSON types that happen to `repr()`/serialise to the
    same text must never hash the same, or a stale cache could return features computed
    under an entirely different config — the exact failure mode this module is built to
    prevent. Concretely, without tagging, a fractional float and the string of that float
    (`20.5` vs `"20.5"`) collided, because both ultimately produced the JSON string
    `"20.5"`. Tagging makes that structurally impossible: a float becomes `["n", "20.5"]`,
    the string becomes `["s", "20.5"]` — different at the first element, so `json.dumps`
    never emits the same bytes for both. Containers (dict/list) are not tagged themselves;
    every leaf they contain is tagged, which is sufficient — see the note at the bottom.

    Rules (see the module docstring and report for the rationale):
    - dicts: sort keys recursively, canonicalise values. Two dicts with the same
      key/value pairs hash identically regardless of insertion order.
    - lists/tuples: order is preserved and treated as significant (`[1, 2] != [2, 1]`),
      since ordered fields such as delta-window taps could plausibly appear in a config.
      Each element is canonicalised recursively; tuples are normalised to lists so a
      list and an equal-content tuple hash the same.
    - numpy scalars: converted to their native Python equivalent first (`np.int64(512)`
      behaves like `512`), then the numeric rule below applies.
    - int vs. float with the same numeric value (`1` vs `1.0`): normalised to the same
      tagged representation, `["n", 1]`. Config values arrive from YAML or from
      post-processing that may promote an int to a float (e.g. `20` vs `20.0` for
      `n_coeff`); if this hashed differently the cache would silently split into two
      entries for what an experimenter considers the "same" config. We normalise every
      `int`/`float` that has no fractional part to `["n", <plain int>]`, and every other
      float to `["n", repr(value)]` (a string payload — stable, round-trips exactly,
      avoids float-formatting drift across platforms, and stays inside the `"n"` tag so
      it can never be confused with an actual string value, which is tagged `"s"`).
    - `bool`: tagged `"b"` and checked *before* the int/float rule (Python's `bool` is an
      `int` subclass and `True == 1`), because otherwise `deltas: true` could silently
      conflate with a hypothetical `deltas: 1` — a real config typo we do not want the
      cache to shrug off.
    - `None`: tagged `"z"`, distinct from the strings `"None"` or `"null"` (tagged `"s"`).
    - `str`: tagged `"s"`.

    Why tagging leaves is enough and containers need no tag of their own: after this
    recursion, every dict/list in the output tree has, at every depth, either another
    dict/list, or a 2-element `[tag, payload]` list at its leaves — never a bare scalar.
    A real config list such as `["f", "20.5"]` (two plain strings) is *not* left as-is; it
    recurses into `[["s", "f"], ["s", "20.5"]]`, which cannot be confused with a single
    tagged float leaf `["n", "20.5"]` (different length and different first element).
    """
    if isinstance(value, bool):
        return ["b", value]
    if isinstance(value, np.generic):
        return _canonicalise(value.item())
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return {str(k): _canonicalise(v) for k, v in items}
    if isinstance(value, (list, tuple)):
        return [_canonicalise(v) for v in value]
    if value is None:
        return ["z", None]
    if isinstance(value, int):
        return ["n", value]
    if isinstance(value, float):
        if value.is_integer():
            return ["n", int(value)]
        return ["n", repr(value)]
    if isinstance(value, str):
        return ["s", value]
    return ["?", repr(value)]


def cache_key(utt_id: str, config: dict) -> str:
    """Stable content hash over `(utt_id, config)`.

    Two calls with configs that differ only in dict-key insertion order, or in int-vs-float
    representation of an unchanged numeric value, produce the identical key. Any change to
    a value that is not one of those cosmetic differences — including a change buried in a
    nested dict — produces a different key.

    The hash is SHA-256 of the canonical JSON encoding, truncated to 32 hex chars (128
    bits). At the scale of this project (thousands of utterances x a handful of variants),
    a 128-bit truncation leaves collision probability astronomically below any other source
    of error in the pipeline; see the report's claim block for the exact figure.
    """
    canonical = _canonicalise({"utt_id": utt_id, "config": config})
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


class FeatureCache:
    """On-disk store mapping a `cache_key()` to a `.npy` array plus an auditable sidecar.

    Layout under `root`: `<key>.npy` (the array, written atomically) and `<key>.json`
    (the full `meta` dict the caller supplied, the array's shape/dtype, and a UTC
    timestamp) — the sidecar is what makes a cached entry auditable after the fact,
    independent of whatever code produced it.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _npy_path(self, key: str) -> Path:
        return self.root / f"{key}.npy"

    def _json_path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> np.ndarray | None:
        """Return the cached array for `key`, or `None` if there is no entry.

        A missing `.npy` (never written, or only a `.tmp` exists because a previous write
        was interrupted) is a clean miss: `None`. A `.npy` that *exists* but fails to load
        is treated as corruption, not a miss, and raises `OSError` / `ValueError` from
        `numpy.load` — see the module docstring for why this is deliberate.
        """
        path = self._npy_path(key)
        if not path.exists():
            return None
        return np.load(path, allow_pickle=False)

    def put(self, key: str, array: np.ndarray, meta: dict) -> Path:
        """Write `array` under `key`, plus a sidecar JSON with `meta`, shape, dtype, timestamp.

        The `.npy` is written to a temp file in the same directory and then atomically
        renamed (`os.replace`) into place, so a crash mid-write leaves only an orphaned
        `.tmp` file — never a truncated file at the final path that a later `get()` could
        load and mistake for valid data. The sidecar is written the same way.
        """
        npy_path = self._npy_path(key)
        json_path = self._json_path(key)

        fd, tmp_name = tempfile.mkstemp(dir=self.root, prefix=f"{key}.", suffix=".npy.tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            # np.save appends ".npy" to the filename if it doesn't already end with it,
            # which would silently create a differently-named file next to our intended
            # temp path. Passing an open file object bypasses that suffix logic entirely.
            with open(tmp_path, "wb") as f:
                np.save(f, array, allow_pickle=False)
            os.replace(tmp_path, npy_path)
        finally:
            tmp_path.unlink(missing_ok=True)

        sidecar = {
            "key": key,
            "meta": meta,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "timestamp": datetime.now(UTC).isoformat(),
        }
        fd, tmp_json_name = tempfile.mkstemp(dir=self.root, prefix=f"{key}.", suffix=".json.tmp")
        os.close(fd)
        tmp_json_path = Path(tmp_json_name)
        try:
            tmp_json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
            os.replace(tmp_json_path, json_path)
        finally:
            tmp_json_path.unlink(missing_ok=True)

        return npy_path

    def stats(self) -> dict:
        """Summary of what is on disk: entry count and total bytes of `.npy` files."""
        npy_files = list(self.root.glob("*.npy"))
        return {
            "n_entries": len(npy_files),
            "bytes_on_disk": sum(f.stat().st_size for f in npy_files),
        }
