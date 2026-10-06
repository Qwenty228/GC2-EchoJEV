"""On-disk, git-committed cache of JEV request/response pairs.

Every real JEV call costs money, so this cache is **committed to git** (see task brief) —
files must stay small JSON, sorted keys, no secrets. `JevCache.get_or_call` never calls the
client for a key already on disk (`tests/test_jev_cache.py`, tested with a counting client).

## Cache key

`cache_key(clip_id, config_hash)` = `sha256(f"{clip_id}:{config_hash}")`, where
`config_hash(request, question_version, variant, route, model_version)` hashes the canonical
(sorted-key) JSON of `{question_id, question_version, variant, route, model_version, state}`
— i.e. every element the task brief lists ("question id + question text version, state
variant, route, model version, and the state JSON itself"). Because `state` (per-clip) is
part of `config_hash`, the hash is already effectively per-clip; `cache_key` additionally
folds in `clip_id` directly so two different clips that happened to produce byte-identical
states still get distinct keys (they would otherwise share a cache entry, which is correct
only by coincidence).

## Layout

`data/jev_cache/<config_hash[:12]>/<clip_id>__<question_id>.json`, containing:
`request` (sans secrets — `JevRequest` never carries any), `response`, `route`,
`model_version`, `latency_s`, `cost_usd`, `timestamp_utc`.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from audio_df.jev.client import JevClient, JevRequest, JevResponse

__all__ = ["JevCache", "config_hash", "cache_key"]


def config_hash(
    request: JevRequest,
    question_version: str,
    variant: str,
    route: str,
    model_version: str,
) -> str:
    """Hash every element that must invalidate a cached answer if it changes (module doc)."""
    payload = {
        "question_id": request.question_id,
        "question_version": question_version,
        "variant": variant,
        "route": route,
        "model_version": model_version,
        "state": request.state_json,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def cache_key(clip_id: str, cfg_hash: str) -> str:
    """`sha256(clip_id : config_hash)` — see module docstring."""
    return hashlib.sha256(f"{clip_id}:{cfg_hash}".encode()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".jevcache-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


class JevCache:
    """Content-addressed cache of `(clip_id, request-under-config)` -> `JevResponse`."""

    def __init__(self, root: str | Path = "data/jev_cache") -> None:
        self.root = Path(root)

    def path_for(
        self,
        clip_id: str,
        request: JevRequest,
        *,
        question_version: str,
        variant: str,
        route: str,
        model_version: str,
    ) -> Path:
        """The on-disk path a given `(clip_id, request, ...)` combination resolves to."""
        cfg_hash = config_hash(request, question_version, variant, route, model_version)
        return self.root / cfg_hash[:12] / f"{clip_id}__{request.question_id}.json"

    def get_or_call(
        self,
        clip_id: str,
        request: JevRequest,
        client: JevClient,
        *,
        question_version: str,
        variant: str,
    ) -> JevResponse:
        """Return the cached `JevResponse` for this key, else call `client.ask` and cache it.

        Never calls `client.ask` when the key is already on disk.
        """
        path = self.path_for(
            clip_id,
            request,
            question_version=question_version,
            variant=variant,
            route=client.route,
            model_version=client.model_version,
        )
        if path.exists():
            return self._load(path)

        response = client.ask(request)
        self._store(path, request, response)
        return response

    @staticmethod
    def _load(path: Path) -> JevResponse:
        payload = json.loads(path.read_text())
        r = payload["response"]
        return JevResponse(
            question_id=r["question_id"],
            question_type=r["question_type"],
            result=r["result"],
            route=r["route"],
            model_version=r["model_version"],
            latency_s=r["latency_s"],
            cost_usd=r["cost_usd"],
            raw=r.get("raw", {}),
        )

    @staticmethod
    def _store(path: Path, request: JevRequest, response: JevResponse) -> None:
        payload = {
            "request": asdict(request),
            "response": asdict(response),
            "route": response.route,
            "model_version": response.model_version,
            "latency_s": response.latency_s,
            "cost_usd": response.cost_usd,
            "timestamp_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        }
        _atomic_write_json(path, payload)
