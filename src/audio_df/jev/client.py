"""The JEV ("TypeSafe Jev API") client interface: text/JSON evidence in, typed probabilities
out.

This module defines the **shape** of a JEV call (`JevRequest`/`JevResponse`), the `JevClient`
ABC every implementation honours, and two concrete clients:

- `DryRunJevClient` — a deterministic, clearly-labelled stub. Its answers are a function of
  the evidence state's `detector.level` word only (never real JEV output); `route` is the
  literal string `"dry-run"` and `model_version` is `"dry-run-0"` specifically so a dry-run
  result can never be mistaken for a real one anywhere downstream (score files, the cache,
  the registry) — every place that reads `route`/`model_version` should treat `"dry-run"` as
  a loud, unmissable flag.
- `TypesafeJevClient` — the real client, **not yet implemented**. The JEV API's actual
  request/response schema, auth, and pricing have not been shared with this agent yet (the
  task brief is explicit: "the user has NOT yet sent the JEV docs"). `TypesafeJevClient.ask`
  therefore raises `NotImplementedError` naming exactly what is missing, rather than a silent
  stub that would look like a working integration. Construction still validates the one thing
  we do know today: `JEV_API_KEY` must be set. See the module-level TODO below for the docs
  question list.

**Security.** No client implementation may log, cache, or otherwise persist the API key or
any auth header. `TypesafeJevClient` reads `JEV_API_KEY` into a local variable inside
`__init__`, never assigns it to `self`, and never appears in `repr`/`vars()` — see
`tests/test_jev_client.py::test_typesafe_client_never_stores_the_api_key`.

TODO (blocks `TypesafeJevClient.ask`) — from the JEV docs, once shared:
  1. The request wire format: does the JEV endpoint take one question per call, or a batch?
     What field names does it expect for `prompt_text`/`state_json`/`options`/`scale`?
  2. Auth: header name and scheme (bearer? custom header?), base URL per `route`
     (`"typesafe"` vs `"jev-ai.pro"`), and whether `model_version` is a request field or
     implied by the endpoint/route.
  3. Response schema: exact field names for the typed probability distributions per question
     type (noul / choice / score), and where latency/cost are reported (response body?
     billing header? not reported at all, requiring us to time it client-side?).
  4. Pricing: cost per call (flat? per-token? per-question-type?) — needed to make the
     runner's `--max-cost-usd` guard exact rather than the current `--n <= 20` placeholder.
  5. Rate limits / retry semantics, so `ask()` can decide what is worth retrying.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from audio_df.jev.state import LEVEL_WORDS

__all__ = [
    "QuestionType",
    "Route",
    "JevRequest",
    "JevResponse",
    "JevClient",
    "DryRunJevClient",
    "TypesafeJevClient",
]

QuestionType = Literal["noul", "choice", "score"]
Route = Literal["typesafe", "jev-ai.pro", "dry-run"]

_DRY_RUN_ROUTE = "dry-run"
_DRY_RUN_MODEL_VERSION = "dry-run-0"


@dataclass(frozen=True)
class JevRequest:
    """One typed question over one clip's evidence state.

    `state_json` is whatever `audio_df.jev.state.build_state` produced — plain, JSON-
    serialisable data, never a clip id, path, label, attack id, or speaker.
    """

    question_id: str
    question_type: QuestionType
    prompt_text: str
    state_json: dict[str, Any]
    options: list[str] | None = None
    scale: tuple[int, int] | None = None


@dataclass(frozen=True)
class JevResponse:
    """A typed answer. `result` keys depend on `question_type`:

    - `"noul"`: `{"yes": p, "no": 1 - p}`.
    - `"choice"`: `{"bonafide": p, "spoof": p, "insufficient_evidence": p}` (sums to ~1).
    - `"score"`: `{"0": p, "1": p, ..., str(scale[1]): p}` plus `"expected_value"`.

    `route`/`model_version` identify *which* client produced this — `"dry-run"` /
    `"dry-run-0"` for the stub, never to be confused with a real answer. `raw` is a
    sanitised (no secrets) copy of whatever extra the client wants to keep for debugging.
    """

    question_id: str
    question_type: QuestionType
    result: dict[str, float]
    route: str
    model_version: str
    latency_s: float
    cost_usd: float | None
    raw: dict[str, Any] = field(default_factory=dict)


class JevClient(ABC):
    """Interface every JEV client implements. No network/SDK dependency lives here."""

    route: str
    model_version: str

    @abstractmethod
    def ask(self, request: JevRequest) -> JevResponse:
        """Answer one `JevRequest`. Must not mutate `request`."""


def _score_distribution(p_bonafide: float, lo: int, hi: int) -> dict[str, float]:
    """A deterministic, peaked distribution over `{lo, ..., hi}`, peak monotone in `p_bonafide`.

    Peak index is `round(p_bonafide * (hi - lo)) + lo`; weight falls off as `exp(-|v - peak|)`,
    normalised to sum to 1. Used only by `DryRunJevClient` — not a claim about what a real JEV
    score distribution looks like.
    """
    peak = lo + round(p_bonafide * (hi - lo))
    peak = min(max(peak, lo), hi)
    raw = {v: float(np.exp(-abs(v - peak))) for v in range(lo, hi + 1)}
    total = sum(raw.values())
    dist = {str(v): w / total for v, w in raw.items()}
    dist["expected_value"] = sum(v * w for v, w in raw.items()) / total
    return dist


class DryRunJevClient(JevClient):
    """Deterministic stub client — NEVER a real JEV result.

    The answer is a function of `state_json["detector"]["level"]` only (one of
    `audio_df.jev.state.LEVEL_WORDS`), mapped to a bona-fide probability
    `p = (level_index + 0.5) / len(LEVEL_WORDS)` — strictly increasing in the level's
    ordinal, so a more bona-fide-leaning A2 score always yields a higher `p(no)` on the
    noul question (the orientation the runner and `noul_to_a3_score` rely on). An unknown or
    missing level falls back to the middle index, never crashes on a malformed state.
    """

    route: str = _DRY_RUN_ROUTE
    model_version: str = _DRY_RUN_MODEL_VERSION

    def ask(self, request: JevRequest) -> JevResponse:
        t0 = time.monotonic()
        p_bonafide = self._p_bonafide(request.state_json)
        result = self._typed_result(request, p_bonafide)
        latency_s = time.monotonic() - t0
        return JevResponse(
            question_id=request.question_id,
            question_type=request.question_type,
            result=result,
            route=self.route,
            model_version=self.model_version,
            latency_s=latency_s,
            cost_usd=0.0,
            raw={"dry_run": True, "note": "DRY RUN stub answer — not a real JEV result"},
        )

    @staticmethod
    def _p_bonafide(state: dict[str, Any]) -> float:
        level = (state.get("detector") or {}).get("level")
        idx = LEVEL_WORDS.index(level) if level in LEVEL_WORDS else len(LEVEL_WORDS) // 2
        p = (idx + 0.5) / len(LEVEL_WORDS)
        return float(np.clip(p, 1e-4, 1.0 - 1e-4))

    @staticmethod
    def _typed_result(request: JevRequest, p_bonafide: float) -> dict[str, float]:
        if request.question_type == "noul":
            return {"yes": 1.0 - p_bonafide, "no": p_bonafide}
        if request.question_type == "choice":
            insufficient = 0.05
            rest = 1.0 - insufficient
            return {
                "bonafide": p_bonafide * rest,
                "spoof": (1.0 - p_bonafide) * rest,
                "insufficient_evidence": insufficient,
            }
        if request.question_type == "score":
            lo, hi = request.scale or (0, 3)
            return _score_distribution(p_bonafide, lo, hi)
        raise ValueError(f"unknown question_type {request.question_type!r}")


class TypesafeJevClient(JevClient):
    """The real JEV client — `ask()` is an explicit, loud stub (see module docstring TODO).

    Reads `JEV_API_KEY` from the environment at construction time, never storing it (or any
    derived auth header) on `self`. Raises `RuntimeError` immediately, with an actionable
    message, if the key is missing — this is checked eagerly so a misconfigured environment
    fails at client construction, not three hours into a run.
    """

    def __init__(self, model_version: str = "unknown", route: str = "typesafe") -> None:
        import os

        api_key = os.environ.get("JEV_API_KEY")
        if not api_key:
            raise RuntimeError(
                "JEV_API_KEY environment variable is not set. TypesafeJevClient needs a real "
                "JEV ('TypeSafe Jev API') key to run for real. Set JEV_API_KEY and retry — "
                "note that ask() itself still raises NotImplementedError until the JEV API "
                "docs are shared (see this module's docstring)."
            )
        # Deliberately not stored: `del api_key` makes the "never on self" intent explicit
        # even though the local binding would fall out of scope anyway at return.
        del api_key
        self.model_version = model_version
        self.route = route

    def ask(self, request: JevRequest) -> JevResponse:
        raise NotImplementedError(
            "TypesafeJevClient.ask is awaiting the JEV API docs from the user — see the "
            "TODO list in audio_df/jev/client.py's module docstring for exactly what is "
            "needed (wire format, auth, response schema, pricing)."
        )
