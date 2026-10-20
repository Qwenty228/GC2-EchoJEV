"""TypeSafe Jev decision adapter; independent of audio and benchmark metrics.

Each request carries its own state, optional guidance, and typed questions.
The default questions/guidance live in configs/jev.yaml. Question outputs
are independent; a Score measures a rubric, not confidence in a Choice.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path
from dotenv import load_dotenv

from audio_df.helpers.config import load_config

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


@dataclass(frozen=True)
class JEVConfig:
    """Default Jev question definitions loaded from configs/jev.yaml."""

    prompt: str = ""
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class JEVState:
    """One decision case. Overrides apply only to this request.

    `questions` replaces the default question set when supplied; each entry
    has `type`, `instructions`, and optional `criteria` (Choice/Score).
    """

    context: Mapping[str, Any]
    prompt: str | None = None
    questions: Mapping[str, Mapping[str, Any]] | None = None


def _make_questions(specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    try:
        from typesafe_sdk import Choice, Noul, Score
    except ImportError as exc:
        raise RuntimeError("Install the SDK with: uv add typesafe-sdk") from exc

    if not specs:
        raise ValueError("JEV requires at least one question")
    result: dict[str, Any] = {}
    for name, spec in specs.items():
        kind = str(spec.get("type", "")).lower()
        instructions = spec.get("instructions")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError(f"Question {name!r} needs nonempty instructions")
        criteria = spec.get("criteria")
        if kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError(f"Choice {name!r} requires at least two criteria")
            result[name] = Choice(instructions=instructions, criteria=criteria)
        elif kind == "score":
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                raise ValueError(f"Score {name!r} requires 2–10 ordered criteria")
            result[name] = Score(instructions=instructions, criteria=criteria)
        elif kind == "noul":
            if criteria is not None:
                raise ValueError(f"Noul {name!r}: omit criteria or add SDK NoulCriteria support")
            result[name] = Noul(instructions=instructions)
        else:
            raise ValueError(f"Unknown question type {kind!r} for {name!r}")
    return result


def decide(
    state: JEVState | Mapping[str, Any],
    *,
    config: JEVConfig | None = None,
    model: str | None = None,
    client: Any = None,
) -> Any:
    """Ask Choice, Score and/or Noul questions about a specific decision case.

    A supplied `client` is injected (normally a mock in offline tests).
    Otherwise a real TypeSafe request uses TYPESAFE_API_KEY.
    """
    case = state if isinstance(state, JEVState) else JEVState(context=state)
    if not case.context:
        raise ValueError("JEV context must not be empty")
    cfg = config if config is not None else load_config(JEVConfig)
    guidance = cfg.prompt if case.prompt is None else case.prompt
    specs = cfg.questions if case.questions is None else case.questions
    questions = _make_questions(specs)
    # Jev has no global system prompt; supply guidance as part of state.
    payload = {"context": dict(case.context), "decision_guidance": guidance}
    if client is not None:
        return client.system_one(state=payload, questions=questions)
    if not os.getenv("TYPESAFE_API_KEY"):
        raise RuntimeError("Set TYPESAFE_API_KEY before calling Jev")
    try:
        from typesafe_sdk import TypeSafeClient
    except ImportError as exc:
        raise RuntimeError("Install the SDK with: uv add typesafe-sdk") from exc
    selected = model or os.getenv("JEV_MODEL", "jev-latest")
    with TypeSafeClient(model=selected) as sdk:
        return sdk.system_one(state=payload, questions=questions)
