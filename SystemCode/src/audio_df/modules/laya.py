"""Convai LAYA decision adapter (Phase 2 P2 decision policy).

An open-source, local, free decision engine alternative to JEV.
Executes non-autoregressive 'System 1' inference over structured evidence.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from audio_df.helpers.config import load_config
from audio_df.modules.jev import JEVState

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)


@dataclass(frozen=True)
class LAYAConfig:
    """Default LAYA question definitions loaded from configs/laya.yaml."""

    prompt: str = ""
    questions: dict[str, dict[str, Any]] = field(default_factory=dict)
    model: str = "convaiinnovations/laya"
    device: str = "auto"


# Re-export state for convenience
LAYAState = JEVState


class DryRunLayaClient:
    """Deterministic offline mock client for smoke tests and dry runs."""

    def system_one(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        context = state.get("context", {}) if isinstance(state, dict) else {}
        margin = float(context.get("signed_margin", 0.0))
        # Margin-based decision policy:
        # Near operating threshold (|margin| < 0.15) -> abstain as uncertain
        # Strong positive margin -> authentic (bonafide)
        # Strong negative margin -> synthetic (spoof)
        if abs(margin) < 0.15:
            verdict = "uncertain"
            probs = {"bonafide": 0.33, "spoof": 0.33, "uncertain": 0.34}
            qual = 0
            ambig = 0.85
        elif margin > 0:
            verdict = "bonafide"
            probs = {"bonafide": 0.88, "spoof": 0.08, "uncertain": 0.04}
            qual = 2
            ambig = 0.10
        else:
            verdict = "spoof"
            probs = {"bonafide": 0.08, "spoof": 0.88, "uncertain": 0.04}
            qual = 2
            ambig = 0.10

        return {
            "model": "laya-dry-run",
            "usage": {"input_tokens": 20, "output_tokens": 1},
            "answers": {
                "decision": {
                    "type": "choice",
                    "choice": verdict,
                    "probabilities": probs,
                    "confidence": 0.88,
                },
                "evidence_quality": {
                    "type": "score",
                    "score": qual,
                    "confidence": 0.85,
                },
                "ambiguity": {
                    "type": "noul",
                    "noul": ambig,
                    "confidence": 0.85,
                },
            },
        }


_CACHED_AGENT: Any = None
_CACHED_MODEL_NAME: str | None = None


def get_laya_agent(model_name: str, device: str = "auto") -> Any:
    """Load and cache the LAYA local agent singleton."""
    global _CACHED_AGENT, _CACHED_MODEL_NAME

    if model_name == "dry-run" or os.getenv("LAYA_DRY_RUN", "0") == "1":
        return DryRunLayaClient()

    local_model_path = Path(__file__).resolve().parents[3] / "data" / "models" / "laya"
    if model_name == "convaiinnovations/laya" and local_model_path.is_dir():
        model_name = str(local_model_path)

    if _CACHED_AGENT is not None and model_name == _CACHED_MODEL_NAME:
        return _CACHED_AGENT

    try:
        import laya
    except ImportError as exc:
        raise RuntimeError("Install laya with: uv add laya") from exc

    resolved_device = device
    if device == "auto":
        import torch

        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    _CACHED_AGENT = laya.load(model_id_or_path=model_name, device=resolved_device)
    _CACHED_MODEL_NAME = model_name
    return _CACHED_AGENT


def decide(
    state: JEVState | Mapping[str, Any],
    *,
    config: LAYAConfig | None = None,
    model: str | None = None,
    client: Any = None,
    **kwargs: Any,
) -> Any:
    """Execute a local, zero-cost decision on the evidence state using Laya."""
    case = state if isinstance(state, JEVState) else JEVState(context=state)
    if not case.context:
        raise ValueError("LAYA context must not be empty")

    cfg = config if config is not None else load_config(LAYAConfig)
    guidance = cfg.prompt if case.prompt is None else case.prompt
    questions = dict(cfg.questions if case.questions is None else case.questions)

    if not questions:
        raise ValueError("LAYA requires at least one question")

    payload = {"context": dict(case.context), "decision_guidance": guidance}

    if client is not None:
        return client.system_one(state=payload, questions=questions)

    selected_model = model or os.getenv("LAYA_MODEL", cfg.model)
    agent = get_laya_agent(selected_model, device=cfg.device)
    return agent.system_one(state=payload, questions=questions)
