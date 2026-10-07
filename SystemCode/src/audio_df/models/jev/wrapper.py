"""JEV as a wrapper around any detector: the A3 idea, decoupled from A2.

`JevEscalation(base, ...)` scores a clip with `base`, builds the label-free evidence state
(`models.jev.state.build_state`) from that score and the audio, asks JEV the configured
typed questions, and turns the yes/no ("noul") answer into a score,
`log(p(no) / p(yes))` (`models.jev.questions.noul_to_a3_score`, higher = more bona fide).

Two modes, set by `escalate_margin`:

- `None` (default) asks JEV about **every** clip and reports the JEV score. This is how the
  A3 runs (E5) are scored, and what an ablation "main model with / without JEV" compares.
- a number accepts the base verdict when `|score - base.threshold| >= escalate_margin`
  and asks JEV only inside that band ("accept when confident, escalate when unsure"). The
  reported score is then the base score for accepted clips and the JEV score for escalated
  ones, so it is a decision, not a single calibrated scale: use `None` for EER
  experiments.

The JEV threshold is 0.0 (p(no) = p(yes)). JEV sees text only, never audio, and is a
typed classifier service, not a large language model.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from audio_df.models.base import Detector, read_bundle_meta, write_bundle_meta
from audio_df.models.jev.cache import JevCache
from audio_df.models.jev.client import JevClient, build_client
from audio_df.models.jev.questions import (
    QUESTION_SETS,
    QUESTION_TEXT_VERSION,
    build_request,
    noul_to_a3_score,
)
from audio_df.models.jev.state import build_state

__all__ = ["JevEscalation"]

JEV_THRESHOLD = 0.0


class JevEscalation(Detector):
    """Wrap `base` with JEV. `jev_config` is the `configs/a3_jev.yaml` dict (thresholds,
    client, sample rate); its `thresholds.detector.eer_threshold` is always replaced by
    `base.threshold`, so the evidence state is relative to the base detector's own line."""

    system = "jev_escalation"

    def __init__(
        self,
        base: Detector,
        jev_config: dict[str, Any],
        *,
        state_variant: str = "numbers",
        question_set: str = "noul_only",
        escalate_margin: float | None = None,
        client: JevClient | None = None,
        cache: JevCache | None = None,
    ) -> None:
        if question_set not in QUESTION_SETS:
            known = sorted(QUESTION_SETS)
            raise ValueError(f"unknown question_set {question_set!r}; known: {known}")
        threshold = JEV_THRESHOLD if escalate_margin is None else base.threshold
        super().__init__(threshold, f"{base.model_version}+jev:{QUESTION_TEXT_VERSION}")
        self.base = base
        self.sample_rate = base.sample_rate
        self.jev_config = copy.deepcopy(jev_config)
        self.jev_config.setdefault("thresholds", {}).setdefault("detector", {})
        self.jev_config["thresholds"]["detector"]["eer_threshold"] = base.threshold
        self.jev_config["sample_rate"] = base.sample_rate
        self.state_variant = state_variant
        self.question_set = question_set
        self.escalate_margin = escalate_margin
        kind = self.jev_config.get("client", {}).get("kind", "dry-run")
        self.client = client if client is not None else build_client(kind, self.jev_config)
        self.cache = cache

    def ask(self, x: np.ndarray, base_score: float, clip_id: str | None = None) -> dict[str, Any]:
        """Ask every configured question about one clip; return the answers and JEV score.

        With a `cache` and a `clip_id`, answers come from (and go to) the on-disk cache, so
        a rerun never pays for the same call twice.
        """
        state = build_state(
            clip_id or "-", base_score, np.asarray(x), self.jev_config, variant=self.state_variant
        )
        answers: dict[str, dict[str, float]] = {}
        meta: dict[str, Any] = {}
        for question in QUESTION_SETS[self.question_set]:
            request = build_request(question, state)
            if self.cache is not None and clip_id is not None:
                response = self.cache.get_or_call(
                    clip_id,
                    request,
                    self.client,
                    question_version=QUESTION_TEXT_VERSION,
                    variant=self.state_variant,
                )
            else:
                response = self.client.ask(request)
            answers[response.question_id] = dict(response.result)
            meta = {"route": response.route, "jev_model_version": response.model_version}
        if "noul" not in answers:
            raise ValueError("the question set produced no 'noul' answer to score from")
        return {
            "jev_score": noul_to_a3_score(answers["noul"]),
            "answers": answers,
            "state": state,
            **meta,
        }

    def score_clip(self, x: np.ndarray, clip_id: str | None = None) -> tuple[float, dict[str, Any]]:
        base_score, base_evidence = self.base.score_waveform(x)
        evidence: dict[str, Any] = {
            "base_system": self.base.system,
            "base_score": base_score,
            "base_threshold": self.base.threshold,
            "base_evidence": base_evidence,
        }
        margin = abs(base_score - self.base.threshold)
        if self.escalate_margin is not None and margin >= self.escalate_margin:
            evidence["escalated"] = False
            return base_score, evidence
        jev = self.ask(x, base_score, clip_id)
        evidence.update(escalated=True, **jev)
        if self.escalate_margin is not None:
            # Escalated clips are decided by JEV: map its score onto the base threshold so
            # `decide(score, self.threshold)` uses JEV's own line (0.0).
            return self.base.threshold + (jev["jev_score"] - JEV_THRESHOLD), evidence
        return float(jev["jev_score"]), evidence

    def score_waveform(self, x: np.ndarray) -> tuple[float, dict[str, Any]]:
        return self.score_clip(x)

    # -- bundle ---------------------------------------------------------------

    def save(self, bundle_dir: str | Path) -> Path:
        bundle_dir = Path(bundle_dir)
        self.base.save(bundle_dir / "base")
        write_bundle_meta(
            bundle_dir,
            {
                "system": self.system,
                "model_version": self.model_version,
                "threshold": self.threshold,
                "jev_config": self.jev_config,
                "state_variant": self.state_variant,
                "question_set": self.question_set,
                "escalate_margin": self.escalate_margin,
            },
        )
        return bundle_dir

    @classmethod
    def load(cls, bundle_dir: str | Path) -> JevEscalation:
        from audio_df.inference import load_detector  # the registry imports this module

        bundle_dir = Path(bundle_dir)
        meta = read_bundle_meta(bundle_dir)
        return cls(
            load_detector(bundle_dir / "base"),
            meta["jev_config"],
            state_variant=meta["state_variant"],
            question_set=meta["question_set"],
            escalate_margin=meta["escalate_margin"],
        )
