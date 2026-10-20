"""The three JEV questions asked over an A3 evidence state, and the question sets that
combine them.

Question text is **versioned** (`QUESTION_TEXT_VERSION`): the cache key covers the version,
so changing a question's wording invalidates cached answers rather than silently mixing
answers to two different questions under one key (`audio_df.models.jev.cache`).

**Score orientation (binding).** A3's per-clip score (written to `scores/jev_dev500.csv`,
project convention: higher = more bona fide) is derived from the **noul** question only,
as `log(p(no) / p(yes))` — the log-odds of "no, the evidence does not indicate synthetic
speech". `p(no)` is the bona-fide-leaning answer, so this is higher-is-bonafide, matching
every other score file in this project (`audio_df.runtime.exp_io` module docstring). See
`noul_to_a3_score` for the exact clipping applied to avoid `log(0)`.
"""

from __future__ import annotations

from dataclasses import dataclass

from audio_df.modules.jev.client import JevRequest, QuestionType

__all__ = [
    "QUESTION_TEXT_VERSION",
    "JevQuestionSpec",
    "NOUL_QUESTION",
    "CHOICE_QUESTION",
    "SCORE_QUESTION",
    "QUESTION_SETS",
    "build_request",
    "noul_to_a3_score",
]

QUESTION_TEXT_VERSION = "v1"


@dataclass(frozen=True)
class JevQuestionSpec:
    """One versioned question, independent of any particular clip's state."""

    id: str
    question_type: QuestionType
    text: str
    version: str = QUESTION_TEXT_VERSION
    options: tuple[str, ...] | None = None
    scale: tuple[int, int] | None = None


NOUL_QUESTION = JevQuestionSpec(
    id="noul",
    question_type="noul",
    text="Does the evidence indicate synthetic speech?",
)

CHOICE_QUESTION = JevQuestionSpec(
    id="choice",
    question_type="choice",
    text="Given the evidence, is this clip bona fide, spoof, or is the evidence insufficient?",
    options=("bonafide", "spoof", "insufficient_evidence"),
)

SCORE_QUESTION = JevQuestionSpec(
    id="score",
    question_type="score",
    text=(
        "On a scale of 0 (strongly inconsistent) to 3 (strongly consistent), how consistent "
        "is this evidence with genuine (bona fide) speech?"
    ),
    scale=(0, 3),
)

# `configs/a3_jev.yaml: variants.question_set` names one of these.
QUESTION_SETS: dict[str, tuple[JevQuestionSpec, ...]] = {
    "noul_only": (NOUL_QUESTION,),
    "all3": (NOUL_QUESTION, CHOICE_QUESTION, SCORE_QUESTION),
}


def build_request(question: JevQuestionSpec, state_json: dict) -> JevRequest:
    """Turn a question spec + evidence state into a client-ready `JevRequest`."""
    return JevRequest(
        question_id=question.id,
        question_type=question.question_type,
        prompt_text=question.text,
        state_json=state_json,
        options=list(question.options) if question.options is not None else None,
        scale=question.scale,
    )


def noul_to_a3_score(result: dict[str, float], eps: float = 1e-6) -> float:
    """`log(p(no) / p(yes))`, clipped away from 0/1 so this never returns `±inf`.

    Higher = more bona fide (see module docstring). `result` is a noul `JevResponse.result`
    dict, `{"yes": p, "no": 1 - p}`.
    """
    import math

    p_no = min(max(float(result["no"]), eps), 1.0 - eps)
    p_yes = min(max(float(result["yes"]), eps), 1.0 - eps)
    return math.log(p_no / p_yes)
