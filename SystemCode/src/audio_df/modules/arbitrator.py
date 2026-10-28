"""Arbitrator: a judge (LAYA or JEV) decides which verdict to trust when detectors disagree.

The judge is fixed in configs/arbitrator.yaml. A request carries each detector's evidence in listing
order (name, verdict, P(bona fide), error rate); it never carries a clip id or label.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from audio_df.helpers.config import load_config
from audio_df.modules import jev, laya
from audio_df.modules.jev import JEVState

__all__ = ["ArbitratorConfig", "VARIANTS", "request", "decide"]


@dataclass(frozen=True)
class ArbitratorConfig:
    judge: str = "laya"


JUDGES = {"laya": laya.decide, "jev": jev.decide}
VARIANTS = ("verdicts", "probabilities", "reliability", "reliability order", "combined")
FIELDS = {
    "verdicts": ("verdict",),
    "probabilities": ("verdict", "p_bonafide"),
    "reliability": ("verdict", "p_bonafide", "error_rate_pct"),
}
QUESTION = {"decision": {
    "type": "choice", "instructions": "Which verdict should be trusted for this clip?",
    "criteria": {"bonafide": "The trustworthy evidence says authentic speech.",
                 "spoof": "The trustworthy evidence says spoofed speech.",
                 "uncertain": "The evidence conflicts too much to decide."}}}
BASE = ("Several spoofing detectors judged the same clip and disagree. "
        "Decide which verdict to trust.")
COMBINED = ("Detector evidence for one clip was combined, weighting each detector by its "
            "reliability. Answer uncertain when weighted_p_bonafide is near 0.5.")


def _shown(e: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {k: round(float(e[k]), 3) if k == "p_bonafide" else e[k] for k in keys}


def request(evidence: list[dict[str, Any]], variant: str) -> JEVState:
    """Evidence: one dict per detector, in listing order, with name, verdict, p_bonafide and
    error_rate_pct.

    verdicts / probabilities / reliability: each detector under its name, more fields per variant.
    reliability order: anonymised, most reliable first.
    combined: one reliability-weighted P(bona fide) and the vote count.
    """
    if variant == "reliability order":
        ranked = sorted(evidence, key=lambda e: e["error_rate_pct"])
        fields = FIELDS["probabilities"]
        ctx = {f"detector_{i + 1}": _shown(e, fields) for i, e in enumerate(ranked)}
        ranked_prompt = BASE + " Detectors are listed from most to least reliable."
        return JEVState(ctx, ranked_prompt, QUESTION)
    if variant == "combined":
        weights = [1 / e["error_rate_pct"] for e in evidence]
        p = sum(w * e["p_bonafide"] for w, e in zip(weights, evidence, strict=True)) / sum(weights)
        n_bona = sum(e["verdict"] == "bonafide" for e in evidence)
        ctx = {"weighted_p_bonafide": round(float(p), 3),
               "detectors_saying_bonafide": f"{n_bona} of {len(evidence)}"}
        return JEVState(ctx, COMBINED, QUESTION)
    ctx = {e["name"]: _shown(e, FIELDS[variant]) for e in evidence}
    extra = " Trust detectors with a lower error_rate_pct more." if variant == "reliability" else ""
    return JEVState(ctx, BASE + extra, QUESTION)


def decide(state: JEVState, config: ArbitratorConfig | None = None) -> dict[str, Any]:
    """The configured judge's decision: choice (bonafide, spoof or uncertain) and probabilities."""
    cfg = config if config is not None else load_config(ArbitratorConfig)
    response = JUDGES[cfg.judge](state)
    if hasattr(response, "model_dump"):  # JEV returns an SDK object, LAYA a dict
        response = response.model_dump(mode="json")
    return response["answers"]["decision"]
