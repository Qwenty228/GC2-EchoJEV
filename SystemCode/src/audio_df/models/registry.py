"""System name -> detector class. A new system (the fusion model) registers here once."""

from __future__ import annotations

from audio_df.models.a1_lfcc_gmm import LfccGmmDetector
from audio_df.models.a2_ssl_head import SslHeadDetector
from audio_df.models.base import Detector
from audio_df.models.jev.wrapper import JevEscalation

__all__ = ["SYSTEMS", "detector_class"]

SYSTEMS: dict[str, type[Detector]] = {
    LfccGmmDetector.system: LfccGmmDetector,
    SslHeadDetector.system: SslHeadDetector,
    JevEscalation.system: JevEscalation,
}


def detector_class(system: str) -> type[Detector]:
    if system not in SYSTEMS:
        raise KeyError(f"unknown system {system!r}; registered: {sorted(SYSTEMS)}")
    return SYSTEMS[system]
