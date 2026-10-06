"""build_state: no label leakage, both variants JSON-serialisable and deterministic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from audio_df.models.jev.state import LEVEL_WORDS, build_state

_CFG = {
    "sample_rate": 16000,
    "thresholds": {
        "detector": {
            "eer_threshold": 0.0,
            "margins": [0.05, 0.2],
            "confidence_words": ["low", "medium", "high"],
        },
        "acoustic": {
            "high_band_hz": 4000,
            "high_band": [0.05, 0.15],
            "high_band_words": ["low", "medium", "high"],
            "silence_top_db": 30,
            "silence": [0.05, 0.2],
            "silence_words": ["little", "some", "a lot"],
            "duration": [2.0, 6.0],
            "duration_words": ["short", "medium", "long"],
        },
    },
}


def _tone(freq=440.0, sr=16000, seconds=1.0, amp=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("variant", ["numbers", "words"])
def test_json_serialisable(variant):
    audio = _tone()
    state = build_state("SOME_CLIP_ID_42", 0.3, audio, _CFG, variant=variant)
    encoded = json.dumps(state)
    assert isinstance(encoded, str)


@pytest.mark.parametrize("variant", ["numbers", "words"])
def test_deterministic(variant):
    audio = _tone()
    a = build_state("clip-x", 0.3, audio, _CFG, variant=variant)
    b = build_state("clip-x", 0.3, audio, _CFG, variant=variant)
    assert a == b


@pytest.mark.parametrize("variant", ["numbers", "words"])
@pytest.mark.parametrize("score", [-1.0, -0.1, 0.0, 0.1, 1.0])  # all five LEVEL_WORDS
def test_no_label_leakage(variant, score):
    """The state may carry A2's opinion and acoustic measurements, never the clip's identity.

    A2's level word ("strongly spoof", "bona fide", ...) is the detector's *opinion* and is
    meant to be in the state, so searching the JSON for "spoof"/"bonafide" would wrongly fail
    every fake-leaning clip. Instead, check what leakage actually means:
    (1) the state does not depend on which clip it is: two different clip ids with the same
        score and audio give byte-identical JSON, so no identity can be in it;
    (2) no key anywhere names an identity or label field;
    (3) the clip id string itself never appears.
    """
    audio = _tone()
    clip_a = "LA_D_9999999_SPEAKER_SECRET"
    clip_b = "ITW_31778_some_celebrity"
    encoded_a = json.dumps(build_state(clip_a, score, audio, _CFG, variant=variant), sort_keys=True)
    encoded_b = json.dumps(build_state(clip_b, score, audio, _CFG, variant=variant), sort_keys=True)

    assert encoded_a == encoded_b, "state changes with clip identity"
    assert clip_a not in encoded_a and clip_b not in encoded_b

    def _keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from _keys(v)

    forbidden_keys = {"label", "attack_id", "attack", "speaker", "speaker_id", "path", "clip_id",
                      "utt_id", "split"}
    leaked = forbidden_keys & set(_keys(json.loads(encoded_a)))
    assert not leaked, f"identity/label keys in state: {leaked}"


@pytest.mark.parametrize("variant", ["numbers", "words"])
def test_state_keys_are_exactly_the_documented_ones(variant):
    """A fixed key set, so a future field (e.g. a file name) cannot slip in unnoticed."""
    state = build_state("clip-1", -1.0, _tone(), _CFG, variant=variant)
    assert set(state) == {"variant", "detector", "acoustic"}
    detector_keys = {"level", "confidence"} | ({"score"} if variant == "numbers" else set())
    assert set(state["detector"]) == detector_keys
    assert set(state["acoustic"]) == {
        "high_band_energy", "silence_leading", "silence_trailing", "silence_total", "duration_s"
    }


def test_numbers_variant_has_numeric_score():
    audio = _tone()
    state = build_state("clip-1", 0.42, audio, _CFG, variant="numbers")
    assert state["detector"]["score"] == pytest.approx(0.42)
    assert isinstance(state["acoustic"]["high_band_energy"], float)


def test_words_variant_has_no_numeric_score():
    audio = _tone()
    state = build_state("clip-1", 0.42, audio, _CFG, variant="words")
    assert "score" not in state["detector"]
    assert isinstance(state["acoustic"]["high_band_energy"], str)
    assert isinstance(state["detector"]["confidence"], str)


@pytest.mark.parametrize(
    "score,expected_level",
    [
        (-1.0, "strongly spoof"),
        (-0.1, "spoof"),
        (0.0, "borderline"),
        (0.1, "bona fide"),
        (1.0, "strongly bona fide"),
    ],
)
def test_level_word_ordering(score, expected_level):
    audio = _tone()
    state = build_state("clip-1", score, audio, _CFG, variant="numbers")
    assert state["detector"]["level"] == expected_level
    assert expected_level in LEVEL_WORDS


def test_level_word_monotone_in_score():
    audio = _tone()
    scores = [-1.0, -0.1, 0.0, 0.1, 1.0]
    indices = [
        LEVEL_WORDS.index(build_state("c", s, audio, _CFG, variant="numbers")["detector"]["level"])
        for s in scores
    ]
    assert indices == sorted(indices)


def test_high_band_energy_responds_to_frequency_content():
    sr = 16000
    audio_cfg = dict(_CFG)
    low_tone = _tone(freq=200.0, sr=sr)
    high_tone = _tone(freq=6000.0, sr=sr)
    low_state = build_state("c", 0.0, low_tone, audio_cfg, variant="numbers")
    high_state = build_state("c", 0.0, high_tone, audio_cfg, variant="numbers")
    assert high_state["acoustic"]["high_band_energy"] > low_state["acoustic"]["high_band_energy"]


def test_invalid_variant_raises():
    audio = _tone()
    with pytest.raises(ValueError):
        build_state("c", 0.0, audio, _CFG, variant="bogus")


def test_numbers_confidence_is_not_the_raw_score():
    """confidence must not just re-derive `score` (that field already exists separately)."""
    audio = _tone()
    state = build_state("clip-1", 0.6125730221093393, audio, _CFG, variant="numbers")
    assert state["detector"]["confidence"] != pytest.approx(state["detector"]["score"])


def test_numbers_confidence_is_bounded_and_zero_at_threshold():
    audio = _tone()
    at_threshold = build_state("c", 0.0, audio, _CFG, variant="numbers")
    assert at_threshold["detector"]["confidence"] == pytest.approx(0.0, abs=1e-9)

    far = build_state("c", 100.0, audio, _CFG, variant="numbers")
    # Saturates numerically to (essentially) 1.0 for a huge margin -- the transform's limit,
    # not evidence of a bug; the strict "< 1.0, monotone increasing" property is checked at
    # more moderate margins in test_numbers_confidence_monotone_in_margin below.
    assert 0.0 <= far["detector"]["confidence"] <= 1.0


def test_numbers_confidence_is_symmetric_on_the_spoof_side():
    """Confidence is distance from the threshold in either direction, never negative."""
    audio = _tone()
    bona = build_state("c", 0.3, audio, _CFG, variant="numbers")["detector"]["confidence"]
    spoof = build_state("c", -0.3, audio, _CFG, variant="numbers")["detector"]["confidence"]
    assert spoof == pytest.approx(bona)
    assert spoof > 0.0


def test_numbers_confidence_monotone_in_margin():
    audio = _tone()
    scores = [0.0, 0.05, 0.1, 0.3, 1.0, 10.0]
    confidences = [
        build_state("c", s, audio, _CFG, variant="numbers")["detector"]["confidence"]
        for s in scores
    ]
    assert confidences == sorted(confidences)
    assert len(set(confidences)) == len(confidences)
