"""A synthetic, ASVspoof-2019-LA-shaped protocol+audio tree, for notebooks and tests that must
run with no dataset present.

This module writes **no real speech** anywhere. `bonafide` utterances are a harmonic comb
(speech-like partials) plus colored noise inside a slow amplitude envelope; `spoof` utterances
are the *same* base signal with an attack-specific high-shelf gain (boost or cut above 3 kHz)
applied. Everything downstream (`data.build_manifest`, `leakage.py`, `features.py`) is exercised
against a real on-disk `.wav`/protocol-file tree in the exact ASVspoof 2019 LA layout, not a
hand-built DataFrame -- so the manifest-building code path itself is exercised, not bypassed.

## The fixture's contract (planted properties notebook narratives rely on)

Three properties are deliberately planted here, and are exactly what `notebooks/01_dataset_audit
.ipynb`, `02_spectral_eda.ipynb` and `03_cepstral_eda.ipynb` demonstrate against. If this module
changes, re-run `tests/test_synthetic.py` before trusting any notebook that imports it -- these
notebooks make specific claims (e.g. "attack A04 has a lower centroid than bonafide") that are
facts about *this generator*, not about ASVspoof, and must stay true for the notebook text to
remain accurate:

1. **Per-attack high-shelf gain, alternating sign.** `attack_gain_db` cycles through
   `[+6, +6, +6, -6, -6, 0]` dB (applied above 3 kHz) across the attack list, so some attacks
   push spectral energy up and others down relative to bonafide. This is what makes the
   generator-conditioned reversal in notebook 02's by-attack spectral-centroid plot, and the
   `rms` leakage flag in notebook 01, real and reproducible rather than illustrative-only.
2. **Attack-family split: train/dev share attacks, eval is disjoint.** `train_dev_attacks`
   (`A01`-`A06`) appear only in the `train`/`dev` splits; `eval_attacks` (`A07`-`A13` by
   default) appear only in `eval`. This mirrors (in miniature) ASVspoof 2019 LA's real
   asymmetry -- train/dev use A01-A06, eval uses A07-A19 (see
   `Miscellaneous/research/knowledge/datasets/asvspoof-2019-la.md`) -- which notebook 01's
   "unseen-attack generalisation" discussion depends on being structurally true here too.
3. **Speaker partition: disjoint across splits.** Each split draws its speaker IDs from a
   disjoint numeric offset (`train` 0-3, `dev` 100-103, `eval` 200-203), so
   `data.assert_no_speaker_overlap(train_df, dev_df)` genuinely passes on this fixture rather
   than passing by accident.

`gain_cycle` is exposed as a parameter (not hardcoded) so a caller can pass `gain_cycle=None`
(or a list of zeros) to get a fixture with no engineered spectral difference at all, e.g. for a
test that wants to confirm the leakage audit does *not* fire when there is nothing to leak.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

__all__ = ["build_synthetic_asvspoof_tree"]

_DEFAULT_GAIN_CYCLE = (6.0, 6.0, 6.0, -6.0, -6.0, 0.0)
_MANIFEST_COLUMNS = [
    "utt_id",
    "path",
    "label",
    "speaker_id",
    "attack_id",
    "source_corpus",
    "sample_rate",
    "duration_s",
    "codec",
    "split",
    "language",
]


def _synth_utt(
    rng: np.random.Generator, duration_s: float, sr: int, gain_db: float | None = None
) -> np.ndarray:
    """One synthetic utterance: harmonic comb + noise in an amplitude envelope, optionally with
    a high-shelf gain (boost/cut above 3 kHz) applied to simulate an attack-specific spectral
    tilt."""
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    f0 = rng.uniform(90, 220)
    harmonics = sum(
        (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi)) for k in range(1, 6)
    )
    noise = rng.standard_normal(n) * 0.02
    env = np.clip(np.sin(np.pi * np.minimum(t / max(duration_s, 1e-6), 1.0)), 0.05, 1.0)
    x = (harmonics * 0.05 + noise) * env
    if gain_db is not None and gain_db != 0.0:
        sos = butter(4, 3000, btype="highpass", fs=sr, output="sos")
        high = np.asarray(sosfilt(sos, x))
        x = x + high * (10 ** (gain_db / 20.0) - 1.0)
    pad = int(0.05 * sr)  # small, class-independent leading/trailing silence
    x = np.concatenate([np.zeros(pad, dtype=x.dtype), x, np.zeros(pad, dtype=x.dtype)])
    return x.astype(np.float32)


def build_synthetic_asvspoof_tree(
    root: Path,
    seed: int = 1337,
    sr: int = 16000,
    n_speakers_per_split: int = 4,
    n_bonafide_per_speaker: int = 3,
    gain_cycle: tuple[float, ...] | None = _DEFAULT_GAIN_CYCLE,
) -> dict:
    """Write a small ASVspoof-2019-LA-shaped protocol+audio tree under `root` and return a
    `configs/data.yaml`-shaped cfg dict pointing at it, suitable for `data.build_manifest(cfg)`.

    See the module docstring's "fixture's contract" section for the three properties this
    deliberately plants (per-attack gain, attack-family split, speaker partition) -- every
    notebook/test that depends on one of those properties should say so explicitly, since they
    are facts about this generator, not about ASVspoof.

    `gain_cycle=None` (or an all-zero tuple) disables the engineered spectral tilt entirely,
    for callers that want a fixture with no planted spectral difference between classes.
    """
    rng = np.random.default_rng(seed)

    # train/dev share attacks A01-A06 (seen), eval uses disjoint attacks A07-A13 (unseen) --
    # same shape as the real corpus's A01-A06 (train/dev) vs A07-A19 (eval-only) asymmetry,
    # see Miscellaneous/research/knowledge/datasets/asvspoof-2019-la.md, just fewer attack families.
    train_dev_attacks = [f"A0{i}" for i in range(1, 7)]
    eval_attacks = [f"A0{i}" for i in range(7, 10)] + [f"A1{i}" for i in range(0, 4)]
    all_attacks = train_dev_attacks + eval_attacks

    if gain_cycle:
        attack_gain_db = {a: gain_cycle[i % len(gain_cycle)] for i, a in enumerate(all_attacks)}
    else:
        attack_gain_db = dict.fromkeys(all_attacks, 0.0)

    root.mkdir(parents=True, exist_ok=True)
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    splits_attacks = {"train": train_dev_attacks, "dev": train_dev_attacks, "eval": eval_attacks}
    split_filenames = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    speaker_offsets = {"train": 0, "dev": 100, "eval": 200}

    utt_counter = 0
    for split, attacks in splits_attacks.items():
        lines = []
        flac_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)
        speaker_offset = speaker_offsets[split]
        speakers = [f"LA_{speaker_offset + s:04d}" for s in range(n_speakers_per_split)]

        for sp in speakers:
            for _ in range(n_bonafide_per_speaker):
                utt_counter += 1
                utt_id = f"LA_U_{utt_counter:06d}"
                duration = rng.uniform(1.0, 2.5)
                x = _synth_utt(rng, duration, sr, gain_db=None)
                sf.write(str(flac_dir / f"{utt_id}.wav"), x, sr)
                lines.append(f"{sp} {utt_id} - - bonafide")

        for sp in speakers:
            for attack in attacks:
                utt_counter += 1
                utt_id = f"LA_U_{utt_counter:06d}"
                duration = rng.uniform(1.0, 2.5)
                x = _synth_utt(rng, duration, sr, gain_db=attack_gain_db[attack])
                sf.write(str(flac_dir / f"{utt_id}.wav"), x, sr)
                lines.append(f"{sp} {utt_id} - {attack} spoof")

        (protocol_dir / split_filenames[split]).write_text("\n".join(lines) + "\n")

    return {
        "dataset": {
            "name": "asvspoof2019_la_synthetic",
            "root": str(root),
            "protocol_dir": str(protocol_dir),
            "splits": split_filenames,
        },
        "manifest": {"columns": list(_MANIFEST_COLUMNS)},
        "seed": seed,
        "train_dev_attacks": train_dev_attacks,
        "eval_attacks": eval_attacks,
        "attack_gain_db": attack_gain_db,
    }
