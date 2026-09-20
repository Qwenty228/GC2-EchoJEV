"""Manifests, ASVspoof protocol parsing, and split hygiene.

This module turns an ASVspoof protocol file plus a directory of audio into a flat manifest
DataFrame (one row per utterance, every column `configs/data.yaml:manifest.columns` names),
and provides the split-safety checks the handoff's Stage 0 audit requires: a stable content
hash to prove two runs used the same split, whole-attack-family holdout for unseen-attack
evaluation, and a speaker-overlap guard.

This module never trains anything and never computes audio features (see
`tasks/CONVENTIONS.md`'s module table). It only reads file metadata, so it uses
`soundfile.info()` directly for sample rate/duration rather than importing `audio.py`, which
is being written concurrently by another agent in this repo; once that module lands, probing
here should probably delegate to it (noted as a future refactor, not done in this task).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import soundfile as sf

__all__ = [
    "DatasetNotFoundError",
    "parse_asvspoof_protocol",
    "build_manifest",
    "split_hash",
    "holdout_attack_families",
    "assert_no_speaker_overlap",
    "summarise_manifest",
]

# Where to send someone whose dataset is missing. ASVspoof 2019 predates the 2021 challenge
# site referenced elsewhere in this repo's docs; this is the 2019-specific download page.
_ASVSPOOF_DOWNLOAD_PAGE = "https://www.asvspoof.org/index2019.html"

# Audio container extensions we'll look for under a split's audio directory. The official
# ASVspoof 2019 LA release ships `.flac`; `.wav` is included so synthetic test fixtures (and
# any local re-encode) resolve the same way without a separate code path.
_AUDIO_EXTENSIONS = (".flac", ".wav")

# ASVspoof 2019 LA source speech is VCTK (English) for all three subsets (train/dev/eval) —
# see research/knowledge/datasets/asvspoof-2019-la.md, Evaluation Plan §3, p.3. There is no
# per-utterance language field in the protocol file, so this is a corpus-level constant, not
# something read per row.
_ASVSPOOF_LA_LANGUAGE = "en"
_ASVSPOOF_LA_SOURCE_CORPUS = "ASVspoof2019_LA"


class DatasetNotFoundError(FileNotFoundError):
    """The ASVspoof dataset (or one of its protocol files) is not present locally.

    Deliberately a distinct, named exception rather than a bare `FileNotFoundError`: a raw
    traceback pointing at some internal `open()` call is not an actionable error message for
    someone who just cloned this repo and has not downloaded the corpus yet.
    """


# --------------------------------------------------------------------------------------------
# ASVspoof 2019 LA protocol line format — SINGLE SOURCE OF TRUTH.
#
# Confirmed in research/knowledge/datasets/asvspoof-2019-la.md (researcher note, confidence:
# high) against two independent official/near-official parsers: the ASVspoof-challenge org's
# own 2021 RawNet2 baseline `data_utils.py`, and the widely used `clovaai/aasist` repo's
# `data_utils.py` — both parse the 2019 LA protocol files identically as 5 whitespace-
# separated fields:
#
#   SPEAKER_ID  AUDIO_FILE_NAME  -(unused placeholder)  SYSTEM_ID  KEY
#
# where SYSTEM_ID is "-" for bonafide trials or "A01".."A19" for spoof trials, and KEY is the
# literal string "bonafide" or "spoof". Field index 2 is a genuine placeholder column (always
# "-" in the pure 2019 LA files) that must be consumed and discarded, not mistaken for
# attack_id.
#
# If this format is ever found to be wrong, fix it HERE ONLY — every caller in this module
# goes through `_split_protocol_line`.
# --------------------------------------------------------------------------------------------
_PROTOCOL_N_FIELDS = 5
_PROTOCOL_FIELD_INDEX = {
    "speaker_id": 0,
    "utt_id": 1,
    # index 2: unused placeholder, always "-" in the 2019 LA protocol files.
    "attack_id": 3,
    "label": 4,
}


def _split_protocol_line(line: str, *, source: Path, line_no: int) -> dict[str, str]:
    """Parse one ASVspoof 2019 LA protocol line into its named fields.

    Isolates the column-order assumption described above. Raises `ValueError` (not a silent
    best-effort parse) if a line does not have exactly 5 whitespace-separated fields, since a
    malformed line silently mis-parsed would corrupt attack_id/label without any symptom.
    """
    fields = line.split()
    if len(fields) != _PROTOCOL_N_FIELDS:
        raise ValueError(
            f"{source}:{line_no}: expected {_PROTOCOL_N_FIELDS} whitespace-separated fields "
            f"in ASVspoof protocol line, got {len(fields)}: {line!r}"
        )
    return {name: fields[idx] for name, idx in _PROTOCOL_FIELD_INDEX.items()}


def _dataset_not_found(path: Path, what: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"{what} not found at '{path}'.\n"
        f"Download the ASVspoof 2019 dataset from {_ASVSPOOF_DOWNLOAD_PAGE} and extract it "
        "so that this path exists (see configs/data.yaml: dataset.root / dataset.protocol_dir)."
    )


def parse_asvspoof_protocol(path: str | Path, split: str) -> pd.DataFrame:
    """Parse one ASVspoof 2019 LA `.trn.txt`/`.trl.txt` protocol file into a DataFrame.

    Returns columns `["utt_id", "speaker_id", "attack_id", "label", "split"]`. Bonafide rows
    keep the protocol's own literal `"-"` in `attack_id`, matching `configs/data.yaml`'s
    convention (`attack_id ... "-" for bonafide`) exactly — no translation is applied.
    """
    path = Path(path)
    if not path.exists():
        raise _dataset_not_found(path, f"ASVspoof protocol file for split '{split}'")

    rows = []
    with path.open() as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parsed = _split_protocol_line(line, source=path, line_no=line_no)
            rows.append(parsed)

    df = pd.DataFrame(rows, columns=["speaker_id", "utt_id", "attack_id", "label"])
    df["split"] = split
    return df[["utt_id", "speaker_id", "attack_id", "label", "split"]]


def _resolve_audio_path(split_audio_dir: Path, utt_id: str, *, split: str) -> Path:
    """Find the audio file for `utt_id` under a split's audio directory.

    Directory layout assumption (not covered by the researcher's protocol-format note): the
    official ASVspoof 2019 LA release lays out audio as
    `<root>/ASVspoof2019_LA_<split>/flac/<utt_id>.flac`. This is isolated in one place so it
    can be corrected without touching the protocol-parsing logic above.
    """
    for ext in _AUDIO_EXTENSIONS:
        candidate = split_audio_dir / f"{utt_id}{ext}"
        if candidate.exists():
            return candidate
    raise _dataset_not_found(
        split_audio_dir / f"{utt_id}{_AUDIO_EXTENSIONS[0]}",
        f"audio file for utterance '{utt_id}' (split '{split}')",
    )


def build_manifest(cfg: dict) -> pd.DataFrame:
    """Build the full manifest: every column in `cfg['manifest']['columns']`, one row per
    utterance across every split in `cfg['dataset']['splits']`.

    `cfg` is the parsed contents of `configs/data.yaml` (or an equivalent dict, as tests
    construct against a fake dataset tree under `tmp_path`). Missing metadata that ASVspoof
    2019 LA does not provide (there is no codec/language field in the protocol) is filled with
    an explicit constant rather than silently dropped — see module-level `_ASVSPOOF_LA_*`
    constants.
    """
    dataset_cfg = cfg["dataset"]
    root = Path(dataset_cfg["root"])
    protocol_dir = Path(dataset_cfg["protocol_dir"])

    if not root.exists():
        raise _dataset_not_found(root, "ASVspoof dataset root")
    if not protocol_dir.exists():
        raise _dataset_not_found(protocol_dir, "ASVspoof protocol directory")

    frames: list[pd.DataFrame] = []
    for split, filename in dataset_cfg["splits"].items():
        protocol_path = protocol_dir / filename
        proto_df = parse_asvspoof_protocol(protocol_path, split=split)
        split_audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"

        records = []
        for row in proto_df.itertuples(index=False):
            audio_path = _resolve_audio_path(split_audio_dir, row.utt_id, split=split)
            info = sf.info(str(audio_path))
            records.append(
                {
                    "utt_id": row.utt_id,
                    "path": str(audio_path),
                    "label": row.label,
                    "speaker_id": row.speaker_id,
                    "attack_id": row.attack_id,
                    "source_corpus": _ASVSPOOF_LA_SOURCE_CORPUS,
                    "sample_rate": info.samplerate,
                    "duration_s": info.frames / info.samplerate,
                    "codec": info.format,
                    "split": split,
                    "language": _ASVSPOOF_LA_LANGUAGE,
                }
            )
        frames.append(pd.DataFrame.from_records(records))

    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    columns = list(cfg["manifest"]["columns"])
    return manifest[columns]


def split_hash(df: pd.DataFrame) -> str:
    """A content hash stable across row order and across repeated calls on equal content.

    Sorts columns and rows before hashing so two DataFrames holding the same rows in a
    different order produce the same hash — the registry uses this to prove two runs used the
    same split without caring how either run happened to order its rows.
    """
    columns = sorted(df.columns)
    canonical = df[columns].sort_values(by=columns).reset_index(drop=True)
    payload = canonical.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def holdout_attack_families(
    df: pd.DataFrame, families: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split `df` into `(remainder, holdout)`, removing WHOLE attack families.

    Never samples individual utterances from a generator: the handoff is explicit that
    unseen-attack evaluation is meaningless if some utterances of a "held-out" generator leak
    into the training remainder. Asserts the held-out families are fully absent from the
    remainder before returning.
    """
    families_set = set(families)
    is_holdout = df["attack_id"].isin(families_set)
    holdout = df[is_holdout].reset_index(drop=True)
    remainder = df[~is_holdout].reset_index(drop=True)

    leaked = families_set & set(remainder["attack_id"])
    if leaked:
        raise AssertionError(
            f"holdout_attack_families: families {sorted(leaked)} still present in the "
            "remainder after holdout — this must never happen for whole-family holdout"
        )
    return remainder, holdout


def assert_no_speaker_overlap(train: pd.DataFrame, dev: pd.DataFrame) -> None:
    """Raise `ValueError` naming every speaker id present in both `train` and `dev`.

    A bare `assert not overlap` would tell a future debugger nothing about which speaker
    leaked; this lists them so the offending manifest rows can be found immediately.
    """
    overlap = sorted(set(train["speaker_id"]) & set(dev["speaker_id"]))
    if overlap:
        raise ValueError(
            f"speaker overlap between train and dev: {overlap} appear in both splits "
            "(ASVspoof protocols are speaker-disjoint by design — this indicates a custom "
            "split was built incorrectly)"
        )


def summarise_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Counts by class (label), attack family, and speaker, grouped by split.

    One row per (split, label, attack_id) with the utterance count and the number of distinct
    speakers contributing to that cell — enough to eyeball class balance, per-attack coverage,
    and whether a handful of speakers dominate any one cell.
    """
    summary = (
        df.groupby(["split", "label", "attack_id"], dropna=False)
        .agg(
            n_utterances=("utt_id", "count"),
            n_speakers=("speaker_id", "nunique"),
        )
        .reset_index()
        .sort_values(["split", "label", "attack_id"])
        .reset_index(drop=True)
    )
    return summary
