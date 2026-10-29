"""Manifests, ASVspoof protocol parsing, and split hygiene.

This module turns an ASVspoof protocol file plus a directory of audio into a flat manifest
DataFrame (one row per utterance, every column `configs/data.yaml:manifest.columns` names),
and provides the split-safety checks the handoff's Stage 0 audit requires: a stable content
hash to prove two runs used the same split, whole-attack-family holdout for unseen-attack
evaluation, and a speaker-overlap guard.

This module never trains anything and never computes audio features (see
`CONVENTIONS.md`'s module table). It only reads file metadata, so it uses
`soundfile.info()` directly for sample rate/duration rather than importing `audio.py`, which
is being written concurrently by another agent in this repo; once that module lands, probing
here should probably delegate to it (noted as a future refactor, not done in this task).
"""

from __future__ import annotations

import hashlib
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

__all__ = [
    "DatasetNotFoundError",
    "load_manifest",
    "parse_asvspoof_protocol",
    "build_manifest",
    "split_hash",
    "holdout_attack_families",
    "assert_no_speaker_overlap",
    "summarise_manifest",
    "build_itw_manifest",
    "build_la2021_manifest",
    "build_df2021_manifest",
    "parse_la2021_cm_key",
    "build_spoofceleb_manifest",
    "build_phonemedf_manifest",
    "stratified_subset",
    "uniform_stratified_subset",
    "manifest_checksums",
    "write_checksums",
    "verify_checksums",
]

# Audio container extensions we'll look for under a split's audio directory. The official
# ASVspoof 2019 LA release ships `.flac`; `.wav` is included so synthetic test fixtures (and
# any local re-encode) resolve the same way without a separate code path.
_AUDIO_EXTENSIONS = (".flac", ".wav")

# ASVspoof 2019 LA source speech is VCTK (English) for all three subsets (train/dev/eval) —
# see Miscellaneous/research/knowledge/datasets/asvspoof-2019-la.md, Evaluation Plan §3, p.3. There
# is no per-utterance language field in the protocol file, so this is a corpus-level constant, not
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
# Confirmed in Miscellaneous/research/knowledge/datasets/asvspoof-2019-la.md (researcher note,
# confidence: high) against two independent official/near-official parsers: the ASVspoof-challenge
# org's own 2021 RawNet2 baseline `data_utils.py`, and the widely used `clovaai/aasist` repo's
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
        "Run `uv run download --asvspoof2019` so that this path exists "
        "(see configs/data.yaml: dataset.root / dataset.protocol_dir)."
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


def _audio_metadata(path: Path) -> tuple[int, float, str]:
    """Picklable audio-header reader; does not decode waveform samples."""
    info = sf.info(str(path))
    return info.samplerate, info.frames / info.samplerate, info.format


def _read_metadata(paths: list[Path], *, workers: int, desc: str):
    """Ordered metadata iterator, with progress and optional process parallelism."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if workers == 1:
        return tqdm(map(_audio_metadata, paths), total=len(paths), desc=desc, unit="clip")
    # A generator keeps the pool open until the caller finishes consuming results.
    def generate():
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from tqdm(
                pool.map(_audio_metadata, paths, chunksize=64),
                total=len(paths), desc=desc, unit="clip",
            )
    return generate()


def build_manifest(cfg: dict, workers: int = 4) -> pd.DataFrame:
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

        rows = list(proto_df.itertuples(index=False))
        audio_paths = [
            _resolve_audio_path(split_audio_dir, str(row.utt_id), split=split)
            for row in rows
        ]
        records = []
        for row, audio_path, (sample_rate, duration_s, codec) in zip(
            rows, audio_paths,
            _read_metadata(audio_paths, workers=workers, desc=f"ASVspoof {split}"),
            strict=True,
        ):
            records.append(
                {
                    "utt_id": row.utt_id,
                    "path": audio_path.as_posix(),
                    "label": row.label,
                    "speaker_id": row.speaker_id,
                    "attack_id": row.attack_id,
                    "source_corpus": _ASVSPOOF_LA_SOURCE_CORPUS,
                    "sample_rate": sample_rate,
                    "duration_s": duration_s,
                    "codec": codec,
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


# ================================================================================================
# In-the-Wild (Müller et al. 2022, arXiv:2203.16263) — eval-only, no official split, no
# per-utterance generator label. See Miscellaneous/research/knowledge/datasets/in-the-wild.md for
# the full note this section is derived from.
# ================================================================================================

_ITW_SOURCE_CORPUS = "InTheWild"
_ITW_LANGUAGE = "en"
_ITW_SPLIT = "itw"

# The original release ships meta.csv either directly under the extracted root, or nested one
# level inside a `release_in_the_wild/` directory (the name of the archive's top-level folder
# on the original mirror). Both are checked, in this order, before giving up.
_ITW_META_SUBDIR_CANDIDATES = (".", "release_in_the_wild")

# meta.csv label strings (original release) map onto this project's manifest convention.
# "bona-fide" -> "bonafide" is the only normalisation needed; anything else is unrecognised.
_ITW_LABEL_MAP = {
    "bona-fide": "bonafide",
    "bonafide": "bonafide",
    "spoof": "spoof",
}


def _itw_not_found(path: Path, what: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"{what} not found at '{path}'.\n"
        "Run `uv run download --itw` so that either '<in_the_wild.root>/meta.csv' or "
        "'<in_the_wild.root>/release_in_the_wild/meta.csv' "
        "exists (see configs/data.yaml: in_the_wild.root / in_the_wild.meta_file)."
    )


def _resolve_itw_meta(root: Path, meta_file: str) -> tuple[Path, Path]:
    """Locate `meta_file` under one of the two known In-the-Wild layouts.

    Returns `(meta_path, audio_dir)` where `audio_dir` is the directory the audio files live
    alongside `meta_file` (audio and meta always sit in the same directory in both known
    layouts). Raises `DatasetNotFoundError` naming every location checked if neither layout
    has the meta file — including when audio exists but the meta file does not, since that is
    exactly the "half-downloaded / wrong archive" case this error message needs to catch.
    """
    if not root.exists():
        raise _itw_not_found(root, "In-the-Wild dataset root")

    checked = []
    for subdir in _ITW_META_SUBDIR_CANDIDATES:
        candidate_dir = root / subdir if subdir != "." else root
        meta_path = candidate_dir / meta_file
        checked.append(meta_path)
        if meta_path.exists():
            return meta_path, candidate_dir

    raise _itw_not_found(
        checked[0],
        f"In-the-Wild meta file '{meta_file}' (checked {[str(p) for p in checked]})",
    )


def build_itw_manifest(cfg: dict, workers: int = 4) -> pd.DataFrame:
    """Build the In-the-Wild manifest with EXACTLY the ASVspoof manifest's column set.

    `cfg` is the parsed contents of `configs/data.yaml` (must carry both `in_the_wild` and
    `manifest.columns`). In-the-Wild carries no generator/attack label at all (see
    `Miscellaneous/research/knowledge/datasets/in-the-wild.md`): every spoof row gets the explicit
    sentinel `attack_id="unknown"` (never a fabricated generator id), and every bonafide row keeps
    this project's usual `"-"`, matching the ASVspoof convention so the two manifests concatenate
    cleanly. `speaker_id` is the original release's `speaker` column (its own free-text
    identifier, not an ASVspoof-style code). `split` is the literal `"itw"` — this project
    imposes no train/dev/eval partition on In-the-Wild (the dataset ships none).
    """
    itw_cfg = cfg["in_the_wild"]
    root = Path(itw_cfg["root"])
    meta_file = itw_cfg.get("meta_file", "meta.csv")
    expected_clips = itw_cfg.get("expected_clips")

    meta_path, audio_dir = _resolve_itw_meta(root, meta_file)
    meta = pd.read_csv(meta_path)

    required_cols = {"file", "speaker", "label"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(
            f"In-the-Wild meta file '{meta_path}' is missing required column(s) "
            f"{sorted(missing_cols)}; expected at least {sorted(required_cols)}."
        )

    # Resolve and validate paths first, preserving meta.csv row order.
    paths = []
    for file_name in meta["file"].astype(str):
        audio_path = audio_dir / file_name
        if not audio_path.exists():
            fallback = audio_dir / f"{file_name}.wav"
            if fallback.exists():
                audio_path = fallback
            else:
                raise _itw_not_found(audio_path, f"In-the-Wild audio file for meta row '{file_name}'")
        paths.append(audio_path)

    records = []
    for row, audio_path, (sample_rate, duration_s, codec) in zip(
        meta.itertuples(index=False), paths,
        _read_metadata(paths, workers=workers, desc="In-the-Wild"),
        strict=True,
    ):
        raw_label = str(row.label).strip().lower()
        label = _ITW_LABEL_MAP.get(raw_label)
        if label is None:
            raise ValueError(
                f"unrecognised In-the-Wild label {row.label!r} for file {row.file!r} in "
                f"'{meta_path}' (expected 'bona-fide' or 'spoof')"
            )
        attack_id = "-" if label == "bonafide" else "unknown"

        file_name = str(row.file)
        records.append(
            {
                "utt_id": f"ITW_{Path(file_name).stem}",
                "path": audio_path.as_posix(),
                "label": label,
                "speaker_id": str(row.speaker),
                "attack_id": attack_id,
                "source_corpus": _ITW_SOURCE_CORPUS,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
                "codec": codec,
                "split": _ITW_SPLIT,
                "language": _ITW_LANGUAGE,
            }
        )

    manifest = pd.DataFrame.from_records(records)

    if expected_clips is not None and len(manifest) != expected_clips:
        warnings.warn(
            f"In-the-Wild manifest built {len(manifest)} rows from '{meta_path}', but "
            f"configs/data.yaml:in_the_wild.expected_clips={expected_clips}. Not fatal — "
            "the local release may be a different version — but worth checking.",
            stacklevel=2,
        )

    columns = list(cfg["manifest"]["columns"])
    return manifest[columns] if len(manifest) else pd.DataFrame(columns=columns)


# ================================================================================================
# ASVspoof 2021 LA eval — test-only. Labels come from the post-challenge CM key
# (keys/LA/CM/trial_metadata.txt); the .trl.txt shipped with the audio lists utt_ids only.
# ================================================================================================

_LA2021_SOURCE_CORPUS = "ASVspoof2021_LA"
_LA2021_LANGUAGE = "en"
# 2019 and 2021 eval utt_ids share the LA_E_ + 7 digit pattern, so prefix to keep them distinct.
_LA2021_UTT_PREFIX = "LA21_"
_LA2021_KEY_FIELDS = ("speaker_id", "utt_id", "codec", "transmission", "attack_id",
                      "label", "trim", "phase")


def _la2021_not_found(path: Path, what: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"{what} not found at '{path}'.\n"
        "Run `uv run download --asvspoof2021la`, then re-encode the audio with "
        "`uv run python scripts/reencode_flac.py data/raw/ASVspoof2021/ASVspoof2021_LA_eval/flac "
        "data/raw/ASVspoof2021/ASVspoof2021_LA_eval/flac_ffmpeg`, so this path exists "
        "(see configs/data.yaml: asvspoof2021.root / audio_dir / cm_key)."
    )


def parse_la2021_cm_key(path: str | Path) -> pd.DataFrame:
    """Parse the 8-field ASVspoof 2021 LA CM key; raise on malformed lines or labels."""
    path = Path(path)
    if not path.exists():
        raise _la2021_not_found(path, "ASVspoof 2021 LA CM key")
    rows = []
    with path.open() as f:
        for line_no, raw_line in enumerate(f, start=1):
            fields = raw_line.split()
            if not fields:
                continue
            if len(fields) != len(_LA2021_KEY_FIELDS):
                raise ValueError(
                    f"{path}:{line_no}: expected {len(_LA2021_KEY_FIELDS)} whitespace-separated "
                    f"fields in ASVspoof 2021 CM key line, got {len(fields)}: {raw_line.strip()!r}"
                )
            rows.append(dict(zip(_LA2021_KEY_FIELDS, fields)))
    df = pd.DataFrame(rows, columns=list(_LA2021_KEY_FIELDS))
    unknown = set(df["label"]) - {"bonafide", "spoof"}
    if unknown:
        raise ValueError(f"unrecognised labels {sorted(unknown)} in '{path}'")
    return df


def build_la2021_manifest(cfg: dict, workers: int = 4) -> pd.DataFrame:
    """Build the ASVspoof 2021 LA eval manifest with EXACTLY the ASVspoof manifest's columns.

    `split` is `asvspoof2021_<phase>` (eval / progress / hidden), so the caller can write one
    CSV per phase. `codec` holds the key's transmission codec (none, alaw, ulaw, gsm, g722,
    opus, pstn) rather than the file format, which is always FLAC. Bona fide `attack_id` is
    the key's own `"-"`, matching the 2019 convention.
    """
    la_cfg = cfg["asvspoof2021"]
    root = Path(la_cfg["root"])
    audio_dir = root / la_cfg.get("audio_dir", "ASVspoof2021_LA_eval/flac")
    expected_clips = la_cfg.get("expected_clips")
    if not audio_dir.exists():
        raise _la2021_not_found(audio_dir, "ASVspoof 2021 LA audio directory")

    key = parse_la2021_cm_key(root / la_cfg.get("cm_key", "keys/LA/CM/trial_metadata.txt"))
    paths = []
    for utt_id in key["utt_id"]:
        audio_path = audio_dir / f"{utt_id}.flac"
        if not audio_path.exists():
            raise _la2021_not_found(audio_path, f"audio file for utterance '{utt_id}'")
        paths.append(audio_path)

    records = []
    for row, audio_path, (sample_rate, duration_s, _fmt) in zip(
        key.itertuples(index=False), paths,
        _read_metadata(paths, workers=workers, desc="ASVspoof 2021 LA"),
        strict=True,
    ):
        records.append(
            {
                "utt_id": f"{_LA2021_UTT_PREFIX}{row.utt_id}",
                "path": audio_path.as_posix(),
                "label": row.label,
                "speaker_id": row.speaker_id,
                "attack_id": row.attack_id,
                "source_corpus": _LA2021_SOURCE_CORPUS,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
                "codec": row.codec,
                "split": f"asvspoof2021_{row.phase}",
                "language": _LA2021_LANGUAGE,
            }
        )

    manifest = pd.DataFrame.from_records(records)

    if expected_clips is not None and len(manifest) != expected_clips:
        warnings.warn(
            f"ASVspoof 2021 LA manifest built {len(manifest)} rows, but "
            f"configs/data.yaml:asvspoof2021.expected_clips={expected_clips}.",
            stacklevel=2,
        )

    columns = list(cfg["manifest"]["columns"])
    return manifest[columns] if len(manifest) else pd.DataFrame(columns=columns)


# ================================================================================================
# ASVspoof 2021 DF eval — test-only, from the Hugging Face mirror unpacked by
# scripts/download_data.py: <root>/flac/*.flac plus <root>/labels.csv (official labels/protocol).
# ================================================================================================

_DF2021_SOURCE_CORPUS = "ASVspoof2021_DF"
_DF2021_LANGUAGE = "en"


def build_df2021_manifest(cfg: dict, workers: int = 4) -> pd.DataFrame:
    """Build the ASVspoof 2021 DF eval manifest with EXACTLY the ASVspoof manifest's columns.

    `split` is `asvspoof2021df_<subset>` (eval / progress / hidden); eval is the official
    scoring subset. `codec` is the DF media codec (e.g. nocodec, low_m4a). DF utt_ids
    (DF_E_*) cannot collide with 2019 or 2021 LA ids, so they are kept as released.
    """
    df_cfg = cfg["asvspoof2021df"]
    root = Path(df_cfg["root"])
    labels_path = root / df_cfg.get("labels", "labels.csv")
    if not labels_path.exists():
        raise DatasetNotFoundError(
            f"ASVspoof 2021 DF labels not found at '{labels_path}'.\n"
            "Run `uv run download --asvspoof2021df` (see configs/data.yaml: asvspoof2021df.root)."
        )
    labels = pd.read_csv(labels_path, dtype=str)
    paths = [root / p for p in labels["path"]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise DatasetNotFoundError(
            f"{len(missing)} ASVspoof 2021 DF audio files listed in '{labels_path}' are missing "
            f"(first: '{missing[0]}'). Rerun `uv run download --asvspoof2021df`."
        )

    records = []
    for row, audio_path, (sample_rate, duration_s, _fmt) in zip(
        labels.itertuples(index=False), paths,
        _read_metadata(paths, workers=workers, desc="ASVspoof 2021 DF"),
        strict=True,
    ):
        records.append(
            {
                "utt_id": row.utterance_id,
                "path": audio_path.as_posix(),
                "label": row.label,
                "speaker_id": row.speaker_id,
                "attack_id": "-" if row.label == "bonafide" else row.attack_id,
                "source_corpus": _DF2021_SOURCE_CORPUS,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
                "codec": row.codec,
                "split": f"asvspoof2021df_{row.subset}",
                "language": _DF2021_LANGUAGE,
            }
        )

    manifest = pd.DataFrame.from_records(records)
    expected_clips = df_cfg.get("expected_clips")
    if expected_clips is not None and len(manifest) != expected_clips:
        warnings.warn(
            f"ASVspoof 2021 DF manifest built {len(manifest)} rows, but "
            f"configs/data.yaml:asvspoof2021df.expected_clips={expected_clips}.",
            stacklevel=2,
        )

    columns = list(cfg["manifest"]["columns"])
    return manifest[columns] if len(manifest) else pd.DataFrame(columns=columns)


# ================================================================================================
# SpoofCeleb — metadata-driven train/development/evaluation manifests.
# ================================================================================================

_SPOOFCELEB_SOURCE_CORPUS = "SpoofCeleb"
_SPOOFCELEB_LANGUAGE = "en"


def _spoofceleb_not_found(path: Path, what: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"{what} not found at '{path}'.\n"
        "Download and extract SpoofCeleb from "
        "https://huggingface.co/datasets/jungjee/spoofceleb (access is gated) so that the "
        "configured flac/, metadata/, and protocol/ paths exist "
        "(see configs/data.yaml:spoofceleb.root)."
    )


def build_spoofceleb_manifest(cfg: dict, workers: int = 4) -> pd.DataFrame:
    """Build the SpoofCeleb manifest from its split metadata CSVs and nested FLAC layout.

    The CSV's `file` field is relative to `flac/<split>` and has the form
    `<attack_id>/<speaker_id>/<utterance>.flac`. Attack `a00` is bona fide; all other
    attack IDs are spoof generators. The returned columns match the shared manifest schema.
    """
    spoof_cfg = cfg["spoofceleb"]
    root = Path(spoof_cfg["root"])
    audio_dir = root / spoof_cfg.get("audio_dir", "flac")
    metadata_dir = root / spoof_cfg.get("metadata_dir", "metadata")

    if not root.is_dir():
        raise _spoofceleb_not_found(root, "SpoofCeleb dataset root")
    if not audio_dir.is_dir():
        raise _spoofceleb_not_found(audio_dir, "SpoofCeleb audio directory")
    if not metadata_dir.is_dir():
        raise _spoofceleb_not_found(metadata_dir, "SpoofCeleb metadata directory")

    columns = list(cfg["manifest"]["columns"])
    frames: list[pd.DataFrame] = []
    for split, filename in spoof_cfg["metadata_files"].items():
        metadata_path = metadata_dir / filename
        if not metadata_path.is_file():
            raise _spoofceleb_not_found(metadata_path, f"SpoofCeleb metadata file for '{split}'")

        metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
        required_cols = {"file", "speaker", "attack"}
        missing_cols = required_cols - set(metadata.columns)
        if missing_cols:
            raise ValueError(
                f"SpoofCeleb metadata file '{metadata_path}' is missing required column(s) "
                f"{sorted(missing_cols)}; expected at least {sorted(required_cols)}."
            )

        rows = list(metadata.itertuples(index=False))
        audio_paths = []
        for row in rows:
            relative_path = Path(row.file)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"invalid SpoofCeleb relative audio path {row.file!r} in '{metadata_path}'"
                )
            attack_id = str(row.attack).strip()
            speaker_id = str(row.speaker).strip()
            if not attack_id or not speaker_id or not row.file:
                raise ValueError(
                    f"empty file, speaker, or attack value in '{metadata_path}'"
                )
            if len(relative_path.parts) < 3 or relative_path.parts[0] != attack_id:
                raise ValueError(
                    f"SpoofCeleb metadata path {row.file!r} must begin with its attack ID "
                    f"{attack_id!r} and include a speaker directory ('{metadata_path}')"
                )
            if relative_path.parts[1] != speaker_id:
                raise ValueError(
                    f"SpoofCeleb metadata path {row.file!r} has speaker directory "
                    f"{relative_path.parts[1]!r}, but metadata lists {speaker_id!r} "
                    f"('{metadata_path}')"
                )
            audio_path = audio_dir / split / relative_path
            if not audio_path.is_file():
                raise _spoofceleb_not_found(
                    audio_path, f"SpoofCeleb audio file for metadata row {row.file!r}"
                )
            audio_paths.append(audio_path)

        records = []
        for row, audio_path, (sample_rate, duration_s, codec) in zip(
            rows,
            audio_paths,
            _read_metadata(audio_paths, workers=workers, desc=f"SpoofCeleb {split}"),
            strict=True,
        ):
            attack_id = str(row.attack).strip()
            relative_id = Path(row.file).with_suffix("").as_posix().replace("/", "_")
            label = "bonafide" if attack_id == "a00" else "spoof"
            records.append(
                {
                    "utt_id": f"SC_{split}_{relative_id}",
                    "path": audio_path.as_posix(),
                    "label": label,
                    "speaker_id": str(row.speaker).strip(),
                    "attack_id": "-" if label == "bonafide" else attack_id,
                    "source_corpus": _SPOOFCELEB_SOURCE_CORPUS,
                    "sample_rate": sample_rate,
                    "duration_s": duration_s,
                    "codec": codec,
                    "split": split,
                    "language": _SPOOFCELEB_LANGUAGE,
                }
            )
        frames.append(pd.DataFrame.from_records(records, columns=columns))

    manifest = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    return manifest[columns]


# ================================================================================================
# PhonemeDF — paired real/synthetic audio with deterministic speaker-disjoint splits.
# ================================================================================================

_PHONEMEDF_SOURCE_CORPUS = "PhonemeDF"
_PHONEMEDF_LANGUAGE = "en"


def _phonemedf_not_found(path: Path, what: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"{what} not found at '{path}'.\n"
        "Place the extracted PhonemeDF folders under the configured "
        "configs/data.yaml:phonemedf.root path."
    )


def build_phonemedf_manifest(cfg: dict, workers: int = 4) -> pd.DataFrame:
    """Build paired PhonemeDF manifests with source-speaker-disjoint splits.

    Real utterances and every available generated counterpart are assigned to the same split.
    Speakers are deterministically partitioned 80/10/10 into train/development/evaluation.
    """
    phonemedf_cfg = cfg["phonemedf"]
    root = Path(phonemedf_cfg["root"])
    real_dir = root / phonemedf_cfg.get("real_audio_dir", "PhonemeDF_real/real/wav")
    tts_dir = root / phonemedf_cfg.get("tts_audio_dir", "PhonemeDF_synthetic_tts/tts")
    vc_dir = root / phonemedf_cfg.get("vc_audio_dir", "PhonemeDF_synthetic_vc/vc")

    for folder, description in (
        (root, "PhonemeDF dataset root"),
        (real_dir, "PhonemeDF real audio directory"),
        (tts_dir, "PhonemeDF TTS directory"),
        (vc_dir, "PhonemeDF voice-conversion directory"),
    ):
        if not folder.is_dir():
            raise _phonemedf_not_found(folder, description)

    real_paths = sorted(
        path for path in real_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".wav"
    )
    if not real_paths:
        raise _phonemedf_not_found(real_dir, "PhonemeDF real WAV files")

    speaker_by_stem: dict[str, str] = {}
    for path in real_paths:
        stem = path.stem
        if "-" not in stem:
            raise ValueError(f"PhonemeDF utterance ID {stem!r} has no speaker prefix")
        if stem in speaker_by_stem:
            raise ValueError(f"Duplicate PhonemeDF real utterance ID {stem!r}")
        speaker_by_stem[stem] = stem.split("-", maxsplit=1)[0]

    speakers = sorted(set(speaker_by_stem.values()))
    if len(speakers) < 3:
        raise ValueError(
            "PhonemeDF needs at least three speakers to create non-empty "
            "train/development/evaluation splits"
        )
    split_seed = phonemedf_cfg.get("split_seed", 1337)
    shuffled_speakers = list(np.random.default_rng(split_seed).permutation(speakers))
    development_count = max(1, round(len(speakers) * 0.1))
    evaluation_count = max(1, round(len(speakers) * 0.1))
    train_count = len(speakers) - development_count - evaluation_count
    split_by_speaker = {
        speaker: "train" for speaker in shuffled_speakers[:train_count]
    }
    split_by_speaker.update(
        {
            speaker: "development"
            for speaker in shuffled_speakers[
                train_count : train_count + development_count
            ]
        }
    )
    split_by_speaker.update(
        {
            speaker: "evaluation"
            for speaker in shuffled_speakers[train_count + development_count :]
        }
    )
    if train_count < 1:
        raise ValueError("PhonemeDF needs at least three speakers for non-empty splits")

    audio_records: list[tuple[Path, dict[str, str]]] = []

    def add_record(path: Path, *, family: str, generator: str, stem: str) -> None:
        speaker_id = speaker_by_stem[stem]
        if family == "real":
            label, attack_id = "bonafide", "-"
            utt_id = f"PHDF_real_{stem}"
        else:
            label, attack_id = "spoof", f"{family}-{generator}"
            utt_id = f"PHDF_{family}_{generator}_{stem}"
        audio_records.append(
            (
                path,
                {
                    "utt_id": utt_id,
                    "label": label,
                    "speaker_id": speaker_id,
                    "attack_id": attack_id,
                    "split": split_by_speaker[speaker_id],
                },
            )
        )

    for path in real_paths:
        add_record(path, family="real", generator="", stem=path.stem)

    for family, family_dir in (("tts", tts_dir), ("vc", vc_dir)):
        generators = sorted(path for path in family_dir.iterdir() if path.is_dir())
        if not generators:
            raise _phonemedf_not_found(family_dir, f"PhonemeDF {family} generators")
        for generator_dir in generators:
            generated_paths = sorted(
                path for path in generator_dir.rglob("*")
                if path.is_file() and path.suffix.lower() == ".wav"
            )
            for path in generated_paths:
                if path.stem not in speaker_by_stem:
                    raise ValueError(
                        f"PhonemeDF generated audio {path} has no matching real source "
                        f"utterance {path.stem!r}"
                    )
                add_record(
                    path,
                    family=family,
                    generator=generator_dir.name,
                    stem=path.stem,
                )

    audio_paths = [path for path, _ in audio_records]
    rows = [record for _, record in audio_records]
    records = []
    for row, path, (sample_rate, duration_s, codec) in zip(
        rows,
        audio_paths,
        _read_metadata(audio_paths, workers=workers, desc="PhonemeDF"),
        strict=True,
    ):
        records.append(
            {
                **row,
                "path": path.as_posix(),
                "source_corpus": _PHONEMEDF_SOURCE_CORPUS,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
                "codec": codec,
                "language": _PHONEMEDF_LANGUAGE,
            }
        )

    columns = list(cfg["manifest"]["columns"])
    manifest = pd.DataFrame.from_records(records, columns=columns)
    if manifest["utt_id"].duplicated().any():
        duplicates = manifest.loc[manifest["utt_id"].duplicated(), "utt_id"].head().tolist()
        raise ValueError(f"Duplicate PhonemeDF utterance IDs: {duplicates}")
    return manifest.sort_values("utt_id").reset_index(drop=True)


# ================================================================================================
# Deterministic stratified subsampling.
# ================================================================================================


def stratified_subset(
    df: pd.DataFrame,
    n: int,
    strata: tuple[str, ...] | list[str] = ("label", "attack_id"),
    seed: int = 1337,
) -> pd.DataFrame:
    """Return `n` rows of `df`, allocated across `strata` proportionally by largest remainder.

    Allocation (deterministic, no RNG involved):
      1. If `n >= number of non-empty strata`, every stratum is first guaranteed exactly 1 row
         — this is the "every stratum with >=1 row gets >=1 row if n allows" guarantee; pure
         proportional largest-remainder allocation does not provide this for small strata
         competing against much larger ones, so it is applied as an explicit first pass.
      2. The remaining budget (`n` minus what step 1 used) is allocated across strata
         proportionally to their size in `df`, using the largest-remainder method: floor the
         proportional share per stratum, then hand out the leftover units one each to the
         strata with the largest fractional remainder (ties broken by stratum key, so the
         result is reproducible), never exceeding a stratum's remaining capacity.

    Row *selection* within each stratum's allocated count uses
    `np.random.default_rng(seed)`, so the same `(df, n, strata, seed)` always returns the same
    rows, and a different `seed` reliably returns a different sample when a stratum has more
    rows than its allocation. The result is sorted by `utt_id` for a stable, order-independent
    return value.
    """
    strata = list(strata)
    columns = list(df.columns)
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()

    n = min(n, len(df))
    grouped = df.groupby(strata, dropna=False)
    group_keys = sorted(grouped.groups.keys(), key=str)
    sizes = {k: len(grouped.groups[k]) for k in group_keys}
    n_groups = len(group_keys)
    total = sum(sizes.values())

    if n >= n_groups:
        guarantee = dict.fromkeys(group_keys, 1)
        remaining = n - n_groups
    else:
        guarantee = dict.fromkeys(group_keys, 0)
        remaining = n

    capacity = {k: sizes[k] - guarantee[k] for k in group_keys}

    if total:
        ideal = {k: remaining * sizes[k] / total for k in group_keys}
    else:
        ideal = dict.fromkeys(group_keys, 0.0)
    base = {k: min(int(np.floor(ideal[k])), capacity[k]) for k in group_keys}
    budget_left = remaining - sum(base.values())

    def _frac(k: object) -> float:
        return ideal[k] - base[k] if base[k] < capacity[k] else -1.0

    # Largest-remainder pass. A single left-to-right pass suffices when no stratum is capacity-
    # constrained (the usual case: the floor error per stratum is < 1, summing to < n_groups
    # total, so each needs at most one extra unit). When `remaining` is close to or equal to
    # the total available capacity, though, a stratum can already be capped at its capacity by
    # the floor step (e.g. floor(ideal) happens to equal capacity), stranding its fractional
    # remainder with nowhere to go — that excess must then flow to *other* strata with room, so
    # this keeps looping over the priority order (skipping capped strata) until the budget is
    # exhausted; it always terminates because sum(capacity) >= remaining by construction.
    order = sorted(group_keys, key=lambda k: (-_frac(k), str(k)))
    while budget_left > 0:
        progressed = False
        for k in order:
            if budget_left <= 0:
                break
            if base[k] < capacity[k]:
                base[k] += 1
                budget_left -= 1
                progressed = True
        if not progressed:
            break

    counts = {k: guarantee[k] + base[k] for k in group_keys}

    rng = np.random.default_rng(seed)
    parts = []
    for k in group_keys:
        c = counts[k]
        if c <= 0:
            continue
        group_df = grouped.get_group(k)
        idx = rng.choice(len(group_df), size=c, replace=False)
        parts.append(group_df.iloc[idx])

    result = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    result = result[columns].sort_values("utt_id").reset_index(drop=True)
    return result


def uniform_stratified_subset(
    df: pd.DataFrame,
    max_per_stratum: int,
    strata: tuple[str, ...] | list[str] = ("label", "attack_id"),
    seed: int = 1337,
) -> pd.DataFrame:
    """Sample up to `max_per_stratum` rows from every non-empty stratum.

    Unlike `stratified_subset`, allocation is uniform across strata rather than proportional
    to their source sizes. Sampling is deterministic for a fixed frame order and seed.
    """
    if max_per_stratum < 1:
        raise ValueError("max_per_stratum must be >= 1")
    strata = list(strata)
    if not strata:
        raise ValueError("strata must contain at least one column")
    missing = set(strata) - set(df.columns)
    if missing:
        raise ValueError(f"stratified subset is missing column(s): {sorted(missing)}")
    if df.empty:
        return df.copy()

    grouped = df.groupby(strata, dropna=False, sort=True)
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in grouped:
        count = min(max_per_stratum, len(group))
        if count == len(group):
            parts.append(group)
        else:
            indices = rng.choice(len(group), size=count, replace=False)
            parts.append(group.iloc[indices])
    result = pd.concat(parts, ignore_index=True)
    if "utt_id" in result.columns:
        result = result.sort_values("utt_id")
    return result.reset_index(drop=True)


# ================================================================================================
# Checksums — a reproducibility record for gitignored, generated manifest CSVs.
#
# Format: standard `sha256sum`-compatible lines (`<hex>  <relative-path>`), so a plain
# `shasum -a 256 -c SHA256SUMS` (from the directory `manifest_checksums`'s paths were made
# relative to) verifies file integrity with no project-specific tooling. Extra per-file
# metadata (row count, split_hash) rides along as `#`-prefixed comment lines interleaved after
# each sha line — both BSD `shasum -a 256 -c` and GNU `sha256sum -c` were confirmed (see task
# report) to skip blank/`#` lines and still report "OK" / exit 0, so this stays ONE file
# rather than splitting metadata into a second file.
# ================================================================================================


def manifest_checksums(
    paths: Sequence[str | Path], base_dir: str | Path | None = None
) -> list[dict]:
    """Compute `(file, sha256, rows, split_hash)` for each manifest CSV in `paths`.

    `sha256` hashes the file's raw bytes (what `shasum -c` will re-check). `rows` and
    `split_hash` are computed from the *parsed* CSV via `split_hash()` above, so they change
    if and only if the manifest's actual content changes, not its on-disk formatting.
    `file` is written relative to `base_dir` (default: current working directory), matching
    how `shasum -a 256 -c` expects the second column when invoked from `base_dir`.
    """
    base_dir = (Path(base_dir) if base_dir is not None else Path.cwd()).resolve()
    entries = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        df = pd.read_csv(path)
        entries.append(
            {
                "file": path.relative_to(base_dir).as_posix(),
                "sha256": sha256,
                "rows": len(df),
                "split_hash": split_hash(df),
            }
        )
    return entries


def write_checksums(entries: list[dict], out_path: str | Path) -> None:
    """Write `entries` (from `manifest_checksums`) to `out_path` in SHA256SUMS format."""
    lines = []
    for entry in entries:
        lines.append(f"{entry['sha256']}  {entry['file']}")
        lines.append(
            f"# file={entry['file']} rows={entry['rows']} split_hash={entry['split_hash']}"
        )
    Path(out_path).write_text("\n".join(lines) + "\n")


def verify_checksums(path: str | Path, base_dir: str | Path | None = None) -> list[str]:
    """Recompute sha256 for every file listed in the SHA256SUMS file at `path`.

    Returns a list of human-readable mismatch/missing-file messages — empty means every file
    verified. `#`-prefixed and blank lines are ignored, matching `shasum -c`'s own tolerance
    (confirmed in the task report) so this and a plain `shasum -a 256 -c path` agree.
    """
    base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
    text = Path(path).read_text()

    mismatches = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected_sha, _, rel_path = line.partition("  ")
        if not rel_path:
            continue
        full_path = base_dir / rel_path
        if not full_path.exists():
            mismatches.append(f"{rel_path}: missing at '{full_path}'")
            continue
        actual_sha = hashlib.sha256(full_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            mismatches.append(
                f"{rel_path}: sha256 mismatch (expected {expected_sha}, got {actual_sha})"
            )
    return mismatches


# Prepared-manifest loading used by the modular runtime.
def load_manifest(path: str | Path, root: str | Path = ".") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run: uv run data-prepare")
    df = pd.read_csv(path)
    needed = {"utt_id", "path", "label"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} missing columns: {sorted(needed - set(df.columns))}")
    if df.utt_id.duplicated().any():
        raise ValueError(f"Duplicate utterance IDs in {path}")
    if not set(df.label).issubset({"bonafide", "spoof"}):
        raise ValueError(f"Unknown labels in {path}")
    root = Path(root).resolve()
    df = df.copy()
    df["path"] = df["path"].map(lambda p: str((root / str(p)).resolve()))
    return df
