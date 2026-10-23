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
    "parse_la2021_cm_key",
    "stratified_subset",
    "manifest_checksums",
    "write_checksums",
    "verify_checksums",
]

# Where to send someone whose dataset is missing. ASVspoof 2019 predates the 2021 challenge
# site referenced elsewhere in this repo's docs; this is the 2019-specific download page.
_ASVSPOOF_DOWNLOAD_PAGE = "https://www.asvspoof.org/index2019.html"

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

_ITW_SOURCE_URL = "https://huggingface.co/datasets/mueller91/In-The-Wild"
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
        f"Download In-the-Wild from {_ITW_SOURCE_URL} and extract it so that either "
        "'<in_the_wild.root>/meta.csv' or '<in_the_wild.root>/release_in_the_wild/meta.csv' "
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

_LA2021_SOURCE_URL = "https://www.asvspoof.org/index2021.html"
_LA2021_SOURCE_CORPUS = "ASVspoof2021_LA"
_LA2021_LANGUAGE = "en"
# 2019 and 2021 eval utt_ids share the LA_E_ + 7 digit pattern, so prefix to keep them distinct.
_LA2021_UTT_PREFIX = "LA21_"
_LA2021_KEY_FIELDS = ("speaker_id", "utt_id", "codec", "transmission", "attack_id",
                      "label", "trim", "phase")


def _la2021_not_found(path: Path, what: str) -> DatasetNotFoundError:
    return DatasetNotFoundError(
        f"{what} not found at '{path}'.\n"
        f"Download ASVspoof 2021 LA eval audio and keys from {_LA2021_SOURCE_URL} and extract "
        "them so this path exists (see configs/data.yaml: asvspoof2021.root / audio_dir / cm_key)."
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
