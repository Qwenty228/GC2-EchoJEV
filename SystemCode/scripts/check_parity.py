"""Check that the Detector classes reproduce the scores of the original A1/A2 runs.

    uv run python scripts/check_parity.py [--n 200] [--seed 0]

- **A1, end to end from audio.** `LfccGmmDetector.from_run` scores `--n` dev clips (and
  `--n` eval clips with silence trimmed, against the trimmed run) from their audio files;
  every score must equal the run's score file exactly.
- **A2, cached features.** `SslHeadDetector.score_pooled` scores every cached dev and eval
  clip; every score must equal the run's score file exactly (same computation, same device).
- **A2, end to end from audio.** `--n` dev clips through the backbone. The cache stores
  pooled states in fp16, so these agree to within fp16 rounding; the check reports the
  largest difference and requires every label to match.

Needs the local runs (`runs/E-20261001-e1-a1-raw`, `-e1-a1-trimmed`, `-e1-a2-raw`), the
SSL cache and the raw audio. Exits 1 on any failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from audio_df.features.ssl import cache_dir_for, load_cache
from audio_df.inference import predict_file
from audio_df.models.a1_lfcc_gmm import LfccGmmDetector
from audio_df.models.a2_ssl_head import SslHeadDetector
from audio_df.models.base import Detector, decide

A1_RAW = Path("runs/E-20261001-e1-a1-raw")
A1_TRIMMED = Path("runs/E-20261001-e1-a1-trimmed")
A2_RAW = Path("runs/E-20261001-e1-a2-raw")


def _read_scores(path: Path) -> pd.DataFrame:
    # pandas' default float parser can be one ulp off; round_trip reads the file exactly.
    return pd.read_csv(path, float_precision="round_trip")


def _sample(manifest: str, scores_csv: Path, n: int, seed: int) -> pd.DataFrame:
    scores = _read_scores(scores_csv)
    rows = pd.read_csv(manifest).merge(
        scores.rename(columns={"clip_id": "utt_id", "score": "run_score"}),
        on="utt_id",
        suffixes=("", "_run"),
    )
    return rows.sample(n=min(n, len(rows)), random_state=seed)


def _end_to_end(detector: Detector, rows: pd.DataFrame) -> np.ndarray:
    return np.array([predict_file(detector, p).score for p in rows["path"]], dtype=np.float64)


def check_a1(n: int, seed: int) -> bool:
    ok = True
    for run, silence, testset in [(A1_RAW, None, "dev"), (A1_TRIMMED, "trimmed", "eval")]:
        detector = LfccGmmDetector.from_run(A1_RAW, silence_mode=silence)
        rows = _sample(f"data/manifests/{testset}.csv", run / "scores" / f"{testset}.csv", n, seed)
        got = _end_to_end(detector, rows)
        n_equal = int(np.sum(got == rows["run_score"].to_numpy()))
        passed = n_equal == len(rows)
        ok &= passed
        print(
            f"A1 {silence or 'raw'} {testset}, audio -> score: {n_equal}/{len(rows)} identical "
            f"(max |diff| {np.max(np.abs(got - rows['run_score'].to_numpy())):.3g}) "
            f"{'PASS' if passed else 'FAIL'}"
        )
    return ok


def check_a2_cached(detector: SslHeadDetector) -> bool:
    ok = True
    for testset in ("dev", "eval"):
        cache = load_cache(
            cache_dir_for(detector.model["tag"], "raw", testset, root="data/cache/ssl")
        )
        ids = [str(i) for i in cache["ids"]]
        got = pd.Series(detector.score_pooled(cache["pooled"]), index=ids)
        run = _read_scores(A2_RAW / "scores" / f"{testset}.csv").set_index("clip_id")["score"]
        got = got.loc[run.index]
        n_equal = int(np.sum(got.to_numpy() == run.to_numpy()))
        passed = n_equal == len(run)
        ok &= passed
        print(
            f"A2 cached {testset}: {n_equal}/{len(run)} identical "
            f"(max |diff| {np.max(np.abs(got.to_numpy() - run.to_numpy())):.3g}) "
            f"{'PASS' if passed else 'FAIL'}"
        )
    return ok


def check_a2_audio(detector: SslHeadDetector, n: int, seed: int) -> bool:
    rows = _sample("data/manifests/dev.csv", A2_RAW / "scores" / "dev.csv", n, seed)
    got = _end_to_end(detector, rows)
    want = rows["run_score"].to_numpy()
    labels_match = all(
        decide(g, detector.threshold) == decide(w, detector.threshold)
        for g, w in zip(got, want, strict=True)
    )
    diff = np.abs(got - want)
    print(
        f"A2 dev, audio -> score: max |diff| {diff.max():.3g}, median {np.median(diff):.3g} "
        f"(fp16 cache rounding); labels {'all match' if labels_match else 'DIFFER'} "
        f"{'PASS' if labels_match else 'FAIL'}"
    )
    return labels_match


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=200, help="clips per end-to-end check")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ok = check_a1(args.n, args.seed)
    a2 = SslHeadDetector.from_run(A2_RAW)
    ok &= check_a2_cached(a2)
    ok &= check_a2_audio(a2, args.n, args.seed)
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
