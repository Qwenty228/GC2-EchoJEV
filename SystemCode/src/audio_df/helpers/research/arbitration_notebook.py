"""Notebook 05/06 helpers: arbitration test case, audio facts, cached JEV calls, the playground."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from audio_df.helpers.audio import load_audio
from audio_df.helpers.eval_audio import TRIM_RECIPE
from audio_df.helpers.metrics import compute_eer
from audio_df.modules import jev, laya
from audio_df.modules.jev import JEVState
from audio_df.pipelines.arbitration import ORDERS, RUNS

SR = 16000


def build_cases(results: dict, models=("a1", "a2", "a4"), n: int = 200, keep: str = "ITW_11249",
                seed: int = 0) -> list[int]:
    """In-the-Wild file numbers where the detectors split at their dev EER thresholds: n clips, half
    bona fide and half spoof, `keep` (bona fide) counted among them."""
    verdicts = {}
    for m in models:
        s = results[(m, "trimmed")]["scores"]
        thr = compute_eer(s["dev"].score, s["dev"].label, higher_is_spoof=False)[1]
        verdicts[m] = s["itw"].set_index("clip_id").score >= thr
    v = pd.DataFrame(verdicts)
    label = s["itw"].set_index("clip_id").label.loc[v.index]
    split = v.index[v.nunique(axis=1) > 1]
    print(f"detectors split on {len(split):,} of {len(v):,} In-the-Wild clips")
    pool = label.loc[split].drop(keep)
    rest = [*pool[pool == "bonafide"].sample(n // 2 - 1, random_state=seed).index,
            *pool[pool == "spoof"].sample(n // 2, random_state=seed).index]
    return [int(c.removeprefix("ITW_")) for c in [keep, *rest]]


def recalibrate(cases: pd.DataFrame, root: Path, seed: int = 0) -> pd.DataFrame:
    """`cases` with each detector's verdict (EER threshold) and P(bona fide) (Platt) refit on
    In-the-Wild, two folds by speaker: a clip's fit comes from the other half of the speakers."""
    root = Path(root)
    speaker = pd.read_csv(root / "data/manifests/itw.csv", usecols=["utt_id", "speaker_id"]
                          ).set_index("utt_id").speaker_id
    names = speaker.drop_duplicates().sample(frac=1, random_state=seed).to_numpy()
    fold = speaker.map({s: i % 2 for i, s in enumerate(names)})
    out, case_fold = cases.copy(), fold.loc[cases.clip_id].to_numpy()
    for m, (_, run) in RUNS.items():
        itw = pd.read_csv(root / run / "scores" / "itw.csv").set_index("clip_id")
        itw_fold = fold.loc[itw.index].to_numpy()
        test = itw.loc[cases.clip_id, ["score"]]
        for f in (0, 1):
            fit, here = itw[itw_fold != f], case_fold == f
            thr = compute_eer(fit.score, fit.label, higher_is_spoof=False)[1]
            platt = LogisticRegression().fit(fit[["score"]], fit.label == "bonafide")
            out.loc[here, f"v_{m}"] = np.where(test.score[here] >= thr, "bonafide", "spoof")
            out.loc[here, f"p_{m}"] = platt.predict_proba(test[here])[:, 1]
    return out


def audio_facts(path) -> dict[str, float]:
    """What an analyst can read off the waveform: duration, silence, bandwidth, high band."""
    x = load_audio(path, SR)
    speech = librosa.effects.split(x, top_db=TRIM_RECIPE["top_db"],
                                   frame_length=TRIM_RECIPE["frame_length"],
                                   hop_length=TRIM_RECIPE["hop_length"])
    voiced = np.concatenate([x[s:e] for s, e in speech]) if len(speech) else x
    power = (np.abs(librosa.stft(voiced, n_fft=512)) ** 2).mean(axis=1)  # mean speech spectrum
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    return {"duration_s": round(len(x) / SR, 2),
            "leading_silence_s": round(speech[0, 0] / SR, 2) if len(speech) else 0.0,
            # highest frequency within 50 dB of the spectral peak: ~8 kHz studio, ~4 kHz narrowband
            "bandwidth_khz": round(float(freqs[np.nonzero(power >= power.max() * 1e-5)[0].max()])
                                   / 1000, 2),
            "energy_above_4khz_pct": round(100 * power[freqs >= 4000].sum() / power.sum(), 2)}


def summarise(pred, label, keep=None) -> pd.Series:
    """Coverage, accuracy on decided clips, errors and abstentions; `keep` selects rows to score."""
    keep = np.ones(len(label), bool) if keep is None else keep
    pred, label = np.asarray(pred)[keep], np.asarray(label)[keep]
    decided = pred != "uncertain"
    correct = (pred == label) & decided
    return pd.Series({"coverage": decided.mean(),
                      "accuracy on decided": correct.sum() / max(decided.sum(), 1),
                      "errors": int((decided & ~correct).sum()),
                      "abstained": int((~decided).sum())})


def ask_jev(ctx: dict, prompt: str, questions: dict, cache: Path, run: bool = False) -> dict | None:
    """JEV answers for one request, cached by request content; None when uncached and run=False."""
    payload = {"context": ctx, "prompt": prompt, "questions": questions}
    path = Path(cache) / (hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
                          + ".json")
    if path.exists():
        return json.loads(path.read_text())
    if not run:
        return None
    answers = jev.decide(JEVState(context=ctx, prompt=prompt, questions=questions)).answers
    if not isinstance(answers, dict):
        answers = json.loads(json.dumps(answers, default=lambda o: o.__dict__))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(answers))
    return answers


def try_request(clips: pd.DataFrame, det: dict, prompt: str, question: dict,
                context: Callable[[dict, tuple], dict],
                judge: Callable[[Any], dict] = laya.decide) -> pd.DataFrame:
    """Playground: the judge's answer per clip with detectors listed A1 first, and whether the
    answer flips when they are listed A1 last."""
    rows = []
    for r in clips.to_dict("records"):
        answers = {o: judge(JEVState(context(r, order), prompt, question))["answers"]["decision"]
                   for o, order in ORDERS.items()}
        first = answers["A1 first"]
        rows.append({"clip": r["clip_id"].removeprefix("ITW_"), "truth": r["label"],
                     **{m.upper(): r[f"v_{m}"] for m in det}, "judge": first["choice"],
                     **{f"P {k}": v for k, v in first["probabilities"].items()},
                     "flips": answers["A1 last"]["choice"] != first["choice"]})
    table = pd.DataFrame(rows).round(2)
    print(f"right {(table.judge == table.truth).sum()}/{len(table)} | flips {table.flips.sum()} | "
          f"uncertain {(table.judge == 'uncertain').sum()}")
    return table
