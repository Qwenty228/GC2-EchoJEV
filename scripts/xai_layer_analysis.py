"""X6: which WavLM Base+ layers carry the evidence for A2? CPU-only, head-only, from caches.

    uv run python scripts/xai_layer_analysis.py [--root experiments]

Variants (all scored on dev, eval raw, eval trimmed, ITW raw; thresholds from each variant's
own raw dev scores, via `audio_df.scoring.score_experiment`, the same code path as
`scripts/score_experiment.py`):
  full_seed{1337,2337,3337}  standard 13-layer head, retrained
  layer00..layer12           single-layer heads (SSLHead(n_layers=1) on x[:, l:l+1]), seed 1337
  loo00..loo12               leave-one-layer-out at scoring on the E1 head, no retraining
  lr_layer12, lr_layer00     standardised sklearn LogisticRegression on one layer, train only

Runs under `experiment_run` with a lock that is NOT the heavy-job lock: CPU only, seconds per
head, meant to run alongside a queued heavy (MPS) job.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audio_df.exp_io import make_exp_id, registry_update, write_scores
from audio_df.experiments.lifecycle import experiment_run
from audio_df.experiments.run_a2 import align_cache_with_manifest, load_config
from audio_df.metrics import prf1_at_threshold
from audio_df.models.ssl_head import SSLHead
from audio_df.resources import memory_status
from audio_df.scoring import score_experiment
from audio_df.ssl_cache import cache_dir_for, load_cache
from audio_df.xai import layers as xl

BASE_RUN = "experiments/E-20261001-e1-a2-raw"
SEEDS = (1337, 2337, 3337)
N_LAYERS = 13
TESTSETS = {  # score-file name -> (manifest key, silence mode)
    "dev": ("dev", "raw"),
    "eval": ("eval", "raw"),
    "eval_trimmed": ("eval", "trimmed"),
    "itw": ("itw", "raw"),
}


def _load(cfg, manifest_key: str, mode: str, split: str | None = None):
    mpath = (
        cfg["data"]["train_manifest"] if split == "train" else cfg["data"]["testsets"][manifest_key]
    )
    name = split or manifest_key
    cache = load_cache(cache_dir_for(cfg["model"]["tag"], mode, name, root=cfg["cache_dir"]))
    pooled, ordered = align_cache_with_manifest(cache, pd.read_csv(mpath))
    return pooled, ordered


def _df(ordered: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame(
        {"clip_id": ordered["utt_id"].astype(str), "label": ordered["label"], "score": scores}
    )
    if "attack_id" in ordered.columns:
        out["attack_id"] = ordered["attack_id"].astype(str)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--lock-path", default="data/cache/xai_light.lock")
    parser.add_argument("--config", default="configs/a2_wavlm_base_plus.yaml")
    args = parser.parse_args()

    torch.set_num_threads(4)
    status = memory_status()
    print(f"[xai] memory status at start: {status}")
    if status.get("pressure_name") not in ("normal",):
        raise SystemExit(f"memory pressure {status.get('pressure_name')!r} is not normal; stop")

    cfg = load_config(args.config)
    exp_id = make_exp_id(date.today(), "e2", "a2", "raw", "layers")
    variants_cfg = (
        [f"full_seed{s}" for s in SEEDS]
        + [f"layer{i:02d}" for i in range(N_LAYERS)]
        + [f"loo{i:02d}" for i in range(N_LAYERS)]
        + ["lr_layer12", "lr_layer00"]
    )
    full_config = {
        **cfg,
        "experiment": "e2",
        "model_from": BASE_RUN,
        "variants": variants_cfg,
        "device": "cpu",
        "test_sets": {k: list(v) for k, v in TESTSETS.items()},
    }

    with experiment_run(
        exp_id,
        full_config,
        job_name="xai-layers",
        config_summary="X6 layers: full x3 seeds, 13 single-layer heads, LOO, LR probe; CPU",
        root=args.root,
        lock_path=args.lock_path,
    ) as run:
        run.note(
            "Lock: data/cache/xai_light.lock, NOT the heavy-job lock. CPU-only (torch threads 4, "
            "no MPS), seconds per head, run alongside the queued A1 heavy job."
        )
        run.note(
            "Thresholds: each variant's own raw dev EER threshold (score_experiment default), "
            "applied unchanged to eval raw, eval trimmed and ITW. LR probes are trained on the "
            "train split only (Pascu et al. trained on train+dev)."
        )
        t0 = time.monotonic()
        x_train, train_o = _load(cfg, "train", "raw", split="train")
        y_train = (train_o["label"] == "bonafide").to_numpy(np.float32)
        x_dev, dev_o = _load(cfg, "dev", "raw")
        y_dev = (dev_o["label"] == "bonafide").to_numpy(np.float32)
        tests = {"dev": (x_dev, dev_o)}
        for name, (mk, mode) in TESTSETS.items():
            if name != "dev":
                tests[name] = _load(cfg, mk, mode)
        print(f"[xai] caches loaded in {time.monotonic() - t0:.0f}s")

        vroot = run.exp_dir / "variants"
        metrics: dict[str, dict] = {}
        rows: list[dict] = []
        extra: dict[str, dict] = {}

        def finish(name: str, scorer, info: dict | None = None) -> None:
            vdir = vroot / name
            for ts, (x, o) in tests.items():
                write_scores(vdir, ts, _df(o, scorer(x)))
            m = score_experiment(vdir)
            metrics[name] = m
            thr = m["threshold_source"]["value"]
            itw = tests["itw"][1]
            s_itw = pd.read_csv(vdir / "scores" / "itw.csv")
            pr = prf1_at_threshold(s_itw["score"], s_itw["label"], thr=thr, higher_is_spoof=False)
            n_bf = int((s_itw["label"] == "bonafide").sum())
            rej = pr["fp"] / n_bf
            rows.append(xl.summary_row(name, m, rej))
            extra[name] = {"itw_bonafide_rejected": rej, "itw_n_bonafide": n_bf, **(info or {})}
            del itw
            r = rows[-1]
            print(
                f"[xai] {name}: dev {r['dev_eer']*100:.2f} eval {r['eval_eer']*100:.2f} "
                f"trim {r['eval_trimmed_eer']*100:.2f} itw {r['itw_eer']*100:.2f} "
                f"({time.monotonic() - t0:.0f}s)"
            )

        head_cfg = cfg["head"]
        # (a) full head, 3 seeds
        for s in SEEDS:
            h = xl.train_layer_head(
                x_train, y_train, x_dev, y_dev, layer=None, seed=s, head_cfg=head_cfg,
                n_layers=N_LAYERS, hidden_dim=cfg["model"]["hidden_dim"],
            )
            finish(
                f"full_seed{s}",
                lambda x, h=h: xl.score_head(h, x),
                {"layer_weights": h.layer_weights().detach().tolist()},
            )
        # (b) single-layer heads
        for i in range(N_LAYERS):
            h = xl.train_layer_head(
                x_train, y_train, x_dev, y_dev, layer=i, seed=cfg["seed"], head_cfg=head_cfg,
                n_layers=N_LAYERS, hidden_dim=cfg["model"]["hidden_dim"],
            )
            finish(f"layer{i:02d}", lambda x, h=h, i=i: xl.score_head(h, x, layer=i))
        # (c) leave-one-layer-out on the E1 head
        base = SSLHead(
            N_LAYERS, cfg["model"]["hidden_dim"], cfg["head"]["proj_dim"],
            cfg["head"]["mlp_hidden"], cfg["head"]["dropout"],
        )
        base.load_state_dict(torch.load(f"{BASE_RUN}/head_state_dict.pt", map_location="cpu"))
        for i in range(N_LAYERS):
            loo = xl.leave_one_layer_out(base, i)
            finish(f"loo{i:02d}", lambda x, h=loo: xl.score_head(h, x))
        # (d) LR probes
        for i in (12, 0):
            probe = xl.fit_lr_probe(x_train, y_train, layer=i)
            finish(f"lr_layer{i:02d}", probe)

        # reproduction check against the E1 run's own score files
        repro = {}
        for ts in ("dev", "eval"):
            a = pd.read_csv(f"{BASE_RUN}/scores/{ts}.csv").set_index("clip_id")["score"]
            b = pd.read_csv(vroot / "full_seed1337" / "scores" / f"{ts}.csv").set_index("clip_id")[
                "score"
            ]
            b = b.loc[a.index]
            repro[ts] = {
                "max_abs_diff": float((a - b).abs().max()),
                "mean_abs_diff": float((a - b).abs().mean()),
                "pearson": float(np.corrcoef(a, b)[0, 1]),
            }
        e1m = json.loads(Path(f"{BASE_RUN}/metrics.json").read_text())["testsets"]
        repro["eer_e1"] = {k: e1m[k]["eer"] for k in ("dev", "eval")}
        repro["eer_retrained_seed1337"] = {
            k: metrics["full_seed1337"]["testsets"][k]["eer"] for k in ("dev", "eval")
        }
        (run.exp_dir / "reproduction_check.json").write_text(json.dumps(repro, indent=2))
        (run.exp_dir / "variant_extra.json").write_text(json.dumps(extra, indent=2))
        run.note(f"reproduction check vs {BASE_RUN} (CPU retrain, seed 1337): {json.dumps(repro)}")

        # summary tables
        summ = pd.DataFrame(rows).set_index("variant")
        summ.to_csv(run.exp_dir / "summary.csv")
        full = summ.loc[[f"full_seed{s}" for s in SEEDS]]
        spread = pd.DataFrame(
            {"mean": full.mean(), "sd": full.std(ddof=1), "min": full.min(), "max": full.max()}
        ).T
        md = "# E2 layer analysis (A2, WavLM Base+) -- EER / rates in percent, AUC as fraction\n\n"
        view = summ.copy()
        for c in view.columns:
            if c != "itw_auc":
                view[c] = view[c] * 100
        spread_view = spread.copy()
        for c in spread_view.columns:
            if c != "itw_auc":
                spread_view[c] = spread_view[c] * 100
        md += _md(view) + "\n## Seed spread of the full head (3 seeds)\n\n" + _md(spread_view)
        hm = ["full_seed1337"] + [f"layer{i:02d}" for i in range(N_LAYERS)]
        hm_cols = {v: metrics[v] for v in hm}
        tabs = {
            "eval": xl.per_attack_table(hm_cols, "eval"),
            "eval_trimmed": xl.per_attack_table(hm_cols, "eval_trimmed"),
        }
        for ts, t in tabs.items():
            t.to_csv(run.exp_dir / f"per_attack_{ts}.csv")
            md += f"\n## Per-attack EER (%), {ts}\n\n"
            md += xl.to_markdown(t, pct=True, index_name="attack")
        (run.exp_dir / "summary.md").write_text(md)
        _heatmap(tabs, run.exp_dir / "per_attack_heatmap.png")

        registry_update(
            exp_id,
            root=args.root,
            key_eers=(
                f"full1337 eval={summ.loc['full_seed1337','eval_eer']*100:.2f}; "
                f"itw={summ.loc['full_seed1337','itw_eer']*100:.2f}; "
                f"lr12 itw={summ.loc['lr_layer12','itw_eer']*100:.2f}"
            ),
        )
        print(md)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| variant | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        cells = [f"{r[c]:.3f}" if c == "itw_auc" else f"{r[c]:.2f}" for c in cols]
        lines.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _heatmap(tabs: dict[str, pd.DataFrame], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    for ax, (name, t) in zip(axes, tabs.items(), strict=True):
        im = ax.imshow(t.to_numpy() * 100, aspect="auto", cmap="viridis", vmin=0, vmax=50)
        ax.set_xticks(range(len(t.columns)), [c.replace("full_seed", "full") for c in t.columns],
                      rotation=60, fontsize=8)
        ax.set_yticks(range(len(t.index)), t.index)
        ax.set_title(f"per-attack EER (%), {name}")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                ax.text(
                    j, i, f"{t.iat[i, j] * 100:.0f}", ha="center", va="center", color="w",
                    fontsize=6,
                )
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
