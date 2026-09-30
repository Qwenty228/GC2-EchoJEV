"""Regenerate `reports/tables/E1.md/.csv` .. `E6.md/.csv` from `experiments/*`.

    uv run python scripts/make_tables.py \\
        [--experiments-root experiments] [--registry experiments/REGISTRY.md] \\
        [--out-dir reports/tables]

Reads every `experiments/<id>/{config.yaml,metrics.json}` for rows whose
`experiments/REGISTRY.md` status is exactly `done` (rows marked `running`,
`failed`, or `superseded` are excluded). Computes nothing: every number already
lives in some `metrics.json`, written once by `audio_df.scoring.score_experiment`.
Ordering is always sorted by explicit key tuples (system, then silence_mode /
testset name / attack id, ...), never insertion order, so re-running this script
on the same registry produces byte-identical tables.

All public functions are importable (used directly by `tests/test_make_tables.py`);
the CLI at the bottom is a thin wrapper guarded by `if __name__ == "__main__":`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

__all__ = [
    "load_done_experiments",
    "build_e1_table",
    "build_e2_table",
    "build_e3_itw_table",
    "build_e3_per_attack_table",
    "build_e4_table",
    "build_e5_table",
    "build_e6_table",
    "write_table",
    "make_tables",
]

_UNSEEN_ATTACKS = [f"A{i:02d}" for i in range(7, 20)]  # A07..A19


def _parse_registry_rows(registry_path: Path) -> list[dict[str, str]]:
    if not registry_path.exists():
        return []
    text = registry_path.read_text()
    lines = text.splitlines()
    header_idx = next((i for i, ln in enumerate(lines) if ln.startswith("| id ")), None)
    if header_idx is None:
        return []
    columns = [c.strip() for c in lines[header_idx].strip().strip("|").split("|")]
    rows = []
    for ln in lines[header_idx + 2 :]:
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != len(columns):
            continue
        rows.append(dict(zip(columns, cells, strict=True)))
    return rows


def load_done_experiments(
    experiments_root: str | Path = "experiments",
    registry_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load `{exp_id, registry_row, config, metrics}` for every `status: done` row.

    Rows with any other status (`running`, `failed`, `superseded`) are skipped.
    An experiment id present in the registry as `done` but missing `metrics.json`
    or `config.yaml` on disk is skipped as well (nothing to report), not an error --
    a registry row can legitimately outlive a metrics file during development.
    """
    experiments_root = Path(experiments_root)
    registry_path = (
        Path(registry_path) if registry_path is not None else experiments_root / "REGISTRY.md"
    )
    rows = _parse_registry_rows(registry_path)

    out = []
    for row in rows:
        if row.get("status") != "done":
            continue
        exp_id = row["id"]
        exp_dir = experiments_root / exp_id
        config_path = exp_dir / "config.yaml"
        metrics_path = exp_dir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        with config_path.open() as f:
            config = yaml.safe_load(f)
        with metrics_path.open() as f:
            metrics = json.load(f)
        out.append({"exp_id": exp_id, "registry_row": row, "config": config, "metrics": metrics})
    out.sort(key=lambda r: r["exp_id"])
    return out


def _silence_mode(config: dict[str, Any]) -> str:
    return str(config.get("preprocessing", {}).get("silence_mode", "unknown"))


def _system(config: dict[str, Any]) -> str:
    return str(config.get("system", "unknown"))


def build_e1_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """System x silence_mode on `eval`: EER, minDCF, ROC-AUC, F1."""
    rows = []
    for exp in experiments:
        if exp["config"].get("experiment") != "e1":
            continue
        testsets = exp["metrics"].get("testsets", {})
        eval_result = testsets.get("eval")
        if eval_result is None:
            continue
        rows.append(
            {
                "system": _system(exp["config"]),
                "silence_mode": _silence_mode(exp["config"]),
                "eer": eval_result.get("eer"),
                "min_dcf": eval_result.get("min_dcf"),
                "roc_auc": eval_result.get("roc_auc"),
                "f1": eval_result.get("f1"),
                "exp_id": exp["exp_id"],
            }
        )
    df = pd.DataFrame(
        rows, columns=["system", "silence_mode", "eer", "min_dcf", "roc_auc", "f1", "exp_id"]
    )
    return df.sort_values(["system", "silence_mode"]).reset_index(drop=True)


def build_e2_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:  # noqa: ARG001
    """E2 (ablation: heads A/B/fused) is scheduled for later; always a placeholder."""
    return pd.DataFrame([{"status": "not run"}])


def build_e3_itw_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """E3: In-the-Wild EER/minDCF per system."""
    rows = []
    for exp in experiments:
        if exp["config"].get("experiment") != "e3":
            continue
        itw = exp["metrics"].get("testsets", {}).get("itw")
        if itw is None:
            continue
        rows.append(
            {
                "system": _system(exp["config"]),
                "eer": itw.get("eer"),
                "min_dcf": itw.get("min_dcf"),
                "exp_id": exp["exp_id"],
            }
        )
    df = pd.DataFrame(rows, columns=["system", "eer", "min_dcf", "exp_id"])
    return df.sort_values(["system"]).reset_index(drop=True)


def build_e3_per_attack_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """E3: per-attack eval EER for the unseen attacks A07..A19, one row per system."""
    rows = []
    for exp in experiments:
        if exp["config"].get("experiment") != "e3":
            continue
        eval_result = exp["metrics"].get("testsets", {}).get("eval")
        if eval_result is None:
            continue
        per_attack = (eval_result.get("per_attack_eer") or {}).get("per_attack", {})
        row = {"system": _system(exp["config"]), "exp_id": exp["exp_id"]}
        for attack in _UNSEEN_ATTACKS:
            row[attack] = per_attack.get(attack)
        rows.append(row)
    df = pd.DataFrame(rows, columns=["system", *_UNSEEN_ATTACKS, "exp_id"])
    return df.sort_values(["system"]).reset_index(drop=True)


def build_e4_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """E4: system x degradation EER/minDCF, `eval` (clean subset) sorted first as reference."""
    rows = []
    for exp in experiments:
        if exp["config"].get("experiment") != "e4":
            continue
        testsets = exp["metrics"].get("testsets", {})
        for name, result in testsets.items():
            if name == "dev":
                continue
            rows.append(
                {
                    "system": _system(exp["config"]),
                    "testset": name,
                    "eer": result.get("eer"),
                    "min_dcf": result.get("min_dcf"),
                    "exp_id": exp["exp_id"],
                }
            )
    df = pd.DataFrame(rows, columns=["system", "testset", "eer", "min_dcf", "exp_id"])
    if df.empty:
        return df

    def _sort_key(name: str) -> tuple[int, str]:
        return (0, name) if name == "eval" else (1, name)

    df["_order"] = df["testset"].map(_sort_key)
    df = df.sort_values(["system", "_order"]).drop(columns="_order").reset_index(drop=True)
    return df


def build_e5_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """E5: A2 vs A3 on the JEV subset (`jev_dev500` test set)."""
    rows = []
    for exp in experiments:
        if exp["config"].get("experiment") != "e5":
            continue
        result = exp["metrics"].get("testsets", {}).get("jev_dev500")
        if result is None:
            continue
        rows.append(
            {
                "system": _system(exp["config"]),
                "eer": result.get("eer"),
                "min_dcf": result.get("min_dcf"),
                "roc_auc": result.get("roc_auc"),
                "f1": result.get("f1"),
                "exp_id": exp["exp_id"],
            }
        )
    df = pd.DataFrame(rows, columns=["system", "eer", "min_dcf", "roc_auc", "f1", "exp_id"])
    return df.sort_values(["system"]).reset_index(drop=True)


def build_e6_table(experiments: list[dict[str, Any]]) -> pd.DataFrame:
    """E6: params/ms-per-clip/RTF/peak-memory from `metrics.json["efficiency"]`."""
    rows = []
    for exp in experiments:
        if exp["config"].get("experiment") != "e6":
            continue
        efficiency = exp["metrics"].get("efficiency")
        row = {"system": _system(exp["config"]), "exp_id": exp["exp_id"]}
        if efficiency is None:
            row.update({"params": None, "ms_per_clip": None, "rtf": None, "peak_memory_mb": None})
        else:
            row.update(
                {
                    "params": efficiency.get("params"),
                    "ms_per_clip": efficiency.get("ms_per_clip"),
                    "rtf": efficiency.get("rtf"),
                    "peak_memory_mb": efficiency.get("peak_memory_mb"),
                }
            )
        rows.append(row)
    df = pd.DataFrame(
        rows,
        columns=["system", "params", "ms_per_clip", "rtf", "peak_memory_mb", "exp_id"],
    )
    return df.sort_values(["system"]).reset_index(drop=True)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "no runs yet\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_table(df: pd.DataFrame, name: str, out_dir: str | Path = "reports/tables") -> None:
    """Write `<out_dir>/<name>.md` and `<name>.csv`. Empty tables still write files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text(_df_to_markdown(df))
    df.to_csv(out_dir / f"{name}.csv", index=False)


def make_tables(
    experiments_root: str | Path = "experiments",
    registry_path: str | Path | None = None,
    out_dir: str | Path = "reports/tables",
) -> dict[str, pd.DataFrame]:
    """Build and write all six tables; returns `{"E1": df, ..., "E6": df}`."""
    experiments = load_done_experiments(experiments_root, registry_path)
    tables = {
        "E1": build_e1_table(experiments),
        "E2": build_e2_table(experiments),
        "E3_itw": build_e3_itw_table(experiments),
        "E3_per_attack": build_e3_per_attack_table(experiments),
        "E4": build_e4_table(experiments),
        "E5": build_e5_table(experiments),
        "E6": build_e6_table(experiments),
    }
    for name, df in tables.items():
        write_table(df, name, out_dir=out_dir)
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=Path("experiments"))
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()

    tables = make_tables(
        experiments_root=args.experiments_root,
        registry_path=args.registry,
        out_dir=args.out_dir,
    )
    for name, df in tables.items():
        print(f"{name}: {len(df)} row(s) -> {args.out_dir / name}.md/.csv")


if __name__ == "__main__":
    main()
