"""Minimal manifest loader. Never invents a train/test split."""
from pathlib import Path
import pandas as pd

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
