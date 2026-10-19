"""Score manifests and compute metrics using the original EchoJEV definitions."""
from pathlib import Path
import json
import pandas as pd
from audio_df2.helpers.metrics import compute_eer, roc_auc, min_dcf, prf1_at_threshold

def score_manifest(df, model, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in df.itertuples(index=False):
        result = {"clip_id": str(row.utt_id), "label": str(row.label),
                  "score": float(model.score_file(row.path))}
        if hasattr(row, "attack_id"):
            result["attack_id"] = str(row.attack_id)
        rows.append(result)
    pd.DataFrame(rows).to_csv(output, index=False)
    return pd.DataFrame(rows)

def evaluate_scores(dev, test, output):
    threshold_eer, threshold = compute_eer(dev.score, dev.label, higher_is_spoof=False)
    eer, _ = compute_eer(test.score, test.label, higher_is_spoof=False)
    metrics = {"eer": float(eer), "min_dcf": float(min_dcf(test.score, test.label, higher_is_spoof=False)[0]),
               "roc_auc": float(roc_auc(test.score, test.label, higher_is_spoof=False)),
               "f1": float(prf1_at_threshold(test.score, test.label, threshold, higher_is_spoof=False)["f1"]),
               "dev_eer": float(threshold_eer), "threshold": float(threshold)}
    # Full minDCF depends on configured target prior and costs; no fabricated default.
    Path(output).write_text(json.dumps(metrics, indent=2))
    return metrics
