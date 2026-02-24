from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score



def evaluate_binary_classifier(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    unique = np.unique(y_true)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(unique) > 1 else None
    avg_precision = float(average_precision_score(y_true, y_prob)) if len(unique) > 1 else None

    metrics = {
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0),
    }
    return metrics



def save_metrics(metrics: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2))
