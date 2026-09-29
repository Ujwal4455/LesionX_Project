from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def compute_metrics(y_true, y_pred, threshold=0.5):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {}
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_pred))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_pred))
    else:
        metrics["auroc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    metrics["f1"] = float(f1_score(y_true, (y_pred >= threshold).astype(int)))
    return metrics
