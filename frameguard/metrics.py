"""Video-level metrics and validation-only operating-point selection."""

import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve, precision_recall_curve, average_precision_score)


def select_threshold(labels, scores):
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection needs both real and manipulated validation videos")
    fpr, tpr, thresholds = roc_curve(labels, scores)
    candidates = [(float(t - f), float(threshold)) for f, t, threshold in zip(fpr, tpr, thresholds)
                  if np.isfinite(threshold) and 0 <= threshold <= 1]
    return max(candidates, key=lambda item: (item[0], -abs(item[1] - 0.5)))[1]


def classification_metrics(labels, scores, threshold):
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("Expected equal nonempty labels and scores")
    if not set(labels).issubset({0, 1}) or not np.isfinite(scores).all() or min(scores) < 0 or max(scores) > 1:
        raise ValueError("Invalid labels or manipulation scores")
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {"count": len(labels), "real": int(tn + fp), "manipulated": int(tp + fn),
            "accuracy": float((tp + tn) / len(labels)),
            "specificity": float(tn / (tn + fp)) if tn + fp else None,
            "balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2) if tp + fn and tn + fp else None,
            "roc_auc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "precision": float(precision_score(labels, predicted, zero_division=0)),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "f1": float(f1_score(labels, predicted, zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "threshold": float(threshold), "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def diagnostic_curves(labels, scores):
    if len(set(labels)) != 2:
        return {"roc": None, "precision_recall": None, "average_precision": None}
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {"roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "precision_recall": {"recall": recall.tolist(), "precision": precision.tolist()},
            "average_precision": float(average_precision_score(labels, scores))}

