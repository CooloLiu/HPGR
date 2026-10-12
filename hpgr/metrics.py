"""Anomaly-detection metrics: AUC / AP / F1 and the threshold that maximizes F1."""
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score


def evaluate_anomaly_detection(anomaly_scores, labels):
    """Evaluate anomaly-detection performance and locate the best threshold."""
    # Labels are binarized: any non-zero label counts as an anomaly (1).
    labels_binary = (labels != 0).astype(int)

    # With a single class present, AUC and friends are undefined.
    if len(np.unique(labels_binary)) < 2:
        print("Warning: only one class present, evaluation is not meaningful.")
        best_threshold = np.quantile(anomaly_scores, 0.95)
        return {
            'roc_auc': 0.5,
            'average_precision': 0.0,
            'f1_score': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'best_threshold': best_threshold
        }

    roc_auc = roc_auc_score(labels_binary, anomaly_scores)
    ap_score = average_precision_score(labels_binary, anomaly_scores)

    # Sweep the thresholds and keep the F1-maximizing one.
    precisions, recalls, thresholds = precision_recall_curve(labels_binary, anomaly_scores)

    f1_scores = np.divide(2 * precisions * recalls, precisions + recalls,
                          out=np.zeros_like(precisions), where=(precisions + recalls) != 0)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    return {
        'roc_auc': roc_auc,
        'average_precision': ap_score,
        'f1_score': best_f1,
        'precision': best_precision,
        'recall': best_recall,
        'best_threshold': best_threshold
    }
