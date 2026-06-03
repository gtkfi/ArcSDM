from typing import Optional, Sequence

import arcpy
import numpy as np


def parse_classifier_metric_names(metric_names: Optional[str]) -> Sequence[str]:
    return [metric.strip().lower() for metric in str(metric_names or "").split(";") if metric.strip()]


def classification_metric_value(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if metric == "accuracy":
        return float((y_true == y_pred).mean())

    classes = np.unique(y_true)
    class_values = []
    for class_value in classes:
        true_positive = np.sum((y_pred == class_value) & (y_true == class_value))
        false_positive = np.sum((y_pred == class_value) & (y_true != class_value))
        false_negative = np.sum((y_pred != class_value) & (y_true == class_value))

        precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) > 0 else 0.0
        recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) > 0 else 0.0

        if metric == "precision":
            class_values.append(precision)
        elif metric == "recall":
            class_values.append(recall)
        elif metric == "f1":
            class_values.append(float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0)
        else:
            return None

    return float(np.mean(class_values))


def log_classifier_test_metrics(metric_names: Optional[str], y_true: Optional[np.ndarray], y_pred: np.ndarray) -> None:
    if y_true is None:
        return

    metric_labels = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "F1",
    }
    for metric in parse_classifier_metric_names(metric_names):
        metric_value = classification_metric_value(metric, y_true, y_pred)
        if metric_value is not None:
            arcpy.AddMessage(f"Test {metric_labels[metric]}: {metric_value:.4f}")
