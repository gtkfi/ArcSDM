from typing import Optional, Sequence

import arcpy
import numpy as np


def regression_metric_value(metric: str, y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if metric == "mse":
        return float(np.mean((y_true - y_pred) ** 2))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric in ["mae", "l1"]:
        return float(np.mean(np.abs(y_true - y_pred)))
    if metric in ["r-squared", "r2", "r^2"]:
        y_mean = float(np.mean(y_true))
        ss_tot = float(np.sum((y_true - y_mean) ** 2))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        return 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return None


def parse_regression_metric_names(metric_names: Optional[str]) -> Sequence[str]:
    return [metric.strip().lower() for metric in str(metric_names or "").split(";") if metric.strip()]


def log_regression_metric(metric: str, y_true: np.ndarray, y_pred: np.ndarray, label_prefix: str) -> None:
    value = regression_metric_value(metric, y_true, y_pred)
    if value is None:
        return

    if metric == "mse":
        arcpy.AddMessage(f"{label_prefix} MSE: {value:.6f}")
    elif metric == "rmse":
        arcpy.AddMessage(f"{label_prefix} RMSE: {value:.6f}")
    elif metric in ["mae", "l1"]:
        arcpy.AddMessage(f"{label_prefix} MAE: {value:.6f}")
    elif metric in ["r-squared", "r2", "r^2"]:
        arcpy.AddMessage(f"{label_prefix} R-squared: {value:.6f}")


def log_regression_validation_metric(metric_name: Optional[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    metric = str(metric_name or "").strip().lower()
    log_regression_metric(metric, y_true, y_pred, "Validation")


def log_regression_test_metrics(metric_names: Optional[str], y_true: np.ndarray, y_pred: np.ndarray) -> None:
    for metric in parse_regression_metric_names(metric_names):
        log_regression_metric(metric, y_true, y_pred, "Test")
