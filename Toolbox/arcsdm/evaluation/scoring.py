"""
Original source: https://github.com/GispoCoding/eis_toolkit
"""

from numbers import Number

import arcpy
import numpy as np
import pandas as pd
from typing import Dict, Optional, Sequence, Union
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
)


def score_predictions(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    metrics: Union[str, Sequence[str]],
    decimals: Optional[int] = None,
) -> Union[Number, Dict[str, Number]]:
    """
    Score model predictions with given metrics.

    One or multiple metrics can be defined for scoring.

    Supported classifier metrics: "accuracy", "precision", "recall", "f1".
    Supported regressor metrics: "mse", "rmse", "mae", "r2".

    Args:
        y_true: Target values ("ground truth") against which scoring is performed.
        y_pred: Predicted labels.
        metrics: The metrics to use for scoring the model. Select only metrics applicable
            for the model type.
        decimals: Number of decimals used in rounding the scores. If None, scores are not rounded.
            Defaults to None.

    Returns:
        Metric scores as a dictionary if multiple metrics, otherwise just the metric value.
    """
    if isinstance(metrics, str):
        score = _score_predictions(y_true, y_pred, metrics)
        return round(score, decimals) if decimals is not None else score
    else:
        out_metrics = {}
        for metric in metrics:
            score = _score_predictions(y_true, y_pred, metric)
            out_metrics[metric] = round(score, decimals) if decimals is not None else score
        return out_metrics


def _score_predictions(
    y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series], metric: str
) -> Number:
    num_classes = len(np.unique(y_true))

    # Multiclass classification
    if num_classes > 2:
        average_method = "micro"
    # Binary classification
    else:
        average_method = "binary"

    if metric == "mae":
        score = mean_absolute_error(y_true, y_pred)
    elif metric == "mse":
        score = mean_squared_error(y_true, y_pred)
    elif metric == "rmse":
        score = root_mean_squared_error(y_true, y_pred)
    elif metric == "r2":
        score = r2_score(y_true, y_pred)
    elif metric == "accuracy":
        score = accuracy_score(y_true, y_pred)
    elif metric == "precision":
        score = precision_score(y_true, y_pred, average=average_method)
    elif metric == "recall":
        score = recall_score(y_true, y_pred, average=average_method)
    elif metric == "f1":
        score = f1_score(y_true, y_pred, average=average_method)
    else:
        arcpy.AddError(f"Unrecognized metric: {metric}")
        raise arcpy.ExecuteError

    return score