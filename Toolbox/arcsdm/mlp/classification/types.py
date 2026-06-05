"""Shared typing definitions for MLP classification modules."""

from typing import Optional, Tuple, TypedDict, Union

import numpy as np


HiddenLayerSpec = Tuple[int, Optional[str], Optional[float]]
LastLayerConfig = Union[str, Tuple[Optional[str], Optional[float]], None]


class MLPClassifierPredictionResult(TypedDict):
    y_true: Optional[np.ndarray]
    y_pred: np.ndarray
    predicted_probabilities: np.ndarray
    prob_raster_array: np.ndarray
    class_raster_array: np.ndarray
    ref_raster_path: str
