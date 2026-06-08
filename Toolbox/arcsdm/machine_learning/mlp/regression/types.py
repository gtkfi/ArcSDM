"""Shared typing definitions for MLP regression modules."""

from typing import Optional, Sequence, Tuple, TypedDict, Union

import numpy as np


HiddenLayerSpec = Tuple[int, Optional[str], Optional[float]]
LastLayerConfig = Union[str, Tuple[Optional[str], Optional[float]], None]


class MLPRegressorPredictionResult(TypedDict):
    predictions: np.ndarray
    prediction_raster_array: np.ndarray
    ref_raster_path: str


ValidationTableFields = Tuple[Sequence[str], str]
