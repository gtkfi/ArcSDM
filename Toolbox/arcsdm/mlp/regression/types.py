from typing import Optional, Sequence, Tuple, TypedDict

import numpy as np


HiddenLayerSpec = Tuple[int, Optional[str], Optional[float]]
LastLayerConfig = Tuple[Optional[str], Optional[float]]


class MLPRegressorPredictionResult(TypedDict):
    predictions: np.ndarray
    prediction_raster_array: np.ndarray
    ref_raster_path: str


ValidationTableFields = Tuple[Sequence[str], str]
