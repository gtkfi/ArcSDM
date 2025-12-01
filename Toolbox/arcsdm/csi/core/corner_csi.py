
import numpy as np
import pandas as pd
from typing import List
from arcsdm.csi.analysis.cosine_similarity import cosine_similarity

from arcsdm.csi.helpers.sanitize_features import sanitize_features

def calculate_corner_csi(labeled_df: pd.DataFrame, feature_fields: List[str], csv_nodata: float) -> np.ndarray:
    """
    Compute the corner CSI matrix for all labeled points using selected features.
    Returns:
        symmetric matrix with diagonal values set to 1.0.
    """
    F = sanitize_features(labeled_df, feature_fields, csv_nodata)
    n = len(F)
    corner_csi = np.full((n, n), None, dtype=np.float64)
    features_array = F.to_numpy(dtype=np.float64)
    valid_masks = [~(np.isnan(features_array[i]) | (features_array[i] == csv_nodata)) for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                corner_csi[i, j] = 1.0
            elif np.any(valid_masks[i]) and np.any(valid_masks[j]):
                corner_csi[i, j] = float(cosine_similarity(features_array[i], features_array[j], csv_nodata))
    return corner_csi
