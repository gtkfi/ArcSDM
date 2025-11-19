
import numpy as np
import pandas as pd
from typing import List
from arcsdm.csi.analysis.cosine_similarity import cosine_similarity

def calculate_centroid_to_centroid_csi(centroids_df: pd.DataFrame, feature_fields: List[str], csv_nodata: float) -> np.ndarray:
    """
    Compute pairwise CSI between class centroids for selected features.
    Returns a symmetric similarity matrix with diagonal values set to 1.0.
    """
    n_classes = len(centroids_df)
    if n_classes == 0:
        return np.array([])
    centroid_vectors = []
    for feature_field in feature_fields:
        col_name = f'centroid_{feature_field}'
        if col_name in centroids_df.columns:
            centroid_vectors.append(centroids_df[col_name].values)
        else:
            centroid_vectors.append(np.full(n_classes, csv_nodata))
    centroid_array = np.array(centroid_vectors).T
    csi_matrix = np.full((n_classes, n_classes), csv_nodata, dtype=np.float64)
    for i in range(n_classes):
        csi_matrix[i, i] = 1.0
        for j in range(i + 1, n_classes):
            csi_value = cosine_similarity(centroid_array[i], centroid_array[j], csv_nodata)
            csi_matrix[i, j] = float(csi_value)
            csi_matrix[j, i] = float(csi_value)
    return csi_matrix
