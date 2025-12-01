import numpy as np
import pandas as pd
from typing import List
from arcsdm.csi.helpers.sanitize_features import sanitize_features

def calculate_class_centroids(labeled_df: pd.DataFrame, feature_fields: List[str], label_field_names: List[str], csv_nodata: float) -> pd.DataFrame:
    """
    Calculate class centroids for selected features.
    Returns:
        DataFrame with class centroids and number of points per class.
    """

    if not label_field_names:
        return pd.DataFrame()
    features_clean = sanitize_features(labeled_df, feature_fields, csv_nodata)
    primary_label_field = label_field_names[0]
    if primary_label_field not in labeled_df.columns:
        return pd.DataFrame()
    classes = labeled_df[primary_label_field].dropna().unique()
    centroids_data = []
    for class_value in classes:
        class_mask = labeled_df[primary_label_field] == class_value
        class_features = features_clean[class_mask]
        n_points = len(class_features)
        if n_points == 0:
            continue
        centroid = {'class': class_value, 'n_points': n_points}
        for feature_field in feature_fields:
            feature_values = class_features[feature_field].values
            valid_values = feature_values[~np.isnan(feature_values)]
            if len(valid_values) > 0:
                centroid[f'centroid_{feature_field}'] = float(np.mean(valid_values))
            else:
                centroid[f'centroid_{feature_field}'] = csv_nodata
        centroids_data.append(centroid)
    centroids_df = pd.DataFrame(centroids_data)
    return centroids_df
