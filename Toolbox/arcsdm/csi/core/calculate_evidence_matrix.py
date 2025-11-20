
import os
import arcpy
import numpy as np
import pandas as pd
from typing import List, Dict

from arcsdm.csi.analysis.cosine_similarity import cosine_similarity

from arcsdm.csi.helpers.sanitize_features import sanitize_features
from arcsdm.csi.helpers.extract_raster_values import extract_raster_values


def calculate_evidence_matrix(
    labeled_df: pd.DataFrame,
    feature_fields: List[str],
    rasters_list: List[str],
    has_geometry: bool,
    csv_nodata: float
) -> Dict[str, np.ndarray]:
    """
    Calculate evidence matrix: labeled points vs raster values.
    Returns matrix of shape (n_labeled_points, n_rasters).
    """
    if not rasters_list:
        return {}

    # Focus only on labeled points
    n_points = len(labeled_df)
    n_rasters = len(rasters_list)

    arcpy.AddMessage(f"Calculating evidence matrix: {n_points} labeled points × {n_rasters} rasters")

    # Initialize evidence matrix
    evidence_matrix = np.full((n_points, n_rasters), csv_nodata, dtype=float)
    evidence_results = {}

    # Get feature vectors for labeled points
    features_clean_df = sanitize_features(labeled_df, feature_fields, csv_nodata)
    features_clean = features_clean_df.to_numpy(dtype=float)

    for raster_idx, raster_path in enumerate(rasters_list):
        arcpy.AddMessage(f"Processing raster {raster_idx + 1}/{n_rasters}: {os.path.basename(raster_path)}")

        # Extract raster values at labeled point locations
        raster_values = extract_raster_values(raster_path, labeled_df, has_geometry)

        # Calculate CSI between each labeled point and raster values
        for point_idx in range(n_points):
            raster_val = raster_values[point_idx]

            # Skip invalid raster values
            if np.isnan(raster_val) or raster_val == csv_nodata:
                continue

            # Get feature vector for this point
            point_features = features_clean[point_idx]

            # Skip if no valid features
            valid_features = ~np.isnan(point_features)
            if not np.any(valid_features):
                continue

            # For now, use first valid feature for CSI calculation with raster
            # Could be enhanced to use all features or specific combinations
            first_valid_feature = point_features[valid_features][0]

            # Calculate CSI between point feature and raster value
            csi_value = cosine_similarity(
                np.array([first_valid_feature]), 
                np.array([raster_val]), 
                csv_nodata
            )

            evidence_matrix[point_idx, raster_idx] = float(csi_value)

        # Store individual raster results
        raster_name = f"raster_{raster_idx + 1}_{os.path.basename(raster_path)}"
        evidence_results[raster_name] = evidence_matrix[:, raster_idx].copy()

        valid_count = np.sum(evidence_matrix[:, raster_idx] != csv_nodata)
        arcpy.AddMessage(f"  Calculated {valid_count} valid CSI values")

    # Store the full evidence matrix
    evidence_results['evidence_matrix'] = evidence_matrix

    arcpy.AddMessage(f"Evidence matrix shape: {evidence_matrix.shape}")
    return evidence_results
