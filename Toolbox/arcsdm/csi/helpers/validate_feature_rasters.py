
import arcpy
import os
from typing import List, Dict, Optional


def validate_feature_rasters(
    rasters_list: List[str],
    feature_fields: List[str]
) -> Optional[Dict[str, str]]:
    """
    Validate that rasters correspond to feature space variables.
    Returns:
        mapping of feature field to raster path.
    """
    if len(rasters_list) != len(feature_fields):
        arcpy.AddError(f"Number of rasters ({len(rasters_list)}) must match number of feature fields ({len(feature_fields)})")
        return None

    feature_raster_map = {}
    for i, (field, raster_path) in enumerate(zip(feature_fields, rasters_list)):
        if not arcpy.Exists(raster_path):
            arcpy.AddError(f"Raster does not exist: {raster_path}")
            return None

        feature_raster_map[field] = raster_path
        arcpy.AddMessage(f"Feature '{field}' mapped to raster: {os.path.basename(raster_path)}")

    return feature_raster_map