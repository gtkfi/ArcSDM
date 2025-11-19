
import arcpy
import numpy as np
import pandas as pd
from typing import List
from arcsdm.csi.core.pixel_vectors import create_pixel_vectors
from arcsdm.csi.core.calculate_pixel_to_label_csi import calculate_pixel_to_label_csi

from arcsdm.csi.helpers.validate_feature_rasters import validate_feature_rasters
from arcsdm.csi.helpers.sanitize_features import sanitize_features
from arcsdm.csi.helpers.save_results import save_csi_rasters

def pixel_to_label_csi(
    labeled_df: pd.DataFrame,
    feature_fields: List[str],
    rasters_list: List[str],
    out_raster_folder: str,
    label_field_names: List[str],
    csv_nodata: float
) -> bool:
    """
    - Validate feature rasters match feature space
    - Create pixel vectors from rasters
    - Calculate CSI between each pixel and each labeled point
    - Output p rasters (one per labeled point)
    """
    arcpy.AddMessage("\n" + "="*60)
    arcpy.AddMessage("PART 2: Pixel-to-Label CSI Analysis")
    arcpy.AddMessage("="*60)

    # Step 1: Validate feature rasters
    arcpy.AddMessage("Step 1: Validating feature rasters...")
    feature_raster_map = validate_feature_rasters(rasters_list, feature_fields)
    if feature_raster_map is None:
        arcpy.AddError("Feature raster validation failed")
        return False

    # Step 2: Create pixel vectors
    arcpy.AddMessage("Step 2: Creating pixel vectors...")
    pixel_vectors, reference_props = create_pixel_vectors(feature_raster_map, feature_fields)
    if pixel_vectors is None:
        arcpy.AddError("Failed to create pixel vectors")
        return False

    # Step 3: Prepare labeled feature vectors
    arcpy.AddMessage("Step 3: Preparing labeled feature vectors...")
    labeled_features_df = sanitize_features(labeled_df, feature_fields, csv_nodata)
    labeled_features = labeled_features_df.to_numpy(dtype='float32')

    # Validate labeled features
    valid_labels = []
    for i, label_vector in enumerate(labeled_features):
        if not np.any(np.isnan(label_vector)) and not np.any(label_vector == csv_nodata):
            valid_labels.append(i)

    if len(valid_labels) == 0:
        arcpy.AddError("No valid labeled feature vectors found")
        return False

    arcpy.AddMessage(f"Using {len(valid_labels)} valid labeled points out of {len(labeled_features)}")

    # Filter to valid labels only
    labeled_features = labeled_features[valid_labels]
    labeled_df_filtered = labeled_df.iloc[valid_labels].reset_index(drop=True)

    # Step 4: Calculate pixel-to-label CSI
    arcpy.AddMessage("Step 4: Calculating pixel-to-label CSI...")
    csi_arrays = calculate_pixel_to_label_csi(
        pixel_vectors, labeled_features, csv_nodata
    )

    # Step 5: Save output rasters
    arcpy.AddMessage("Step 5: Saving CSI rasters...")
    save_csi_rasters(
        csi_arrays, labeled_df_filtered, reference_props, out_raster_folder,
        label_field_names, csv_nodata
    )

    arcpy.AddMessage(f"\nPart 2 completed successfully!")
    arcpy.AddMessage(f"Created {len(csi_arrays)} CSI rasters in: {out_raster_folder}")

    return True