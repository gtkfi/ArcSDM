
import numpy as np
import arcpy
from typing import Dict, List, Tuple, Optional, Any
from arcsdm.csi.helpers.raster_properties import get_raster_properties


def create_pixel_vectors(
    feature_raster_map: Dict[str, str],
    feature_fields: List[str]
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Create pixel vectors from feature rasters.
    Returns 3D array: (nrows, ncols, n_features)
    Uses mask if set in environment.
    """
    arcpy.AddMessage("Creating pixel vectors from feature rasters...")

    # Load all raster properties
    raster_props = {}
    reference_props = None

    for field in feature_fields:
        raster_path = feature_raster_map[field]
        props = get_raster_properties(raster_path)
        if props is None:
            return None, None

        raster_props[field] = props

        # Use first raster as reference for dimensions and extent
        if reference_props is None:
            reference_props = props

    # Validate all rasters have same dimensions
    nrows, ncols = reference_props['nrows'], reference_props['ncols']
    for field, props in raster_props.items():
        if props['nrows'] != nrows or props['ncols'] != ncols:
            arcpy.AddError(f"Raster dimension mismatch for {field}: "
                          f"Expected {nrows}x{ncols}, got {props['nrows']}x{props['ncols']}")
            return None, None

    # Create 3D array for pixel vectors
    pixel_vectors = np.full((nrows, ncols, len(feature_fields)), np.nan, dtype='float64')

    for i, field in enumerate(feature_fields):
        pixel_vectors[:, :, i] = raster_props[field]['array']
        arcpy.AddMessage(f"Loaded feature '{field}' into pixel vector layer {i+1}")

    # Apply mask if set in environment
    mask_array = None
    if arcpy.env.mask:
        try:
            arcpy.AddMessage(f"Applying mask from: {arcpy.env.mask}")

            mask = arcpy.Describe(arcpy.env.mask)
            if mask.dataType != 'RasterDataset':
                # create temporary raster from feature class
                arcpy.AddMessage("Mask is not a raster dataset - creating temporary raster from feature class")
                temp_mask_raster = arcpy.FeatureToRaster_conversion(arcpy.env.mask, mask.shapeFieldName, "in_memory\\temp_mask", reference_props['cell_size']).getOutput(0)

            mask_raster = arcpy.Raster(temp_mask_raster if mask.dataType != 'RasterDataset' else arcpy.env.mask)
            mask_nodata = mask_raster.noDataValue
            mask_array_raw = arcpy.RasterToNumPyArray(mask_raster)
            mask_array = (mask_array_raw != mask_nodata)

            # Validate mask dimensions
            if mask_array.shape != (nrows, ncols):
                arcpy.AddWarning(f"Mask dimensions {mask_array.shape} don't match raster dimensions {(nrows, ncols)}")
                # Apply mask to all feature layers at once (vectorized)
                pixel_vectors[~mask_array] = np.nan

                valid_pixels = np.sum(mask_array)
                arcpy.AddMessage(f"Mask applied: {valid_pixels} valid pixels out of {nrows * ncols}")
                valid_pixels = np.sum(mask_array)
                arcpy.AddMessage(f"Mask applied: {valid_pixels} valid pixels out of {nrows * ncols}")
        except Exception as e:
            arcpy.AddWarning(f"Failed to apply mask: {e}")
            mask_array = None
    else:
        arcpy.AddMessage("No mask set in environment")

    # Store mask info in reference props
    reference_props['mask_array'] = mask_array

    return pixel_vectors, reference_props
