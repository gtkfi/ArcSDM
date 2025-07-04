import os
import arcpy

from arcpy import sa
from arcpy import conversion

def rasterize_vector(raster_file_path: str, label_file: str, target_label_attr: str):
    """
    Rasterize a vector file to a raster file with the same extent and cell size as the reference raster.
    """
    reference_raster = arcpy.Raster(raster_file_path)
    desc_ref_raster = arcpy.Describe(reference_raster)

    filename = os.path.basename(desc_ref_raster.catalogPath)
    arcpy.AddWarning(f"Setting Environment extent to be the same as in {filename}")
    arcpy.env.extent = desc_ref_raster.extent
    arcpy.AddWarning(f"Setting Environment cell size to be the same as in {filename}")
    arcpy.env.cellSize = desc_ref_raster.meanCellWidth
    arcpy.AddWarning(f"Setting Environment Snap Raster to be {filename}")
    arcpy.env.snapRaster = raster_file_path

    # Create an empty raster with the same extent and cell size as the reference raster
    empty_raster = sa.CreateConstantRaster(0, "INTEGER", desc_ref_raster.meanCellWidth, desc_ref_raster.extent)

    # Rasterize points to raster
    points_raster = conversion.PointToRaster(
        in_features=label_file,
        value_field=target_label_attr,
        cellsize=desc_ref_raster.meanCellWidth,
    )

    # Combine empty raster and points raster
    rasterized_vector = sa.Con(sa.IsNull(points_raster), empty_raster, (points_raster + empty_raster))
    
    return rasterized_vector