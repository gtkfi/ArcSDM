
import arcpy
import numpy as np
from typing import Dict, Any, Optional

def get_raster_properties(
    raster_path: str
) -> Optional[Dict[str, Any]]:
    """Get raster spatial properties for processing
    Returns:
        dictionary of raster properties including array, extent, cell size, spatial reference, and nodata value
    """
    try:
        raster = arcpy.Raster(raster_path)
        extent = raster.extent
        cell_width = raster.meanCellWidth
        cell_height = raster.meanCellHeight
        spatial_ref = raster.spatialReference

        # Convert to numpy array
        array = arcpy.RasterToNumPyArray(raster, nodata_to_value=np.nan).astype("float64")

        return {
            'array': array,
            'extent': extent,
            'cell_width': cell_width,
            'cell_height': cell_height,
            'spatial_ref': spatial_ref,
            'nrows': array.shape[0],
            'ncols': array.shape[1],
            'nodata_value': raster.noDataValue
        }
    except Exception as e:
        arcpy.AddError(f"Error reading raster {raster_path}: {e}")
        return None