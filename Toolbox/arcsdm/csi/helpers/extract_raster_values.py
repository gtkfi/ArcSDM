import numpy as np
import pandas as pd
import arcpy
from typing import List, Optional

def extract_raster_values(
    raster_path: str,
    points_df: pd.DataFrame,
    coords: Optional[List[str]] = None,
    has_geometry: bool = False
) -> np.ndarray:
    """Extract raster values at point locations.
    
    Args:
        raster_path: Path to raster file
        points_df: DataFrame with point data
        coords: List of two column names [x_col, y_col] containing coordinates
        has_geometry: Whether DataFrame has SHAPE@XY geometry column
    Returns:
        array of raster values at point locations
    """
    ras = arcpy.Raster(raster_path)
    arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")
    ex = ras.extent
    cw, ch = ras.meanCellWidth, ras.meanCellHeight
    nrows, ncols = arr.shape
    out = np.full(len(points_df), np.nan, dtype="float32")
    
    xs, ys = None, None
    
    # Option 1: Use coordinate column names from coords list
    if coords and len(coords) == 2:
        x_col, y_col = coords[0], coords[1]
        if x_col in points_df.columns and y_col in points_df.columns:
            xs = points_df[x_col].to_numpy(dtype="float32")
            ys = points_df[y_col].to_numpy(dtype="float32")
        else:
            arcpy.AddWarning(f"Coordinate columns '{coords}' not found in DataFrame")

    # Option 2: Auto-detect likely coordinate columns if coords not provided or not found
    if (xs is None or ys is None):
        likely_x = [c for c in points_df.columns if c.lower() in ["x", "lon", "longitude", "easting"]]
        likely_y = [c for c in points_df.columns if c.lower() in ["y", "lat", "latitude", "northing"]]
        if likely_x and likely_y:
            xs = points_df[likely_x[0]].to_numpy(dtype="float32")
            ys = points_df[likely_y[0]].to_numpy(dtype="float32")

    # Option 3: Fall back to geometry if nothing else worked
    if (xs is None or ys is None) and has_geometry and "SHAPE@XY" in points_df.columns:
        xy = points_df["SHAPE@XY"].to_numpy()
        xs = np.array([t[0] if isinstance(t, tuple) and len(t) == 2 else np.nan for t in xy], dtype="float32")
        ys = np.array([t[1] if isinstance(t, tuple) and len(t) == 2 else np.nan for t in xy], dtype="float32")
    
    if xs is None or ys is None:
        arcpy.AddWarning("Could not extract coordinates from points")
        return out
    
    cols = np.floor((xs - ex.XMin) / cw).astype("int64", copy=False)
    rows = np.floor((ex.YMax - ys) / ch).astype("int64", copy=False)
    valid = (
        np.isfinite(xs) & np.isfinite(ys) &
        (rows >= 0) & (cols >= 0) & (rows < nrows) & (cols < ncols)
    )
    out[valid] = arr[rows[valid], cols[valid]]
    return out