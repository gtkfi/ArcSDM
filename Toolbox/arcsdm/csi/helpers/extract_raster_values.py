import numpy as np
import pandas as pd
import arcpy
from arcsdm.csi.helpers.detect_xy_columns import detect_xy_columns

def extract_raster_values(raster_path: str, points_df: pd.DataFrame, has_geometry: bool) -> np.ndarray:
    """Extract raster values at point locations."""

    ras = arcpy.Raster(raster_path)
    arr = arcpy.RasterToNumPyArray(ras, nodata_to_value=np.nan).astype("float32")
    ex = ras.extent
    cw, ch = ras.meanCellWidth, ras.meanCellHeight
    nrows, ncols = arr.shape
    out = np.full(len(points_df), np.nan, dtype="float32")
    xs = ys = None
    if has_geometry and "SHAPE@XY" in points_df.columns:
        xy = points_df["SHAPE@XY"].to_numpy()
        xs = np.array([t[0] if isinstance(t, tuple) and len(t) == 2 else np.nan for t in xy], dtype="float32")
        ys = np.array([t[1] if isinstance(t, tuple) and len(t) == 2 else np.nan for t in xy], dtype="float32")
    else:
        xcol, ycol = detect_xy_columns(points_df)
        if xcol and ycol:
            xs = pd.to_numeric(points_df[xcol], errors="coerce").to_numpy(dtype="float32")
            ys = pd.to_numeric(points_df[ycol], errors="coerce").to_numpy(dtype="float32")
    if xs is None or ys is None:
        return out
    cols = np.floor((xs - ex.XMin) / cw).astype("int64", copy=False)
    rows = np.floor((ex.YMax - ys) / ch).astype("int64", copy=False)
    valid = (
        np.isfinite(xs) & np.isfinite(ys) &
        (rows >= 0) & (cols >= 0) & (rows < nrows) & (cols < ncols)
    )
    out[valid] = arr[rows[valid], cols[valid]]
    return out
