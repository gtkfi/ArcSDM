import pandas as pd
from typing import Optional, Tuple

def detect_xy_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    norm = {c.replace(" ", "").lstrip("\ufeff").strip().lower(): c for c in df.columns}
    pairs = [
        ("x","y"), ("lon","lat"), ("longitude","latitude"),
        ("easting","northing"), ("xcoord","ycoord"),
        ("projx","projy"), ("utm_x","utm_y"), ("x_","y_"),
        ("point_x", "point_y"), ("coord_x", "coord_y"),
        ("x_coord", "y_coord"), ("xcoordinate", "ycoordinate"),
    ]
    for a, b in pairs:
        if a in norm and b in norm:
            return norm[a], norm[b]
    if "X" in df.columns and "Y" in df.columns:
        return "X", "Y"
    return None, None
