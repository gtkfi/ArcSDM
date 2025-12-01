import pandas as pd
from typing import List

def rows_with_labels(df: pd.DataFrame, label_cols: List[str], csv_nodata: float, apply_nodata: bool = True) -> pd.Series:
    """Return boolean Series indicating rows with at least one valid label.
    If apply_nodata is False, only filter out NaNs/Nones/nulls, not csv_nodata values.
    Returns:
        pd.Series of booleans indicating rows with valid labels.
    """
    if not label_cols:
        return pd.Series(False, index=df.index)
    masks = []
    for col in label_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            if apply_nodata:
                m = s.notna() & (s.astype(float) != float(csv_nodata))
            else:
                m = s.notna()
        else:
            txt = s.astype("string").str.strip()
            if apply_nodata:
                m = txt.notna() & (txt != "") & ~txt.str.lower().isin({"nan", "none", "null", str(csv_nodata).lower()})
            else:
                m = txt.notna() & (txt != "") & ~txt.str.lower().isin({"nan", "none", "null"})
        masks.append(m)
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return mask
