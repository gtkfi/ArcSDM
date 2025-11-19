import pandas as pd
from typing import List

def rows_with_labels(df: pd.DataFrame, label_cols: List[str], csv_nodata: float) -> pd.Series:
    """Return boolean Series indicating rows with at least one valid label."""

    if not label_cols:
        return pd.Series(False, index=df.index)
    masks = []
    for col in label_cols:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            m = s.notna() & (s.astype(float) != float(csv_nodata))
        else:
            txt = s.astype("string").str.strip()
            m = txt.notna() & (txt != "") & ~txt.str.lower().isin({"nan", "none", "null", str(csv_nodata).lower()})
        masks.append(m)
    mask = masks[0]
    for m in masks[1:]:
        mask = mask | m
    return mask
