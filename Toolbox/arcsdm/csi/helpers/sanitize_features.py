import pandas as pd
import numpy as np
from typing import List

def sanitize_features(df: pd.DataFrame, fields: List[str], csv_nodata: float) -> pd.DataFrame:
    """Sanitize feature columns by converting to numeric and replacing csv_nodata with NaN.
    Returns:
        pd.DataFrame with sanitized feature columns.
    """
    clean = df[fields].apply(pd.to_numeric, errors="coerce").replace(csv_nodata, np.nan)
    return clean
