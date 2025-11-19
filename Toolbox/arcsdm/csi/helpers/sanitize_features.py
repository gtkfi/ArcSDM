import pandas as pd
import numpy as np
from typing import List

def sanitize_features(df: pd.DataFrame, fields: List[str], csv_nodata: float) -> pd.DataFrame:
    clean = df[fields].apply(pd.to_numeric, errors="coerce").replace(csv_nodata, np.nan)
    return clean
