import numpy as np


def extract_tbp_features(df, available_columns=None):
    if df is None or len(df) == 0:
        return None
    candidate = available_columns or df.columns.tolist()
    cols = [c for c in candidate if c in df.columns]
    if not cols:
        return None
    return df[cols].copy()
