from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class MetadataPreprocessor:
    def __init__(self, numeric_columns=None, categorical_columns=None):
        self.numeric_columns = numeric_columns or []
        self.categorical_columns = categorical_columns or []
        self.numeric_imputer = SimpleImputer(strategy="median")
        self.categorical_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def fit(self, df):
        if not self.numeric_columns and not self.categorical_columns:
            return

        if self.numeric_columns:
            num = df[self.numeric_columns].copy().replace([np.inf, -np.inf], np.nan)
            self.numeric_imputer.fit(num)
            self.scaler.fit(self.numeric_imputer.transform(num))

        if self.categorical_columns:
            cat = df[self.categorical_columns].copy().fillna("missing")
            self.categorical_imputer.fit(cat)
            self.encoder.fit(self.categorical_imputer.transform(cat))

    def transform(self, df):
        if not self.numeric_columns and not self.categorical_columns:
            return np.zeros((len(df), 0), dtype=np.float32)

        parts = []

        if self.numeric_columns:
            num = df[self.numeric_columns].copy().replace([np.inf, -np.inf], np.nan)
            num = self.numeric_imputer.transform(num)
            num = self.scaler.transform(num)
            parts.append(num)

        if self.categorical_columns:
            cat = df[self.categorical_columns].copy().fillna("missing")
            cat = self.categorical_imputer.transform(cat)
            cat = self.encoder.transform(cat)
            parts.append(cat)

        if len(parts) == 1:
            return parts[0].astype(np.float32)
        return np.concatenate(parts, axis=1).astype(np.float32)
