from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def patient_disjoint_split(df, patient_col, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42):
    if patient_col not in df.columns:
        raise ValueError(f"Required column not found. Available columns: {list(df.columns)}")

    patients = pd.Series(df[patient_col].dropna().unique()).astype(str).tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_patients = len(patients)
    train_n = int(np.floor(n_patients * train_frac))
    val_n = int(np.floor(n_patients * val_frac))
    test_n = max(1, n_patients - train_n - val_n) if n_patients > 1 else 0

    train_patients = set(patients[:train_n])
    val_patients = set(patients[train_n:train_n + val_n])
    test_patients = set(patients[train_n + val_n:train_n + val_n + test_n])

    assert len(train_patients & val_patients) == 0
    assert len(train_patients & test_patients) == 0
    assert len(val_patients & test_patients) == 0

    train_df = df[df[patient_col].astype(str).isin(train_patients)].copy()
    val_df = df[df[patient_col].astype(str).isin(val_patients)].copy()
    test_df = df[df[patient_col].astype(str).isin(test_patients)].copy()

    return train_df, val_df, test_df


def grouped_kfold(df, patient_col, n_splits=5):
    patients = df[patient_col].astype(str).dropna().unique()
    folds = []
    for i in range(n_splits):
        pass
    return folds
