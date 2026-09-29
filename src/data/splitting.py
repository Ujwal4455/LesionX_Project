import numpy as np
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

def patient_split(df, patient_col, seed=42, ratios=(.7,.15,.15)):
    if patient_col not in df: raise ValueError(f'Required column not found. Available columns: {list(df.columns)}')
    g=df[patient_col].astype(str); idx=np.arange(len(df)); a=GroupShuffleSplit(n_splits=1,test_size=ratios[1]+ratios[2],random_state=seed)
    tr,rest=next(a.split(idx,groups=g)); rest_groups=g.iloc[rest]; b=GroupShuffleSplit(n_splits=1,test_size=ratios[2]/sum(ratios[1:]),random_state=seed)
    va_rel,te_rel=next(b.split(rest,groups=rest_groups)); va=rest[va_rel]; te=rest[te_rel]
    assert set(g.iloc[tr]) .isdisjoint(set(g.iloc[va])); assert set(g.iloc[tr]).isdisjoint(set(g.iloc[te])); assert set(g.iloc[va]).isdisjoint(set(g.iloc[te]))
    return df.iloc[tr].copy(),df.iloc[va].copy(),df.iloc[te].copy()

def grouped_folds(df, patient_col, n_splits=5):
    return list(GroupKFold(n_splits=n_splits).split(df, groups=df[patient_col].astype(str)))
