def match_lesions(df, patient_col, lesion_col, visit_col=None):
    if patient_col not in df.columns or lesion_col not in df.columns:
        return {"status": "disabled", "message": "Patient or lesion columns unavailable."}
    if visit_col is not None and visit_col not in df.columns:
        return {"status": "disabled", "message": "Temporal matching disabled: visit field missing."}
    return {"status": "supported" if visit_col else "disabled", "matches": []}
