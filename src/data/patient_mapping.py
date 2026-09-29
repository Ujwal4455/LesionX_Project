import numpy as np
import pandas as pd


def patient_lesion_mapping(df, patient_col, lesion_col):
    if patient_col not in df.columns or lesion_col not in df.columns:
        raise ValueError(f"Required column not found. Available columns: {list(df.columns)}")

    patient_to_lesions = {}
    lesion_to_patient = {}
    patient_to_images = {}
    lesion_to_images = {}
    patient_to_visits = {}

    for _, row in df.iterrows():
        patient = row.get(patient_col)
        lesion = row.get(lesion_col)
        if pd.isna(patient) or pd.isna(lesion):
            continue
        patient = str(patient)
        lesion = str(lesion)

        patient_to_lesions.setdefault(patient, set()).add(lesion)
        lesion_to_patient[lesion] = patient

        image_col = row.index[row.notna()]
        # This function only creates the core patient/lesion mapping.
        # Image and visit mapping are added when the dataset provides those columns.

    return {
        "patient_to_lesions": {k: sorted(v) for k, v in patient_to_lesions.items()},
        "lesion_to_patient": lesion_to_patient,
        "patient_to_images": patient_to_images,
        "lesion_to_images": lesion_to_images,
        "patient_to_visits": patient_to_visits,
    }
