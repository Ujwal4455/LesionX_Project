from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd


def _detect_field(columns, candidates):
    normalized = {c.lower().replace(" ", "_"): c for c in columns}
    for keyword in candidates:
        for key, original in normalized.items():
            if keyword in key:
                return original
    return None


def inspect_dataset(root):
    if not root:
        raise ValueError("Dataset path is empty. Set dataset.root in config.yaml.")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError("Dataset path does not exist. Please update dataset.root in config.yaml.")

    files = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".parquet", ".json"}:
            files.append(path)

    if not files:
        raise ValueError(f"No dataset files found under {root_path}")

    detected = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame([data])
            else:
                sep = "\t" if file_path.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(file_path, sep=sep)
            detected.append((file_path, df))
        except Exception:
            continue

    if not detected:
        raise ValueError("No readable dataset files were found.")

    selected_path, df = detected[0]
    columns = list(df.columns)

    fields = {
        "patient_id": _detect_field(columns, ["patient", "subject", "pid", "patient_id"]),
        "lesion_id": _detect_field(columns, ["lesion", "lesion_id", "segment", "instance"]),
        "image_path": _detect_field(columns, ["image", "img", "filepath", "path", "image_path"]),
        "target": _detect_field(columns, ["target", "label", "diagnosis", "class"]),
        "visit_id": _detect_field(columns, ["visit", "visit_id"]),
        "date": _detect_field(columns, ["date", "timestamp", "time", "visit_date"]),
        "clinical_info": _detect_field(columns, ["clinical", "history", "symptom", "notes"]),
        "tbp": _detect_field(columns, ["tbp", "melan", "clinical", "measurement"]),
        "metadata": _detect_field(columns, ["metadata", "meta", "features"]),
    }

    print(f"Dataset root: {root_path}")
    print(f"Files scanned: {len(files)}")
    print(f"Selected file: {selected_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {columns}")
    print("\nDetected likely fields:")
    for key, value in fields.items():
        print(f"  {key}: {value}")

    print("\nNumber of patients:", df[fields["patient_id"]].nunique() if fields["patient_id"] else "N/A")
    print("Number of lesions:", df[fields["lesion_id"]].nunique() if fields["lesion_id"] else "N/A")
    print("Number of images:", len(df) if fields["image_path"] else "N/A")
    print("Number of classes:", df[fields["target"]].nunique() if fields["target"] else "N/A")

    if fields["target"] is not None:
        print("\nClass distribution:")
        print(df[fields["target"]].value_counts(dropna=False).to_string())

    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print("\nDuplicate records:", int(df.duplicated().sum()))

    if fields["image_path"]:
        missing = 0
        for img_path in df[fields["image_path"]].dropna():
            if not Path(str(img_path)).exists() and not (root_path / str(img_path)).exists():
                missing += 1
        print("Missing image files:", missing)

    print("\nAvailable metadata and temporal fields:")
    for col in columns:
        print(f"  - {col}")

    temporal = ["visit_id", "date", "timestamp", "time", "visit"]
    print("\nAvailable temporal fields:")
    found = [name for name in temporal if any(name in c.lower() for c in columns)]
    print(found if found else "None found")

    return {
        "dataset_root": str(root_path),
        "selected_file": str(selected_path),
        "shape": df.shape,
        "columns": columns,
        "fields": fields,
        "dataframe": df,
    }
