
import pandas as pd
import sys
from pathlib import Path

import argparse

REQUIRED_FILES = {
    "patients": {
        "path": "hosp/patients.csv",
        "columns": ["subject_id", "anchor_age", "gender", "anchor_year_group"],
    },
    "admissions": {
        "path": "hosp/admissions.csv",
        "columns": ["subject_id", "hadm_id", "admittime", "dischtime", "admission_type", "admission_location"],
    },
    "microbiologyevents": {
        "path": "hosp/microbiologyevents.csv",
        "columns": ["subject_id", "hadm_id", "micro_specimen_id", "charttime", "chartdate", "spec_type_desc", "org_name", "ab_name", "interpretation"],
    },
}

def check_file(file_path, required_columns):
    # Accept .csv or .csv.gz
    if not file_path.exists():
        gz_path = file_path.with_suffix(file_path.suffix + '.gz')
        if gz_path.exists():
            file_path = gz_path
        else:
            print(f"File not found: {file_path} or {gz_path}")
            return
    print(f"\nChecking file: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    columns = list(df.columns)
    print(f"Columns in file: {columns}")
    total = len(df)
    print(f"Total rows: {total}")

    # Check for missing required columns in the file header
    missing_in_header = [col for col in required_columns if col not in columns]
    if missing_in_header:
        print(f"Missing required columns in header: {missing_in_header}")
        return

    all_present = 0
    missing_1 = {}
    missing_2 = []
    all_present_subject_ids = set()

    for idx, row in df.iterrows():
        missing = [col for col in required_columns if pd.isnull(row[col]) or row[col] == '']
        if len(missing) == 0:
            all_present += 1
            # Track subject_id if present
            if 'subject_id' in row:
                all_present_subject_ids.add(row['subject_id'])
        elif len(missing) == 1:
            col = missing[0]
            if col not in missing_1:
                missing_1[col] = []
            missing_1[col].append(idx)
        elif len(missing) == 2:
            missing_2.append((idx, missing))

    print(f"Rows with all required columns present: {all_present}")
    if 'subject_id' in required_columns:
        print(f"Unique patients (subject_id) with all required columns present: {len(all_present_subject_ids)}")
    print(f"Rows missing exactly 1 required column:")
    for col, idxs in missing_1.items():
        print(f"  Column '{col}': {len(idxs)} rows (examples: {idxs[:10]}{'...' if len(idxs)>10 else ''})")
    print(f"Rows missing exactly 2 required columns: {len(missing_2)}")
    if missing_2:
        print("  Example (row, missing columns):")
        for i, (idx, cols) in enumerate(missing_2[:10]):
            print(f"    Row {idx}: {cols}")
        if len(missing_2) > 10:
            print("    ...")

def main():
    parser = argparse.ArgumentParser(description="Check required columns in MIMIC-IV files.")
    parser.add_argument("--mimic_path", type=str, required=True, help="Path to MIMIC-IV data directory (containing hosp/ and icu/)")
    args = parser.parse_args()

    mimic_path = Path(args.mimic_path)
    for file_key, info in REQUIRED_FILES.items():
        file_path = mimic_path / info["path"]
        check_file(file_path, info["columns"])

if __name__ == "__main__":
    main()
