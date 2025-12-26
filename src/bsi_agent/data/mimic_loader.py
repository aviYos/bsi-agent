"""MIMIC-IV data loading utilities."""

import gzip
from pathlib import Path
from typing import Optional

import pandas as pd


class MIMICLoader:
    """Load and manage MIMIC-IV data files."""

    def __init__(self, mimic_path: str | Path):
        """
        Initialize MIMIC loader.

        Args:
            mimic_path: Path to MIMIC-IV root directory (contains hosp/, icu/ folders)
        """
        self.mimic_path = Path(mimic_path)
        self._validate_path()
        self._cache: dict[str, pd.DataFrame] = {}

    def _validate_path(self) -> None:
        """Validate MIMIC-IV directory structure exists."""
        if not self.mimic_path.exists():
            raise FileNotFoundError(f"MIMIC path not found: {self.mimic_path}")

    def _load_csv(
        self,
        module: str,
        table: str,
        usecols: Optional[list[str]] = None,
        dtype: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Load a MIMIC-IV CSV file.

        Args:
            module: MIMIC module ('hosp' or 'icu')
            table: Table name without extension
            usecols: Columns to load (None for all)
            dtype: Column data types

        Returns:
            DataFrame with table data
        """
        cache_key = f"{module}/{table}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try both .csv.gz and .csv
        file_path = self.mimic_path / module / f"{table}.csv.gz"
        if not file_path.exists():
            file_path = self.mimic_path / module / f"{table}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Table not found: {file_path}")

        df = pd.read_csv(
            file_path,
            usecols=usecols,
            dtype=dtype,
            low_memory=False,
        )
        return df

    def load_patients(self) -> pd.DataFrame:
        """Load patients table."""
        return self._load_csv(
            "hosp",
            "patients",
            dtype={"subject_id": int},
        )

    def load_admissions(self) -> pd.DataFrame:
        """Load admissions table."""
        df = self._load_csv(
            "hosp",
            "admissions",
            dtype={"subject_id": int, "hadm_id": int},
        )
        # Parse datetime columns
        for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def load_labevents(
        self,
        subject_ids: Optional[list[int]] = None,
        itemids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Load laboratory events.

        Args:
            subject_ids: Filter to specific patients
            itemids: Filter to specific lab items

        Returns:
            DataFrame with lab events
        """
        df = self._load_csv(
            "hosp",
            "labevents",
            dtype={"subject_id": int, "hadm_id": float, "itemid": int},
        )
        df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

        if subject_ids is not None:
            df = df[df["subject_id"].isin(subject_ids)]
        if itemids is not None:
            df = df[df["itemid"].isin(itemids)]

        return df

    def load_microbiologyevents(self) -> pd.DataFrame:
        """Load microbiology events (cultures, susceptibilities)."""
        df = self._load_csv(
            "hosp",
            "microbiologyevents",
            dtype={"subject_id": int, "hadm_id": float, "micro_specimen_id": int},
        )
        # Parse datetime columns
        for col in ["chartdate", "charttime", "storedate", "storetime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def load_prescriptions(
        self,
        subject_ids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Load prescriptions (medications)."""
        df = self._load_csv(
            "hosp",
            "prescriptions",
            dtype={"subject_id": int, "hadm_id": int},
        )
        for col in ["starttime", "stoptime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if subject_ids is not None:
            df = df[df["subject_id"].isin(subject_ids)]

        return df

    def load_chartevents(
        self,
        subject_ids: Optional[list[int]] = None,
        itemids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """
        Load chart events (vital signs from ICU).

        Note: This is a large file, filtering recommended.

        Args:
            subject_ids: Filter to specific patients
            itemids: Filter to specific items (vital signs)

        Returns:
            DataFrame with chart events
        """
        # chartevents is huge - load in chunks if filtering
        file_path = self.mimic_path / "icu" / "chartevents.csv.gz"
        if not file_path.exists():
            file_path = self.mimic_path / "icu" / "chartevents.csv"

        if subject_ids is not None or itemids is not None:
            # Load in chunks for filtering
            chunks = []
            for chunk in pd.read_csv(
                file_path,
                chunksize=500000,
                dtype={"subject_id": int, "hadm_id": float, "itemid": int},
            ):
                if subject_ids is not None:
                    chunk = chunk[chunk["subject_id"].isin(subject_ids)]
                if itemids is not None:
                    chunk = chunk[chunk["itemid"].isin(itemids)]
                if len(chunk) > 0:
                    chunks.append(chunk)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        else:
            df = pd.read_csv(
                file_path,
                dtype={"subject_id": int, "hadm_id": float, "itemid": int},
            )

        if len(df) > 0:
            df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")

        return df

    def load_d_labitems(self) -> pd.DataFrame:
        """Load lab items dictionary."""
        return self._load_csv("hosp", "d_labitems")

    def load_d_items(self) -> pd.DataFrame:
        """Load chart items dictionary (for ICU)."""
        return self._load_csv("icu", "d_items")

    def load_icustays(self) -> pd.DataFrame:
        """Load ICU stays."""
        df = self._load_csv(
            "icu",
            "icustays",
            dtype={"subject_id": int, "hadm_id": int, "stay_id": int},
        )
        for col in ["intime", "outtime"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def load_diagnoses_icd(self) -> pd.DataFrame:
        """Load ICD diagnoses."""
        return self._load_csv(
            "hosp",
            "diagnoses_icd",
            dtype={"subject_id": int, "hadm_id": int, "seq_num": int},
        )


# Common lab item IDs in MIMIC-IV
LAB_ITEMIDS = {
    # Complete Blood Count
    "wbc": 51301,  # White Blood Cells
    "hemoglobin": 51222,
    "hematocrit": 51221,
    "platelets": 51265,
    # Metabolic Panel
    "sodium": 50983,
    "potassium": 50971,
    "chloride": 50902,
    "bicarbonate": 50882,
    "bun": 51006,  # Blood Urea Nitrogen
    "creatinine": 50912,
    "glucose": 50931,
    # Liver Function
    "bilirubin_total": 50885,
    "alt": 50861,
    "ast": 50878,
    "alkaline_phosphatase": 50863,
    "albumin": 50862,
    # Infection Markers
    "lactate": 50813,
    "procalcitonin": 51652,
    "crp": 50889,  # C-Reactive Protein
    # Coagulation
    "inr": 51237,
    "pt": 51274,
    "ptt": 51275,
}

# Common vital sign item IDs in MIMIC-IV ICU
VITAL_ITEMIDS = {
    "heart_rate": 220045,
    "sbp": 220050,  # Systolic BP (non-invasive)
    "dbp": 220051,  # Diastolic BP (non-invasive)
    "mbp": 220052,  # Mean BP (non-invasive)
    "respiratory_rate": 220210,
    "temperature_c": 223761,  # Temp Celsius
    "temperature_f": 223762,  # Temp Fahrenheit
    "spo2": 220277,  # Oxygen saturation
    "fio2": 223835,  # Fraction inspired oxygen
}
