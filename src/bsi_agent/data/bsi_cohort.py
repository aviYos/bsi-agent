"""BSI (Bloodstream Infection) cohort extraction from MIMIC-IV."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .mimic_loader import MIMICLoader, LAB_ITEMIDS, VITAL_ITEMIDS


# Common contaminants to exclude
CONTAMINANT_ORGANISMS = [
    "CORYNEBACTERIUM SPECIES",
    "COAGULASE NEGATIVE STAPHYLOCOCCI",  # Often contaminant if single culture
    "MICROCOCCUS SPECIES",
    "BACILLUS SPECIES",
    "PROPIONIBACTERIUM",
    "VIRIDANS STREPTOCOCCI",  # Context-dependent
]

# True pathogens (almost never contaminants)
TRUE_PATHOGENS = [
    "STAPHYLOCOCCUS AUREUS",
    "ESCHERICHIA COLI",
    "KLEBSIELLA PNEUMONIAE",
    "KLEBSIELLA OXYTOCA",
    "PSEUDOMONAS AERUGINOSA",
    "ENTEROCOCCUS FAECALIS",
    "ENTEROCOCCUS FAECIUM",
    "STREPTOCOCCUS PNEUMONIAE",
    "CANDIDA ALBICANS",
    "CANDIDA GLABRATA",
    "ACINETOBACTER BAUMANNII",
    "ENTEROBACTER CLOACAE",
    "SERRATIA MARCESCENS",
    "PROTEUS MIRABILIS",
    "STENOTROPHOMONAS MALTOPHILIA",
]

# Antibiotic drug classes for prescription filtering
ANTIBIOTIC_KEYWORDS = [
    "VANCOMYCIN",
    "PIPERACILLIN",
    "TAZOBACTAM",
    "CEFEPIME",
    "CEFTRIAXONE",
    "CEFAZOLIN",
    "MEROPENEM",
    "IMIPENEM",
    "CIPROFLOXACIN",
    "LEVOFLOXACIN",
    "GENTAMICIN",
    "TOBRAMYCIN",
    "AMIKACIN",
    "DAPTOMYCIN",
    "LINEZOLID",
    "METRONIDAZOLE",
    "AZITHROMYCIN",
    "DOXYCYCLINE",
    "AMPICILLIN",
    "SULBACTAM",
    "FLUCONAZOLE",
    "MICAFUNGIN",
    "CASPOFUNGIN",
]


@dataclass
class BSICase:
    """Represents a single BSI case for the agent."""

    subject_id: int
    hadm_id: int
    case_id: str

    # Patient demographics
    age: int
    gender: str
    anchor_year_group: str

    # Admission info
    admission_type: str
    admission_location: str
    admit_time: pd.Timestamp
    discharge_time: Optional[pd.Timestamp]

    # BSI specifics
    culture_time: pd.Timestamp
    specimen_type: str
    organism: str  # Ground truth
    gram_stain: Optional[str]
    susceptibilities: dict[str, str]  # antibiotic -> S/I/R

    # Clinical data (DataFrames)
    labs: pd.DataFrame
    vitals: pd.DataFrame
    medications: pd.DataFrame

    # Metadata
    is_polymicrobial: bool
    other_organisms: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "subject_id": self.subject_id,
            "hadm_id": self.hadm_id,
            "case_id": self.case_id,
            "age": self.age,
            "gender": self.gender,
            "anchor_year_group": self.anchor_year_group,
            "admission_type": self.admission_type,
            "admission_location": self.admission_location,
            "admit_time": self.admit_time.isoformat() if self.admit_time else None,
            "discharge_time": self.discharge_time.isoformat() if self.discharge_time else None,
            "culture_time": self.culture_time.isoformat() if self.culture_time else None,
            "specimen_type": self.specimen_type,
            "organism": self.organism,
            "gram_stain": self.gram_stain,
            "susceptibilities": self.susceptibilities,
            "labs": self.labs.to_dict(orient="records"),
            "vitals": self.vitals.to_dict(orient="records"),
            "medications": self.medications.to_dict(orient="records"),
            "is_polymicrobial": self.is_polymicrobial,
            "other_organisms": self.other_organisms,
        }


class BSICohortExtractor:
    """Extract BSI cases from MIMIC-IV for agent training/evaluation."""

    def __init__(self, mimic_path: str | Path):
        """
        Initialize extractor.

        Args:
            mimic_path: Path to MIMIC-IV data directory
        """
        self.loader = MIMICLoader(mimic_path)
        self._micro_df: Optional[pd.DataFrame] = None
        self._patients_df: Optional[pd.DataFrame] = None
        self._admissions_df: Optional[pd.DataFrame] = None

    def _load_base_tables(self) -> None:
        """Load core tables needed for cohort extraction."""
        if self._micro_df is None:
            print("Loading microbiology events...")
            self._micro_df = self.loader.load_microbiologyevents()

        if self._patients_df is None:
            print("Loading patients...")
            self._patients_df = self.loader.load_patients()

        if self._admissions_df is None:
            print("Loading admissions...")
            self._admissions_df = self.loader.load_admissions()

    def extract_positive_blood_cultures(
        self,
        exclude_contaminants: bool = True,
        require_susceptibilities: bool = True,
    ) -> pd.DataFrame:
        """
        Extract positive blood culture events.

        Args:
            exclude_contaminants: Whether to exclude likely contaminants
            require_susceptibilities: Only include cultures with AST results

        Returns:
            DataFrame with positive blood cultures
        """
        self._load_base_tables()

        # Filter to blood cultures
        blood_specimens = ["BLOOD CULTURE", "BLOOD", "BLOOD CULTURE ( MYCO/F LYTIC BOTTLE)"]
        bc_df = self._micro_df[
            self._micro_df["spec_type_desc"].str.upper().isin(blood_specimens)
        ].copy()

        # Filter to positive cultures (organism identified)
        positive_bc = bc_df[bc_df["org_name"].notna()].copy()

        if exclude_contaminants:
            # Exclude known contaminants (simple approach)
            # More sophisticated: require 2+ bottles for CoNS
            positive_bc = positive_bc[
                ~positive_bc["org_name"].str.upper().isin(
                    [c.upper() for c in CONTAMINANT_ORGANISMS]
                )
            ]

        if require_susceptibilities:
            # Only keep cultures that have susceptibility testing
            has_suscept = positive_bc.groupby("micro_specimen_id").apply(
                lambda x: x["ab_name"].notna().any()
            )
            valid_specimens = has_suscept[has_suscept].index
            positive_bc = positive_bc[
                positive_bc["micro_specimen_id"].isin(valid_specimens)
            ]

        print(f"Found {positive_bc['micro_specimen_id'].nunique()} positive blood cultures")
        return positive_bc

    def _get_gram_stain(self, org_name: str) -> str:
        """Infer gram stain from organism name."""
        org_upper = org_name.upper()

        gram_positive = [
            "STAPHYLOCOCCUS", "STREPTOCOCCUS", "ENTEROCOCCUS",
            "CORYNEBACTERIUM", "LISTERIA", "BACILLUS", "CLOSTRIDIUM",
        ]
        gram_negative = [
            "ESCHERICHIA", "KLEBSIELLA", "PSEUDOMONAS", "ACINETOBACTER",
            "ENTEROBACTER", "SERRATIA", "PROTEUS", "SALMONELLA",
            "SHIGELLA", "CITROBACTER", "HAEMOPHILUS", "NEISSERIA",
        ]
        fungi = ["CANDIDA", "ASPERGILLUS", "CRYPTOCOCCUS"]

        for gp in gram_positive:
            if gp in org_upper:
                if "COCCI" in org_upper or "COCCUS" in org_upper:
                    return "Gram-positive cocci"
                return "Gram-positive"

        for gn in gram_negative:
            if gn in org_upper:
                return "Gram-negative rods"

        for f in fungi:
            if f in org_upper:
                return "Yeast"

        return "Unknown"

    def _get_morphology_hint(self, org_name: str) -> str:
        """Get morphology hint for Gram stain result."""
        org_upper = org_name.upper()

        if "STAPHYLOCOCCUS" in org_upper:
            return "Gram-positive cocci in clusters"
        elif "STREPTOCOCCUS" in org_upper:
            return "Gram-positive cocci in chains"
        elif "ENTEROCOCCUS" in org_upper:
            return "Gram-positive cocci in pairs/chains"
        elif any(x in org_upper for x in ["ESCHERICHIA", "KLEBSIELLA", "ENTEROBACTER"]):
            return "Gram-negative rods"
        elif "PSEUDOMONAS" in org_upper:
            return "Gram-negative rods (non-fermenter)"
        elif "CANDIDA" in org_upper:
            return "Yeast cells"
        else:
            return self._get_gram_stain(org_name)

    def _extract_susceptibilities(
        self,
        micro_df: pd.DataFrame,
        micro_specimen_id: int,
    ) -> dict[str, str]:
        """Extract antibiotic susceptibility results for a specimen."""
        specimen_data = micro_df[micro_df["micro_specimen_id"] == micro_specimen_id]

        susceptibilities = {}
        for _, row in specimen_data.iterrows():
            if pd.notna(row.get("ab_name")) and pd.notna(row.get("interpretation")):
                ab_name = row["ab_name"]
                interp = row["interpretation"]  # S, I, R, or other
                susceptibilities[ab_name] = interp

        return susceptibilities

    def extract_bsi_cases(
        self,
        max_cases: Optional[int] = None,
        hours_before_culture: int = 48,
        hours_after_culture: int = 0,
        include_derived_gram_stain: bool = False,
    ) -> list[BSICase]:
        """
        Extract complete BSI cases with all clinical context.

        Args:
            max_cases: Maximum number of cases to extract (None for all)
            hours_before_culture: Hours of data to include before culture
            hours_after_culture: Hours of data to include after culture
            include_derived_gram_stain: If True, infer Gram stain from organism name
                (label-derived; disabled by default to prevent leakage)

        Returns:
            List of BSICase objects
        """
        self._load_base_tables()

        # Get positive blood cultures
        positive_bc = self.extract_positive_blood_cultures()

        # Get unique culture events (group by specimen)
        culture_events = (
            positive_bc.groupby("micro_specimen_id")
            .agg({
                "subject_id": "first",
                "hadm_id": "first",
                "charttime": "first",
                "chartdate": "first",
                "spec_type_desc": "first",
                "org_name": "first",
            })
            .reset_index()
        )

        # Check for polymicrobial infections
        org_counts = positive_bc.groupby("micro_specimen_id")["org_name"].nunique()
        polymicrobial_specimens = set(org_counts[org_counts > 1].index)

        if max_cases:
            culture_events = culture_events.head(max_cases)

        print(f"Processing {len(culture_events)} BSI cases...")

        # Load additional data
        print("Loading lab events...")
        subject_ids = culture_events["subject_id"].unique().tolist()
        lab_itemids = list(LAB_ITEMIDS.values())
        labs_df = self.loader.load_labevents(subject_ids=subject_ids, itemids=lab_itemids)

        print("Loading prescriptions...")
        prescriptions_df = self.loader.load_prescriptions(subject_ids=subject_ids)
        # Filter to antibiotics
        antibiotic_mask = prescriptions_df["drug"].str.upper().apply(
            lambda x: any(ab in str(x) for ab in ANTIBIOTIC_KEYWORDS) if pd.notna(x) else False
        )
        antibiotics_df = prescriptions_df[antibiotic_mask]

        print("Loading vital signs (this may take a while)...")
        vital_itemids = list(VITAL_ITEMIDS.values())
        try:
            vitals_df = self.loader.load_chartevents(
                subject_ids=subject_ids,
                itemids=vital_itemids,
            )
        except FileNotFoundError:
            print("Warning: chartevents not found, vitals will be empty")
            vitals_df = pd.DataFrame()

        # Build lab item mapping
        try:
            d_labitems = self.loader.load_d_labitems()
            lab_id_to_name = dict(zip(d_labitems["itemid"], d_labitems["label"]))
        except FileNotFoundError:
            lab_id_to_name = {v: k for k, v in LAB_ITEMIDS.items()}

        # Build vital item mapping
        try:
            d_items = self.loader.load_d_items()
            vital_id_to_name = dict(zip(d_items["itemid"], d_items["label"]))
        except FileNotFoundError:
            vital_id_to_name = {v: k for k, v in VITAL_ITEMIDS.items()}

        cases = []
        for _, culture in culture_events.iterrows():
            subject_id = int(culture["subject_id"])
            hadm_id = culture["hadm_id"]
            micro_specimen_id = culture["micro_specimen_id"]

            # Get culture time
            culture_time = culture["charttime"]
            if pd.isna(culture_time):
                culture_time = pd.to_datetime(culture["chartdate"])
            if pd.isna(culture_time):
                continue

            # Time window
            time_start = culture_time - timedelta(hours=hours_before_culture)
            time_end = culture_time + timedelta(hours=hours_after_culture)

            # Get patient info
            patient = self._patients_df[
                self._patients_df["subject_id"] == subject_id
            ].iloc[0] if len(self._patients_df[self._patients_df["subject_id"] == subject_id]) > 0 else None

            if patient is None:
                continue

            # Get admission info
            if pd.notna(hadm_id):
                admission = self._admissions_df[
                    self._admissions_df["hadm_id"] == int(hadm_id)
                ]
                if len(admission) > 0:
                    admission = admission.iloc[0]
                else:
                    admission = None
            else:
                admission = None

            # Get labs in time window
            patient_labs = labs_df[
                (labs_df["subject_id"] == subject_id) &
                (labs_df["charttime"] >= time_start) &
                (labs_df["charttime"] <= time_end)
            ].copy()
            if len(patient_labs) > 0:
                patient_labs["lab_name"] = patient_labs["itemid"].map(lab_id_to_name)

            # Get vitals in time window
            if len(vitals_df) > 0:
                patient_vitals = vitals_df[
                    (vitals_df["subject_id"] == subject_id) &
                    (vitals_df["charttime"] >= time_start) &
                    (vitals_df["charttime"] <= time_end)
                ].copy()
                if len(patient_vitals) > 0:
                    patient_vitals["vital_name"] = patient_vitals["itemid"].map(vital_id_to_name)
            else:
                patient_vitals = pd.DataFrame()

            # Get antibiotics in time window
            patient_abx = antibiotics_df[
                (antibiotics_df["subject_id"] == subject_id) &
                (antibiotics_df["starttime"] >= time_start) &
                (antibiotics_df["starttime"] <= time_end)
            ].copy()

            # Get susceptibilities
            susceptibilities = self._extract_susceptibilities(
                positive_bc, micro_specimen_id
            )

            # Check polymicrobial
            is_polymicrobial = micro_specimen_id in polymicrobial_specimens
            other_organisms = []
            if is_polymicrobial:
                specimen_orgs = positive_bc[
                    positive_bc["micro_specimen_id"] == micro_specimen_id
                ]["org_name"].unique()
                other_organisms = [
                    o for o in specimen_orgs if o != culture["org_name"]
                ]

            # Build case
            case = BSICase(
                subject_id=subject_id,
                hadm_id=int(hadm_id) if pd.notna(hadm_id) else 0,
                case_id=f"BSI_{subject_id}_{micro_specimen_id}",
                age=patient.get("anchor_age", 0) if patient is not None else 0,
                gender=patient.get("gender", "Unknown") if patient is not None else "Unknown",
                anchor_year_group=patient.get("anchor_year_group", "") if patient is not None else "",
                admission_type=admission["admission_type"] if admission is not None else "Unknown",
                admission_location=admission.get("admission_location", "Unknown") if admission is not None else "Unknown",
                admit_time=admission["admittime"] if admission is not None else culture_time,
                discharge_time=admission.get("dischtime") if admission is not None else None,
                culture_time=culture_time,
                specimen_type=culture["spec_type_desc"],
                organism=culture["org_name"],
                gram_stain=(
                    self._get_morphology_hint(culture["org_name"])
                    if include_derived_gram_stain
                    else None
                ),
                susceptibilities=susceptibilities,
                labs=patient_labs,
                vitals=patient_vitals,
                medications=patient_abx,
                is_polymicrobial=is_polymicrobial,
                other_organisms=other_organisms,
            )
            cases.append(case)

        print(f"Extracted {len(cases)} complete BSI cases")
        return cases

    def save_cases(
        self,
        cases: list[BSICase],
        output_path: str | Path,
    ) -> None:
        """Save extracted cases to parquet file."""
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame-friendly format
        records = [case.to_dict() for case in cases]

        # Save as JSON lines for complex nested structure
        jsonl_path = output_path.with_suffix(".jsonl")
        with open(jsonl_path, "w") as f:
            for record in records:
                f.write(json.dumps(record, default=str) + "\n")

        print(f"Saved {len(cases)} cases to {jsonl_path}")

    def load_cases(self, input_path: str | Path) -> list[dict]:
        """Load cases from JSONL file."""
        import json

        input_path = Path(input_path)
        cases = []
        with open(input_path, "r") as f:
            for line in f:
                cases.append(json.loads(line))
        return cases
