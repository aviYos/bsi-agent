"""EHR Environment that simulates patient data access for the agent."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import pandas as pd


class DataAvailability(Enum):
    """When different data types become available."""
    IMMEDIATE = 0  # Available at presentation
    HOURS_6 = 6    # Available after 6 hours
    HOURS_12 = 12  # Gram stain typically
    HOURS_24 = 24
    HOURS_48 = 48  # Final culture results


@dataclass
class EnvironmentState:
    """Tracks what information has been revealed to the agent."""
    current_hour: int = 0
    revealed_demographics: bool = False
    revealed_admission: bool = False
    revealed_vitals: bool = False
    revealed_labs: bool = False
    revealed_medications: bool = False
    revealed_gram_stain: bool = False
    revealed_culture_result: bool = False
    revealed_susceptibilities: bool = False
    questions_asked: list[str] = field(default_factory=list)
    turn_count: int = 0


class EHREnvironment:
    """
    Simulates an EHR environment that reveals patient data progressively.

    The environment manages what information is available at different
    time points and formats data as natural language for the agent.
    """

    def __init__(
        self,
        case: dict,
        gram_stain_hour: int = 12,
        culture_result_hour: int = 48,
    ):
        """
        Initialize environment with a BSI case.

        Args:
            case: BSI case dictionary (from BSICohortExtractor)
            gram_stain_hour: Hours after culture when Gram stain available
            culture_result_hour: Hours after culture when final result available
        """
        self.case = case
        self.gram_stain_hour = gram_stain_hour
        self.culture_result_hour = culture_result_hour
        self.state = EnvironmentState()

        # Parse nested data
        self.labs_df = pd.DataFrame(case.get("labs", []))
        self.vitals_df = pd.DataFrame(case.get("vitals", []))
        self.medications_df = pd.DataFrame(case.get("medications", []))

        # Convert timestamps if needed
        for df in [self.labs_df, self.vitals_df, self.medications_df]:
            if len(df) > 0 and "charttime" in df.columns:
                df["charttime"] = pd.to_datetime(df["charttime"])

    def get_initial_presentation(self) -> str:
        """
        Get the initial patient presentation.

        This is always available at the start of the case.
        """
        self.state.revealed_demographics = True
        self.state.revealed_admission = True
        self.state.turn_count += 1

        case = self.case

        # Build presentation text
        lines = []

        # Demographics
        age = case.get("age", "Unknown")
        gender = case.get("gender", "Unknown")
        gender_text = "male" if gender == "M" else "female" if gender == "F" else "patient"

        lines.append(f"Patient is a {age}-year-old {gender_text}.")

        # Admission info
        admission_type = case.get("admission_type", "Unknown")
        admission_location = case.get("admission_location", "Unknown")

        if admission_type and admission_type != "Unknown":
            lines.append(f"Admission type: {admission_type}.")

        if admission_location and admission_location != "Unknown":
            lines.append(f"Admitted from: {admission_location}.")

        # Add initial vitals if available
        initial_vitals = self._get_vitals_summary(hours_limit=6)
        if initial_vitals:
            lines.append(f"Initial vitals: {initial_vitals}")

        # Add initial labs if available
        initial_labs = self._get_labs_summary(hours_limit=6)
        if initial_labs:
            lines.append(f"Initial labs: {initial_labs}")

        # Mention blood culture drawn
        lines.append("Blood cultures were drawn on presentation.")

        return " ".join(lines)

    def _get_vitals_summary(
        self,
        hours_limit: Optional[int] = None,
    ) -> str:
        """Generate natural language summary of vital signs."""
        if len(self.vitals_df) == 0:
            return ""

        df = self.vitals_df.copy()

        # Filter by time if needed
        if hours_limit and "charttime" in df.columns:
            culture_time = pd.to_datetime(self.case.get("culture_time"))
            if culture_time:
                cutoff = culture_time - timedelta(hours=48) + timedelta(hours=hours_limit)
                df = df[df["charttime"] <= cutoff]

        if len(df) == 0:
            return ""

        summaries = []

        # Group by vital type and summarize
        vital_mapping = {
            "heart_rate": ("HR", "bpm"),
            "sbp": ("SBP", "mmHg"),
            "dbp": ("DBP", "mmHg"),
            "temperature_c": ("Temp", "°C"),
            "temperature_f": ("Temp", "°F"),
            "respiratory_rate": ("RR", "/min"),
            "spo2": ("SpO2", "%"),
        }

        for vital_col, (label, unit) in vital_mapping.items():
            if "vital_name" in df.columns:
                vital_data = df[df["vital_name"].str.lower().str.contains(vital_col.replace("_", " "), na=False)]
            else:
                continue

            if len(vital_data) > 0 and "valuenum" in vital_data.columns:
                values = vital_data["valuenum"].dropna()
                if len(values) > 0:
                    latest = values.iloc[-1]
                    if vital_col == "temperature_f":
                        # Convert to Celsius
                        latest = (latest - 32) * 5/9
                        label = "Temp"
                        unit = "°C"
                    summaries.append(f"{label} {latest:.1f}{unit}")

        return ", ".join(summaries) if summaries else ""

    def _get_labs_summary(
        self,
        hours_limit: Optional[int] = None,
        focus_on_abnormal: bool = True,
    ) -> str:
        """Generate natural language summary of laboratory results."""
        if len(self.labs_df) == 0:
            return ""

        df = self.labs_df.copy()

        # Filter by time if needed
        if hours_limit and "charttime" in df.columns:
            culture_time = pd.to_datetime(self.case.get("culture_time"))
            if culture_time:
                cutoff = culture_time - timedelta(hours=48) + timedelta(hours=hours_limit)
                df = df[df["charttime"] <= cutoff]

        if len(df) == 0:
            return ""

        summaries = []

        # Normal ranges for interpretation
        normal_ranges = {
            "wbc": (4.5, 11.0, "K/uL"),
            "white blood cells": (4.5, 11.0, "K/uL"),
            "hemoglobin": (12.0, 17.5, "g/dL"),
            "platelets": (150, 400, "K/uL"),
            "creatinine": (0.6, 1.2, "mg/dL"),
            "bun": (7, 20, "mg/dL"),
            "sodium": (136, 145, "mEq/L"),
            "potassium": (3.5, 5.0, "mEq/L"),
            "lactate": (0.5, 2.0, "mmol/L"),
            "glucose": (70, 100, "mg/dL"),
            "bilirubin": (0.1, 1.2, "mg/dL"),
            "procalcitonin": (0, 0.5, "ng/mL"),
        }

        # Get latest value for each lab
        if "lab_name" not in df.columns:
            return ""

        for lab_name in df["lab_name"].dropna().unique():
            lab_data = df[df["lab_name"] == lab_name]
            if len(lab_data) == 0 or "valuenum" not in lab_data.columns:
                continue

            values = lab_data["valuenum"].dropna()
            if len(values) == 0:
                continue

            latest = values.iloc[-1]
            lab_lower = lab_name.lower()

            # Find matching normal range
            interpretation = ""
            for key, (low, high, unit) in normal_ranges.items():
                if key in lab_lower:
                    if latest < low:
                        interpretation = " (low)"
                    elif latest > high:
                        interpretation = " (high)"
                    break

            # Only include if abnormal or important
            important_labs = ["wbc", "lactate", "creatinine", "procalcitonin", "platelet"]
            is_important = any(imp in lab_lower for imp in important_labs)

            if focus_on_abnormal:
                if interpretation or is_important:
                    summaries.append(f"{lab_name}: {latest:.1f}{interpretation}")
            else:
                summaries.append(f"{lab_name}: {latest:.1f}")

        return ", ".join(summaries[:8]) if summaries else ""  # Limit to 8 labs

    def _get_medications_summary(self) -> str:
        """Generate summary of antibiotic medications."""
        if len(self.medications_df) == 0:
            return "No antibiotics documented."

        drugs = self.medications_df["drug"].dropna().unique()
        if len(drugs) == 0:
            return "No antibiotics documented."

        # Simplify drug names
        simplified = []
        for drug in drugs[:5]:  # Limit to 5
            # Take first word or common name
            name = drug.split()[0] if isinstance(drug, str) else str(drug)
            simplified.append(name.title())

        return f"Antibiotics started: {', '.join(simplified)}."

    def process_query(self, query: str) -> str:
        """
        Process an agent query and return appropriate information.

        Args:
            query: Natural language query from the agent

        Returns:
            Response with requested information (or indication it's not available)
        """
        self.state.turn_count += 1
        self.state.questions_asked.append(query)

        query_lower = query.lower()

        # Check what type of information is being requested
        if any(word in query_lower for word in ["vital", "blood pressure", "temperature", "heart rate", "bp", "temp"]):
            return self._respond_vitals()

        elif any(word in query_lower for word in ["lab", "wbc", "white blood", "creatinine", "lactate", "blood work"]):
            return self._respond_labs()

        elif any(word in query_lower for word in ["antibiotic", "medication", "drug", "treatment", "therapy"]):
            return self._respond_medications()

        elif any(word in query_lower for word in ["gram stain", "gram-stain", "preliminary", "stain"]):
            return self._respond_gram_stain()

        elif any(word in query_lower for word in ["culture", "organism", "pathogen", "result", "final", "susceptib", "sensitivity"]):
            return self._respond_culture()

        elif any(word in query_lower for word in ["allergy", "allergies", "allergic"]):
            return self._respond_allergies()

        elif any(word in query_lower for word in ["history", "comorbid", "medical history", "past medical"]):
            return self._respond_history()

        elif any(word in query_lower for word in ["source", "focus", "infection source", "line", "catheter", "central"]):
            return self._respond_infection_source()

        else:
            return "I can provide information about: vital signs, laboratory results, medications/antibiotics, Gram stain results, culture results, allergies, or medical history. Please specify what you'd like to know."

    def _respond_vitals(self) -> str:
        """Respond to vitals query."""
        self.state.revealed_vitals = True
        vitals = self._get_vitals_summary()
        if vitals:
            return f"Current vital signs: {vitals}"
        return "Vital signs not available in the record."

    def _respond_labs(self) -> str:
        """Respond to labs query."""
        self.state.revealed_labs = True
        labs = self._get_labs_summary(focus_on_abnormal=False)
        if labs:
            return f"Laboratory results: {labs}"
        return "Laboratory results not available."

    def _respond_medications(self) -> str:
        """Respond to medications query."""
        self.state.revealed_medications = True
        return self._get_medications_summary()

    def _respond_gram_stain(self) -> str:
        """Respond to Gram stain query."""
        if self.state.current_hour < self.gram_stain_hour:
            return f"Gram stain results not yet available. Expected in approximately {self.gram_stain_hour - self.state.current_hour} hours."

        self.state.revealed_gram_stain = True
        gram_stain = self.case.get("gram_stain", "Not performed")
        return f"Gram stain of blood culture: {gram_stain}."

    def _respond_culture(self) -> str:
        """Respond to culture result query."""
        # Check if asking about susceptibilities specifically
        if self.state.current_hour < self.culture_result_hour:
            if self.state.current_hour >= self.gram_stain_hour:
                gram = self.case.get("gram_stain", "pending")
                return f"Final culture identification pending. Gram stain shows: {gram}. Susceptibilities not yet available."
            return f"Blood cultures incubating. Results expected in approximately {self.culture_result_hour - self.state.current_hour} hours."

        self.state.revealed_culture_result = True
        self.state.revealed_susceptibilities = True

        organism = self.case.get("organism", "Unknown")
        susceptibilities = self.case.get("susceptibilities", {})

        response = f"Blood culture final result: {organism}."

        if susceptibilities:
            susc_list = []
            for ab, interp in list(susceptibilities.items())[:6]:  # Limit to 6
                susc_list.append(f"{ab}: {interp}")
            response += f" Susceptibilities: {', '.join(susc_list)}."

        return response

    def _respond_allergies(self) -> str:
        """Respond to allergy query."""
        # For now, return no known allergies (can be enhanced with real data)
        return "No known drug allergies documented."

    def _respond_history(self) -> str:
        """Respond to medical history query."""
        # Basic response based on admission info
        admission_type = self.case.get("admission_type", "Unknown")
        age = self.case.get("age", "Unknown")

        response = f"Patient is {age} years old. "
        if "EMERGENCY" in str(admission_type).upper():
            response += "Presented to the emergency department. "
        elif "ELECTIVE" in str(admission_type).upper():
            response += "Elective admission. "

        return response + "Detailed medical history not available in current data extract."

    def _respond_infection_source(self) -> str:
        """Respond to infection source query."""
        admission_location = self.case.get("admission_location", "")

        response = "Possible infection sources being investigated. "
        if "ICU" in str(admission_location).upper():
            response += "Patient has been in ICU (possible line-associated infection). "

        return response + "No definitive source identified yet."

    def advance_time(self, hours: int) -> str:
        """
        Advance simulation time.

        Args:
            hours: Hours to advance

        Returns:
            Status message about time passage
        """
        self.state.current_hour += hours
        messages = []

        if self.state.current_hour >= self.gram_stain_hour and not self.state.revealed_gram_stain:
            messages.append("Gram stain results are now available.")

        if self.state.current_hour >= self.culture_result_hour and not self.state.revealed_culture_result:
            messages.append("Final culture and susceptibility results are now available.")

        if messages:
            return f"Time advanced by {hours} hours. " + " ".join(messages)
        return f"Time advanced by {hours} hours. Current time: hour {self.state.current_hour}."

    def get_ground_truth(self) -> dict:
        """Get ground truth for evaluation (should not be revealed to agent)."""
        return {
            "organism": self.case.get("organism"),
            "susceptibilities": self.case.get("susceptibilities", {}),
            "is_polymicrobial": self.case.get("is_polymicrobial", False),
            "other_organisms": self.case.get("other_organisms", []),
        }

    def get_state_summary(self) -> dict:
        """Get summary of current environment state."""
        return {
            "current_hour": self.state.current_hour,
            "turn_count": self.state.turn_count,
            "revealed": {
                "demographics": self.state.revealed_demographics,
                "vitals": self.state.revealed_vitals,
                "labs": self.state.revealed_labs,
                "medications": self.state.revealed_medications,
                "gram_stain": self.state.revealed_gram_stain,
                "culture_result": self.state.revealed_culture_result,
            },
            "questions_asked": self.state.questions_asked,
        }
