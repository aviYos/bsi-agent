"""Text rendering utilities for converting structured data to natural language."""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


class ClinicalTextRenderer:
    """
    Renders clinical data as natural language text.

    Focuses on concise, clinically relevant summaries that highlight
    abnormalities and trends rather than exhaustive data dumps.
    """

    # Reference ranges for lab interpretation
    LAB_REFERENCE_RANGES = {
        "white blood cells": {"low": 4.5, "high": 11.0, "unit": "K/uL", "critical_low": 2.0, "critical_high": 30.0},
        "wbc": {"low": 4.5, "high": 11.0, "unit": "K/uL", "critical_low": 2.0, "critical_high": 30.0},
        "hemoglobin": {"low": 12.0, "high": 17.5, "unit": "g/dL", "critical_low": 7.0, "critical_high": 20.0},
        "hematocrit": {"low": 36.0, "high": 50.0, "unit": "%"},
        "platelets": {"low": 150, "high": 400, "unit": "K/uL", "critical_low": 50, "critical_high": 1000},
        "sodium": {"low": 136, "high": 145, "unit": "mEq/L", "critical_low": 120, "critical_high": 160},
        "potassium": {"low": 3.5, "high": 5.0, "unit": "mEq/L", "critical_low": 2.5, "critical_high": 6.5},
        "chloride": {"low": 98, "high": 106, "unit": "mEq/L"},
        "bicarbonate": {"low": 22, "high": 29, "unit": "mEq/L", "critical_low": 10},
        "co2": {"low": 22, "high": 29, "unit": "mEq/L"},
        "bun": {"low": 7, "high": 20, "unit": "mg/dL"},
        "blood urea nitrogen": {"low": 7, "high": 20, "unit": "mg/dL"},
        "creatinine": {"low": 0.6, "high": 1.2, "unit": "mg/dL", "critical_high": 4.0},
        "glucose": {"low": 70, "high": 100, "unit": "mg/dL", "critical_low": 40, "critical_high": 500},
        "lactate": {"low": 0.5, "high": 2.0, "unit": "mmol/L", "critical_high": 4.0},
        "bilirubin": {"low": 0.1, "high": 1.2, "unit": "mg/dL"},
        "ast": {"low": 10, "high": 40, "unit": "U/L"},
        "alt": {"low": 7, "high": 56, "unit": "U/L"},
        "alkaline phosphatase": {"low": 44, "high": 147, "unit": "U/L"},
        "albumin": {"low": 3.5, "high": 5.0, "unit": "g/dL"},
        "procalcitonin": {"low": 0, "high": 0.5, "unit": "ng/mL", "critical_high": 2.0},
        "inr": {"low": 0.9, "high": 1.1, "unit": ""},
        "pt": {"low": 11, "high": 13.5, "unit": "sec"},
        "ptt": {"low": 25, "high": 35, "unit": "sec"},
    }

    # Vital sign reference ranges
    VITAL_REFERENCE_RANGES = {
        "heart rate": {"low": 60, "high": 100, "unit": "bpm", "critical_low": 40, "critical_high": 150},
        "systolic": {"low": 90, "high": 140, "unit": "mmHg", "critical_low": 80, "critical_high": 180},
        "diastolic": {"low": 60, "high": 90, "unit": "mmHg"},
        "temperature": {"low": 36.0, "high": 37.5, "unit": "°C", "critical_high": 39.0},
        "respiratory rate": {"low": 12, "high": 20, "unit": "/min", "critical_high": 30},
        "spo2": {"low": 95, "high": 100, "unit": "%", "critical_low": 90},
        "oxygen saturation": {"low": 95, "high": 100, "unit": "%", "critical_low": 90},
    }

    @classmethod
    def interpret_value(
        cls,
        value: float,
        lab_name: str,
        reference_ranges: Optional[dict] = None,
    ) -> str:
        """
        Interpret a lab/vital value relative to reference ranges.

        Returns: Interpretation string (e.g., "elevated", "critically low")
        """
        if reference_ranges is None:
            reference_ranges = cls.LAB_REFERENCE_RANGES

        lab_lower = lab_name.lower()

        # Find matching reference range
        for key, ranges in reference_ranges.items():
            if key in lab_lower:
                low = ranges.get("low", float("-inf"))
                high = ranges.get("high", float("inf"))
                critical_low = ranges.get("critical_low", float("-inf"))
                critical_high = ranges.get("critical_high", float("inf"))

                if value <= critical_low:
                    return "critically low"
                elif value >= critical_high:
                    return "critically elevated"
                elif value < low:
                    return "low"
                elif value > high:
                    return "elevated"
                else:
                    return "normal"

        return ""

    @classmethod
    def render_lab_value(
        cls,
        lab_name: str,
        value: float,
        include_interpretation: bool = True,
    ) -> str:
        """Render a single lab value with optional interpretation."""
        interpretation = cls.interpret_value(value, lab_name, cls.LAB_REFERENCE_RANGES)

        # Get unit
        unit = ""
        for key, ranges in cls.LAB_REFERENCE_RANGES.items():
            if key in lab_name.lower():
                unit = ranges.get("unit", "")
                break

        text = f"{lab_name}: {value:.1f}"
        if unit:
            text += f" {unit}"
        if include_interpretation and interpretation and interpretation != "normal":
            text += f" ({interpretation})"

        return text

    @classmethod
    def render_labs_summary(
        cls,
        labs_df: pd.DataFrame,
        lab_name_col: str = "lab_name",
        value_col: str = "valuenum",
        time_col: str = "charttime",
        focus_abnormal: bool = True,
        max_labs: int = 10,
    ) -> str:
        """
        Render a summary of laboratory results.

        Args:
            labs_df: DataFrame with lab results
            lab_name_col: Column containing lab names
            value_col: Column containing numeric values
            time_col: Column containing timestamps
            focus_abnormal: Whether to prioritize abnormal values
            max_labs: Maximum number of labs to include

        Returns:
            Natural language summary of labs
        """
        if len(labs_df) == 0:
            return "No laboratory results available."

        # Get latest value for each lab
        latest_labs = (
            labs_df.sort_values(time_col)
            .groupby(lab_name_col)
            .last()
            .reset_index()
        )

        rendered = []
        abnormal_labs = []
        normal_labs = []

        for _, row in latest_labs.iterrows():
            lab_name = row[lab_name_col]
            value = row[value_col]

            if pd.isna(value):
                continue

            interpretation = cls.interpret_value(value, lab_name, cls.LAB_REFERENCE_RANGES)
            text = cls.render_lab_value(lab_name, value)

            if interpretation and interpretation != "normal":
                abnormal_labs.append((text, "critical" in interpretation))
            else:
                normal_labs.append(text)

        # Sort abnormal labs by severity (critical first)
        abnormal_labs.sort(key=lambda x: x[1], reverse=True)
        abnormal_texts = [t[0] for t in abnormal_labs]

        if focus_abnormal:
            rendered = abnormal_texts[:max_labs]
            remaining = max_labs - len(rendered)
            if remaining > 0:
                rendered.extend(normal_labs[:remaining])
        else:
            rendered = (abnormal_texts + normal_labs)[:max_labs]

        if not rendered:
            return "Laboratory results within normal limits."

        return "Labs: " + "; ".join(rendered) + "."

    @classmethod
    def render_vitals_summary(
        cls,
        vitals_df: pd.DataFrame,
        vital_name_col: str = "vital_name",
        value_col: str = "valuenum",
        time_col: str = "charttime",
    ) -> str:
        """Render a summary of vital signs."""
        if len(vitals_df) == 0:
            return "Vital signs not available."

        # Get latest value for each vital
        latest_vitals = (
            vitals_df.sort_values(time_col)
            .groupby(vital_name_col)
            .last()
            .reset_index()
        )

        rendered = []
        for _, row in latest_vitals.iterrows():
            vital_name = row[vital_name_col]
            value = row[value_col]

            if pd.isna(value):
                continue

            interpretation = cls.interpret_value(value, vital_name, cls.VITAL_REFERENCE_RANGES)

            # Get unit
            unit = ""
            for key, ranges in cls.VITAL_REFERENCE_RANGES.items():
                if key in vital_name.lower():
                    unit = ranges.get("unit", "")
                    break

            text = f"{vital_name}: {value:.0f}"
            if unit:
                text += f" {unit}"
            if interpretation and interpretation != "normal":
                text += f" ({interpretation})"

            rendered.append(text)

        if not rendered:
            return "Vital signs not documented."

        return "Vitals: " + ", ".join(rendered) + "."

    @classmethod
    def render_trend(
        cls,
        values: list[float],
        timestamps: list[datetime],
        lab_name: str,
    ) -> str:
        """
        Render a trend description for a series of values.

        Args:
            values: List of numeric values
            timestamps: Corresponding timestamps
            lab_name: Name of the measurement

        Returns:
            Natural language trend description
        """
        if len(values) < 2:
            return f"{lab_name}: {values[0]:.1f}" if values else ""

        first_val = values[0]
        last_val = values[-1]
        max_val = max(values)
        min_val = min(values)

        # Calculate percentage change
        pct_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0

        # Determine trend direction
        if abs(pct_change) < 10:
            trend = "stable"
        elif pct_change > 0:
            trend = "increasing" if pct_change < 50 else "rapidly increasing"
        else:
            trend = "decreasing" if pct_change > -50 else "rapidly decreasing"

        # Build description
        if trend == "stable":
            return f"{lab_name} stable around {last_val:.1f}"
        else:
            return f"{lab_name} {trend} from {first_val:.1f} to {last_val:.1f}"

    @classmethod
    def render_patient_summary(
        cls,
        age: int,
        gender: str,
        admission_type: str,
        chief_complaint: Optional[str] = None,
    ) -> str:
        """Render a patient summary sentence."""
        gender_text = "male" if gender == "M" else "female" if gender == "F" else "patient"

        text = f"Patient is a {age}-year-old {gender_text}"

        if admission_type:
            if "EMERGENCY" in admission_type.upper():
                text += " presenting to the emergency department"
            elif "URGENT" in admission_type.upper():
                text += " with urgent admission"

        if chief_complaint:
            text += f" with {chief_complaint}"

        return text + "."

    @classmethod
    def render_gram_stain(cls, organism: str) -> str:
        """
        Render Gram stain result based on organism (simulates what would be seen).

        This simulates the preliminary Gram stain before final identification.
        """
        org_upper = organism.upper()

        if "STAPHYLOCOCCUS" in org_upper:
            return "Gram-positive cocci in clusters"
        elif "STREPTOCOCCUS" in org_upper:
            return "Gram-positive cocci in chains"
        elif "ENTEROCOCCUS" in org_upper:
            return "Gram-positive cocci in pairs and short chains"
        elif any(x in org_upper for x in ["ESCHERICHIA", "KLEBSIELLA", "ENTEROBACTER", "SERRATIA", "PROTEUS"]):
            return "Gram-negative rods"
        elif "PSEUDOMONAS" in org_upper:
            return "Gram-negative rods (non-lactose fermenting on preliminary)"
        elif "ACINETOBACTER" in org_upper:
            return "Gram-negative coccobacilli"
        elif "CANDIDA" in org_upper:
            return "Yeast cells with budding"
        else:
            return "Organisms seen, identification pending"

    @classmethod
    def render_susceptibilities(
        cls,
        susceptibilities: dict[str, str],
        max_antibiotics: int = 8,
    ) -> str:
        """Render antibiotic susceptibility results."""
        if not susceptibilities:
            return "Susceptibility testing pending."

        # Prioritize common antibiotics
        priority_antibiotics = [
            "VANCOMYCIN", "OXACILLIN", "CEFAZOLIN", "CEFTRIAXONE", "CEFEPIME",
            "PIPERACILLIN/TAZOBACTAM", "MEROPENEM", "CIPROFLOXACIN", "GENTAMICIN",
            "AMPICILLIN", "PENICILLIN", "LINEZOLID", "DAPTOMYCIN",
        ]

        # Sort susceptibilities
        sorted_susc = []
        remaining = []

        for ab, result in susceptibilities.items():
            ab_upper = ab.upper()
            is_priority = any(p in ab_upper for p in priority_antibiotics)
            if is_priority:
                sorted_susc.append((ab, result))
            else:
                remaining.append((ab, result))

        sorted_susc.extend(remaining)
        sorted_susc = sorted_susc[:max_antibiotics]

        # Render
        parts = []
        for ab, result in sorted_susc:
            # Simplify result
            result_text = result[0] if result else "?"  # S, I, or R
            parts.append(f"{ab}: {result_text}")

        return "Susceptibilities: " + ", ".join(parts) + "."
