"""
Summary Generator (Model A)

Generates comprehensive medical summaries from structured BSI case data.
"""

import json
from openai import OpenAI


SUMMARY_PROMPT = """You are a clinical documentation specialist. Given the following structured patient data, write a comprehensive medical summary for a patient with a suspected bloodstream infection.

IMPORTANT RULES:
- ONLY describe the information that IS present in the data below.
- Do NOT mention, note, or comment on any categories of information that are absent, missing, or unavailable. Simply omit them from your summary entirely.
- Do NOT write phrases like "no medications are specified", "gram stain was not available", "vitals are not reported", etc.
- Write as if the provided data is the complete clinical picture available at this time.
- Do NOT mention the actual pathogen/organism if shown - that is for diagnosis.

Style Instruction:
{style}

Patient Data:
{patient_data}

Write a detailed clinical summary using ONLY the information provided above and following the style instruction:"""


import pandas as pd
import math

def format_case_data(case: dict) -> str:
    """
    Format case data for the prompt.
    Extracts comprehensive details including labs (with ref ranges/flags), 
    vitals, medications, and microbiology susceptibilities.
    """
    lines = []

    # --- 1. Demographics & Admission Info ---
    lines.append("=== PATIENT DEMOGRAPHICS & ADMISSION ===")
    lines.append(f"Case ID: {case.get('case_id', 'Unknown')}")
    lines.append(f"Age: {case.get('age', 'Unknown')}")
    lines.append(f"Gender: {case.get('gender', 'Unknown')}")
    
    if case.get('admission_type'):
        lines.append(f"Admission Type: {case['admission_type']}")
    if case.get('admission_location'):
        lines.append(f"Admission Location: {case['admission_location']}")
    if case.get('admit_time'):
        lines.append(f"Admit Time: {case['admit_time']}")

    # --- 2. Microbiology (The Infection Context) ---
    lines.append("\n=== MICROBIOLOGY & CULTURES ===")
    if case.get('culture_time'):
        lines.append(f"Culture Collection Time: {case['culture_time']}")
    if case.get('specimen_type'):
        lines.append(f"Specimen Type: {case['specimen_type']}")
    
    # Gram Stain
    if case.get('gram_stain'):
        lines.append(f"Gram Stain: {case['gram_stain']}")

    # Susceptibilities (Crucial for treatment questions)
    susceptibilities = case.get('susceptibilities', {})
    if susceptibilities:
        lines.append("Antibiotic Susceptibilities:")
        # Sort for consistent reading
        for abx, sensitivity in sorted(susceptibilities.items()):
            lines.append(f"  - {abx}: {sensitivity}")

    # --- 3. Laboratory Results ---
    labs = case.get('labs', [])
    if labs:
        lines.append(f"\n=== LABORATORY RESULTS ({len(labs)} items) ===")
        # Sort labs by charttime to show progression, if needed. 
        # Assuming list order is relevant, we keep it, but limit count to avoid context window explosion.
        
        for lab in labs[:50]:  # Increased limit slightly, adjust if context is too long
            lab_name = lab.get('lab_name') or f"Item {lab.get('itemid')}"
            
            # Value handling
            val = lab.get('valuenum')
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = lab.get('value', 'N/A')
            
            # Unit handling
            unit = lab.get('valueuom')
            if unit is None or (isinstance(unit, float) and math.isnan(unit)):
                unit = ""
            
            # Time handling
            chart_time = lab.get('charttime', '')
            
            # Reference Range
            ref_low = lab.get('ref_range_lower')
            ref_high = lab.get('ref_range_upper')
            ref_str = ""
            if ref_low is not None and ref_high is not None and not (pd.isna(ref_low) or pd.isna(ref_high)):
                ref_str = f"[Ref: {ref_low}-{ref_high}]"
            
            # Flags (Abnormal)
            flag = lab.get('flag')
            flag_str = f"[{flag.upper()}]" if flag and not pd.isna(flag) else ""
            
            # Comments
            comments = lab.get('comments')
            comment_str = ""
            if comments and not pd.isna(comments):
                comment_str = f" Note: {comments}"

            # Construct the line
            # Format: - Hemoglobin: 9.9 g/dL [Ref: 12.0-16.0] [ABNORMAL] (2187-08-29 17:30:00) Note: ...
            lines.append(f"  - {lab_name}: {val} {unit} {ref_str} {flag_str} ({chart_time}){comment_str}")

    # --- 4. Vital Signs ---
    vitals = case.get('vitals', [])
    if vitals:
        lines.append(f"\n=== VITAL SIGNS ({len(vitals)} items) ===")
        for vital in vitals[:40]:
            vital_name = vital.get('vital_name') or f"Item {vital.get('itemid')}"
            
            val = vital.get('valuenum')
            if val is None or (isinstance(val, float) and math.isnan(val)):
                val = vital.get('value', 'N/A')
                
            unit = vital.get('valueuom')
            if unit is None or (isinstance(unit, float) and math.isnan(unit)):
                unit = ""
            
            chart_time = vital.get('charttime', '')
            
            lines.append(f"  - {vital_name}: {val} {unit} ({chart_time})")

    # --- 5. Medications ---
    meds = case.get('medications', [])
    if meds:
        lines.append(f"\n=== MEDICATIONS/ANTIBIOTICS ({len(meds)} items) ===")
        for med in meds[:25]:
            drug = med.get('drug', 'Unknown Drug')
            dose = med.get('dose_val_rx', '')
            unit = med.get('dose_unit_rx', '')
            route = med.get('route', '')
            
            # Start/Stop times can be useful context
            start = med.get('starttime', '')
            stop = med.get('stoptime', '')
            timing = ""
            if start:
                timing = f"(Start: {start})"
            
            lines.append(f"  - {drug} {dose} {unit} via {route} {timing}")

    return "\n".join(lines)


class SummaryGenerator:
    """Generate medical summaries using GPT-4o (Model A)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _format_case_data(self, case: dict) -> str:
        """Format case data for the prompt."""
        return format_case_data(case)

    def generate_summary(self, case: dict, style: str = "Standard clinical summary", temperature: float = 0.3) -> str:
        """
        Generate a full medical summary for a BSI case.

        Args:
            case: Dictionary with patient data
            style: Style instruction string
            temperature: Model temperature (lower = more deterministic)

        Returns:
            Generated medical summary text
        """
        patient_data = self._format_case_data(case)
        prompt = SUMMARY_PROMPT.format(patient_data=patient_data, style=style)



        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()
