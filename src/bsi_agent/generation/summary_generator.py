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

Patient Data:
{patient_data}

Write a detailed clinical summary using ONLY the information provided above:"""


class SummaryGenerator:
    """Generate medical summaries using GPT-4o (Model A)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _format_case_data(self, case: dict) -> str:
        """Format case data for the prompt."""
        lines = []

        # Demographics
        lines.append(f"Age: {case.get('age', 'Unknown')}")
        lines.append(f"Gender: {case.get('gender', 'Unknown')}")

        # Admission
        lines.append(f"Admission Type: {case.get('admission_type', 'Unknown')}")
        lines.append(f"Admission Location: {case.get('admission_location', 'Unknown')}")
        if case.get('admit_time'):
            lines.append(f"Admit Time: {case['admit_time']}")
        if case.get('culture_time'):
            lines.append(f"Blood Culture Collected: {case['culture_time']}")

        # Labs
        labs = case.get('labs', [])
        if labs:
            lines.append("\nLaboratory Results:")
            for lab in labs[:30]:  # Limit to avoid token overflow
                lab_name = lab.get('lab_name') or f"Item {lab.get('itemid')}"
                value = lab.get('valuenum', lab.get('value', 'N/A'))
                unit = lab.get('valueuom', '')
                time = lab.get('charttime', '')
                lines.append(f"  - {lab_name}: {value} {unit} ({time})")

        # Vitals
        vitals = case.get('vitals', [])
        if vitals:
            lines.append("\nVital Signs:")
            for vital in vitals[:20]:
                vital_name = vital.get('vital_name') or f"Item {vital.get('itemid')}"
                value = vital.get('valuenum', vital.get('value', 'N/A'))
                unit = vital.get('valueuom', '')
                lines.append(f"  - {vital_name}: {value} {unit}")

        # Medications
        meds = case.get('medications', [])
        if meds:
            lines.append("\nMedications (Antibiotics):")
            for med in meds[:15]:
                drug = med.get('drug', 'Unknown')
                dose = med.get('dose_val_rx', '')
                unit = med.get('dose_unit_rx', '')
                route = med.get('route', '')
                lines.append(f"  - {drug} {dose} {unit} ({route})")

        # Gram stain hint (if available)
        if case.get('gram_stain'):
            lines.append(f"\nGram Stain: {case['gram_stain']}")

        return "\n".join(lines)

    def generate_summary(self, case: dict, temperature: float = 0.3) -> str:
        """
        Generate a full medical summary for a BSI case.

        Args:
            case: Dictionary with patient data
            temperature: Model temperature (lower = more deterministic)

        Returns:
            Generated medical summary text
        """
        patient_data = self._format_case_data(case)
        prompt = SUMMARY_PROMPT.format(patient_data=patient_data)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()
