"""
Pathogen Classifier (Model C)

Classifies likely pathogens from medical summaries.
"""

import re
from openai import OpenAI


CLASSIFIER_PROMPT = """You are an infectious disease specialist. Based on the following clinical summary of a patient with a bloodstream infection, predict the most likely causative pathogen(s).

Clinical Summary:
{summary}

Provide your prediction as a ranked list of the top 10 most likely pathogens:
1. [Pathogen name]
2. [Pathogen name]
3. [Pathogen name]
4. [Pathogen name]
5. [Pathogen name]
6. [Pathogen name]
7. [Pathogen name]
8. [Pathogen name]
9. [Pathogen name]
10. [Pathogen name]

Use standard microbiological nomenclature (e.g., "Staphylococcus aureus", "Escherichia coli", "Klebsiella pneumoniae").
Only output the numbered list, nothing else."""


class PathogenClassifier:
    """Classify pathogens from medical summaries using GPT-4o (Model C)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def classify(self, summary: str, temperature: float = 0.3) -> list[str]:
        """
        Classify likely pathogens from a medical summary.

        Args:
            summary: Medical summary text
            temperature: Model temperature

        Returns:
            List of top 10 predicted pathogens (ranked)
        """
        prompt = CLASSIFIER_PROMPT.format(summary=summary)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=600,
        )

        raw_response = response.choices[0].message.content.strip()
        return self._parse_predictions(raw_response)

    def _parse_predictions(self, response: str) -> list[str]:
        """Parse the model's response to extract pathogen names."""
        predictions = []

        # Match lines like "1. Pathogen name" or "1) Pathogen name"
        pattern = r'^\d+[\.\)]\s*(.+)$'

        for line in response.split('\n'):
            line = line.strip()
            match = re.match(pattern, line)
            if match:
                pathogen = match.group(1).strip()
                # Clean up any trailing punctuation or extra text
                pathogen = re.sub(r'\s*[-–—].*$', '', pathogen)  # Remove explanations after dash
                pathogen = pathogen.strip('.,;')
                if pathogen:
                    predictions.append(pathogen)
        return predictions[:10]  # Ensure max 10
