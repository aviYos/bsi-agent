"""
Answer Generator (Model A)

Generates answers to diagnostic questions using full patient data.
"""

from openai import OpenAI


ANSWER_PROMPT = """You are a clinical assistant with access to the complete patient record. 
Your task is to answer the specific clinical question based **ONLY** on the provided data.

CRITICAL RULES:
1. **Objective Data Only**: Provide specific numbers, values, and observations (e.g., "WBC is 14.5", "Temperature 39.1°C").
2. **Handle Missing Data**: If the requested information is NOT in the patient record below, you must state: "This information is not available in the record." Do NOT guess or assume normal values.
3. **No Interpretations**: Do not say "suggestive of sepsis" or "consistent with infection." Just state the facts.
4. **Redaction Protocol**: Do NOT name the specific pathogen (bacteria/fungus) even if the data implies it.

Complete Patient Record:
{patient_data}

Question: {question}

Focused Answer (Objective facts only):"""


class AnswerGenerator:
    """Generate answers using GPT-4o (Model A)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, patient_data: str, question: str, temperature: float = 0.3) -> str:
        """
        Generate an answer to a diagnostic question.

        Args:
            patient_data: The complete medical data (raw case or summary)
            question: The diagnostic question to answer
            temperature: Model temperature (lower = more deterministic)

        Returns:
            An answer to the question
        """
        prompt = ANSWER_PROMPT.format(
            patient_data=patient_data,
            question=question
        )


        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300,
        )


        return response.choices[0].message.content.strip()
