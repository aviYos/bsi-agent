"""
Answer Generator (Model A)

Generates answers to diagnostic questions using full patient data.
"""

from openai import OpenAI


ANSWER_PROMPT = """You are a clinical assistant with access to the complete patient record. Answer the following question using specific information from the patient data.

CRITICAL RULES:
- Do NOT name, identify, or suggest any specific pathogen, organism, or bacteria
- Do NOT say things like "consistent with X infection" or "suggestive of X"
- ONLY provide objective clinical findings: lab values, vital signs, observations
- Let the clinician draw their own diagnostic conclusions

Complete Patient Record:
{full_summary}

Question: {question}

Provide a focused answer with ONLY objective clinical details (no pathogen names or suggestions):"""


class AnswerGenerator:
    """Generate answers using GPT-4o (Model A)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, full_summary: str, question: str, temperature: float = 0.3) -> str:
        """
        Generate an answer to a diagnostic question.

        Args:
            full_summary: The complete medical summary
            question: The diagnostic question to answer
            temperature: Model temperature (lower = more deterministic)

        Returns:
            An answer to the question
        """
        prompt = ANSWER_PROMPT.format(
            full_summary=full_summary,
            question=question
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=300,
        )

        return response.choices[0].message.content.strip()
