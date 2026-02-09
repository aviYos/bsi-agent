"""
Question Generator (Model B)

Generates diagnostic questions based on partial medical summaries.
"""

from openai import OpenAI


QUESTION_PROMPT = """You are an infectious disease specialist reviewing a partial patient summary for a patient with a suspected bloodstream infection. Some clinical information is missing from this summary.

Your task is to ask ONE specific question about the most important missing information that would help you identify the causative pathogen.

Focus on asking about:
- Laboratory values (CBC, metabolic panel, inflammatory markers)
- Vital signs (temperature, heart rate, blood pressure)
- Gram stain morphology (e.g., gram-positive cocci in clusters)
- Current antibiotics or recent antibiotic exposure
- Source of infection (urinary, respiratory, line-related, etc.)
- Antibiotic susceptibility patterns

IMPORTANT - Do NOT ask about any of the following:
- The specific organism or pathogen that grew in the culture
- Direct culture results or culture identification
- "What bacteria was identified?" or similar direct-answer questions
- The final microbiological diagnosis
Your question should seek indirect clinical clues, not the answer itself.

Partial Summary:
{partial_summary}

Ask a single, specific clinical question that would help identify the pathogen (just the question, no explanation):"""


class QuestionGenerator:
    """Generate diagnostic questions using GPT-4o (Model B)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, partial_summary: str, temperature: float = 0.7) -> str:
        """
        Generate a diagnostic question from a partial summary.

        Args:
            partial_summary: The partial medical summary
            temperature: Model temperature (higher = more diverse)

        Returns:
            A diagnostic question
        """
        prompt = QUESTION_PROMPT.format(partial_summary=partial_summary)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150,
        )

        question = response.choices[0].message.content.strip()
        return question.strip('"\'')
