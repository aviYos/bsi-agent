"""
Question Generator (Model B)

Generates diagnostic questions based on partial medical summaries.
"""

from openai import OpenAI

QUESTION_PROMPT = """You are an infectious disease specialist reviewing a partial patient summary for a patient with a suspected bloodstream infection.

Writing Instructions:
{style_instructions}

Current Status:
The patient summary below is incomplete. Some specific laboratory values, vital signs, or history details have been redacted or are missing.

Your task is to ask ONE specific question about a missing clinical data point that would help you narrow down the specific pathogen causing the infection.

Focus on asking about:
- Missing critical laboratory values (e.g., specific missing CBC or Chemistry values)
- Missing vital signs that indicate severity (e.g., temperature curve, hypotension)
- Specific risk factors or comorbidities (e.g., indwelling devices, recent surgery)
- The source of infection (e.g., urinary, pulmonary, line-associated)

IMPORTANT - Do NOT ask about:
- The name of the organism
- Culture results or Gram stain morphology
- Antibiotic susceptibility or resistance profiles (AST)
- General, vague questions like "How is the patient?"

Partial Summary:
{partial_summary}

Ask a single, specific clinical question targeting the missing information:
"""


class QuestionGenerator:

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        partial_summary: str,
        style_instructions: str,
        temperature: float = 0.7
    ) -> str:

        prompt = QUESTION_PROMPT.format(
            partial_summary=partial_summary,
            style_instructions=style_instructions
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150,
        )

        question = response.choices[0].message.content.strip()

        return question.strip('"\'')

