"""
Question Generator (Model B)

Generates diagnostic questions based on partial medical summaries.
"""

from openai import OpenAI

QUESTION_PROMPT = """You are an infectious disease specialist reviewing a partial patient summary.

Writing Instructions:
{style_instructions}

Current Status:
The patient summary below is incomplete.

**AVAILABLE DATA HINTS:**
The following specific data points exist in the full record but are currently hidden from you. **You MUST ask about one of these topics** to ensure the record can provide an answer:
[ {available_hints} ]

Your task is to ask ONE specific question about one of the missing items listed above that is most critical for diagnosis.

Partial Summary:
{partial_summary}

Ask a single, specific clinical question targeting the available hidden info:
"""


QUESTION_PROMPT_TRAINING = """You are an infectious disease specialist reviewing a partial patient summary for a patient with a suspected bloodstream infection.

The summary below has specific missing data points (such as specific labs, vitals, or history) that are critical for diagnosis.

Your task is to ask ONE specific clinical question about a missing value that would help identify the causative pathogen.

Focus on asking about:
- Specific missing Laboratory values (e.g., Lactate, WBC, Platelets, Creatinine)
- Missing Vital signs (e.g., Temperature, Blood Pressure)
- Specific risk factors (e.g., indwelling lines, recent surgery, immunosuppression)
- The source of infection
- Culture results or Gram stain
- Antibiotic susceptibility or resistance profiles (AST)

IMPORTANT - Do NOT ask about:
- The name of the organism or pathogen
- General/vague questions like "How is the patient?"

Partial Summary:
{partial_summary}

Ask a single, specific clinical question (just the question, no explanation):"""


class QuestionGenerator:

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        partial_summary: str,
        style: str,
        available_hints: str,
        temperature: float = 0.7
    ) -> str:

        prompt = QUESTION_PROMPT.format(
            partial_summary=partial_summary,
            style_instructions=style,
            available_hints=available_hints
        )
        print("-------------------------------------------")
        print("QUESTION GENERATION PROMPT:")
        print(available_hints)
        print("-------------------------------------------")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=150,
        )

        question = response.choices[0].message.content.strip()

        return question.strip('"\'')

