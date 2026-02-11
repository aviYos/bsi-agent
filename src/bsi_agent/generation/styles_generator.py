import os
import json
import random
from openai import OpenAI


STYLES_PROMPT = """
Create attributes describing how a synthetic medical summary for a patient with suspected bloodstream infection may vary.

Requirements:
- Focus only on writing style, tone, and content structure.
- Do NOT include medical facts or clinical decisions.
- Provide exactly 5 attributes.
- Each attribute should contain 3–5 distinct possible values.

Return output strictly as valid JSON in the following format:
{
    "attribute_name": ["value1", "value2", "value3"]
}
"""



class StylesGenerator:
    """Generate styles using GPT-4o (Model A)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.styles = None


    def generate_styles(self, temperature: float = 0.3) -> dict:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": STYLES_PROMPT}],
            temperature=temperature,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        print("STYLES FOR SUMMARIES:")
        print(response.choices[0].message.content)

        json_output = response.choices[0].message.content.strip()

        self.styles = json.loads(json_output)

        return self.styles


    def sample_random_style_string(self) -> str:
        """
        Randomly select one value from each style attribute
        and return a formatted string.
        """

        if not self.styles:
            raise ValueError("Styles were not generated yet. Call generate_styles() first.")

        selections = []

        for key, values in self.styles.items():
            chosen_value = random.choice(values)
            selections.append(f"{key}: {chosen_value}")

        return ", ".join(selections)

QUESTION_STYLES_PROMPT = """
Create attributes describing how a diagnostic clinical question (asked by an infectious disease specialist about a suspected bloodstream infection) may vary.

Requirements:
- Focus only on how the QUESTION is phrased.
- Do NOT include medical facts or clinical decisions.
- Do NOT include specific pathogens or diagnoses.
- Provide exactly 5 attributes.
- Each attribute must describe a dimension of question style, tone, or reasoning approach.
- Each attribute should contain 3–5 distinct possible values.

Examples of relevant variation dimensions include:
- Question structure
- Specificity level
- Clinical reasoning focus
- Urgency or tone
- Information targeting strategy

Return output strictly as valid JSON in the following format:
{
    "attribute_name": ["value1", "value2", "value3"]
}
"""


class QuestionStylesGenerator:
    """Generate styles using GPT-4o (Model A)."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.styles = None


    def generate_styles(self, temperature: float = 0.3) -> dict:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": QUESTION_STYLES_PROMPT}],
            temperature=temperature,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )

        print("STYLES FOR QUESTIONS:")
        print(response.choices[0].message.content)

        json_output = response.choices[0].message.content.strip()

        self.styles = json.loads(json_output)

        return self.styles


    def sample_random_style_string(self) -> str:
        """
        Randomly select one value from each style attribute
        and return a formatted string.
        """

        if not self.styles:
            raise ValueError("Styles were not generated yet. Call generate_styles() first.")

        selections = []

        for key, values in self.styles.items():
            chosen_value = random.choice(values)
            selections.append(f"{key}: {chosen_value}")

        return ", ".join(selections)
