"""
Partial Summary Generator

Creates partial summaries by hiding ~50% of information by category.
"""

import random
import re


# Category patterns to identify sections in summaries
CATEGORY_PATTERNS = {
    "demographics": [
        r"(\d+[-\s]year[-\s]old\s+(male|female|man|woman))",
        r"(patient is a[n]?\s+\d+)",
    ],
    "admission": [
        r"(admitted\s+on[^.]+\.)",
        r"(admission[^.]+\.)",
        r"(admitted\s+to[^.]+\.)",
    ],
    "labs": [
        r"(laboratory\s+findings[^.]+\.)",
        r"(lab\w*\s+(results?|findings?|tests?)[^.]+\.)",
        r"(white\s+blood\s+cell[^.]+\.)",
        r"(wbc[^.]+\.)",
        r"(hemoglobin[^.]+\.)",
        r"(hematocrit[^.]+\.)",
        r"(platelet[^.]+\.)",
        r"(creatinine[^.]+\.)",
        r"(bilirubin[^.]+\.)",
        r"(lactate[^.]+\.)",
        r"(alt\s|ast\s|alkaline[^.]+\.)",
        r"(liver\s+function[^.]+\.)",
        r"(renal\s+function[^.]+\.)",
        r"(electrolyte[^.]+\.)",
        r"(sodium|potassium|chloride|bicarbonate[^.]+\.)",
        r"(coagulation[^.]+\.)",
        r"(inr|pt\s|ptt[^.]+\.)",
        r"(glucose[^.]+\.)",
        r"(albumin[^.]+\.)",
    ],
    "vitals": [
        r"(vital\s+signs?[^.]+\.)",
        r"(temperature[^.]+\.)",
        r"(blood\s+pressure[^.]+\.)",
        r"(heart\s+rate[^.]+\.)",
        r"(respiratory\s+rate[^.]+\.)",
        r"(oxygen\s+saturation[^.]+\.)",
        r"(fever[^.]+\.)",
        r"(hypotension[^.]+\.)",
        r"(tachycardia[^.]+\.)",
    ],
    "medications": [
        r"(medication[^.]+\.)",
        r"(antibiotic[^.]+\.)",
        r"(started\s+on[^.]+\.)",
        r"(piperacillin[^.]+\.)",
        r"(vancomycin[^.]+\.)",
        r"(cef\w+[^.]+\.)",
        r"(meropenem[^.]+\.)",
        r"(treatment[^.]+antibiotic[^.]+\.)",
    ],
    "gram_stain": [
        r"(gram\s+stain[^.]+\.)",
        r"(gram[-\s]positive[^.]+\.)",
        r"(gram[-\s]negative[^.]+\.)",
        r"(cocci[^.]+\.)",
        r"(rods[^.]+\.)",
        r"(yeast[^.]+\.)",
    ],
}

# Probability of hiding each category
HIDE_PROBABILITY = {
    "demographics": 0.0,   # Never hide
    "admission": 0.2,      # Rarely hide
    "labs": 0.6,           # Often hide
    "vitals": 0.5,         # Sometimes hide
    "medications": 0.6,    # Often hide
    "gram_stain": 0.8,     # Usually hide
}


def _identify_sentences_by_category(summary: str) -> dict[str, list[str]]:
    """Identify which sentences belong to which category."""
    sentences = re.split(r'(?<=[.!?])\s+', summary)

    categorized = {cat: [] for cat in CATEGORY_PATTERNS.keys()}
    categorized["other"] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        found_category = False
        for category, patterns in CATEGORY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    categorized[category].append(sentence)
                    found_category = True
                    break
            if found_category:
                break

        if not found_category:
            categorized["other"].append(sentence)

    return categorized


def create_partial_summary(full_summary: str, seed: int = None) -> tuple[str, dict]:
    """
    Create a partial summary by hiding ~50% of information by category.

    Args:
        full_summary: The complete medical summary
        seed: Random seed for reproducibility

    Returns:
        tuple: (partial_summary, hidden_info)
    """
    if seed is not None:
        random.seed(seed)

    categorized = _identify_sentences_by_category(full_summary)

    visible_sentences = []
    hidden_info = {}

    for category, sentences in categorized.items():
        if category == "other":
            visible_sentences.extend(sentences)
            continue

        hide_prob = HIDE_PROBABILITY.get(category, 0.3)

        if random.random() < hide_prob and sentences:
            hidden_info[category] = sentences
        else:
            visible_sentences.extend(sentences)

    partial_summary = " ".join(visible_sentences)

    return partial_summary, hidden_info


def create_dialogue(partial_summary: str, question: str, answer: str) -> str:
    """
    Create full dialogue by combining x + q + answer.

    Args:
        partial_summary: The partial summary (x)
        question: The diagnostic question (q)
        answer: The answer from Model A

    Returns:
        Full dialogue (d)
    """
    return f"""{partial_summary}

Question: {question}

Answer: {answer}"""
