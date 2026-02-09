"""
Partial Case Generator

Creates partial cases by hiding ~50% of clinical categories from raw case data,
then generates narrative summaries from the remaining data via Model A.
"""

import random


# Categories and their corresponding keys in the raw case dict
CATEGORY_KEYS = {
    "demographics": ["age", "gender"],
    "admission": ["admission_type", "admission_location", "admit_time",
                   "discharge_time", "culture_time"],
    "labs": ["labs"],
    "vitals": ["vitals"],
    "medications": ["medications"],
    "gram_stain": ["gram_stain"],
}

# Probability of KEEPING each category (inverse of old HIDE_PROBABILITY)
KEEP_PROBABILITY = {
    "demographics": 1.0,   # Always keep
    "admission": 0.8,      # Usually keep
    "labs": 0.4,           # Often hidden
    "vitals": 0.5,         # Sometimes hidden
    "medications": 0.4,    # Often hidden
    "gram_stain": 0.2,     # Usually hidden
}

# Default values when a category is hidden
_EMPTY_DEFAULTS = {
    "age": "Unknown",
    "gender": "Unknown",
    "admission_type": None,
    "admission_location": None,
    "admit_time": None,
    "discharge_time": None,
    "culture_time": None,
    "labs": [],
    "vitals": [],
    "medications": [],
    "gram_stain": None,
}


def create_partial_case(
    raw_case: dict,
    seed: int = None,
    keep_probs: dict = None,
) -> tuple[dict, list[str]]:
    """
    Create a partial case by randomly removing ~50% of categories.

    Args:
        raw_case: Raw case dict from bsi_cases.jsonl
        seed: Random seed for reproducibility
        keep_probs: Override keep probabilities per category

    Returns:
        tuple: (partial_case_dict, list_of_hidden_category_names)
    """
    if keep_probs is None:
        keep_probs = KEEP_PROBABILITY

    rng = random.Random(seed)

    partial = dict(raw_case)
    hidden_categories = []

    for category, keys in CATEGORY_KEYS.items():
        prob = keep_probs.get(category, 0.5)
        if rng.random() < prob:
            # KEEP this category
            continue
        else:
            # HIDE this category
            hidden_categories.append(category)
            for key in keys:
                if key in _EMPTY_DEFAULTS:
                    partial[key] = _EMPTY_DEFAULTS[key]

    return partial, hidden_categories


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
