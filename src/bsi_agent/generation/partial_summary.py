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

import random
import copy

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Probability of KEEPING a specific item within a category.
# 1.0 = Always keep
# 0.0 = Always remove
# 0.4 = Keep roughly 40% of the items (lines) in that list
ITEM_KEEP_PROBABILITY = {
    # Atomic Fields (Single values)
    "demographics": 1.0,      # ALWAYS KEEP: Age/Gender are fundamental context
    "admission": 1.0,         # ALWAYS KEEP: Context (ICU vs Ward) is vital
    
    # List Fields (We will filter these lists)
    "labs": 0.4,              # Keep only 40% of labs -> Forces model to ask for missing ones
    "vitals": 0.5,            # Keep 50% of vitals
    "medications": 0.5,       # Keep 50% of meds
    
    # Sensitive Fields (Must be hidden in Partial Summary to prevent easy guessing)
    "gram_stain": 0.0,        # ALWAYS HIDE in partial
    "susceptibilities": 0.0,  # ALWAYS HIDE in partial (Leak prevention)
    "organism": 0.0           # ALWAYS HIDE in partial (The Answer)
}

# Fields that are lists of dictionaries (we filter items inside)
LIST_FIELDS = ["labs", "vitals", "medications"]

# Fields that are single values (we keep or set to None/Unknown)
ATOMIC_FIELDS = {
    "age": "demographics",
    "gender": "demographics",
    "admission_type": "admission",
    "admission_location": "admission",
    "admit_time": "admission",
    "culture_time": "admission",
    "gram_stain": "gram_stain",
    "organism": "organism"
}

# ==============================================================================
# LOGIC
# ==============================================================================

def create_partial_case(raw_case: dict, seed: int = None) -> tuple[dict, list[str]]:
    rng = random.Random(seed)
    partial = copy.deepcopy(raw_case)
    hidden_hints = [] 

    # 1. Atomic Fields
    for field, category in ATOMIC_FIELDS.items():
        # CRITICAL FIX: Skip "organism" for hints. We hide it, but don't hint it.
        if field == "organism":
            # Force remove organism logic is handled in step 4 below
            continue

        prob = ITEM_KEEP_PROBABILITY.get(category, 0.5)
        if partial.get(field) is not None:
            if prob < 1.0 and rng.random() > prob:
                partial[field] = None
                # Add human-readable hint
                hidden_hints.append(field.replace("_", " ").title())

    # 2. List Fields (Labs, Vitals, Meds)
    for field in LIST_FIELDS:
        prob = ITEM_KEEP_PROBABILITY.get(field, 0.5)
        original_list = partial.get(field, [])
        if not original_list: continue
            
        new_list = []
        for item in original_list:
            name = item.get('lab_name') or item.get('vital_name') or item.get('drug') or "Unknown Item"
            if rng.random() < prob:
                new_list.append(item)
            else:
                hidden_hints.append(str(name))
        partial[field] = new_list

    # 3. Susceptibilities
    if raw_case.get('susceptibilities'):
        partial["susceptibilities"] = {}
        # Hint is allowed here if you want them to ask about resistance patterns, 
        # BUT usually better to hide to force clinical reasoning. 
        # Uncomment next line if you want to allow asking for resistance:
        hidden_hints.append("Antibiotic Susceptibility Profile")
    
    if raw_case.get('gram_stain'):
        partial["gram_stain"] = None
        # Hint is allowed here if you want them to ask about gram stain results, 
        # BUT usually better to hide to force clinical reasoning. 
        # Uncomment next line if you want to allow asking for gram stain:
        hidden_hints.append("Gram Stain Result")

    # 4. Organism (ALWAYS HIDE, NEVER HINT)
    # This ensures Model A doesn't see it in the partial text,
    # and Model B doesn't see it in the hints.
    partial["organism"] = None

    return partial, hidden_hints


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
