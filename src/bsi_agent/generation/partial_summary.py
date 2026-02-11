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

def create_partial_case(
    raw_case: dict,
    seed: int = None,
    keep_probs: dict = None,
) -> tuple[dict, list[str]]:
    """
    Create a partial case by randomly filtering ITEMS inside categories.
    
    Instead of removing 'labs' entirely, we remove ~60% of the specific lab rows.
    This creates a 'Swiss Cheese' effect where specific data points are missing.
    """
    if keep_probs is None:
        keep_probs = ITEM_KEEP_PROBABILITY

    rng = random.Random(seed)
    
    # Deep copy to ensure we don't modify the raw_case by accident
    partial = copy.deepcopy(raw_case)
    
    # Track what we touched (for logging/debugging purposes)
    hidden_meta = []

    # --- 1. Handle Atomic Fields (Single Values) ---
    for field, category in ATOMIC_FIELDS.items():
        prob = keep_probs.get(category, 0.5)
        
        # Logic: If random > prob, we hide it.
        # BUT: For 0.0 (like gram_stain), we force hide. 
        # For 1.0 (demographics), we force keep.
        if prob < 1.0 and rng.random() > prob:
            partial[field] = None # or "Unknown" depending on your preference
            hidden_meta.append(field)

    # --- 2. Handle List Fields (Labs, Vitals, Meds) ---
    for field in LIST_FIELDS:
        category = field # keys match categories in this simple setup
        prob = keep_probs.get(category, 0.5)
        
        original_list = partial.get(field, [])
        if not original_list:
            continue
            
        # Filter the list
        # We keep an item if random() < prob
        new_list = [item for item in original_list if rng.random() < prob]
        
        partial[field] = new_list
        
        # Log if we removed things
        if len(new_list) < len(original_list):
            hidden_meta.append(f"partial_{field}")

    # --- 3. Handle Susceptibilities (CRITICAL LEAK PREVENTION) ---
    # format_case_data prints 'susceptibilities' if they exist.
    # We MUST remove them from the partial case dict, otherwise Model A
    # (acting as the summarizer) might see them, or they might leak into 
    # the partial text if you use a direct dump.
    if keep_probs.get("susceptibilities", 0.0) == 0.0:
        partial["susceptibilities"] = {}
        hidden_meta.append("susceptibilities")

    # --- 4. Handle Organism (The Truth) ---
    if keep_probs.get("organism", 0.0) == 0.0:
        partial["organism"] = None
    print(f"Created partial case with hidden fields: {hidden_meta}")
    print(partial)
    return partial, hidden_meta


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
