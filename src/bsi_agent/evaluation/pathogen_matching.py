"""Pathogen name matching utilities."""

# Pathogen name aliases for matching
PATHOGEN_ALIASES = {
    "ESCHERICHIA COLI": ["E. COLI", "E COLI", "ESCHERICHIA"],
    "STAPHYLOCOCCUS AUREUS": ["S. AUREUS", "S AUREUS", "STAPH AUREUS", "STAPH AUREUS COAG +"],
    "KLEBSIELLA PNEUMONIAE": ["K. PNEUMONIAE", "K PNEUMONIAE", "KLEBSIELLA"],
    "KLEBSIELLA OXYTOCA": ["K. OXYTOCA", "K OXYTOCA"],
    "PSEUDOMONAS AERUGINOSA": ["P. AERUGINOSA", "P AERUGINOSA", "PSEUDOMONAS"],
    "ENTEROCOCCUS FAECALIS": ["E. FAECALIS", "E FAECALIS"],
    "ENTEROCOCCUS FAECIUM": ["E. FAECIUM", "E FAECIUM"],
    "ENTEROCOCCUS": ["ENTEROCOCCUS SPECIES", "ENTEROCOCCUS SPP"],
    "STAPHYLOCOCCUS EPIDERMIDIS": ["S. EPIDERMIDIS", "STAPH EPIDERMIDIS", "COAGULASE NEGATIVE STAPHYLOCOCCI", "CONS"],
    "STAPHYLOCOCCUS HOMINIS": ["S. HOMINIS", "STAPH HOMINIS"],
    "STAPHYLOCOCCUS, COAGULASE NEGATIVE": ["COAGULASE NEGATIVE STAPHYLOCOCCI", "CONS", "STAPHYLOCOCCUS EPIDERMIDIS"],
    "SERRATIA MARCESCENS": ["S. MARCESCENS", "SERRATIA"],
    "PROTEUS MIRABILIS": ["P. MIRABILIS", "PROTEUS"],
    "ENTEROBACTER CLOACAE": ["E. CLOACAE", "ENTEROBACTER"],
    "CANDIDA ALBICANS": ["C. ALBICANS", "CANDIDA"],
    "CANDIDA GLABRATA": ["C. GLABRATA"],
    "STREPTOCOCCUS": ["STREP", "STREPTOCOCCUS SPECIES"],
    "ACINETOBACTER BAUMANNII": ["A. BAUMANNII", "ACINETOBACTER"],
}


def normalize_pathogen(name: str) -> str:
    """Normalize pathogen name for comparison."""
    return name.upper().strip()


def pathogen_matches(ground_truth: str, prediction: str) -> bool:
    """Check if prediction matches ground truth, accounting for aliases."""
    gt = normalize_pathogen(ground_truth)
    pred = normalize_pathogen(prediction)

    # Exact match
    if gt == pred:
        return True

    # Check if prediction contains ground truth or vice versa
    if gt in pred or pred in gt:
        return True

    # Check aliases
    for canonical, aliases in PATHOGEN_ALIASES.items():
        canonical_upper = canonical.upper()
        aliases_upper = [a.upper() for a in aliases]

        gt_matches = (gt == canonical_upper or gt in aliases_upper or
                      any(a in gt for a in [canonical_upper] + aliases_upper))
        pred_matches = (pred == canonical_upper or pred in aliases_upper or
                        any(a in pred for a in [canonical_upper] + aliases_upper))

        if gt_matches and pred_matches:
            return True

    return False


def get_pathogen_rank(predictions: list[str], ground_truth: str) -> int:
    """
    Get the rank of ground truth in predictions.

    Args:
        predictions: List of predicted pathogen names (ranked)
        ground_truth: The actual pathogen name

    Returns:
        1, 2, or 3 if found in top 3
        99 if not found
    """
    for i, pred in enumerate(predictions):
        if pathogen_matches(ground_truth, pred):
            return i + 1
    return 99


def is_correct_top3(predictions: list[str], ground_truth: str) -> bool:
    """Check if ground truth is in top 3 predictions."""
    return get_pathogen_rank(predictions, ground_truth) <= 3
