"""Pathogen name matching utilities."""

# Pathogen name aliases for matching.
# Each entry maps a canonical name to a list of known aliases.
# A name can appear as an alias in multiple groups (e.g., S. epidermidis
# is both its own canonical AND an alias of CoNS).
PATHOGEN_ALIASES = {
    "ESCHERICHIA COLI": ["E. COLI", "E COLI"],
    "STAPHYLOCOCCUS AUREUS": ["S. AUREUS", "S AUREUS", "STAPH AUREUS", "STAPH AUREUS COAG +"],
    "KLEBSIELLA PNEUMONIAE": ["K. PNEUMONIAE", "K PNEUMONIAE"],
    "KLEBSIELLA OXYTOCA": ["K. OXYTOCA", "K OXYTOCA"],
    "PSEUDOMONAS AERUGINOSA": ["P. AERUGINOSA", "P AERUGINOSA"],
    "ENTEROCOCCUS FAECALIS": ["E. FAECALIS", "E FAECALIS"],
    "ENTEROCOCCUS FAECIUM": ["E. FAECIUM", "E FAECIUM"],
    "STAPHYLOCOCCUS EPIDERMIDIS": ["S. EPIDERMIDIS", "STAPH EPIDERMIDIS"],
    "STAPHYLOCOCCUS HOMINIS": ["S. HOMINIS", "STAPH HOMINIS"],
    "STAPHYLOCOCCUS HAEMOLYTICUS": ["S. HAEMOLYTICUS", "STAPH HAEMOLYTICUS"],
    "STAPHYLOCOCCUS, COAGULASE NEGATIVE": [
        "COAGULASE NEGATIVE STAPHYLOCOCCI", "COAGULASE-NEGATIVE STAPHYLOCOCCI",
        "CONS", "STAPHYLOCOCCUS EPIDERMIDIS", "S. EPIDERMIDIS", "STAPH EPIDERMIDIS",
    ],
    "SERRATIA MARCESCENS": ["S. MARCESCENS"],
    "PROTEUS MIRABILIS": ["P. MIRABILIS"],
    "ENTEROBACTER CLOACAE": ["E. CLOACAE", "ENTEROBACTER CLOACAE COMPLEX"],
    "STREPTOCOCCUS AGALACTIAE": [
        "S. AGALACTIAE", "GROUP B STREPTOCOCCUS", "GROUP B STREP", "GBS",
        "BETA STREPTOCOCCUS GROUP B", "BETA STREP GROUP B",
    ],
    "STREPTOCOCCUS PYOGENES": ["S. PYOGENES", "GROUP A STREPTOCOCCUS", "GROUP A STREP", "GAS"],
    "STREPTOCOCCUS PNEUMONIAE": ["S. PNEUMONIAE", "PNEUMOCOCCUS"],
    "STREPTOCOCCUS ANGINOSUS": [
        "S. ANGINOSUS", "STREPTOCOCCUS ANGINOSUS (MILLERI) GROUP",
        "MILLERI GROUP", "STREPTOCOCCUS MILLERI",
    ],
    "STREPTOCOCCUS SANGUINIS": ["S. SANGUINIS", "STREP SANGUINIS"],
    "MORGANELLA MORGANII": ["M. MORGANII"],
    "STENOTROPHOMONAS MALTOPHILIA": ["S. MALTOPHILIA"],
    "AEROCOCCUS VIRIDANS": ["A. VIRIDANS"],
    "CANDIDA ALBICANS": ["C. ALBICANS"],
    "CANDIDA GLABRATA": ["C. GLABRATA"],
    "ACINETOBACTER BAUMANNII": ["A. BAUMANNII"],
}


def normalize_pathogen(name: str) -> str:
    """Normalize pathogen name for comparison."""
    return name.upper().strip()


def _resolve_all_canonicals(name: str) -> set[str]:
    """Find all canonical groups this name belongs to (exact match only).

    A name can belong to multiple groups. For example, "STAPHYLOCOCCUS
    EPIDERMIDIS" is both its own canonical AND an alias of CoNS.
    """
    name_upper = name.upper().strip()
    result = set()
    for canonical, aliases in PATHOGEN_ALIASES.items():
        canonical_upper = canonical.upper()
        aliases_upper = {a.upper() for a in aliases}
        if name_upper == canonical_upper or name_upper in aliases_upper:
            result.add(canonical_upper)
    return result


def pathogen_matches(ground_truth: str, prediction: str) -> bool:
    """Check if prediction matches ground truth, accounting for aliases.

    Matching strategy (in order):
    1. Case-insensitive exact match
    2. Both names resolve to at least one common canonical group
    3. Substring match — only when the shorter name is >= 2 words,
       to avoid genus-only false matches (e.g. "Streptococcus" matching
       across different Streptococcus species).
    """
    gt = normalize_pathogen(ground_truth)
    pred = normalize_pathogen(prediction)

    # 1. Exact match
    if gt == pred:
        return True

    # 2. Alias resolution (exact match into alias table)
    gt_groups = _resolve_all_canonicals(gt)
    pred_groups = _resolve_all_canonicals(pred)

    if gt_groups and pred_groups and (gt_groups & pred_groups):
        return True

    # 3. Substring fallback for format variations
    #    e.g. "Escherichia coli (ESBL-producing)" contains "ESCHERICHIA COLI"
    #    Require the shorter string to be >= 2 words to prevent genus-only matching.
    shorter, longer = (gt, pred) if len(gt) <= len(pred) else (pred, gt)
    if shorter in longer and len(shorter.split()) >= 2:
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
