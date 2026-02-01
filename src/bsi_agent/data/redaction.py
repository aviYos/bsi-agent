"""
Redaction utilities to remove explicit pathogen mentions from model inputs.
"""

from __future__ import annotations

import re

from bsi_agent.evaluation.pathogen_matching import PATHOGEN_ALIASES


def _expand_term_variants(term: str) -> set[str]:
    """Generate simple string variants for matching (case-insensitive later)."""
    variants = set()
    if not term:
        return variants

    term = term.strip()
    if not term:
        return variants

    variants.add(term)

    # Remove dots (e.g., "E. coli" -> "E coli")
    no_dots = term.replace(".", "")
    if no_dots and no_dots != term:
        variants.add(no_dots)

    # Replace dots with spaces (e.g., "E. coli" -> "E  coli" -> "E coli")
    spaced = term.replace(".", " ")
    spaced = " ".join(spaced.split())
    if spaced and spaced != term:
        variants.add(spaced)

    return variants


def collect_pathogen_terms(ground_truth: str) -> list[str]:
    """
    Build a list of pathogen name variants to redact based on ground truth.
    Uses alias mappings where available.
    """
    if not ground_truth:
        return []

    gt_upper = ground_truth.upper().strip()
    terms: set[str] = set()

    terms.update(_expand_term_variants(ground_truth))

    for canonical, aliases in PATHOGEN_ALIASES.items():
        canon_upper = canonical.upper()
        aliases_upper = [a.upper() for a in aliases]

        if gt_upper == canon_upper or gt_upper in aliases_upper:
            terms.update(_expand_term_variants(canonical))
            for alias in aliases:
                terms.update(_expand_term_variants(alias))
            break

    # Redact longer terms first to avoid partial overlaps.
    return sorted(terms, key=len, reverse=True)


def redact_pathogen_mentions(
    text: str,
    ground_truth: str,
    replacement: str = "[REDACTED ORGANISM]",
) -> str:
    """Redact explicit pathogen mentions from text."""
    if not text or not ground_truth:
        return text

    redacted = text
    for term in collect_pathogen_terms(ground_truth):
        if not term:
            continue
        pattern = r"(?i)(?<!\w)" + re.escape(term) + r"(?!\w)"
        redacted = re.sub(pattern, replacement, redacted)

    return redacted


def remove_gram_stain_lines(text: str) -> str:
    """Remove sentences that explicitly mention Gram stain."""
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not re.search(r"(?i)\bgram stain\b", s)]
    return " ".join(kept).strip()


def sanitize_summary(text: str, ground_truth: str) -> str:
    """
    Sanitize a summary by removing Gram stain lines and redacting pathogen mentions.
    """
    sanitized = remove_gram_stain_lines(text)
    sanitized = redact_pathogen_mentions(sanitized, ground_truth)
    return sanitized


def find_pathogen_mentions(text: str, ground_truth: str) -> list[str]:
    """Return a list of pathogen terms detected in text."""
    if not text or not ground_truth:
        return []

    found = []
    for term in collect_pathogen_terms(ground_truth):
        if not term:
            continue
        pattern = r"(?i)(?<!\w)" + re.escape(term) + r"(?!\w)"
        if re.search(pattern, text):
            found.append(term)

    return found
