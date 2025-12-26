"""Evaluation metrics for BSI agent."""

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.metrics import brier_score_loss


@dataclass
class EvaluationResult:
    """Results from evaluating the agent on a single case."""

    case_id: str
    ground_truth_organism: str
    predicted_organisms: list[str]  # Ordered by confidence
    predicted_confidences: list[float]
    correct_at_1: bool
    correct_at_3: bool
    correct_at_5: bool
    num_turns: int
    grounding_score: Optional[float] = None
    safety_violations: int = 0
    reasoning_quality: Optional[float] = None


@dataclass
class AggregateMetrics:
    """Aggregate metrics across multiple cases."""

    num_cases: int
    accuracy_at_1: float
    accuracy_at_3: float
    accuracy_at_5: float
    avg_num_turns: float
    brier_score: float
    avg_grounding_score: float
    total_safety_violations: int
    calibration_error: float


def normalize_organism_name(name: str) -> str:
    """Normalize organism name for comparison."""
    name = name.lower().strip()

    # Remove common suffixes/prefixes
    name = re.sub(r'\s+species$', '', name)
    name = re.sub(r'\s+sp\.?$', '', name)
    name = re.sub(r'^methicillin[- ]resistant\s+', '', name)
    name = re.sub(r'^methicillin[- ]sensitive\s+', '', name)
    name = re.sub(r'^mrsa$', 'staphylococcus aureus', name)
    name = re.sub(r'^mssa$', 'staphylococcus aureus', name)

    # Normalize whitespace
    name = ' '.join(name.split())

    return name


def organisms_match(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted organism matches ground truth.

    Handles common variations and abbreviations.
    """
    pred_norm = normalize_organism_name(predicted)
    truth_norm = normalize_organism_name(ground_truth)

    # Exact match
    if pred_norm == truth_norm:
        return True

    # Substring match (for genus-level predictions)
    pred_words = set(pred_norm.split())
    truth_words = set(truth_norm.split())

    # If genus matches, consider it correct
    if pred_words & truth_words:
        return True

    # Common abbreviations
    abbreviations = {
        "e coli": "escherichia coli",
        "s aureus": "staphylococcus aureus",
        "staph aureus": "staphylococcus aureus",
        "k pneumoniae": "klebsiella pneumoniae",
        "p aeruginosa": "pseudomonas aeruginosa",
        "e faecalis": "enterococcus faecalis",
        "e faecium": "enterococcus faecium",
        "c albicans": "candida albicans",
    }

    pred_expanded = abbreviations.get(pred_norm, pred_norm)
    if pred_expanded == truth_norm:
        return True

    truth_expanded = abbreviations.get(truth_norm, truth_norm)
    if pred_norm == truth_expanded:
        return True

    return False


def calculate_top_k_accuracy(
    predicted_organisms: list[str],
    ground_truth: str,
    k: int,
) -> bool:
    """Check if ground truth is in top-k predictions."""
    for pred in predicted_organisms[:k]:
        if organisms_match(pred, ground_truth):
            return True
    return False


def calculate_brier_score(
    predicted_probs: list[float],
    predicted_organisms: list[str],
    ground_truth: str,
) -> float:
    """
    Calculate Brier score for calibration.

    Brier score = mean((predicted_prob - actual_outcome)^2)
    Lower is better, 0 is perfect.
    """
    if not predicted_probs or not predicted_organisms:
        return 1.0  # Worst score if no predictions

    # Find probability assigned to correct answer
    correct_prob = 0.0
    for prob, org in zip(predicted_probs, predicted_organisms):
        if organisms_match(org, ground_truth):
            correct_prob = prob
            break

    # Brier score: (predicted - actual)^2
    # actual = 1 for correct, 0 for incorrect
    return (correct_prob - 1.0) ** 2


def calculate_calibration_error(
    all_predictions: list[tuple[float, bool]],
    num_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        all_predictions: List of (confidence, is_correct) tuples
        num_bins: Number of bins for calibration

    Returns:
        ECE score (lower is better)
    """
    if not all_predictions:
        return 1.0

    # Sort into bins
    bins = [[] for _ in range(num_bins)]
    for conf, correct in all_predictions:
        bin_idx = min(int(conf * num_bins), num_bins - 1)
        bins[bin_idx].append((conf, correct))

    ece = 0.0
    total = len(all_predictions)

    for bin_preds in bins:
        if not bin_preds:
            continue

        avg_conf = np.mean([p[0] for p in bin_preds])
        avg_acc = np.mean([p[1] for p in bin_preds])
        bin_weight = len(bin_preds) / total

        ece += bin_weight * abs(avg_conf - avg_acc)

    return ece


def calculate_grounding_score(
    agent_claims: list[str],
    environment_facts: list[str],
) -> float:
    """
    Calculate how well agent claims are grounded in environment data.

    Simple implementation using keyword overlap.
    More sophisticated: use embedding similarity.

    Args:
        agent_claims: Factual claims made by agent
        environment_facts: Facts provided by environment

    Returns:
        Score from 0 to 1 (1 = fully grounded)
    """
    if not agent_claims:
        return 1.0  # No claims = nothing to ground

    # Combine environment facts
    env_text = " ".join(environment_facts).lower()

    grounded_count = 0
    for claim in agent_claims:
        claim_lower = claim.lower()

        # Extract key terms from claim
        # Simple: check if important words appear in environment
        words = re.findall(r'\b\w+\b', claim_lower)
        medical_words = [w for w in words if len(w) > 3]  # Skip short words

        if not medical_words:
            grounded_count += 1
            continue

        # Check if majority of key words appear in environment
        matches = sum(1 for w in medical_words if w in env_text)
        if matches >= len(medical_words) * 0.5:
            grounded_count += 1

    return grounded_count / len(agent_claims)


def extract_claims_from_text(text: str) -> list[str]:
    """Extract factual claims from agent text."""
    claims = []

    # Split into sentences
    sentences = re.split(r'[.!?]', text)

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Filter to likely factual claims
        claim_indicators = [
            "shows", "indicates", "is", "are", "has", "have",
            "elevated", "low", "high", "normal", "positive", "negative",
            "patient", "culture", "gram", "wbc", "temperature",
        ]

        sent_lower = sent.lower()
        if any(ind in sent_lower for ind in claim_indicators):
            claims.append(sent)

    return claims


def evaluate_single_case(
    case_id: str,
    ground_truth_organism: str,
    agent_differential: list[dict],
    dialogue_turns: int,
    agent_reasoning: str = "",
    environment_messages: list[str] = None,
    safety_violations: int = 0,
) -> EvaluationResult:
    """
    Evaluate agent performance on a single case.

    Args:
        case_id: Case identifier
        ground_truth_organism: Correct organism
        agent_differential: Agent's differential diagnosis
            [{"organism": str, "confidence": float}, ...]
        dialogue_turns: Number of turns taken
        agent_reasoning: Agent's reasoning text
        environment_messages: All environment messages
        safety_violations: Number of safety guardrail violations

    Returns:
        EvaluationResult
    """
    # Extract predictions
    predicted_organisms = [d["organism"] for d in agent_differential]
    predicted_confidences = [d.get("confidence", 0.0) / 100.0 for d in agent_differential]

    # Normalize confidences to sum to 1
    total_conf = sum(predicted_confidences)
    if total_conf > 0:
        predicted_confidences = [c / total_conf for c in predicted_confidences]

    # Calculate accuracy at different K
    correct_at_1 = calculate_top_k_accuracy(predicted_organisms, ground_truth_organism, 1)
    correct_at_3 = calculate_top_k_accuracy(predicted_organisms, ground_truth_organism, 3)
    correct_at_5 = calculate_top_k_accuracy(predicted_organisms, ground_truth_organism, 5)

    # Calculate grounding score
    grounding_score = None
    if agent_reasoning and environment_messages:
        claims = extract_claims_from_text(agent_reasoning)
        if claims:
            grounding_score = calculate_grounding_score(claims, environment_messages)

    return EvaluationResult(
        case_id=case_id,
        ground_truth_organism=ground_truth_organism,
        predicted_organisms=predicted_organisms,
        predicted_confidences=predicted_confidences,
        correct_at_1=correct_at_1,
        correct_at_3=correct_at_3,
        correct_at_5=correct_at_5,
        num_turns=dialogue_turns,
        grounding_score=grounding_score,
        safety_violations=safety_violations,
    )


def aggregate_results(results: list[EvaluationResult]) -> AggregateMetrics:
    """Aggregate evaluation results across multiple cases."""
    if not results:
        return AggregateMetrics(
            num_cases=0,
            accuracy_at_1=0.0,
            accuracy_at_3=0.0,
            accuracy_at_5=0.0,
            avg_num_turns=0.0,
            brier_score=1.0,
            avg_grounding_score=0.0,
            total_safety_violations=0,
            calibration_error=1.0,
        )

    num_cases = len(results)

    # Accuracy metrics
    accuracy_at_1 = sum(r.correct_at_1 for r in results) / num_cases
    accuracy_at_3 = sum(r.correct_at_3 for r in results) / num_cases
    accuracy_at_5 = sum(r.correct_at_5 for r in results) / num_cases

    # Average turns
    avg_num_turns = sum(r.num_turns for r in results) / num_cases

    # Brier score
    brier_scores = []
    calibration_data = []

    for r in results:
        if r.predicted_confidences and r.predicted_organisms:
            bs = calculate_brier_score(
                r.predicted_confidences,
                r.predicted_organisms,
                r.ground_truth_organism,
            )
            brier_scores.append(bs)

            # For calibration: top prediction confidence and correctness
            top_conf = r.predicted_confidences[0] if r.predicted_confidences else 0.0
            calibration_data.append((top_conf, r.correct_at_1))

    brier_score = np.mean(brier_scores) if brier_scores else 1.0

    # Calibration error
    calibration_error = calculate_calibration_error(calibration_data)

    # Grounding score
    grounding_scores = [r.grounding_score for r in results if r.grounding_score is not None]
    avg_grounding_score = np.mean(grounding_scores) if grounding_scores else 0.0

    # Safety violations
    total_safety_violations = sum(r.safety_violations for r in results)

    return AggregateMetrics(
        num_cases=num_cases,
        accuracy_at_1=accuracy_at_1,
        accuracy_at_3=accuracy_at_3,
        accuracy_at_5=accuracy_at_5,
        avg_num_turns=avg_num_turns,
        brier_score=brier_score,
        avg_grounding_score=avg_grounding_score,
        total_safety_violations=total_safety_violations,
        calibration_error=calibration_error,
    )


def format_metrics_report(metrics: AggregateMetrics) -> str:
    """Format metrics as a readable report."""
    lines = [
        "=" * 50,
        "BSI Agent Evaluation Report",
        "=" * 50,
        "",
        f"Number of cases: {metrics.num_cases}",
        "",
        "Accuracy:",
        f"  Top-1: {metrics.accuracy_at_1:.1%}",
        f"  Top-3: {metrics.accuracy_at_3:.1%}",
        f"  Top-5: {metrics.accuracy_at_5:.1%}",
        "",
        "Calibration:",
        f"  Brier Score: {metrics.brier_score:.4f} (lower is better)",
        f"  ECE: {metrics.calibration_error:.4f} (lower is better)",
        "",
        "Efficiency:",
        f"  Avg turns: {metrics.avg_num_turns:.1f}",
        "",
        "Grounding:",
        f"  Avg score: {metrics.avg_grounding_score:.1%}",
        "",
        "Safety:",
        f"  Total violations: {metrics.total_safety_violations}",
        "",
        "=" * 50,
    ]
    return "\n".join(lines)
