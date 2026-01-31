"""
Step 1.4: Evaluate Model C's Classification Accuracy

Computes Top-3 accuracy with proper pathogen name matching.

Usage:
    python scripts/step4_evaluate_classifier.py
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter


# Pathogen name mappings for matching
PATHOGEN_ALIASES = {
    "ESCHERICHIA COLI": ["E. COLI", "E COLI", "ESCHERICHIA"],
    "STAPHYLOCOCCUS AUREUS": ["S. AUREUS", "S AUREUS", "STAPH AUREUS", "STAPH AUREUS COAG +"],
    "KLEBSIELLA PNEUMONIAE": ["K. PNEUMONIAE", "K PNEUMONIAE", "KLEBSIELLA"],
    "KLEBSIELLA OXYTOCA": ["K. OXYTOCA", "K OXYTOCA"],
    "PSEUDOMONAS AERUGINOSA": ["P. AERUGINOSA", "P AERUGINOSA", "PSEUDOMONAS"],
    "ENTEROCOCCUS FAECALIS": ["E. FAECALIS", "E FAECALIS"],
    "ENTEROCOCCUS FAECIUM": ["E. FAECIUM", "E FAECIUM"],
    "ENTEROCOCCUS": ["ENTEROCOCCUS SPECIES", "ENTEROCOCCUS SPP"],
    "STAPHYLOCOCCUS EPIDERMIDIS": ["S. EPIDERMIDIS", "STAPH EPIDERMIDIS", "COAGULASE NEGATIVE STAPHYLOCOCCI", "COAGULASE-NEGATIVE STAPHYLOCOCCI", "CONS"],
    "STAPHYLOCOCCUS HOMINIS": ["S. HOMINIS", "STAPH HOMINIS"],
    "STAPHYLOCOCCUS, COAGULASE NEGATIVE": ["COAGULASE NEGATIVE STAPHYLOCOCCI", "COAGULASE-NEGATIVE STAPHYLOCOCCI", "CONS", "STAPHYLOCOCCUS EPIDERMIDIS"],
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

        # If ground truth matches canonical or any alias
        gt_matches = (gt == canonical_upper or gt in aliases_upper or
                      any(a in gt for a in [canonical_upper] + aliases_upper))

        # If prediction matches canonical or any alias
        pred_matches = (pred == canonical_upper or pred in aliases_upper or
                        any(a in pred for a in [canonical_upper] + aliases_upper))

        if gt_matches and pred_matches:
            return True

    return False


def evaluate_top3(results: list[dict]) -> dict:
    """Evaluate top-3 accuracy."""
    correct = 0
    incorrect_cases = []

    for r in results:
        gt = r["ground_truth"]
        preds = r["predictions"]

        is_correct = any(pathogen_matches(gt, p) for p in preds)

        if is_correct:
            correct += 1
        else:
            incorrect_cases.append({
                "case_id": r["case_id"],
                "ground_truth": gt,
                "predictions": preds,
            })

    accuracy = correct / len(results) if results else 0

    return {
        "total": len(results),
        "correct": correct,
        "accuracy": accuracy,
        "incorrect_cases": incorrect_cases,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier results")
    parser.add_argument(
        "--results_path",
        type=str,
        default="data/processed/classification_results.jsonl",
        help="Path to classification results",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/validation_report.json",
        help="Output path for report",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    results_path = project_root / args.results_path
    output_path = project_root / args.output_path

    print(f"Loading results from: {results_path}")

    # Load results
    results = []
    with open(results_path, "r") as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Evaluating {len(results)} cases")
    print("=" * 50)

    # Evaluate
    eval_result = evaluate_top3(results)

    # Print results
    print(f"\n{'='*50}")
    print(f"TOP-3 ACCURACY: {eval_result['correct']}/{eval_result['total']} = {100*eval_result['accuracy']:.1f}%")
    print(f"{'='*50}")

    if eval_result['accuracy'] >= 0.70:
        print("\n[PASS] Target accuracy (70%) achieved!")
    else:
        print(f"\n[FAIL] Below target accuracy (70%). Gap: {70 - 100*eval_result['accuracy']:.1f}%")

    # Show pathogen-level breakdown
    print("\n" + "-" * 50)
    print("Breakdown by pathogen:")

    # Count by ground truth pathogen
    gt_counts = Counter(r["ground_truth"] for r in results)
    gt_correct = Counter()

    for r in results:
        gt = r["ground_truth"]
        if any(pathogen_matches(gt, p) for p in r["predictions"]):
            gt_correct[gt] += 1

    for pathogen, total in gt_counts.most_common():
        correct = gt_correct.get(pathogen, 0)
        acc = 100 * correct / total if total > 0 else 0
        status = "OK" if acc >= 70 else "LOW"
        print(f"  [{status}] {pathogen}: {correct}/{total} ({acc:.0f}%)")

    # Show some incorrect cases
    if eval_result['incorrect_cases']:
        print("\n" + "-" * 50)
        print(f"Sample incorrect cases ({min(5, len(eval_result['incorrect_cases']))} of {len(eval_result['incorrect_cases'])}):")
        for case in eval_result['incorrect_cases'][:5]:
            print(f"  - GT: {case['ground_truth']}")
            print(f"    Predicted: {case['predictions']}")
            print()

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "top3_accuracy": eval_result["accuracy"],
            "total_cases": eval_result["total"],
            "correct": eval_result["correct"],
            "target_met": eval_result["accuracy"] >= 0.70,
            "incorrect_cases": eval_result["incorrect_cases"],
        }, f, indent=2)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
