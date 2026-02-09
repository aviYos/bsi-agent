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

# Import shared matching logic
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from bsi_agent.evaluation.pathogen_matching import pathogen_matches


def evaluate_top_k(results: list[dict], k: int) -> dict:
    """Evaluate top-k accuracy."""
    correct = 0
    incorrect_cases = []

    for r in results:
        gt = r["ground_truth"]
        preds = r["predictions"][:k]

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
    eval_top1 = evaluate_top_k(results, k=1)
    eval_top3 = evaluate_top_k(results, k=3)

    # Print results
    print(f"\n{'='*50}")
    print(f"TOP-1 ACCURACY: {eval_top1['correct']}/{eval_top1['total']} = {100*eval_top1['accuracy']:.1f}%")
    print(f"TOP-3 ACCURACY: {eval_top3['correct']}/{eval_top3['total']} = {100*eval_top3['accuracy']:.1f}%")
    print(f"{'='*50}")

    if eval_top3['accuracy'] >= 0.70:
        print("\n[PASS] Target top-3 accuracy (70%) achieved!")
    else:
        print(f"\n[FAIL] Below target top-3 accuracy (70%). Gap: {70 - 100*eval_top3['accuracy']:.1f}%")

    # Show pathogen-level breakdown
    print("\n" + "-" * 50)
    print("Breakdown by pathogen:")

    # Count by ground truth pathogen
    gt_counts = Counter(r["ground_truth"] for r in results)
    gt_correct_top1 = Counter()
    gt_correct_top3 = Counter()

    for r in results:
        gt = r["ground_truth"]
        preds = r["predictions"]
        if preds and pathogen_matches(gt, preds[0]):
            gt_correct_top1[gt] += 1
        if any(pathogen_matches(gt, p) for p in preds):
            gt_correct_top3[gt] += 1

    for pathogen, total in gt_counts.most_common():
        c1 = gt_correct_top1.get(pathogen, 0)
        c3 = gt_correct_top3.get(pathogen, 0)
        a1 = 100 * c1 / total if total > 0 else 0
        a3 = 100 * c3 / total if total > 0 else 0
        status = "OK" if a3 >= 70 else "LOW"
        print(f"  [{status}] {pathogen}: top1={c1}/{total} ({a1:.0f}%)  top3={c3}/{total} ({a3:.0f}%)")

    # Show some incorrect cases (top-3 misses)
    if eval_top3['incorrect_cases']:
        print("\n" + "-" * 50)
        print(f"Sample incorrect cases - top-3 ({min(5, len(eval_top3['incorrect_cases']))} of {len(eval_top3['incorrect_cases'])}):")
        for case in eval_top3['incorrect_cases'][:5]:
            print(f"  - GT: {case['ground_truth']}")
            print(f"    Predicted: {case['predictions']}")
            print()

    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "top1_accuracy": eval_top1["accuracy"],
            "top1_correct": eval_top1["correct"],
            "top3_accuracy": eval_top3["accuracy"],
            "top3_correct": eval_top3["correct"],
            "total_cases": eval_top3["total"],
            "target_met": eval_top3["accuracy"] >= 0.70,
            "incorrect_cases_top1": eval_top1["incorrect_cases"],
            "incorrect_cases_top3": eval_top3["incorrect_cases"],
        }, f, indent=2)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
