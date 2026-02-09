"""
Step 1.3: Test Model C's Pathogen Classification

Tests GPT-4o's ability to classify pathogens from full medical summaries.

Usage:
    python scripts/step3_test_classifier.py
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.generation.pathogen_classifier import PathogenClassifier
from bsi_agent.evaluation import pathogen_matches


def load_summaries(path: Path) -> list[dict]:
    """Load summaries from JSONL file."""
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def save_results(results: list[dict], path: Path):
    """Save classification results to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test pathogen classifier")
    parser.add_argument(
        "--summaries_path",
        type=str,
        default="data/processed/full_summaries.jsonl",
        help="Path to summaries",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/classification_results.jsonl",
        help="Output path for results",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    summaries_path = project_root / args.summaries_path
    output_path = project_root / args.output_path
    config_path = project_root / args.config_path

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    api_key = config["dialogue_generation"]["api_key"]
    model = config["dialogue_generation"].get("model", "gpt-4o")

    print(f"Loading summaries from: {summaries_path}")
    summaries = load_summaries(summaries_path)
    print(f"Testing classifier on {len(summaries)} cases with {model}")
    print("-" * 50)

    # Initialize classifier
    classifier = PathogenClassifier(api_key=api_key, model=model)

    # Classify each summary
    results = []
    for item in tqdm(summaries, desc="Classifying"):
        try:
            predictions = classifier.classify(item["full_summary"])
            results.append({
                "case_id": item["case_id"],
                "ground_truth": item["ground_truth_pathogen"],
                "predictions": predictions,
            })
        except Exception as e:
            print(f"\nError processing {item['case_id']}: {e}")
            continue

    # Save results
    save_results(results, output_path)
    print("-" * 50)
    print(f"Saved {len(results)} results to {output_path}")

    # Quick accuracy check (alias-aware top-1 and top-3)
    top1_correct = 0
    top3_correct = 0
    for r in results:
        gt = r["ground_truth"].upper()
        preds = r["predictions"]
        if preds and pathogen_matches(gt, preds[0]):
            top1_correct += 1
        if any(pathogen_matches(gt, p) for p in preds):
            top3_correct += 1

    print(f"\nQuick Top-1 Accuracy: {top1_correct}/{len(results)} = {100*top1_correct/len(results):.1f}%")
    print(f"Quick Top-3 Accuracy: {top3_correct}/{len(results)} = {100*top3_correct/len(results):.1f}%")


if __name__ == "__main__":
    main()
