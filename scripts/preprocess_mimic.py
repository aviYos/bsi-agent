#!/usr/bin/env python3
"""
Preprocess MIMIC-IV data to extract BSI cases.

This script extracts patients with positive blood cultures from MIMIC-IV
and prepares them for dialogue generation and evaluation.
"""

import argparse
import json
import random
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.data.bsi_cohort import BSICohortExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract BSI cases from MIMIC-IV"
    )
    parser.add_argument(
        "--mimic_path",
        type=str,
        required=True,
        help="Path to MIMIC-IV data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed cases",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Maximum number of cases to extract",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of cases for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Fraction of cases for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting",
    )
    parser.add_argument(
        "--hours_before",
        type=int,
        default=48,
        help="Hours of data to include before culture",
    )
    parser.add_argument(
        "--hours_after",
        type=int,
        default=48,
        help="Hours of data to include after culture",
    )

    args = parser.parse_args()

    # Validate MIMIC path
    mimic_path = Path(args.mimic_path)
    if not mimic_path.exists():
        print(f"Error: MIMIC path not found: {mimic_path}")
        print("\nPlease ensure you have:")
        print("1. Downloaded MIMIC-IV from PhysioNet")
        print("2. Placed files in the specified directory")
        print("3. Directory structure: mimic_path/hosp/, mimic_path/icu/")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize extractor
    print("Initializing BSI cohort extractor...")
    extractor = BSICohortExtractor(mimic_path)

    # Extract cases
    print("\nExtracting BSI cases...")
    cases = extractor.extract_bsi_cases(
        max_cases=args.max_cases,
        hours_before_culture=args.hours_before,
        hours_after_culture=args.hours_after,
    )

    if not cases:
        print("No cases extracted. Check MIMIC data and paths.")
        return

    print(f"\nExtracted {len(cases)} BSI cases")

    # Convert to dicts for serialization
    case_dicts = [case.to_dict() for case in cases]

    # Analyze organism distribution
    organism_counts = {}
    for case in case_dicts:
        org = case.get("organism", "Unknown")
        organism_counts[org] = organism_counts.get(org, 0) + 1

    print("\nOrganism distribution:")
    for org, count in sorted(organism_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {org}: {count}")

    # Split into train/val/test
    random.seed(args.seed)
    random.shuffle(case_dicts)

    n_total = len(case_dicts)
    n_train = int(n_total * args.train_ratio)
    n_val = int(n_total * args.val_ratio)

    train_cases = case_dicts[:n_train]
    val_cases = case_dicts[n_train:n_train + n_val]
    test_cases = case_dicts[n_train + n_val:]

    print(f"\nSplit: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}")

    # Save cases
    def save_jsonl(cases, path):
        with open(path, "w") as f:
            for case in cases:
                f.write(json.dumps(case, default=str) + "\n")
        print(f"Saved {len(cases)} cases to {path}")

    save_jsonl(case_dicts, output_dir / "all_cases.jsonl")
    save_jsonl(train_cases, output_dir / "train_cases.jsonl")
    save_jsonl(val_cases, output_dir / "val_cases.jsonl")
    save_jsonl(test_cases, output_dir / "test_cases.jsonl")

    print("\nPreprocessing complete!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
