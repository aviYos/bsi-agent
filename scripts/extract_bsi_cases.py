"""
Step 1.1: Extract BSI Cases from MIMIC-IV

This script extracts positive blood culture cases with all clinical context
(labs, vitals, medications) for validation of Model C's classification ability.

Usage:
    python scripts/step1_extract_bsi_cases.py --max_cases 100
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.data.bsi_cohort import BSICohortExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract BSI cases from MIMIC-IV")
    parser.add_argument(
        "--mimic_path",
        type=str,
        default="full_data/mimic-iv-3.1",
        help="Path to MIMIC-IV data directory",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/bsi_cases.jsonl",
        help="Output path for extracted cases",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=100,
        help="Maximum number of cases to extract",
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
        default=24,
        help="Hours of data to include after culture",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    mimic_path = project_root / args.mimic_path
    output_path = project_root / args.output_path

    print(f"MIMIC-IV path: {mimic_path}")
    print(f"Output path: {output_path}")
    print(f"Max cases: {args.max_cases}")
    print("-" * 50)

    # Extract cases
    extractor = BSICohortExtractor(mimic_path)
    cases = extractor.extract_bsi_cases(
        max_cases=args.max_cases,
        hours_before_culture=args.hours_before,
        hours_after_culture=args.hours_after,
    )

    # Save cases
    extractor.save_cases(cases, output_path)

    # Print summary
    print("-" * 50)
    print(f"Extracted {len(cases)} BSI cases")

    # Show pathogen distribution
    from collections import Counter
    pathogens = Counter(case.organism for case in cases)
    print("\nTop 10 pathogens:")
    for pathogen, count in pathogens.most_common(10):
        print(f"  {pathogen}: {count}")


if __name__ == "__main__":
    main()
