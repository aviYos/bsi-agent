"""
Step 1.2: Generate Full Medical Summaries (Model A)

Generates 100% complete medical summaries from BSI case data using GPT-4o.

Usage:
    python scripts/step2_generate_summaries.py
"""

import argparse
import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.generation.summary_generator import SummaryGenerator
from bsi_agent.generation.styles_generator import StylesGenerator
from bsi_agent.data.redaction import sanitize_summary


def load_cases(path: Path) -> list[dict]:
    """Load BSI cases from JSONL file."""
    cases = []
    with open(path, "r") as f:
        for line in f:
            cases.append(json.loads(line))
    return cases


def save_summaries(summaries: list[dict], path: Path):
    """Save summaries to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in summaries:
            f.write(json.dumps(item) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate medical summaries")
    parser.add_argument(
        "--cases_path",
        type=str,
        default="data/processed/bsi_cases.jsonl",
        help="Path to BSI cases",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/full_summaries.jsonl",
        help="Output path for summaries",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--max_cases",
        type=int,
        default=None,
        help="Limit number of cases to process",
    )
    args = parser.parse_args()

    # Resolve paths
    project_root = Path(__file__).parent.parent
    cases_path = project_root / args.cases_path
    output_path = project_root / args.output_path
    config_path = project_root / args.config_path

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    api_key = config["dialogue_generation"]["api_key"]
    model = config["dialogue_generation"].get("model", "gpt-4o")

    print(f"Loading cases from: {cases_path}")
    cases = load_cases(cases_path)

    if args.max_cases:
        cases = cases[:args.max_cases]

    print(f"Processing {len(cases)} cases with {model}")
    print("-" * 50)

    styles_gen = StylesGenerator(api_key=api_key, model=model)
    styles = styles_gen.generate_styles()

    # Initialize generator
    generator = SummaryGenerator(api_key=api_key, model=model)

    # Generate summaries
    summaries = []
    for case in tqdm(cases, desc="Generating summaries"):
        try:
            summary = generator.generate_summary(case, styles_gen.sample_random_style_string())
            summary = sanitize_summary(summary, case.get("organism", ""))
            summaries.append({
                "case_id": case["case_id"],
                "ground_truth_pathogen": case["organism"],
                "full_summary": summary,
            })
        except Exception as e:
            print(f"\nError processing {case['case_id']}: {e}")
            continue

    # Save results
    save_summaries(summaries, output_path)
    print("-" * 50)
    print(f"Saved {len(summaries)} summaries to {output_path}")


if __name__ == "__main__":
    main()
