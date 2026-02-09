"""
Demo: Run 1 case through Phase 0 (full summary + classify) and Phase 1 (partial summary + question).
Prints all intermediate outputs.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.data.utils import load_jsonl, load_config
from bsi_agent.data.redaction import sanitize_summary, find_pathogen_mentions
from bsi_agent.generation import (
    SummaryGenerator,
    PathogenClassifier,
    QuestionGenerator,
    create_partial_case,
)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_index", type=int, default=0, help="Index of case to demo")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config = load_config(project_root / "configs" / "config.yaml")

    api_key = config["dialogue_generation"]["api_key"]
    model = config["dialogue_generation"].get("model", "gpt-4o")

    # --- Load data ---
    raw_cases = load_jsonl(project_root / "data" / "processed" / "bsi_cases.jsonl")
    full_summaries = load_jsonl(project_root / "data" / "processed" / "full_summaries.jsonl")
    full_lookup = {s["case_id"]: s["full_summary"] for s in full_summaries}

    # Pick case by index (only among those with full summaries)
    matched_cases = [c for c in raw_cases if c["case_id"] in full_lookup]
    case = matched_cases[args.case_index] if args.case_index < len(matched_cases) else None

    if not case:
        print("No matching case found!")
        return

    case_id = case["case_id"]
    organism = case.get("organism", "?")

    # =====================================================
    # PHASE 0: Show raw case + existing full summary + classify
    # =====================================================
    print("=" * 70)
    print("PHASE 0: RAW CASE DATA")
    print("=" * 70)
    print(f"Case ID:    {case_id}")
    print(f"Organism:   {organism}")
    print(f"Age:        {case.get('age')}")
    print(f"Gender:     {case.get('gender')}")
    print(f"Admission:  {case.get('admission_type')}")
    print(f"Gram stain: {case.get('gram_stain')}")
    print(f"Labs:       {len(case.get('labs', []))} values")
    print(f"Vitals:     {len(case.get('vitals', []))} values")
    print(f"Meds:       {len(case.get('medications', []))} entries")

    # Show a few labs
    labs = case.get("labs", [])
    if labs:
        print("\n  Sample labs (first 5):")
        for lab in labs[:5]:
            name = lab.get("lab_name", f"Item {lab.get('itemid')}")
            val = lab.get("valuenum", "N/A")
            unit = lab.get("valueuom", "")
            print(f"    - {name}: {val} {unit}")

    # Show meds
    meds = case.get("medications", [])
    if meds:
        print(f"\n  Medications (first 5):")
        for med in meds[:5]:
            print(f"    - {med.get('drug', '?')} {med.get('dose_val_rx', '')} {med.get('dose_unit_rx', '')} ({med.get('route', '')})")

    # Existing full summary
    print("\n" + "=" * 70)
    print("PHASE 0: FULL SUMMARY (already generated)")
    print("=" * 70)
    full_summary = full_lookup[case_id]
    print(full_summary)

    # Classify from full summary
    print("\n" + "=" * 70)
    print("PHASE 0: CLASSIFY FROM FULL SUMMARY (Model C)")
    print("=" * 70)
    classifier = PathogenClassifier(api_key=api_key, model=model)
    preds_full = classifier.classify(full_summary)
    print(f"Ground truth:  {organism}")
    print(f"Predictions:   {preds_full}")
    match = any(organism.upper() in p.upper() or p.upper() in organism.upper() for p in preds_full)
    print(f"Top-3 correct: {'YES' if match else 'NO'}")
    if preds_full:
        match1 = organism.upper() in preds_full[0].upper() or preds_full[0].upper() in organism.upper()
        print(f"Top-1 correct: {'YES' if match1 else 'NO'}")

    # =====================================================
    # PHASE 1: Create partial case + generate partial summary + question
    # =====================================================
    print("\n" + "=" * 70)
    print("PHASE 1: CREATE PARTIAL CASE (hide categories)")
    print("=" * 70)
    partial_case, hidden = create_partial_case(case, seed=42)
    print(f"Hidden categories: {hidden}")
    print(f"Kept categories:   {[c for c in ['demographics','admission','labs','vitals','medications','gram_stain'] if c not in hidden]}")
    print(f"Labs remaining:    {len(partial_case.get('labs', []))}")
    print(f"Vitals remaining:  {len(partial_case.get('vitals', []))}")
    print(f"Meds remaining:    {len(partial_case.get('medications', []))}")
    print(f"Gram stain:        {partial_case.get('gram_stain')}")

    print("\n" + "=" * 70)
    print("PHASE 1: GENERATE PARTIAL SUMMARY (Model A from partial case)")
    print("=" * 70)
    generator = SummaryGenerator(api_key=api_key, model=model)
    partial_text = generator.generate_summary(partial_case)
    partial_text = sanitize_summary(partial_text, organism)
    print(partial_text)

    leaks = find_pathogen_mentions(partial_text, organism)
    if leaks:
        print(f"\n[WARNING] Pathogen leakage detected: {leaks}")

    # Classify from partial
    print("\n" + "=" * 70)
    print("PHASE 1: CLASSIFY FROM PARTIAL SUMMARY (Model C)")
    print("=" * 70)
    preds_partial = classifier.classify(partial_text)
    print(f"Ground truth:  {organism}")
    print(f"Predictions:   {preds_partial}")

    # Generate question
    print("\n" + "=" * 70)
    print("PHASE 1: GENERATE QUESTION (Model B from partial summary)")
    print("=" * 70)
    q_gen = QuestionGenerator(api_key=api_key, model=model)
    question = q_gen.generate(partial_text)
    print(f"Question: {question}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Case:              {case_id}")
    print(f"Ground truth:      {organism}")
    print(f"Hidden categories: {hidden}")
    print(f"Full preds:        {preds_full}")
    print(f"Partial preds:     {preds_partial}")
    print(f"Question asked:    {question}")


if __name__ == "__main__":
    main()
