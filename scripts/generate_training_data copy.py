"""
Step 2: Generate Training Data

Generates [x, q] training pairs for Model D by:
1. Creating partial summaries (50% hidden)
2. Generating questions (Model B)
3. Generating answers (Model A)
4. Creating dialogues
5. Classifying from partial (x) and dialogue (d)
6. Filtering for good questions (rank_d < rank_x)

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --step 3  # Start from step 3
"""

import argparse
import sys
import json
import datetime
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.data.utils import load_jsonl, save_jsonl, load_config
from bsi_agent.data.redaction import (
    sanitize_summary,
    redact_pathogen_mentions,
    find_pathogen_mentions,
)
from bsi_agent.generation import (
    SummaryGenerator,
    format_case_data,
    QuestionGenerator,
    AnswerGenerator,
    PathogenClassifier,
    create_partial_case,
    create_dialogue,
    StylesGenerator,
    QuestionStylesGenerator
)
from bsi_agent.evaluation import get_pathogen_rank


class Tee:
    """Redirects stdout/stderr to both console and a file."""
    def __init__(self, filename, stream):
        self.file = open(filename, 'a', encoding='utf-8')
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def __del__(self):
        self.file.close()


def step1_create_partial_summaries(
    cases_path: Path,
    output_path: Path,
    api_key: str,
    model: str,
    seed: int = 42,
    max_cases: int = None,
):
    """Step 2.1: Create partial summaries from raw case data."""
    print("\n" + "=" * 60)
    print("STEP 2.1: Create Partial Summaries (from raw cases)")
    print("=" * 60)

    # Load raw cases
    raw_cases = load_jsonl(cases_path)
    print(f"Loaded {len(raw_cases)} raw BSI cases from {cases_path}")

    # Remove dependency on full_summaries
    if max_cases:
        raw_cases = raw_cases[:max_cases]
    print(f"Processing {len(raw_cases)} cases")

    # Gets randomly assigned style for each case (for diversity in summaries)
    styles_gen = StylesGenerator(api_key=api_key, model=model)
    styles = styles_gen.generate_styles()



    # Initialize summary generator (Model A)
    generator = SummaryGenerator(api_key=api_key, model=model)

    results = []
    for i, raw_case in enumerate(tqdm(raw_cases, desc="Creating partial summaries")):
        case_id = raw_case["case_id"]
        ground_truth = raw_case.get("organism", "")

        # Step A: Create partial case (hide categories from raw data)
        partial_case, hidden_categories = create_partial_case(
            raw_case, seed=seed + i
        )

        # Step B: Generate narrative summary from partial case via Model A
        try:
            partial_text = generator.generate_summary(partial_case, styles_gen.sample_random_style_string())

        except Exception as e:
            print(f"\nError generating summary for {case_id}: {e}")
            continue

        # Step C: Sanitize (redact pathogen mentions)
        partial_text = sanitize_summary(partial_text, ground_truth)

        # Step D: Check for leaks
        leaks = find_pathogen_mentions(partial_text, ground_truth)
        if leaks:
            print(f"Warning: possible pathogen leakage in {case_id}: {leaks}")

        # Retrieve full summary for downstream steps
        # full_summary_lookup removed, we rely on raw_case
        full_summary = ""

        results.append({
            "case_id": case_id,
            "ground_truth_pathogen": ground_truth,
            "full_summary": full_summary,
            "partial_summary": partial_text,
            "hidden_categories": hidden_categories,
            "raw_case": raw_case,
        })

    save_jsonl(results, output_path)
    print(f"Saved {len(results)} partial summaries to {output_path}")

    # Stats
    from collections import Counter
    hidden_cats = Counter()
    for r in results:
        for cat in r["hidden_categories"]:
            hidden_cats[cat] += 1

    print("\nCategories hidden:")
    for cat, count in hidden_cats.most_common():
        print(f"  {cat}: {count}/{len(results)} ({100*count/len(results):.0f}%)")


def step2_generate_questions(input_path: Path, output_path: Path, api_key: str, model: str):
    """Step 2.2: Generate questions using Model B."""
    print("\n" + "=" * 60)
    print("STEP 2.2: Generate Questions (Model B)")
    print("=" * 60)

    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} partial summaries")

    # Gets randomly assigned style for each case (for diversity in summaries)
    styles_gen = QuestionStylesGenerator(api_key=api_key, model=model)
    styles = styles_gen.generate_styles()
    print

    generator = QuestionGenerator(api_key=api_key, model=model)
    results = []

    for item in tqdm(data, desc="Generating questions"):
        try:
            question = generator.generate(item["partial_summary"], styles_gen.sample_random_style_string())
            question = redact_pathogen_mentions(
                question,
                item.get("ground_truth_pathogen", ""),
            )
            results.append({
                "case_id": item["case_id"],
                "ground_truth_pathogen": item["ground_truth_pathogen"],
                "full_summary": sanitize_summary(
                    item.get("full_summary", ""),
                    item.get("ground_truth_pathogen", ""),
                ),
                "partial_summary": sanitize_summary(
                    item.get("partial_summary", ""),
                    item.get("ground_truth_pathogen", ""),
                ),
                "question": question,
                "raw_case": item.get("raw_case", {}),
            })
        except Exception as e:
            print(f"\nError processing {item['case_id']}: {e}")

    save_jsonl(results, output_path)
    print(f"Saved {len(results)} questions to {output_path}")


def step3_generate_answers(input_path: Path, output_path: Path, api_key: str, model: str):
    """Step 2.3: Generate answers using Model A."""
    print("\n" + "=" * 60)
    print("STEP 2.3: Generate Answers (Model A)")
    print("=" * 60)

    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} questions")

    generator = AnswerGenerator(api_key=api_key, model=model)
    results = []

    for item in tqdm(data, desc="Generating answers"):
        try:
            raw_case = item.get("raw_case", {})
            ground_truth = item.get("ground_truth_pathogen", "")

            if raw_case:
                # Use raw case if available
                # Format using SummaryGenerator's logic and redact pathogen name
                patient_data_str = format_case_data(raw_case)
                patient_data = redact_pathogen_mentions(patient_data_str, ground_truth)
            else:
                # Fallback to full summary
                patient_data = sanitize_summary(
                    item.get("full_summary", ""),
                    ground_truth,
                )

            answer = generator.generate(patient_data, item["question"])
            answer = redact_pathogen_mentions(
                answer,
                ground_truth,
            )
            results.append({
                "case_id": item["case_id"],
                "ground_truth_pathogen": item["ground_truth_pathogen"],
                "partial_summary": item["partial_summary"],
                "question": item["question"],
                "answer": answer,
            })
        except Exception as e:
            print(f"\nError processing {item['case_id']}: {e}")

    save_jsonl(results, output_path)
    print(f"Saved {len(results)} answers to {output_path}")


def step4_create_dialogues(input_path: Path, output_path: Path):
    """Step 2.4: Create dialogues (d = x + q + answer)."""
    print("\n" + "=" * 60)
    print("STEP 2.4: Create Dialogues")
    print("=" * 60)

    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} answers")

    results = []
    for item in data:
        safe_question = redact_pathogen_mentions(
            item.get("question", ""),
            item.get("ground_truth_pathogen", ""),
        )
        safe_answer = redact_pathogen_mentions(
            item.get("answer", ""),
            item.get("ground_truth_pathogen", ""),
        )
        safe_partial = sanitize_summary(
            item.get("partial_summary", ""),
            item.get("ground_truth_pathogen", ""),
        )
        dialogue = create_dialogue(
            safe_partial,
            safe_question,
            safe_answer,
        )
        results.append({
            "case_id": item["case_id"],
            "ground_truth_pathogen": item["ground_truth_pathogen"],
            "x": safe_partial,
            "q": safe_question,
            "answer": safe_answer,
            "d": dialogue,
        })

    save_jsonl(results, output_path)
    print(f"Saved {len(results)} dialogues to {output_path}")


def step5_classify_partial(input_path: Path, output_path: Path, api_key: str, model: str):
    """Step 2.5: Classify pathogens from partial summaries (x)."""
    print("\n" + "=" * 60)
    print("STEP 2.5: Classify from Partial (Model C)")
    print("=" * 60)

    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} dialogues")

    classifier = PathogenClassifier(api_key=api_key, model=model)
    results = []

    for item in tqdm(data, desc="Classifying from x"):
        try:
            predictions = classifier.classify(item["x"])
            rank = get_pathogen_rank(predictions, item["ground_truth_pathogen"])
            results.append({
                "case_id": item["case_id"],
                "ground_truth": item["ground_truth_pathogen"],
                "predictions_x": predictions,
                "rank_x": rank,
            })
        except Exception as e:
            print(f"\nError processing {item['case_id']}: {e}")

    save_jsonl(results, output_path)

    correct = sum(1 for r in results if r["rank_x"] <= 3)
    print(f"Saved {len(results)} classifications to {output_path}")
    print(f"Accuracy from x: {correct}/{len(results)} = {100*correct/len(results):.1f}%")


def step6_classify_dialogue(input_path: Path, output_path: Path, api_key: str, model: str):
    """Step 2.6: Classify pathogens from full dialogues (d)."""
    print("\n" + "=" * 60)
    print("STEP 2.6: Classify from Dialogue (Model C)")
    print("=" * 60)

    data = load_jsonl(input_path)
    print(f"Loaded {len(data)} dialogues")

    classifier = PathogenClassifier(api_key=api_key, model=model)
    results = []

    for item in tqdm(data, desc="Classifying from d"):
        try:
            predictions = classifier.classify(item["d"])
            rank = get_pathogen_rank(predictions, item["ground_truth_pathogen"])
            results.append({
                "case_id": item["case_id"],
                "ground_truth": item["ground_truth_pathogen"],
                "predictions_d": predictions,
                "rank_d": rank,
            })
        except Exception as e:
            print(f"\nError processing {item['case_id']}: {e}")

    save_jsonl(results, output_path)

    correct = sum(1 for r in results if r["rank_d"] <= 3)
    print(f"Saved {len(results)} classifications to {output_path}")
    print(f"Accuracy from d: {correct}/{len(results)} = {100*correct/len(results):.1f}%")


def step7_filter_good_questions(
    dialogues_path: Path,
    class_x_path: Path,
    class_d_path: Path,
    output_path: Path
):
    """Step 2.7: Filter for good questions (rank_d < rank_x)."""
    print("\n" + "=" * 60)
    print("STEP 2.7: Filter Good Questions")
    print("=" * 60)

    dialogues = load_jsonl(dialogues_path)
    class_x = load_jsonl(class_x_path)
    class_d = load_jsonl(class_d_path)

    # Create lookups
    dialogue_lookup = {d["case_id"]: d for d in dialogues}
    class_x_lookup = {c["case_id"]: c for c in class_x}
    class_d_lookup = {c["case_id"]: c for c in class_d}

    good_questions = []
    bad_questions = []

    for case_id in dialogue_lookup.keys():
        if case_id not in class_x_lookup or case_id not in class_d_lookup:
            continue

        dialogue = dialogue_lookup[case_id]
        cx = class_x_lookup[case_id]
        cd = class_d_lookup[case_id]

        rank_x = cx["rank_x"]
        rank_d = cd["rank_d"]
        is_good = rank_d < rank_x

        result = {
            "case_id": case_id,
            "ground_truth_pathogen": dialogue["ground_truth_pathogen"],
            "x": dialogue["x"],
            "q": dialogue["q"],
            "rank_x": rank_x,
            "rank_d": rank_d,
            "improvement": rank_x - rank_d,
        }

        if is_good:
            good_questions.append(result)
        else:
            bad_questions.append(result)

    save_jsonl(good_questions, output_path)

    total = len(good_questions) + len(bad_questions)
    print(f"\nRESULTS:")
    print(f"  Total cases: {total}")
    print(f"  Good questions: {len(good_questions)} ({100*len(good_questions)/total:.1f}%)")
    print(f"  Bad questions: {len(bad_questions)} ({100*len(bad_questions)/total:.1f}%)")
    print(f"\nSaved {len(good_questions)} good [x, q] pairs to {output_path}")

    # Train/test split (80/20)
    import random as _rng
    _rng.seed(42)
    shuffled = list(good_questions)
    _rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * 0.8))
    train_set = shuffled[:split_idx]
    test_set = shuffled[split_idx:] if split_idx < len(shuffled) else shuffled[:1]

    train_path = output_path.parent / "good_questions_train.jsonl"
    test_path = output_path.parent / "good_questions_test.jsonl"
    save_jsonl(train_set, train_path)
    save_jsonl(test_set, test_path)
    print(f"  Train: {len(train_set)} -> {train_path}")
    print(f"  Test:  {len(test_set)} -> {test_path}")

    if good_questions:
        print("\n" + "-" * 60)
        print("EXAMPLE GOOD QUESTION:")
        ex = good_questions[0]
        print(f"  Case: {ex['case_id']}")
        print(f"  Pathogen: {ex['ground_truth_pathogen']}")
        print(f"  Rank x->d: {ex['rank_x']} -> {ex['rank_d']} (improved by {ex['improvement']})")
        print(f"  Question: {ex['q']}")


def main():
    parser = argparse.ArgumentParser(description="Generate training data for Model D")
    parser.add_argument("--step", type=int, default=1, help="Start from step (1-7)")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of cases")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    # Setup logging
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_data_gen_{timestamp}.log"
    print(f"Logging execution output to: {log_file}")
    
    # Redirect stdout and stderr
    sys.stdout = Tee(log_file, sys.stdout)
    sys.stderr = Tee(log_file, sys.stderr)

    config = load_config(project_root / args.config)

    api_key = config["dialogue_generation"]["api_key"]
    model = config["dialogue_generation"].get("model", "gpt-4o")

    # Define paths
    data_dir = project_root / "data" / "processed"
    paths = {
        "bsi_cases": data_dir / "bsi_cases.jsonl",
        "full_summaries": data_dir / "full_summaries.jsonl",
        "partial_summaries": data_dir / "partial_summaries.jsonl",
        "questions": data_dir / "questions.jsonl",
        "answers": data_dir / "answers.jsonl",
        "dialogues": data_dir / "dialogues.jsonl",
        "classifications_x": data_dir / "classifications_x.jsonl",
        "classifications_d": data_dir / "classifications_d.jsonl",
        "good_questions": data_dir / "good_questions.jsonl",
    }

    print("=" * 60)
    print("STEP 2: GENERATE TRAINING DATA")
    print("=" * 60)
    print(f"Starting from step: {args.step}")
    print(f"Model: {model}")

    # Run steps
    if args.step <= 1:
        step1_create_partial_summaries(
            paths["bsi_cases"],
            # paths["full_summaries"],  <-- Removed
            paths["partial_summaries"],
            api_key,
            model,
            args.seed,
            args.max_cases,
        )

    if args.step <= 2:
        step2_generate_questions(
            paths["partial_summaries"],
            paths["questions"],
            api_key, model
        )

    if args.step <= 3:
        step3_generate_answers(
            paths["questions"],
            paths["answers"],
            api_key, model
        )

    if args.step <= 4:
        step4_create_dialogues(
            paths["answers"],
            paths["dialogues"]
        )

    if args.step <= 5:
        step5_classify_partial(
            paths["dialogues"],
            paths["classifications_x"],
            api_key, model
        )

    if args.step <= 6:
        step6_classify_dialogue(
            paths["dialogues"],
            paths["classifications_d"],
            api_key, model
        )

    if args.step <= 7:
        step7_filter_good_questions(
            paths["dialogues"],
            paths["classifications_x"],
            paths["classifications_d"],
            paths["good_questions"]
        )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nTraining data saved to: {paths['good_questions']}")


if __name__ == "__main__":
    main()
