"""
Parallel Training Data Generation Pipeline (Full Refactor)
---------------------------------------------------------
Drop‑in replacement for generate_training_data.py with:

✔ Threaded parallel LLM calls
✔ Retry + exponential backoff
✔ Rate‑limit friendly jitter
✔ Minimal change to original logic
✔ Same CLI interface
"""

import argparse
import sys
import json
import datetime
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Callable, Any

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
    QuestionStylesGenerator,
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


# =====================================================
# Parallel Utilities
# =====================================================


def retry_with_backoff(func, max_retries=5, base_delay=1.0):
    def wrapper(*args, **kwargs):
        for attempt in range(max_retries):
            try:
                time.sleep(random.uniform(0, 0.3))
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                sleep = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep)
    return wrapper


def parallel_map(items: List[Any], worker_fn: Callable, max_workers: int, desc: str):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(worker_fn, item) for item in items]

        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                r = f.result()
                if r:
                    results.append(r)
            except Exception as e:
                print(f"Worker error: {e}")

    return results


# =====================================================
# Step 1
# =====================================================


def step1_create_partial_summaries(raw_cases, generator, styles_gen, seed, max_workers):

    @retry_with_backoff
    def process(args):
        i, raw_case = args
        case_id = raw_case["case_id"]
        gt = raw_case.get("organism", "")
        # Use new function that returns hints
        partial_case, hidden_hints = create_partial_case(raw_case, seed=seed + i)

        partial_text = generator.generate_summary(
            partial_case, styles_gen.sample_random_style_string()
        )

        partial_text = sanitize_summary(partial_text, gt)

        return {
            "case_id": case_id,
            "ground_truth_pathogen": gt,
            "full_summary": "",
            "partial_summary": partial_text,
            "hidden_hints": hidden_hints, # <--- Save this!
            "raw_case": raw_case,
        }

    indexed = list(enumerate(raw_cases))
    return parallel_map(indexed, process, max_workers, "Step1")


# =====================================================
# Step 2
# =====================================================


def step2_generate_questions(data, generator, styles_gen, max_workers):

    @retry_with_backoff
    def process(item):
        
        # Format hints for the prompt
        hints = item.get("hidden_hints", [])
        hints_str = ""
        if hints:
            # Shuffle to prevent bias and limit to 15 items
            rng = random.Random(item["case_id"])
            rng.shuffle(hints)
            hints_str = ", ".join(hints[:15])

        # Pass hints to the generator
        q = generator.generate(
            partial_summary=item["partial_summary"], 
            style=styles_gen.sample_random_style_string(),
            available_hints=hints_str  # <--- NEW ARGUMENT
        )

        q = redact_pathogen_mentions(q, item["ground_truth_pathogen"])

        return {
            "case_id": item["case_id"],
            "ground_truth_pathogen": item["ground_truth_pathogen"],
            "partial_summary": item["partial_summary"],
            "question": q,
            "raw_case": item.get("raw_case", {}),
        }

    return parallel_map(data, process, max_workers, "Step2")


# =====================================================
# Step 3
# =====================================================


def step3_generate_answers(data, generator, max_workers):

    @retry_with_backoff
    def process(item):
        raw_case = item.get("raw_case", {})
        gt = item["ground_truth_pathogen"]
        
        patient_data_text = ""

        # 1. Prepare the FULL Patient Record (Teacher View)
        if raw_case:
            # Convert the raw dict to a readable string with ALL labs/vitals
            full_text = format_case_data(raw_case)
            
            # 2. Safety Redaction
            # We want Model A to know the labs, but NOT the final diagnosis name.
            # This prevents Model A from slipping up and saying "In cases of E. Coli..."
            patient_data_text = redact_pathogen_mentions(full_text, gt)
        else:
            # Fallback if raw_case is missing (should not happen)
            patient_data_text = sanitize_summary(item.get("full_summary", ""), gt)

        # 3. Generate Answer
        # Model A now sees the FULL data (including the 'hidden' hints from Step 2)
        answer = generator.generate(patient_data_text, item["question"])
        
        # 4. Final Safety Redaction on the Answer
        answer = redact_pathogen_mentions(answer, gt)

        return {
            "case_id": item["case_id"],
            "ground_truth_pathogen": gt,
            "partial_summary": item["partial_summary"],
            "question": item["question"],
            "answer": answer,
        }

    return parallel_map(data, process, max_workers, "Step3")

# =====================================================
# Step 4
# =====================================================


def step4_create_dialogues(data):
    results = []

    for item in data:
        d = create_dialogue(
            item["partial_summary"],
            item["question"],
            item["answer"],
        )

        results.append(
            {
                "case_id": item["case_id"],
                "ground_truth_pathogen": item["ground_truth_pathogen"],
                "x": item["partial_summary"],
                "q": item["question"],
                "answer": item["answer"],
                "d": d,
            }
        )

    return results


# =====================================================
# Step 5 & 6
# =====================================================


def step_classify(data, classifier, key, max_workers):

    @retry_with_backoff
    def process(item):

        preds = classifier.classify(item[key])
        rank = get_pathogen_rank(preds, item["ground_truth_pathogen"])

        return {
            "case_id": item["case_id"],
            "ground_truth": item["ground_truth_pathogen"],
            f"predictions_{key}": preds,
            f"rank_{key}": rank,
        }

    return parallel_map(data, process, max_workers, f"Classify {key}")


# =====================================================
# Step 7
# =====================================================


def step7_filter(dialogues, class_x, class_d):

    dx = {d["case_id"]: d for d in dialogues}
    cx = {c["case_id"]: c for c in class_x}
    cd = {c["case_id"]: c for c in class_d}

    good = []

    for cid in dx:
        if cid not in cx or cid not in cd:
            continue

        rx = cx[cid]["rank_x"]
        rd = cd[cid]["rank_d"]

        if rd < rx:
            good.append(
                {
                    "case_id": cid,
                    "ground_truth_pathogen": dx[cid]["ground_truth_pathogen"],
                    "x": dx[cid]["x"],
                    "q": dx[cid]["q"],
                    "rank_x": rx,
                    "rank_d": rd,
                    "improvement": rx - rd,
                }
            )

    return good


# =====================================================
# Main
# =====================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--max_cases", type=int)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--config", default="configs/config.yaml")
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
    model = config["dialogue_generation"].get("model", "gpt-4.1-nano")

    data_dir = project_root / "data" / "processed"

    paths = {
        "cases": data_dir / "bsi_cases.jsonl",
        "partials": data_dir / "partial_summaries.jsonl",
        "questions": data_dir / "questions.jsonl",
        "answers": data_dir / "answers.jsonl",
        "dialogues": data_dir / "dialogues.jsonl",
        "cx": data_dir / "classifications_x.jsonl",
        "cd": data_dir / "classifications_d.jsonl",
        "good": data_dir / "good_questions.jsonl",
    }

    print("=" * 60)
    print("STEP 2: GENERATE TRAINING DATA (PARALLEL)")
    print("=" * 60)
    print(f"Starting from step: {args.step}")
    print(f"Model: {model}")
    print(f"Workers: {args.workers}")

    raw_cases = load_jsonl(paths["cases"])
    if args.max_cases:
        raw_cases = raw_cases[: args.max_cases]

    styles = StylesGenerator(api_key, model)
    styles.generate_styles()

    qstyles = QuestionStylesGenerator(api_key, model)
    qstyles.generate_styles()



    summary_gen = SummaryGenerator(api_key, model)
    question_gen = QuestionGenerator(api_key, model)
    answer_gen = AnswerGenerator(api_key, model)
    classifier = PathogenClassifier(api_key, model)

    if args.step <= 1:
        print("\n" + "=" * 60)
        print("STEP 2.1: Create Partial Summaries (from raw cases)")
        print("=" * 60)
        partials = step1_create_partial_summaries(
            raw_cases, summary_gen, styles, args.seed, args.workers
        )
        save_jsonl(partials, paths["partials"])
        print(f"Saved {len(partials)} partial summaries to {paths['partials']}")
    else:
        partials = load_jsonl(paths["partials"])
        print(f"Loaded {len(partials)} partial summaries")

    if args.step <= 2:
        print("\n" + "=" * 60)
        print("STEP 2.2: Generate Questions (Model B)")
        print("=" * 60)
        questions = step2_generate_questions(partials, question_gen, qstyles, args.workers)
        save_jsonl(questions, paths["questions"])
        print(f"Saved {len(questions)} questions to {paths['questions']}")
    else:
        questions = load_jsonl(paths["questions"])
        print(f"Loaded {len(questions)} questions")

    if args.step <= 3:
        print("\n" + "=" * 60)
        print("STEP 2.3: Generate Answers (Model A)")
        print("=" * 60)
        answers = step3_generate_answers(questions, answer_gen, args.workers)
        save_jsonl(answers, paths["answers"])
        print(f"Saved {len(answers)} answers to {paths['answers']}")
    else:
        answers = load_jsonl(paths["answers"])
        print(f"Loaded {len(answers)} answers")

    if args.step <= 4:
        print("\n" + "=" * 60)
        print("STEP 2.4: Create Dialogues")
        print("=" * 60)
        dialogues = step4_create_dialogues(answers)
        save_jsonl(dialogues, paths["dialogues"])
        print(f"Saved {len(dialogues)} dialogues to {paths['dialogues']}")
    else:
        dialogues = load_jsonl(paths["dialogues"])
        print(f"Loaded {len(dialogues)} dialogues")

    if args.step <= 5:
        print("\n" + "=" * 60)
        print("STEP 2.5: Classify from Partial (Model C)")
        print("=" * 60)
        cx = step_classify(dialogues, classifier, "x", args.workers)
        save_jsonl(cx, paths["cx"])
        
        correct = sum(1 for r in cx if r["rank_x"] <= 3)
        print(f"Saved {len(cx)} classifications to {paths['cx']}")
        if len(cx) > 0:
            print(f"Accuracy from x: {correct}/{len(cx)} = {100*correct/len(cx):.1f}%")
    else:
        cx = load_jsonl(paths["cx"])
        print(f"Loaded {len(cx)} classifications (x)")

    if args.step <= 6:
        print("\n" + "=" * 60)
        print("STEP 2.6: Classify from Dialogue (Model C)")
        print("=" * 60)
        cd = step_classify(dialogues, classifier, "d", args.workers)
        save_jsonl(cd, paths["cd"])
        
        correct = sum(1 for r in cd if r["rank_d"] <= 3)
        print(f"Saved {len(cd)} classifications to {paths['cd']}")
        if len(cd) > 0:
            print(f"Accuracy from d: {correct}/{len(cd)} = {100*correct/len(cd):.1f}%")
    else:
        cd = load_jsonl(paths["cd"])
        print(f"Loaded {len(cd)} classifications (d)")

    if args.step <= 7:
        print("\n" + "=" * 60)
        print("STEP 2.7: Filter Good Questions")
        print("=" * 60)
        good = step7_filter(dialogues, cx, cd)
        save_jsonl(good, paths["good"])
        print(f"Saved {len(good)} good questions to {paths['good']}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
