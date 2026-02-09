"""
Evaluate Model D question generation using similarity metrics.

Usage:
    python scripts/evaluate_question_generation.py --model <path-or-hf-id>
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.data.utils import load_jsonl
from bsi_agent.evaluation.similarity_metrics import (
    compute_sentence_similarity,
    compute_bleu,
    compute_rouge_l,
    compute_bertscore_f1,
)
from bsi_agent.generation.question_generator import QUESTION_PROMPT


def extract_reference_question(item: dict) -> str:
    if "q" in item:
        return item["q"]
    if "question" in item:
        return item["question"]
    raise KeyError("Missing reference question field (expected 'q' or 'question').")


def extract_question(text: str) -> str:
    """Extract a single question from model output."""
    if not text:
        return ""

    # Prefer explicit QUESTION: ... formatting
    q_match = re.search(r"QUESTION\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if q_match:
        return q_match.group(1).strip().splitlines()[0].strip()

    # Otherwise, take the first sentence ending with '?'
    for sentence in re.split(r"(?<=[?])\s+", text.strip()):
        if "?" in sentence:
            return sentence.strip()

    # Fallback: first line
    return text.strip().splitlines()[0].strip()


def build_prompt(partial_summary: str) -> str:
    return QUESTION_PROMPT.format(partial_summary=partial_summary)


def generate_question(
    model,
    tokenizer,
    partial_summary: str,
    device: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    prompt = build_prompt(partial_summary)

    messages = [{"role": "user", "content": prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        gen_ids = model.generate(
            input_ids,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = gen_ids[0][input_ids.shape[-1]:]
    raw_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return extract_question(raw_text)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model D question generation")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HuggingFace model ID",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/processed/good_questions.jsonl",
        help="Path to [x, q] pairs (jsonl)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/question_eval.json",
        help="Output path for metrics and predictions",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples to evaluate",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=1.0,
        help="Fraction of data to use for evaluation (0-1]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run model on",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (if fine-tuned model)",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    input_path = project_root / args.input_path
    output_path = project_root / args.output_path

    data = load_jsonl(input_path)
    if not data:
        raise ValueError(f"No data found in {input_path}")

    # Subsample if needed
    random.seed(args.seed)
    if args.test_size < 1.0:
        sample_size = max(1, int(len(data) * args.test_size))
        data = random.sample(data, sample_size)

    if args.max_samples:
        data = data[: args.max_samples]

    print(f"Evaluating {len(data)} samples with model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_device_map = args.device.startswith("cuda")

    # Model loading kwargs
    model_kwargs = {
        "torch_dtype": torch.float16 if args.device.startswith("cuda") else torch.float32,
        "device_map": "auto" if use_device_map else None,
        "trust_remote_code": True,
    }
    if args.load_in_4bit and use_device_map:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Load LoRA adapter if specified
    if args.adapter_path:
        print(f"Loading LoRA adapter from: {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    if not use_device_map:
        model.to(args.device)
    model.eval()

    predictions = []
    references = []
    records = []

    for item in data:
        partial_summary = item.get("x") or item.get("partial_summary")
        if not partial_summary:
            raise KeyError("Missing partial summary field (expected 'x' or 'partial_summary').")

        ref_q = extract_reference_question(item)
        pred_q = generate_question(
            model,
            tokenizer,
            partial_summary,
            device=args.device,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )

        predictions.append(pred_q)
        references.append(ref_q)
        records.append({
            "case_id": item.get("case_id"),
            "partial_summary": partial_summary,
            "reference_question": ref_q,
            "predicted_question": pred_q,
        })

    metrics: dict[str, Optional[float]] = {}

    try:
        metrics["sentence_similarity"] = compute_sentence_similarity(predictions, references)
    except Exception as e:
        print(f"Sentence similarity unavailable: {e}")
        metrics["sentence_similarity"] = None

    try:
        metrics["bleu"] = compute_bleu(predictions, references)
    except Exception as e:
        print(f"BLEU unavailable: {e}")
        metrics["bleu"] = None

    try:
        metrics["rouge_l"] = compute_rouge_l(predictions, references)
    except Exception as e:
        print(f"ROUGE-L unavailable: {e}")
        metrics["rouge_l"] = None

    try:
        metrics["bertscore_f1"] = compute_bertscore_f1(predictions, references)
    except Exception as e:
        print(f"BERTScore unavailable: {e}")
        metrics["bertscore_f1"] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "num_samples": len(records),
                "predictions": records,
            },
            f,
            indent=2,
        )

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
