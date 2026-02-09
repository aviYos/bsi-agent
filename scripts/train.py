#!/usr/bin/env python3
"""
Fine-tune a language model on BSI data using QLoRA.

Supports two training modes:
- "xq": Train on [x, q] pairs (partial summary -> question). Default.
- "dialogue": Train on multi-turn dialogues (legacy).

Usage:
    python scripts/train.py --mode xq --data_path data/processed/good_questions_train.jsonl
    python scripts/train.py --mode dialogue --dialogues_path data/synthetic_dialogues/dialogues.jsonl
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl(path: str) -> list[dict]:
    """Load data from JSONL file."""
    items = []
    with open(path, "r") as f:
        for line in f:
            items.append(json.loads(line))
    return items


# =========================================================================
# Mode: xq (partial summary -> question pairs)
# =========================================================================

def format_xq_pair_for_training(
    item: dict,
    tokenizer,
    max_length: int = 2048,
) -> Optional[dict]:
    """
    Format an [x, q] pair for causal LM training.

    Uses the same QUESTION_PROMPT as inference (evaluate_question_generation.py)
    so training and evaluation prompts match exactly.

    Labels are masked (-100) for the user prompt tokens so that loss is only
    computed on the assistant response (the question q).
    """
    from bsi_agent.generation.question_generator import QUESTION_PROMPT

    x = item.get("x") or item.get("partial_summary", "")
    q = item.get("q") or item.get("question", "")

    if not x or not q:
        return None

    user_content = QUESTION_PROMPT.format(partial_summary=x)

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": q},
    ]

    prompt_only = [
        {"role": "user", "content": user_content},
    ]

    try:
        # Tokenize full conversation (user + assistant response)
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_dict=True,
        )

        # Tokenize prompt only (with generation prompt = assistant header)
        # to find where the assistant response starts
        prompt_formatted = tokenizer.apply_chat_template(
            prompt_only,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        input_ids = formatted["input_ids"]
        attention_mask = formatted.get("attention_mask", [1] * len(input_ids))
        prompt_len = len(prompt_formatted["input_ids"])

        # Mask labels: -100 for prompt tokens, real ids for assistant response only
        labels = [-100] * min(prompt_len, len(input_ids)) + input_ids[prompt_len:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    except Exception as e:
        print(f"Error formatting xq pair: {e}")
        return None


def prepare_xq_dataset(
    data: list[dict],
    tokenizer,
    max_length: int = 2048,
) -> Dataset:
    """Prepare HuggingFace Dataset from [x, q] pairs."""
    processed = []
    for item in tqdm(data, desc="Processing xq pairs"):
        formatted = format_xq_pair_for_training(item, tokenizer, max_length)
        if formatted is not None:
            processed.append(formatted)

    print(f"Processed {len(processed)}/{len(data)} pairs successfully")
    return Dataset.from_list(processed)


# =========================================================================
# Mode: dialogue (multi-turn, legacy)
# =========================================================================

def format_dialogue_for_training(
    dialogue_data: dict,
    tokenizer,
    max_length: int = 4096,
    include_system_context: bool = True,
) -> Optional[dict]:
    """Format a multi-turn dialogue for training (legacy mode).

    Labels are masked (-100) for all non-assistant tokens so that loss
    is only computed on assistant responses.
    """
    from bsi_agent.agent.prompts import SYSTEM_PROMPT, ANTIBIOGRAM_CONTEXT, TREATMENT_GUIDELINES_CONTEXT

    dialogue = dialogue_data.get("dialogue", [])

    system_parts = [SYSTEM_PROMPT]
    if include_system_context:
        system_parts.append(ANTIBIOGRAM_CONTEXT)
        system_parts.append(TREATMENT_GUIDELINES_CONTEXT)
    system_content = "\n\n".join(system_parts)

    messages = [{"role": "system", "content": system_content}]

    for turn in dialogue:
        role = turn.get("role", "").lower()
        content = turn.get("content", "")

        if role in ["environment", "user", "system"]:
            messages.append({"role": "user", "content": f"[ENVIRONMENT]: {content}"})
        elif role in ["agent", "assistant"]:
            messages.append({"role": "assistant", "content": content})

    try:
        # Tokenize full conversation
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_dict=True,
        )

        input_ids = formatted["input_ids"]
        attention_mask = formatted.get("attention_mask", [1] * len(input_ids))

        # Build labels: mask everything except assistant responses
        # Tokenize incrementally to find assistant turn boundaries
        labels = [-100] * len(input_ids)
        for i in range(len(messages)):
            if messages[i]["role"] == "assistant":
                # Tokens up to (excluding) this assistant turn
                prefix = tokenizer.apply_chat_template(
                    messages[:i], tokenize=True, add_generation_prompt=True, return_dict=True,
                )
                # Tokens up to (including) this assistant turn
                with_turn = tokenizer.apply_chat_template(
                    messages[:i + 1], tokenize=True, add_generation_prompt=False, return_dict=True,
                )
                start = len(prefix["input_ids"])
                end = len(with_turn["input_ids"])
                for j in range(start, min(end, len(input_ids))):
                    labels[j] = input_ids[j]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    except Exception as e:
        print(f"Error formatting dialogue: {e}")
        return None


def prepare_dialogue_dataset(
    dialogues: list[dict],
    tokenizer,
    max_length: int = 4096,
) -> Dataset:
    """Prepare HuggingFace Dataset from dialogues (legacy mode)."""
    processed = []
    for dialogue_data in tqdm(dialogues, desc="Processing dialogues"):
        formatted = format_dialogue_for_training(dialogue_data, tokenizer, max_length)
        if formatted is not None:
            processed.append(formatted)

    print(f"Processed {len(processed)}/{len(dialogues)} dialogues successfully")
    return Dataset.from_list(processed)


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BSI agent with QLoRA")
    parser.add_argument("--mode", type=str, default="xq", choices=["xq", "dialogue"],
                        help="Training mode: 'xq' for [x,q] pairs (default), 'dialogue' for multi-turn")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to training data JSONL")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--no_quantize", action="store_true", help="Disable 4-bit quantization")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config) if Path(args.config).exists() else {}
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    lora_config_dict = model_config.get("lora", {})
    data_config = config.get("data", {})

    # Resolve parameters (CLI overrides config)
    base_model = args.base_model or model_config.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    output_dir = args.output_dir or training_config.get("output_dir", "outputs/model")
    num_epochs = args.num_epochs or training_config.get("num_epochs", 5)
    batch_size = args.batch_size or training_config.get("batch_size", 4)
    learning_rate = args.learning_rate or training_config.get("learning_rate", 2e-5)
    lora_r = args.lora_r or lora_config_dict.get("r", 16)
    gradient_accumulation = training_config.get("gradient_accumulation_steps", 4)

    # Resolve data path
    if args.data_path:
        data_path = args.data_path
    elif args.mode == "xq":
        data_path = f"{data_config.get('processed_dir', 'data/processed')}/good_questions_train.jsonl"
    else:
        data_path = f"{data_config.get('dialogues_dir', 'data/synthetic_dialogues')}/dialogues.jsonl"

    max_seq_length = 2048 if args.mode == "xq" else training_config.get("max_seq_length", 4096)

    print("=" * 60)
    print("BSI Agent Fine-Tuning")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Base model: {base_model}")
    print(f"Data: {data_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}, Batch: {batch_size}, LR: {learning_rate}")
    print(f"LoRA rank: {lora_r}, Max seq len: {max_seq_length}")
    print(f"Quantize: {not args.no_quantize}")
    print("=" * 60)

    if not Path(data_path).exists():
        print(f"Error: Data file not found: {data_path}")
        return

    # Load data
    print("\nLoading data...")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} items")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare dataset
    print("\nPreparing dataset...")
    if args.mode == "xq":
        dataset = prepare_xq_dataset(data, tokenizer, max_length=max_seq_length)
    else:
        dataset = prepare_dialogue_dataset(data, tokenizer, max_length=max_seq_length)

    if len(dataset) == 0:
        print("Error: No valid training examples after processing")
        return

    # Split into train/val
    if len(dataset) > 1:
        split = dataset.train_test_split(test_size=max(1, int(len(dataset) * 0.1)), seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]
    else:
        train_dataset = dataset
        val_dataset = dataset
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Configure quantization
    if not args.no_quantize and torch.cuda.is_available():
        print("\nConfiguring quantization (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None

    # Load model
    print(f"\nLoading model: {base_model}...")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    if bnb_config:
        model = prepare_model_for_kbit_training(model)
    else:
        # Enable gradients for non-quantized training
        model.enable_input_require_grads()

    # Configure LoRA
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_config_dict.get("lora_alpha", 32),
        lora_dropout=lora_config_dict.get("lora_dropout", 0.05),
        target_modules=lora_config_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=training_config.get("warmup_ratio", 0.1),
        logging_steps=10,
        eval_strategy="steps" if len(train_dataset) > 10 else "epoch",
        eval_steps=100,
        save_strategy="steps" if len(train_dataset) > 10 else "epoch",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"bsi-agent-{args.mode}",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=-100,
    )

    # Trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save final model
    print("\nSaving model...")
    final_output = Path(output_dir) / "final"
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)

    print(f"\nTraining complete!")
    print(f"Model saved to: {final_output}")


if __name__ == "__main__":
    main()
