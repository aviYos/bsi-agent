#!/usr/bin/env python3
"""
Fine-tune a language model on BSI dialogues using QLoRA.

This script implements parameter-efficient fine-tuning using 4-bit quantization
and Low-Rank Adaptation (LoRA) to train a BSI diagnostic agent.
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

from bsi_agent.agent.prompts import SYSTEM_PROMPT, ANTIBIOGRAM_CONTEXT, TREATMENT_GUIDELINES_CONTEXT


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dialogues(dialogues_path: str) -> list[dict]:
    """Load dialogues from JSONL file."""
    dialogues = []
    with open(dialogues_path, "r") as f:
        for line in f:
            dialogues.append(json.loads(line))
    return dialogues


def format_dialogue_for_training(
    dialogue_data: dict,
    tokenizer,
    max_length: int = 4096,
    include_system_context: bool = True,
) -> dict:
    """
    Format a dialogue into the model's chat format for training.

    Args:
        dialogue_data: Dialogue with case info and turns
        tokenizer: Model tokenizer with chat template
        max_length: Maximum sequence length
        include_system_context: Whether to include antibiogram/guidelines

    Returns:
        Dict with input_ids, attention_mask, and labels
    """
    dialogue = dialogue_data.get("dialogue", [])

    # Build system message
    system_parts = [SYSTEM_PROMPT]
    if include_system_context:
        system_parts.append(ANTIBIOGRAM_CONTEXT)
        system_parts.append(TREATMENT_GUIDELINES_CONTEXT)
    system_content = "\n\n".join(system_parts)

    # Build messages list
    messages = [{"role": "system", "content": system_content}]

    for turn in dialogue:
        role = turn.get("role", "").lower()
        content = turn.get("content", "")

        if role in ["environment", "user", "system"]:
            messages.append({"role": "user", "content": f"[ENVIRONMENT]: {content}"})
        elif role in ["agent", "assistant"]:
            messages.append({"role": "assistant", "content": content})

    # Apply chat template
    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            max_length=max_length,
            truncation=True,
            return_dict=True,
        )

        # For causal LM training, labels = input_ids (shifted internally)
        input_ids = formatted["input_ids"]
        attention_mask = formatted.get("attention_mask", [1] * len(input_ids))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.copy(),  # Will be shifted by model
        }
    except Exception as e:
        print(f"Error formatting dialogue: {e}")
        return None


def prepare_dataset(
    dialogues: list[dict],
    tokenizer,
    max_length: int = 4096,
) -> Dataset:
    """
    Prepare a HuggingFace Dataset from dialogues.

    Args:
        dialogues: List of dialogue dictionaries
        tokenizer: Model tokenizer
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset
    """
    processed = []

    for dialogue_data in tqdm(dialogues, desc="Processing dialogues"):
        formatted = format_dialogue_for_training(
            dialogue_data,
            tokenizer,
            max_length=max_length,
        )
        if formatted is not None:
            processed.append(formatted)

    print(f"Processed {len(processed)}/{len(dialogues)} dialogues successfully")

    return Dataset.from_list(processed)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BSI agent with QLoRA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--dialogues_path",
        type=str,
        default=None,
        help="Path to dialogues JSONL (overrides config)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Base model path/name (overrides config)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="LoRA rank (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config) if Path(args.config).exists() else {}

    # Get parameters (CLI overrides config)
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    lora_config_dict = model_config.get("lora", {})
    data_config = config.get("data", {})

    base_model = args.base_model or model_config.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct")
    dialogues_path = args.dialogues_path or f"{data_config.get('dialogues_dir', 'data/synthetic_dialogues')}/dialogues.jsonl"
    output_dir = args.output_dir or training_config.get("output_dir", "outputs/model")
    num_epochs = args.num_epochs or training_config.get("num_epochs", 5)
    batch_size = args.batch_size or training_config.get("batch_size", 4)
    learning_rate = args.learning_rate or training_config.get("learning_rate", 2e-5)
    lora_r = args.lora_r or lora_config_dict.get("r", 16)
    max_seq_length = training_config.get("max_seq_length", 4096)
    gradient_accumulation = training_config.get("gradient_accumulation_steps", 4)

    print("=" * 60)
    print("BSI Agent Fine-Tuning")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Dialogues: {dialogues_path}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_r}")
    print("=" * 60)

    # Check if dialogues exist
    if not Path(dialogues_path).exists():
        print(f"Error: Dialogues file not found: {dialogues_path}")
        print("Please run generate_dialogues.py first")
        return

    # Load dialogues
    print("\nLoading dialogues...")
    dialogues = load_dialogues(dialogues_path)
    print(f"Loaded {len(dialogues)} dialogues")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare dataset
    print("\nPreparing dataset...")
    dataset = prepare_dataset(dialogues, tokenizer, max_length=max_seq_length)

    # Split into train/val
    split = dataset.train_test_split(test_size=0.01, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Configure quantization
    print("\nConfiguring quantization (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    print(f"\nLoading model: {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

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
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name="bsi-agent-qlora",
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True,
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
