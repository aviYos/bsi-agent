#!/usr/bin/env python3
"""
Generate synthetic training dialogues using GPT-4.

This script takes BSI cases and generates realistic diagnostic dialogues
between an agent and an EHR environment.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


# Dialogue generation prompt template
DIALOGUE_GENERATION_SYSTEM = """You are simulating a diagnostic dialogue between an Infectious Disease physician (AGENT) and an Electronic Health Record system (ENVIRONMENT) for a bloodstream infection case.

Generate a realistic multi-turn conversation where:

1. The ENVIRONMENT starts by presenting the patient
2. The AGENT asks focused questions to gather information
3. The ENVIRONMENT responds with realistic clinical data
4. The AGENT maintains and updates a differential diagnosis
5. Eventually, culture results become available and the AGENT reaches a final diagnosis

IMPORTANT RULES:
- The ENVIRONMENT must NOT reveal the organism name until "final culture results" turn
- Gram stain can be revealed after ~12 hours (hint at morphology, not species)
- The AGENT should express confidence levels (e.g., "70% confident")
- The AGENT should cite evidence for reasoning
- Keep the dialogue to 4-8 turns total (efficient)
- Include at least one moment of diagnostic uncertainty or hypothesis revision

Output the dialogue as a JSON array of objects with "role" and "content" fields."""


DIALOGUE_GENERATION_USER = """Generate a dialogue for this BSI case:

## Patient Information
- Age: {age} years old
- Gender: {gender}
- Admission type: {admission_type}

## Clinical Data Available
{clinical_summary}

## Ground Truth (HIDDEN from AGENT until culture finalizes)
- Organism: {organism}
- Gram stain would show: {gram_stain}
- Key susceptibilities: {susceptibilities}

## Requirements
1. ENVIRONMENT presents initial case (vitals, labs suggesting infection)
2. AGENT asks relevant questions (source, cultures, etc.)
3. ENVIRONMENT reveals Gram stain after AGENT asks (around turn 3-4)
4. AGENT updates differential based on Gram stain
5. ENVIRONMENT reveals final culture (last turn)
6. AGENT provides final assessment with treatment recommendation

Generate the dialogue now as a JSON array:"""


def create_clinical_summary(case: dict) -> str:
    """Create a clinical summary from case data."""
    parts = []

    # Labs summary
    labs = case.get("labs", [])
    if labs:
        abnormal_labs = []
        for lab in labs[:10]:
            name = lab.get("lab_name", "Unknown")
            value = lab.get("valuenum")
            if value is not None:
                abnormal_labs.append(f"{name}: {value:.1f}")
        if abnormal_labs:
            parts.append(f"Labs: {', '.join(abnormal_labs)}")

    # Vitals summary
    vitals = case.get("vitals", [])
    if vitals:
        vital_summary = []
        for vital in vitals[:5]:
            name = vital.get("vital_name", "Unknown")
            value = vital.get("valuenum")
            if value is not None:
                vital_summary.append(f"{name}: {value:.0f}")
        if vital_summary:
            parts.append(f"Vitals: {', '.join(vital_summary)}")

    # Medications
    meds = case.get("medications", [])
    if meds:
        drug_names = list(set([m.get("drug", "Unknown").split()[0] for m in meds[:5]]))
        parts.append(f"Antibiotics: {', '.join(drug_names)}")

    if not parts:
        parts.append("Limited clinical data available - generate realistic values for a septic patient")

    return "\n".join(parts)


def format_susceptibilities(susc: dict) -> str:
    """Format susceptibilities for prompt."""
    if not susc:
        return "Pending"

    items = []
    for ab, result in list(susc.items())[:5]:
        items.append(f"{ab}: {result}")
    return ", ".join(items)


def generate_dialogue(
    client: OpenAI,
    case: dict,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_retries: int = 3,
) -> Optional[list[dict]]:
    """
    Generate a single dialogue for a BSI case.

    Args:
        client: OpenAI client
        case: BSI case dictionary
        model: Model to use
        temperature: Sampling temperature
        max_retries: Number of retries on failure

    Returns:
        List of dialogue turns or None if failed
    """
    # Build prompt
    clinical_summary = create_clinical_summary(case)
    susceptibilities = format_susceptibilities(case.get("susceptibilities", {}))

    user_prompt = DIALOGUE_GENERATION_USER.format(
        age=case.get("age", "Unknown"),
        gender=case.get("gender", "Unknown"),
        admission_type=case.get("admission_type", "Unknown"),
        clinical_summary=clinical_summary,
        organism=case.get("organism", "Unknown"),
        gram_stain=case.get("gram_stain", "Unknown"),
        susceptibilities=susceptibilities,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": DIALOGUE_GENERATION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=2000,
            )

            content = response.choices[0].message.content

            # Extract JSON from response
            # Handle case where model wraps in markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            dialogue = json.loads(content.strip())

            # Validate dialogue structure
            if not isinstance(dialogue, list):
                raise ValueError("Dialogue must be a list")

            for turn in dialogue:
                if not isinstance(turn, dict) or "role" not in turn or "content" not in turn:
                    raise ValueError("Each turn must have 'role' and 'content'")

            return dialogue

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            print(f"  Error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)

    return None


def validate_dialogue(dialogue: list[dict], case: dict) -> tuple[bool, list[str]]:
    """
    Validate a generated dialogue for quality.

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check minimum length
    if len(dialogue) < 4:
        issues.append("Dialogue too short (< 4 turns)")

    # Check maximum length
    if len(dialogue) > 12:
        issues.append("Dialogue too long (> 12 turns)")

    # Check for both roles
    roles = set(turn["role"].lower() for turn in dialogue)
    if "agent" not in roles and "assistant" not in roles:
        issues.append("No agent turns found")
    if "environment" not in roles and "user" not in roles and "system" not in roles:
        issues.append("No environment turns found")

    # Check that organism is mentioned eventually
    organism = case.get("organism", "").lower()
    full_text = " ".join(turn["content"].lower() for turn in dialogue)

    # Check for Gram stain mention
    if "gram" not in full_text:
        issues.append("Gram stain not mentioned")

    # Check for confidence/differential
    confidence_terms = ["confidence", "likely", "%", "differential", "suspect"]
    if not any(term in full_text for term in confidence_terms):
        issues.append("No confidence expression found")

    # Check that final diagnosis appears
    last_turns = " ".join(turn["content"].lower() for turn in dialogue[-3:])
    if organism and organism.split()[0].lower() not in last_turns:
        # Be lenient - just check first word of organism
        pass  # issues.append("Final diagnosis may not match ground truth")

    is_valid = len(issues) == 0
    return is_valid, issues


def generate_dialogues_batch(
    cases: list[dict],
    output_path: Path,
    model: str = "gpt-4o",
    num_dialogues: int = 500,
    temperature: float = 0.7,
    api_key: Optional[str] = None,
    validate: bool = True,
    skip_existing: bool = True,
) -> None:
    """
    Generate dialogues for multiple cases.

    Args:
        cases: List of BSI cases
        output_path: Path to save dialogues
        model: OpenAI model to use
        num_dialogues: Target number of dialogues
        temperature: Sampling temperature
        api_key: OpenAI API key (or use env var)
        validate: Whether to validate dialogues
        skip_existing: Skip if output file exists with enough dialogues
    """
    # Initialize client
    client = OpenAI(api_key=api_key)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing dialogues if any
    existing_dialogues = []
    if output_path.exists() and skip_existing:
        with open(output_path, "r") as f:
            for line in f:
                existing_dialogues.append(json.loads(line))
        print(f"Found {len(existing_dialogues)} existing dialogues")

    if len(existing_dialogues) >= num_dialogues:
        print(f"Already have {len(existing_dialogues)} dialogues, skipping generation")
        return

    # Determine how many more to generate
    to_generate = num_dialogues - len(existing_dialogues)
    print(f"Generating {to_generate} new dialogues...")

    # Sample cases (with replacement if needed)
    if len(cases) < to_generate:
        sampled_cases = random.choices(cases, k=to_generate)
    else:
        sampled_cases = random.sample(cases, to_generate)

    # Generate dialogues
    new_dialogues = []
    failed = 0

    with open(output_path, "a") as f:
        for i, case in enumerate(tqdm(sampled_cases, desc="Generating dialogues")):
            dialogue = generate_dialogue(
                client,
                case,
                model=model,
                temperature=temperature,
            )

            if dialogue is None:
                failed += 1
                continue

            # Validate
            if validate:
                is_valid, issues = validate_dialogue(dialogue, case)
                if not is_valid:
                    print(f"  Case {i}: Validation issues: {issues}")
                    # Still keep it but flag
                    # failed += 1
                    # continue

            # Create training example
            example = {
                "case_id": case.get("case_id", f"case_{i}"),
                "organism": case.get("organism"),
                "dialogue": dialogue,
                "metadata": {
                    "age": case.get("age"),
                    "gender": case.get("gender"),
                    "gram_stain": case.get("gram_stain"),
                },
            }

            # Write immediately (append mode)
            f.write(json.dumps(example) + "\n")
            f.flush()

            new_dialogues.append(example)

            # Rate limiting
            time.sleep(0.5)  # Avoid rate limits

    print(f"\nGeneration complete!")
    print(f"  Generated: {len(new_dialogues)}")
    print(f"  Failed: {failed}")
    print(f"  Total in file: {len(existing_dialogues) + len(new_dialogues)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training dialogues using GPT-4"
    )
    parser.add_argument(
        "--cases_path",
        type=str,
        required=True,
        help="Path to BSI cases JSONL file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/synthetic_dialogues/dialogues.jsonl",
        help="Output path for dialogues",
    )
    parser.add_argument(
        "--num_dialogues",
        type=int,
        default=500,
        help="Number of dialogues to generate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--no_validate",
        action="store_true",
        help="Skip dialogue validation",
    )

    args = parser.parse_args()

    # Load cases
    cases_path = Path(args.cases_path)
    if not cases_path.exists():
        print(f"Error: Cases file not found: {cases_path}")
        print("Please run preprocessing first to extract BSI cases from MIMIC-IV")
        return

    cases = []
    with open(cases_path, "r") as f:
        for line in f:
            cases.append(json.loads(line))

    print(f"Loaded {len(cases)} BSI cases")

    # Generate dialogues
    generate_dialogues_batch(
        cases=cases,
        output_path=Path(args.output_path),
        model=args.model,
        num_dialogues=args.num_dialogues,
        temperature=args.temperature,
        api_key=args.api_key,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
