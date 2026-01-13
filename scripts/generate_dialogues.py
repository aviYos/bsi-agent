#!/usr/bin/env python3
"""
Generate synthetic training dialogues using GPT-4.

This script takes BSI cases and generates realistic diagnostic dialogues
between an ID consultant (agent) and a human clinician presenting the case,
in the PRE-CULTURE setting.

CRITICAL: The AGENT turns follow the same protocol used at evaluation time:
- Short reasoning paragraph
- OPTIONAL "QUESTION: ..."
- "FINAL_PATHOGEN_ESTIMATE_JSON:" followed by a JSON object:
    {"pathogen_estimate": [
        {"organism": "...", "confidence": 0.7},
        ...
    ]}
with confidences in [0,1] summing to ~1.0.
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from tqdm import tqdm


# ============================================================================
# Dialogue generation prompt templates
# ============================================================================

DIALOGUE_GENERATION_SYSTEM = """You are simulating a diagnostic dialogue between an Infectious Disease consultant (AGENT) and the bedside clinician responsible for the patient (ENVIRONMENT label) for a bloodstream infection case.

You are modeling the PRE-CULTURE decision-making phase:
- Blood cultures have been drawn but final culture results are NOT yet available.
- The AGENT must make probabilistic, pre-culture pathogen estimates only from clinical data and (optionally) Gram stain.

DIALOGUE STRUCTURE:

1. The clinician (ENVIRONMENT) starts by presenting the patient.
2. The AGENT asks focused questions to gather information over several turns.
3. The clinician responds with realistic clinical data in natural, conversational language.
4. The AGENT maintains and updates a differential diagnosis over time, based ONLY on pre-culture information.
5. The dialogue should include at least one moment of diagnostic uncertainty or hypothesis revision.
6. The dialogue should be efficient: around 4 AGENT turns total.

STRICT PRE-CULTURE RULES:

- The clinician must NOT reveal the organism name or final culture result at any point.
- The clinician may say that "blood cultures are pending" but must NOT state what grew.
- Gram stain information (e.g., "Gram-positive cocci in clusters") CAN be revealed after ~12 hours, but this is still considered pre-culture in terms of species-level identification.

AGENT TURN FORMAT (CRITICAL):

For EVERY AGENT turn ("role": "agent"), the AGENT's "content" MUST follow this exact structure:

1) One short paragraph (up to ~1.5 sentences) of clinical reasoning in plain text.

2) OPTIONAL QUESTION:
   If the AGENT needs more information, it MUST add a SINGLE line of the form:
       QUESTION: <a single, concrete clinical question ending with a question mark?>
   If no further information is needed in that turn, the AGENT MUST NOT include any line starting with "QUESTION:".

3) FINAL PATHOGEN ESTIMATE JSON:
   In each agent round, at the VERY END of the AGENT's content, the AGENT MUST output a single JSON object on a new line,
   prefixed exactly by:

       FINAL_PATHOGEN_ESTIMATE_JSON:
       {"pathogen_estimate": [
         {"organism": "<Exact organism name 1 (from the label set)>", "confidence": 0.70},
         {"organism": "<Exact organism name 2 (from the label set)>", "confidence": 0.20},
         {"organism": "Other / Unknown", "confidence": 0.10}
       ]}

   Requirements for this JSON:
   - It must be valid JSON.
   - "confidence" values must be between 0 and 1 (NOT percentages).
   - The confidences should sum to approximately 1.0 (allow small numeric rounding).
   - The list must be ordered from most likely to less likely.
   - At least one entry must exist.

FINAL TURN ALIGNMENT WITH GROUND TRUTH:

- In the FINAL AGENT TURN of the dialogue, the TOP entry in "pathogen_estimate" MUST have an "organism"
  that EXACTLY matches (string match) the ground-truth organism name for this case.
- The confidence in that top organism in the final turn should be high but not absolute (e.g., 0.70–0.85),
  to reflect pre-culture uncertainty.
- The AGENT MUST still frame this as a pre-culture probabilistic prediction, not as a post-culture fact.

ROLE LABELS:

- Use "agent" as the role for the Infectious Disease consultant.
- Use "environment" as the role for the bedside clinician.
- Do NOT invent any other role labels.

OUTPUT FORMAT:

- Output the dialogue as a JSON array of objects, each with "role" and "content" fields.
- For EVERY AGENT turn, the "content" MUST include:
  - Reasoning paragraph,
  - OPTIONAL "QUESTION: ..." line,
  - and the REQUIRED "FINAL_PATHOGEN_ESTIMATE_JSON:" line followed by the JSON object exactly as specified.
"""


DIALOGUE_GENERATION_USER = """Generate a pre-culture diagnostic dialogue for this BSI case:

## Patient Information
- Age: {age} years old
- Gender: {gender}
- Admission type: {admission_type}

## Clinical Data Available
{clinical_summary}

## Ground Truth (HIDDEN from AGENT, used only to shape its probabilistic reasoning)
- Organism: {organism}
- Gram stain would show: {gram_stain}
- Key susceptibilities: {susceptibilities}

## Requirements
1. The clinician (role label `environment`) presents the initial case (vitals, labs suggesting infection).
2. The consultant (role label `agent`) asks relevant questions (source, risk factors, comorbidities, prior antibiotics, etc.).
3. The clinician reveals Gram stain only if/when the AGENT asks for it (around turn 3–4), but never the final culture result.
4. The consultant updates the differential based on new information (including Gram stain if available).
5. Blood culture results remain pending throughout the dialogue; the organism name is NEVER explicitly stated by the clinician.
6. In EVERY AGENT TURN, the AGENT must follow the format described in the system prompt:
   - Short reasoning paragraph,
   - OPTIONAL 'QUESTION: ...' line,
   - 'FINAL_PATHOGEN_ESTIMATE_JSON:' followed by a JSON object with a 'pathogen_estimate' list.
7. In the FINAL AGENT TURN, the FIRST entry in 'pathogen_estimate' MUST have an 'organism' field that exactly matches the ground-truth organism above, with high but not absolute confidence (e.g., 0.70–0.85).
8. The dialogue should end with a pre-culture assessment and empiric treatment recommendation (not definitive post-culture therapy).

Generate the dialogue now as a JSON array:
"""


# ============================================================================
# Helpers to build prompts from case dicts
# ============================================================================

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

    # Medications (antibiotics)
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


# ============================================================================
# Core dialogue generation
# ============================================================================

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
        model: Model to use (e.g. "gpt-4o")
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


# ============================================================================
# Dialogue validation
# ============================================================================

def validate_dialogue(dialogue: list[dict], case: dict) -> tuple[bool, list[str]]:
    """
    Validate a generated dialogue for quality in the PRE-CULTURE setting
    and consistency with the FINAL_PATHOGEN_ESTIMATE_JSON protocol.

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Basic length checks
    if len(dialogue) < 4:
        issues.append("Dialogue too short (< 4 turns)")
    if len(dialogue) > 12:
        issues.append("Dialogue too long (> 12 turns)")

    # Role presence checks
    roles = set(turn["role"].lower() for turn in dialogue)
    if "agent" not in roles and "assistant" not in roles:
        issues.append("No agent turns found")
    if "environment" not in roles and "user" not in roles:
        issues.append("No environment turns found")

    # Extract organism token
    organism_full = (case.get("organism") or "").strip().lower()
    organism_token = organism_full.split()[0] if organism_full else ""

    # Split agent vs environment turns
    agent_turns = [
        (idx, t)
        for idx, t in enumerate(dialogue)
        if t["role"].lower() in ("agent", "assistant")
    ]
    env_turns = [
        (idx, t)
        for idx, t in enumerate(dialogue)
        if t["role"].lower() in ("environment", "user")
    ]

    full_text = " ".join(turn["content"].lower() for turn in dialogue)

    # Gram stain mention (desirable)
    if "gram" not in full_text:
        issues.append("Gram stain not mentioned")

    # Confidence / differential language (desirable)
    confidence_terms = ["confidence", "confident", "likely", "%", "differential", "suspect"]
    if not any(term in full_text for term in confidence_terms):
        issues.append("No confidence expression found")

    # Clinician must NOT reveal ground-truth organism name
    if organism_token:
        for idx, t in env_turns:
            if organism_token in t["content"].lower():
                issues.append(
                    f"Environment turn {idx} appears to mention the ground-truth organism (must stay pre-culture)"
                )
                break

    # Each agent turn must have FINAL_PATHOGEN_ESTIMATE_JSON and valid-ish JSON
    if not agent_turns:
        issues.append("No agent turns to check for pathogen estimates")
    else:
        import json as _json

        last_agent_estimate_top = None

        for idx, t in agent_turns:
            content = t["content"]
            lower_content = content.lower()

            if "final_pathogen_estimate_json" not in lower_content:
                issues.append(f"Agent turn {idx} missing 'FINAL_PATHOGEN_ESTIMATE_JSON' block")
                continue

            # Extract JSON after the prefix
            m = re.search(
                r"FINAL_PATHOGEN_ESTIMATE_JSON\s*:\s*(\{.*\})",
                content,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if not m:
                issues.append(f"Agent turn {idx} has no parseable JSON after FINAL_PATHOGEN_ESTIMATE_JSON:")
                continue

            json_str = m.group(1).strip()

            # Try to parse; be a bit forgiving on errors
            try:
                obj = _json.loads(json_str)
            except Exception:
                issues.append(f"Agent turn {idx} JSON is not valid")
                continue

            est_list = obj.get("pathogen_estimate", [])
            if not isinstance(est_list, list) or not est_list:
                issues.append(f"Agent turn {idx} has empty or invalid 'pathogen_estimate' list")
                continue

            # Check confidences in [0,1] and ~sum==1
            confidences = []
            for entry in est_list:
                if not isinstance(entry, dict):
                    continue
                c = entry.get("confidence")
                if not isinstance(c, (float, int)):
                    continue
                confidences.append(float(c))
            if confidences:
                s = sum(confidences)
                if not (0.8 <= s <= 1.2):
                    issues.append(f"Agent turn {idx} confidences do not sum to ~1.0 (sum={s:.3f})")

            # Track the top organism in the LAST agent turn
            if (idx, t) == agent_turns[-1]:
                top_entry = est_list[0]
                last_agent_estimate_top = str(top_entry.get("organism", "")).strip().lower()

        # Final agent turn must include ground-truth in first JSON entry
        if organism_token and last_agent_estimate_top:
            if organism_token not in last_agent_estimate_top:
                issues.append(
                    "Final agent turn's JSON 'pathogen_estimate[0].organism' does not contain the ground-truth organism token"
                )

    is_valid = len(issues) == 0
    return is_valid, issues


# ============================================================================
# Batch generation
# ============================================================================

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
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

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
                    # We still keep the dialogue, but you could choose to skip here.

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


# ============================================================================
# CLI
# ============================================================================

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
            line = line.strip()
            if not line:
                continue
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
