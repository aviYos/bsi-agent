#!/usr/bin/env python3
"""
Run the BSI agent interactively on a single case.

This script allows you to step through a BSI case dialogue,
observing the agent's reasoning and the environment's responses.
"""

import argparse
import json
import random
from pathlib import Path

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.agent.bsi_agent import BSIAgent, AgentConfig, OpenAIAgent
from bsi_agent.environment.ehr_environment import EHREnvironment
from bsi_agent.guardrails.safety_checks import SafetyGuardrails, extract_drugs_from_text


def print_separator():
    print("\n" + "=" * 60 + "\n")


def run_interactive_session(
    agent,
    case: dict,
    auto_mode: bool = False,
    max_turns: int = 10,
):
    """
    Run an interactive diagnostic session.

    Args:
        agent: BSI agent instance
        case: BSI case dictionary
        auto_mode: If True, run without pausing
        max_turns: Maximum turns
    """
    # Initialize environment
    env = EHREnvironment(case)

    # Reset agent
    agent.reset()

    # Show ground truth (hidden from agent)
    print_separator()
    print("🔬 GROUND TRUTH (Hidden from agent)")
    print(f"   Organism: {case.get('organism', 'Unknown')}")
    print(f"   Gram stain: {case.get('gram_stain', 'Unknown')}")
    print_separator()

    # Get initial presentation
    initial = env.get_initial_presentation()
    agent.add_environment_message(initial)

    print("📋 ENVIRONMENT (Initial Presentation):")
    print(f"   {initial}")

    if not auto_mode:
        input("\nPress Enter to continue...")

    turn = 0
    while turn < max_turns:
        turn += 1
        print_separator()
        print(f"🔄 TURN {turn}")

        # Agent generates response
        print("\n🤖 AGENT thinking...")
        response = agent.generate_response()

        print(f"\n🤖 AGENT:")
        print(f"   {response.raw_text}")

        # Show parsed information
        if response.differential:
            print("\n   📊 Parsed Differential:")
            for d in response.differential:
                print(f"      - {d['organism']}: {d.get('confidence', '?')}%")

        if response.is_question:
            print(f"\n   ❓ Question detected: {response.question}")

        if response.final_diagnosis:
            print(f"\n   ✅ Final diagnosis: {response.final_diagnosis}")
            print(f"   🎯 Confidence: {response.confidence}")

        if response.treatment_recommendation:
            print(f"\n   💊 Treatment: {response.treatment_recommendation}")

            # Check safety
            drugs = extract_drugs_from_text(response.treatment_recommendation)
            guardrails = SafetyGuardrails(
                patient_allergies=[],
                identified_organism=case.get("organism"),
            )
            violations = guardrails.check_recommendation(drugs)
            if violations:
                print(f"\n   ⚠️ SAFETY ALERTS:")
                for v in violations:
                    print(f"      - {v.message}")

        # Check if done
        if response.final_diagnosis and not response.is_question:
            print_separator()
            print("🏁 Agent reached final diagnosis")
            break

        if not auto_mode:
            user_input = input("\nPress Enter to continue, 'q' to quit, or enter custom response: ")
            if user_input.lower() == 'q':
                break
            if user_input:
                # Custom environment response
                env_response = user_input
            else:
                # Generate environment response
                if response.is_question:
                    env_response = env.process_query(response.raw_text)
                else:
                    # Advance time and provide update
                    env.state.current_hour += 6
                    if env.state.current_hour >= 48:
                        env_response = env._respond_culture()
                    else:
                        env_response = env.advance_time(6)
        else:
            # Auto mode - generate response
            if response.is_question:
                env_response = env.process_query(response.raw_text)
            else:
                env.state.current_hour += 6
                if env.state.current_hour >= 48:
                    env_response = env._respond_culture()
                else:
                    env_response = env.advance_time(6)

        print(f"\n📋 ENVIRONMENT:")
        print(f"   {env_response}")

        agent.add_environment_message(env_response)

    # Final summary
    print_separator()
    print("📊 SESSION SUMMARY")
    print(f"   Total turns: {turn}")
    print(f"   Ground truth: {case.get('organism', 'Unknown')}")
    if response.final_diagnosis:
        correct = case.get('organism', '').lower() in response.final_diagnosis.lower()
        status = "✅ CORRECT" if correct else "❌ INCORRECT"
        print(f"   Agent diagnosis: {response.final_diagnosis} {status}")
    print_separator()


def main():
    parser = argparse.ArgumentParser(description="Run BSI agent interactively")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained model",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--cases_path",
        type=str,
        default="data/processed/test_cases.jsonl",
        help="Path to cases file",
    )
    parser.add_argument(
        "--case_index",
        type=int,
        default=None,
        help="Specific case index to use",
    )
    parser.add_argument(
        "--use_openai",
        action="store_true",
        help="Use OpenAI API instead of local model",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run in auto mode without pausing",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum turns",
    )

    args = parser.parse_args()

    # Load cases
    cases_path = Path(args.cases_path)
    if not cases_path.exists():
        print(f"Cases file not found: {cases_path}")
        print("\nTo test without MIMIC data, create a sample case file.")
        return

    cases = []
    with open(cases_path, "r") as f:
        for line in f:
            cases.append(json.loads(line))

    if not cases:
        print("No cases found in file")
        return

    # Select case
    if args.case_index is not None:
        if args.case_index >= len(cases):
            print(f"Case index {args.case_index} out of range (0-{len(cases)-1})")
            return
        case = cases[args.case_index]
    else:
        case = random.choice(cases)

    print(f"Selected case: {case.get('case_id', 'unknown')}")

    # Initialize agent
    if args.use_openai:
        print(f"Using OpenAI agent: {args.openai_model}")
        agent = OpenAIAgent(model=args.openai_model)
    elif args.model_path:
        print(f"Loading local model: {args.model_path}")
        config = AgentConfig(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            load_in_4bit=True,
        )
        agent = BSIAgent(config)
        agent.load_model()
    else:
        print("Error: Must specify --model_path or --use_openai")
        return

    # Run session
    run_interactive_session(
        agent,
        case,
        auto_mode=args.auto,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
