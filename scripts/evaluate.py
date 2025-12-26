#!/usr/bin/env python3
"""
Evaluate the BSI agent on test cases.

This script runs the trained agent on held-out BSI cases and computes
evaluation metrics including accuracy, calibration, and efficiency.
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bsi_agent.agent.bsi_agent import BSIAgent, AgentConfig, OpenAIAgent
from bsi_agent.environment.ehr_environment import EHREnvironment
from bsi_agent.guardrails.safety_checks import SafetyGuardrails, extract_drugs_from_text
from bsi_agent.evaluation.metrics import (
    evaluate_single_case,
    aggregate_results,
    format_metrics_report,
    EvaluationResult,
)


def run_agent_on_case(
    agent,
    case: dict,
    max_turns: int = 10,
    reveal_culture_at_turn: int = 6,
) -> dict:
    """
    Run the agent through a complete BSI case.

    Args:
        agent: BSI agent instance
        case: BSI case dictionary
        max_turns: Maximum dialogue turns
        reveal_culture_at_turn: When to reveal culture results

    Returns:
        Dict with agent's predictions and dialogue history
    """
    # Initialize environment
    env = EHREnvironment(case)

    # Reset agent
    agent.reset()

    # Get initial presentation
    initial = env.get_initial_presentation()
    agent.add_environment_message(initial)

    dialogue_history = [{"role": "environment", "content": initial}]
    environment_messages = [initial]

    final_response = None

    for turn in range(max_turns):
        # Agent generates response
        response = agent.generate_response()
        dialogue_history.append({"role": "agent", "content": response.raw_text})

        # Check if agent is asking a question or giving final diagnosis
        if response.final_diagnosis or (response.differential and len(response.differential) >= 3):
            final_response = response
            break

        # Check if agent asked a question
        if response.is_question:
            # Process the query
            env_response = env.process_query(response.raw_text)

            # Reveal culture results after certain turns
            if turn >= reveal_culture_at_turn - 1:
                env.state.current_hour = 48
                if "culture" in response.raw_text.lower():
                    env_response = env._respond_culture()

            agent.add_environment_message(env_response)
            dialogue_history.append({"role": "environment", "content": env_response})
            environment_messages.append(env_response)

    # If no final response, use the last one
    if final_response is None and dialogue_history:
        final_response = response

    return {
        "dialogue": dialogue_history,
        "environment_messages": environment_messages,
        "final_response": final_response,
        "num_turns": len([d for d in dialogue_history if d["role"] == "agent"]),
    }


def evaluate_agent(
    agent,
    test_cases: list[dict],
    max_turns: int = 10,
    check_safety: bool = True,
) -> list[EvaluationResult]:
    """
    Evaluate agent on multiple test cases.

    Args:
        agent: BSI agent instance
        test_cases: List of BSI case dictionaries
        max_turns: Maximum turns per case
        check_safety: Whether to check safety guardrails

    Returns:
        List of evaluation results
    """
    results = []

    for case in tqdm(test_cases, desc="Evaluating"):
        # Run agent
        run_result = run_agent_on_case(agent, case, max_turns=max_turns)

        # Extract predictions
        final_response = run_result["final_response"]
        differential = []
        reasoning = ""

        if final_response:
            differential = final_response.differential
            reasoning = final_response.reasoning

            # If no differential parsed, create one from final diagnosis
            if not differential and final_response.final_diagnosis:
                differential = [{
                    "organism": final_response.final_diagnosis,
                    "confidence": (final_response.confidence or 0.7) * 100,
                }]

        # Check safety
        safety_violations = 0
        if check_safety and final_response and final_response.treatment_recommendation:
            drugs = extract_drugs_from_text(final_response.treatment_recommendation)
            guardrails = SafetyGuardrails(
                patient_allergies=[],  # Would come from case
                identified_organism=case.get("organism"),
            )
            violations = guardrails.check_recommendation(drugs)
            safety_violations = len(violations)

        # Evaluate
        result = evaluate_single_case(
            case_id=case.get("case_id", "unknown"),
            ground_truth_organism=case.get("organism", "Unknown"),
            agent_differential=differential,
            dialogue_turns=run_result["num_turns"],
            agent_reasoning=reasoning,
            environment_messages=run_result["environment_messages"],
            safety_violations=safety_violations,
        )

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate BSI agent")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model or base model",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (if separate from model)",
    )
    parser.add_argument(
        "--test_cases",
        type=str,
        required=True,
        help="Path to test cases JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation_results.json",
        help="Output path for results",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum dialogue turns per case",
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
        "--max_cases",
        type=int,
        default=None,
        help="Maximum number of cases to evaluate",
    )
    parser.add_argument(
        "--no_safety",
        action="store_true",
        help="Skip safety checks",
    )

    args = parser.parse_args()

    # Load test cases
    print("Loading test cases...")
    test_cases = []
    with open(args.test_cases, "r") as f:
        for line in f:
            test_cases.append(json.loads(line))

    if args.max_cases:
        test_cases = test_cases[:args.max_cases]

    print(f"Loaded {len(test_cases)} test cases")

    # Initialize agent
    if args.use_openai:
        print(f"Using OpenAI agent: {args.openai_model}")
        agent = OpenAIAgent(model=args.openai_model)
    else:
        print(f"Loading local model: {args.model_path}")
        config = AgentConfig(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            load_in_4bit=True,
        )
        agent = BSIAgent(config)
        agent.load_model()

    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate_agent(
        agent,
        test_cases,
        max_turns=args.max_turns,
        check_safety=not args.no_safety,
    )

    # Aggregate metrics
    metrics = aggregate_results(results)

    # Print report
    print("\n" + format_metrics_report(metrics))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "metrics": {
            "num_cases": metrics.num_cases,
            "accuracy_at_1": metrics.accuracy_at_1,
            "accuracy_at_3": metrics.accuracy_at_3,
            "accuracy_at_5": metrics.accuracy_at_5,
            "avg_num_turns": metrics.avg_num_turns,
            "brier_score": metrics.brier_score,
            "avg_grounding_score": metrics.avg_grounding_score,
            "calibration_error": metrics.calibration_error,
            "total_safety_violations": metrics.total_safety_violations,
        },
        "individual_results": [
            {
                "case_id": r.case_id,
                "ground_truth": r.ground_truth_organism,
                "predictions": r.predicted_organisms,
                "confidences": r.predicted_confidences,
                "correct_at_1": r.correct_at_1,
                "correct_at_3": r.correct_at_3,
                "num_turns": r.num_turns,
                "grounding_score": r.grounding_score,
                "safety_violations": r.safety_violations,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
