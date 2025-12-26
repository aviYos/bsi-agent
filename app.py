#!/usr/bin/env python3
"""
Gradio UI for BSI Diagnostic Agent.

A clinical interface for interacting with the BSI agent, showing:
- Patient case information
- Agent-Environment dialogue
- Differential diagnosis with confidence
- Safety alerts
"""

import json
import os
from pathlib import Path
from typing import Optional
import random

import gradio as gr

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from bsi_agent.agent.bsi_agent import BSIAgent, AgentConfig, OpenAIAgent
from bsi_agent.environment.ehr_environment import EHREnvironment
from bsi_agent.guardrails.safety_checks import SafetyGuardrails, extract_drugs_from_text


# Global state
class AppState:
    def __init__(self):
        self.agent = None
        self.environment = None
        self.current_case = None
        self.cases = []
        self.dialogue_history = []
        self.differential = []
        self.safety_alerts = []

state = AppState()


def load_cases(cases_path: str) -> list[dict]:
    """Load BSI cases from file."""
    cases = []
    path = Path(cases_path)
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                cases.append(json.loads(line))
    return cases


def format_patient_info(case: dict) -> str:
    """Format patient information for display."""
    if not case:
        return "No case loaded"

    lines = [
        f"**Age:** {case.get('age', 'Unknown')} years",
        f"**Gender:** {case.get('gender', 'Unknown')}",
        f"**Admission:** {case.get('admission_type', 'Unknown')}",
        "",
        "---",
        "",
        f"**Case ID:** {case.get('case_id', 'Unknown')}",
    ]

    # Add some lab highlights if available
    labs = case.get("labs", [])
    if labs:
        lines.append("")
        lines.append("**Key Labs:**")
        for lab in labs[:5]:
            name = lab.get("lab_name", "Unknown")
            value = lab.get("valuenum")
            if value:
                lines.append(f"- {name}: {value:.1f}")

    return "\n".join(lines)


def format_ground_truth(case: dict, show: bool = False) -> str:
    """Format ground truth (hidden by default)."""
    if not case:
        return ""

    if not show:
        return "🔒 *Hidden until case completion*"

    return f"""**Organism:** {case.get('organism', 'Unknown')}

**Gram Stain:** {case.get('gram_stain', 'Unknown')}

**Susceptibilities:**
{json.dumps(case.get('susceptibilities', {}), indent=2)[:500]}
"""


def format_differential(differential: list[dict]) -> str:
    """Format differential diagnosis for display."""
    if not differential:
        return "*No differential yet - agent is gathering information*"

    lines = []
    for i, item in enumerate(differential[:5], 1):
        org = item.get("organism", "Unknown")
        conf = item.get("confidence", 0)
        reasoning = item.get("reasoning", "")[:100]

        # Create a simple bar
        bar_length = int(conf / 10)
        bar = "█" * bar_length + "░" * (10 - bar_length)

        lines.append(f"**{i}. {org}**")
        lines.append(f"   {bar} {conf}%")
        if reasoning:
            lines.append(f"   _{reasoning}_")
        lines.append("")

    return "\n".join(lines)


def format_safety_alerts(alerts: list[str]) -> str:
    """Format safety alerts for display."""
    if not alerts:
        return "✅ No safety concerns"

    lines = ["⚠️ **Safety Alerts:**", ""]
    for alert in alerts:
        lines.append(f"- {alert}")

    return "\n".join(lines)


def initialize_agent(use_openai: bool, openai_model: str, local_model_path: str):
    """Initialize the agent."""
    if use_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "❌ Error: OPENAI_API_KEY not set in environment"
        state.agent = OpenAIAgent(model=openai_model)
        return f"✅ Initialized OpenAI agent ({openai_model})"
    else:
        if not local_model_path or not Path(local_model_path).exists():
            return "❌ Error: Local model path not found"
        config = AgentConfig(
            model_path=local_model_path,
            load_in_4bit=True,
        )
        state.agent = BSIAgent(config)
        state.agent.load_model()
        return f"✅ Initialized local agent ({local_model_path})"


def load_new_case(cases_path: str, case_index: Optional[int] = None):
    """Load a new BSI case."""
    # Load cases if not already loaded
    if not state.cases or cases_path:
        state.cases = load_cases(cases_path)

    if not state.cases:
        return (
            "No cases found",
            "",
            "",
            [],
            "Load cases first"
        )

    # Select case
    if case_index is not None and 0 <= case_index < len(state.cases):
        state.current_case = state.cases[case_index]
    else:
        state.current_case = random.choice(state.cases)

    # Reset state
    state.dialogue_history = []
    state.differential = []
    state.safety_alerts = []

    # Initialize environment
    state.environment = EHREnvironment(state.current_case)

    # Get initial presentation
    initial = state.environment.get_initial_presentation()
    state.dialogue_history.append(("Environment", initial))

    # Initialize agent conversation
    if state.agent:
        state.agent.reset()
        state.agent.add_environment_message(initial)

    return (
        format_patient_info(state.current_case),
        format_ground_truth(state.current_case, show=False),
        format_differential([]),
        state.dialogue_history,
        format_safety_alerts([])
    )


def agent_step():
    """Run one step of the agent."""
    if not state.agent:
        state.dialogue_history.append(("System", "⚠️ Agent not initialized"))
        return (
            state.dialogue_history,
            format_differential(state.differential),
            format_safety_alerts(state.safety_alerts)
        )

    if not state.environment:
        state.dialogue_history.append(("System", "⚠️ No case loaded"))
        return (
            state.dialogue_history,
            format_differential(state.differential),
            format_safety_alerts(state.safety_alerts)
        )

    # Agent generates response
    try:
        response = state.agent.generate_response()
    except Exception as e:
        state.dialogue_history.append(("System", f"❌ Error: {str(e)}"))
        return (
            state.dialogue_history,
            format_differential(state.differential),
            format_safety_alerts(state.safety_alerts)
        )

    # Add agent response to history
    state.dialogue_history.append(("Agent", response.raw_text))

    # Update differential if present
    if response.differential:
        state.differential = response.differential
    elif response.final_diagnosis:
        state.differential = [{
            "organism": response.final_diagnosis,
            "confidence": (response.confidence or 0.7) * 100,
            "reasoning": "Final diagnosis"
        }]

    # Check safety if treatment recommended
    if response.treatment_recommendation:
        drugs = extract_drugs_from_text(response.treatment_recommendation)
        guardrails = SafetyGuardrails(
            patient_allergies=[],
            identified_organism=state.current_case.get("organism"),
        )
        violations = guardrails.check_recommendation(drugs)
        state.safety_alerts = [v.message for v in violations]

    # If agent asked a question, get environment response
    if response.is_question and not response.final_diagnosis:
        env_response = state.environment.process_query(response.raw_text)
        state.dialogue_history.append(("Environment", env_response))
        state.agent.add_environment_message(env_response)

    return (
        state.dialogue_history,
        format_differential(state.differential),
        format_safety_alerts(state.safety_alerts)
    )


def reveal_answer():
    """Reveal the ground truth."""
    return format_ground_truth(state.current_case, show=True)


def send_custom_message(message: str):
    """Send a custom environment message."""
    if not message.strip():
        return state.dialogue_history

    state.dialogue_history.append(("Environment (Manual)", message))
    if state.agent:
        state.agent.add_environment_message(message)

    return state.dialogue_history


def create_demo_case():
    """Create a demo case for testing without MIMIC data."""
    demo_case = {
        "case_id": "DEMO_001",
        "age": 67,
        "gender": "M",
        "admission_type": "EMERGENCY",
        "admission_location": "EMERGENCY ROOM",
        "organism": "STAPHYLOCOCCUS AUREUS",
        "gram_stain": "Gram-positive cocci in clusters",
        "susceptibilities": {
            "VANCOMYCIN": "S",
            "OXACILLIN": "R",
            "DAPTOMYCIN": "S",
            "LINEZOLID": "S",
            "CEFAZOLIN": "R"
        },
        "labs": [
            {"lab_name": "White Blood Cells", "valuenum": 18.5},
            {"lab_name": "Lactate", "valuenum": 3.8},
            {"lab_name": "Creatinine", "valuenum": 1.9},
            {"lab_name": "Procalcitonin", "valuenum": 8.5},
        ],
        "vitals": [
            {"vital_name": "Temperature", "valuenum": 39.2},
            {"vital_name": "Heart Rate", "valuenum": 112},
            {"vital_name": "Blood Pressure Systolic", "valuenum": 88},
        ],
        "medications": [
            {"drug": "VANCOMYCIN"},
            {"drug": "PIPERACILLIN-TAZOBACTAM"},
        ],
        "is_polymicrobial": False,
        "other_organisms": [],
    }

    state.cases = [demo_case]
    return "data/demo_case.jsonl"


# Build Gradio Interface
def build_interface():
    with gr.Blocks(
        title="BSI Diagnostic Agent",
        theme=gr.themes.Soft(),
        css="""
        .differential-box {background-color: #f0f7ff; padding: 15px; border-radius: 8px;}
        .safety-box {background-color: #fff0f0; padding: 15px; border-radius: 8px;}
        .patient-box {background-color: #f0fff0; padding: 15px; border-radius: 8px;}
        """
    ) as demo:
        gr.Markdown("# 🏥 BSI Diagnostic Agent")
        gr.Markdown("Interactive bloodstream infection diagnostic assistant")

        with gr.Row():
            # Left column - Patient Info & Controls
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Patient Information")
                patient_info = gr.Markdown(
                    "No case loaded",
                    elem_classes=["patient-box"]
                )

                gr.Markdown("### 🔬 Ground Truth")
                ground_truth = gr.Markdown(
                    "Load a case first",
                    elem_classes=["patient-box"]
                )
                reveal_btn = gr.Button("🔓 Reveal Answer", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Settings")

                with gr.Accordion("Agent Settings", open=False):
                    use_openai = gr.Checkbox(label="Use OpenAI API", value=True)
                    openai_model = gr.Dropdown(
                        choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                        value="gpt-4o",
                        label="OpenAI Model"
                    )
                    local_model_path = gr.Textbox(
                        label="Local Model Path",
                        placeholder="outputs/model/final"
                    )
                    init_btn = gr.Button("Initialize Agent")
                    init_status = gr.Markdown("")

                with gr.Accordion("Case Loading", open=True):
                    cases_path = gr.Textbox(
                        label="Cases File Path",
                        value="data/processed/test_cases.jsonl"
                    )
                    case_index = gr.Number(
                        label="Case Index (optional)",
                        precision=0
                    )
                    with gr.Row():
                        load_btn = gr.Button("📂 Load Case", variant="primary")
                        demo_btn = gr.Button("🎭 Demo Case")

            # Middle column - Dialogue
            with gr.Column(scale=2):
                gr.Markdown("### 💬 Diagnostic Dialogue")
                chatbot = gr.Chatbot(
                    height=500,
                    show_label=False,
                    avatar_images=["🏥", "🤖"],
                )

                with gr.Row():
                    step_btn = gr.Button("▶️ Agent Step", variant="primary", scale=2)
                    auto_btn = gr.Button("⏩ Auto-Run", scale=1)

                with gr.Accordion("Manual Input", open=False):
                    manual_input = gr.Textbox(
                        label="Send custom environment message",
                        placeholder="Enter custom response..."
                    )
                    send_btn = gr.Button("Send")

            # Right column - Analysis
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Differential Diagnosis")
                differential_display = gr.Markdown(
                    "*No differential yet*",
                    elem_classes=["differential-box"]
                )

                gr.Markdown("### ⚠️ Safety Check")
                safety_display = gr.Markdown(
                    "✅ No safety concerns",
                    elem_classes=["safety-box"]
                )

                gr.Markdown("### 📈 Session Stats")
                stats_display = gr.Markdown(
                    "Turns: 0\nQuestions: 0"
                )

        # Event handlers
        init_btn.click(
            fn=initialize_agent,
            inputs=[use_openai, openai_model, local_model_path],
            outputs=[init_status]
        )

        load_btn.click(
            fn=load_new_case,
            inputs=[cases_path, case_index],
            outputs=[patient_info, ground_truth, differential_display, chatbot, safety_display]
        )

        demo_btn.click(
            fn=create_demo_case,
            outputs=[cases_path]
        ).then(
            fn=load_new_case,
            inputs=[cases_path, case_index],
            outputs=[patient_info, ground_truth, differential_display, chatbot, safety_display]
        )

        step_btn.click(
            fn=agent_step,
            outputs=[chatbot, differential_display, safety_display]
        )

        reveal_btn.click(
            fn=reveal_answer,
            outputs=[ground_truth]
        )

        send_btn.click(
            fn=send_custom_message,
            inputs=[manual_input],
            outputs=[chatbot]
        )

        # Auto-run: multiple steps
        def auto_run():
            results = []
            for _ in range(5):  # Max 5 steps
                result = agent_step()
                results = result
                # Check if done (has final diagnosis)
                if state.differential and any(
                    d.get("reasoning") == "Final diagnosis"
                    for d in state.differential
                ):
                    break
            return results

        auto_btn.click(
            fn=auto_run,
            outputs=[chatbot, differential_display, safety_display]
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
