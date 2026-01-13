#!/usr/bin/env python3
"""
Pre-culture BSI Agent Evaluation

- BSIAgent:    HF causal LM (fine-tuned with QLoRA) acting as the diagnostic agent.
- HumanEnvironment: GPT-4o-based environment simulating the bedside clinician/patient.
- Evaluation loop: runs pre-culture dialogues and checks if the agent's final
  CURRENT PATHOGEN ESTIMATE matches the ground-truth organism.

Requirements:
    pip install transformers accelerate peft bitsandbytes openai datasets pyyaml tqdm
"""

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ===========================
# Data structures
# ===========================

@dataclass
class AgentDifferentialEntry:
    organism: str
    confidence: float  # 0–1


@dataclass
class AgentResponse:
    raw_text: str
    differential: List[AgentDifferentialEntry]
    final_diagnosis: Optional[str]
    confidence: Optional[float]          # 0–1 for final_diagnosis
    reasoning: str
    treatment_recommendation: Optional[str]
    is_question: bool


# ===========================
# BSIAgent (local fine-tuned model)
# ===========================

from peft import PeftModel

class BSIAgent:
    """
    BSI diagnostic agent based on a base causal LM + LoRA adapter.

    - base_model: e.g. "meta-llama/Meta-Llama-3-8B-Instruct"
    - adapter_path: directory containing adapter_model.safetensors, adapter_config.json, chat_template.jinja, tokenizer.json, ...
    """

    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ):
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"[BSIAgent] Loading base model: {base_model} on device {self.device}...")
        print(f"[BSIAgent] Loading LoRA adapter from: {adapter_path}...")

        # חשוב: לטעון את ה-tokenizer מה-adapter כדי לקבל את ה-chat_template והטוקנים המעודכנים
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path or base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # טוענים מודל בסיס
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
            device_map="auto" if "cuda" in self.device else None,
            trust_remote_code=True,
        )

        # מזריקים את ה-LoRA
        self.model = PeftModel.from_pretrained(
            base,
            adapter_path,
        )

        # System prompt: (אפשר להחליף בזה מהפרויקט שלך)
        self.system_prompt = system_prompt or (
            "You are an Infectious Diseases consultant acting as a pre-culture BSI diagnostic agent.\n"
            "Base your reasoning ONLY on pre-culture clinical information (history, vitals, labs, "
            "risk factors, and optionally Gram stain). You NEVER have access to final culture results.\n\n"
            "For EACH response, you MUST follow this structure:\n"
            "1) First, write 1 short paragraph (around 1 sentence) of clinical reasoning in plain text.\n"
            "2) If you need more information from the clinician, add ONE line starting with:\n"
            '   QUESTION: <a single, concrete clinical question ending with a question mark?>\n'
            "   If you do NOT need more information, do NOT include any line starting with 'QUESTION:'.\n"
            "3) At the VERY END of your response, you MUST output a single JSON object, on a new line, "
            "prefixed exactly by:\n"
            "   FINAL_PATHOGEN_ESTIMATE_JSON:\n"
            "   followed immediately by a valid JSON object of the form:\n"
            '   {\"pathogen_estimate\": [\n'
            '       {\"organism\": \"Escherichia coli\", \"confidence\": 0.7},\n'
            '       {\"organism\": \"Klebsiella pneumoniae\", \"confidence\": 0.2},\n'
            '       {\"organism\": \"Other / Unknown\", \"confidence\": 0.1}\n'
            "   ]}\n" 
            "CRITICAL: Always complete the JSON. Never truncate it. The JSON must be valid and parseable, with confidences between 0 and 1, ideally summing to 1.0. Check twice that the json is valid.\n"
        )



        self.history: List[Dict[str, str]] = []


    def reset(self):
        """Clear chat history."""
        self.history = []

    def add_environment_message(self, text: str):
        """Add a message from the environment (clinician/patient) as 'user' side."""
        self.history.append({"role": "user", "content": text})

    def _build_messages(self) -> List[Dict[str, str]]:
        """Build messages list for chat template."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        return messages

    def generate_response(self) -> AgentResponse:
        """
        Generate a single agent turn and parse it into structured fields.

        Returns:
            AgentResponse with:
                - raw_text
                - differential (parsed from CURRENT PATHOGEN ESTIMATE)
                - final_diagnosis (top organism)
                - confidence (for top organism, 0–1)
                - reasoning (text before the estimate block)
                - treatment_recommendation (if parsed)
                - is_question
        """
        messages = self._build_messages()

        # נבצע chat_template ונוודא שתמיד נקבל input_ids ו־attention_mask
        templ = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if isinstance(templ, torch.Tensor):
            # חלק מהמודלים מחזירים ישירות Tensor
            input_ids = templ.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
        else:
            # BatchEncoding / dict
            input_ids = templ["input_ids"].to(self.model.device)
            if "attention_mask" in templ:
                attention_mask = templ["attention_mask"].to(self.model.device)
            else:
                attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # חיתוך החלק החדש שנוצר אחרי ההקשר
        input_len = input_ids.shape[1]
        generated_ids = gen_ids[0][input_len:]
        raw_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # מוסיפים להיסטוריה כ־assistant
        self.history.append({"role": "assistant", "content": raw_text})

        # פרסינג לפורמט המובנה שלנו
        response = self._parse_agent_output(raw_text)
        return response


    
    import re
    from typing import List, Optional

    @staticmethod
    def _parse_agent_output(text: str) -> AgentResponse:
        """
        Parse the agent's raw text into structured fields.

        Priority:
        1) Extract pathogen_estimate from JSON-like fragments: "organism": "...", "confidence": number
        2) (Optional) Fallback to textual CURRENT PATHOGEN ESTIMATE if needed
        3) Detect QUESTION: ... as question flag
        """

        differential: List[AgentDifferentialEntry] = []
        reasoning = text.strip()
        treatment_recommendation: Optional[str] = None

        # ---- 1) Extract all organism/confidence pairs from the text (JSON-like) ----
        # This will work even if the JSON is truncated at the end.
        pair_pattern = re.compile(
            r'"organism"\s*:\s*"([^"]+)"\s*,\s*"confidence"\s*:\s*([0-9]*\.?[0-9]+)',
            flags=re.IGNORECASE,
        )

        for m in pair_pattern.finditer(text):
            name = m.group(1).strip()
            conf = float(m.group(2))
            differential.append(
                AgentDifferentialEntry(
                    organism=name,
                    confidence=conf,
                )
            )

        # ---- 2) OPTIONAL: if nothing found, fallback to textual CURRENT PATHOGEN ESTIMATE ----
        lower = text.lower()
        if not differential and "current pathogen estimate" in lower:
            idx = lower.rfind("current pathogen estimate")
            reasoning = text[:idx].strip()
            estimate_block = text[idx:].strip()
            lines = estimate_block.splitlines()

            for line in lines[1:]:
                if not line.strip():
                    continue
                m = re.match(
                    r"\s*[-\u2022]?\s*([^:]+):\s*([0-9]+(?:\.[0-9]+)?)\s*%(.*)",
                    line
                )
                if m:
                    label = m.group(1).strip()
                    conf = float(m.group(2)) / 100.0
                    desc = m.group(3).strip()

                    if label.lower().startswith("organism") and desc:
                        organism_name = desc.strip(" .")
                    else:
                        organism_name = label

                    differential.append(
                        AgentDifferentialEntry(organism=organism_name, confidence=conf)
                    )

        # ---- 3) Optional: extract treatment recommendation (if you had this before) ----
        treat_match = re.search(
            r"(recommend(?:ation)?[:\-].+)",
            reasoning,
            flags=re.IGNORECASE | re.DOTALL
        )
        if treat_match:
            treatment_recommendation = treat_match.group(1).strip()

        # ---- 4) Final diagnosis & confidence ----
        final_diagnosis = differential[0].organism if differential else None
        top_conf = differential[0].confidence if differential else None

        # ---- 5) Question detection via QUESTION: ----
        q_match = re.search(r"QUESTION\s*:(.+)", text, flags=re.IGNORECASE | re.DOTALL)
        is_question = q_match is not None

        return AgentResponse(
            raw_text=text,
            differential=differential,
            final_diagnosis=final_diagnosis,
            confidence=top_conf,
            reasoning=reasoning,
            treatment_recommendation=treatment_recommendation,
            is_question=is_question,
        )



# ===========================
# HumanEnvironment (GPT-4o)
# ===========================

class HumanEnvironment:
    """
    GPT-4o-based environment that simulates the bedside clinician/patient.

    Design goals:
    - Presents only pre-culture clinical information.
    - NEVER reveals the organism name or final culture result.
    - May reveal Gram stain if relevant, but not species-level culture result.
    """

    def __init__(
        self,
        case: Dict[str, Any],
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ):
        self.case = case
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        self._history: List[Dict[str, str]] = []  # for environment-internal chat with GPT

        # Precompute a concise clinical summary from the case
        self.case_context = self._build_case_context(case)

        self.system_prompt = (
            "You are simulating the bedside clinician (or the patient) in a bloodstream infection (BSI) case.\n"
            "You know the full clinical details, but you DO NOT reveal final culture results or the exact organism name.\n"
            "You may describe symptoms, vital signs, lab values, comorbidities, prior antibiotics, and risk factors.\n"
            "You may mention that blood cultures were drawn and are pending, and you MAY mention Gram stain findings,\n"
            "but you MUST NOT say what organism grew in the culture.\n"
            "Keep answers concise and clinically realistic."
        )

    @staticmethod
    def _build_case_context(case: Dict[str, Any]) -> str:
        """Build a textual summary of the case from raw fields."""
        parts = []

        age = case.get("age")
        gender = case.get("gender")
        admission_type = case.get("admission_type")

        if age is not None or gender is not None:
            demo = []
            if age is not None:
                demo.append(f"{age}-year-old")
            if gender:
                demo.append(gender)
            parts.append(" ".join(demo))

        if admission_type:
            parts.append(f"Admission type: {admission_type}")

        # Optional: labs and vitals summaries if available
        # [SAFETY]: Filter out culture results from labs to prevent data leaks
        labs = case.get("labs", [])
        if labs:
            lab_strs = []
            for lab in labs[:10]:
                name = lab.get("lab_name", "Unknown")
                
                # Skip explicit culture results in labs if they appear
                if "culture" in name.lower() or "organism" in name.lower():
                    continue

                value = lab.get("valuenum")
                if value is not None:
                    lab_strs.append(f"{name}: {value}")
            if lab_strs:
                parts.append("Labs: " + ", ".join(lab_strs))

        vitals = case.get("vitals", [])
        if vitals:
            vitals_strs = []
            for v in vitals[:5]:
                name = v.get("vital_name", "Unknown")
                value = v.get("valuenum")
                if value is not None:
                    vitals_strs.append(f"{name}: {value}")
            if vitals_strs:
                parts.append("Vitals: " + ", ".join(vitals_strs))

        meds = case.get("medications", [])
        if meds:
            drug_names = list({m.get("drug", "Unknown").split()[0] for m in meds[:5]})
            parts.append("Antibiotics received: " + ", ".join(drug_names))
            
        # [Improvement]: Allow Gram stain (pre-culture) but not Organism (post-culture)
        gram_stain = case.get("gram_stain")
        if gram_stain:
            parts.append(f"Gram Stain: {gram_stain}")

        if not parts:
            parts.append("You are caring for a septic patient; invent realistic details consistent with BSI.")

        return "\n".join(parts)

    def get_initial_presentation(self) -> str:
        """
        Returns an initial natural-language presentation of the case,
        without revealing organism name or final cultures.
        """
        user_prompt = (
            "Using the following case summary, introduce the patient and the situation to an Infectious Diseases consultant "
            "in 2–4 concise sentences. Do NOT mention any specific organism name or final culture results.\n\n"
            f"CASE SUMMARY:\n{self.case_context}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
            max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()
        # Save to internal history
        self._history = [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": user_prompt},
                         {"role": "assistant", "content": text}]
        return text

    def process_query(self, agent_text: str) -> str:
        """
        Given the agent's turn (question / reasoning), respond as the clinician.

        Guarantee:
        - No organism name
        - No explicit final culture result
        """
        # Try to extract explicit QUESTION: ... from the agent's text
        q_match = re.search(r"QUESTION\s*:(.+)", agent_text, flags=re.IGNORECASE | re.DOTALL)
        if q_match:
            question = q_match.group(1).strip()
        else:
            # Fallback: use the entire agent_text if no formatted question is found
            question = agent_text.strip()

        user_prompt = (
            "The consultant asked the following question based on the current case:\n"
            f"\"{question}\"\n\n"
            "Respond as the bedside clinician, providing any additional information that would be "
            "available pre-culture. You may mention Gram stain if appropriate. "
            "Do NOT reveal any specific organism name or final culture result."
        )

        messages = self._history + [{"role": "user", "content": user_prompt}]

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.6,
            max_tokens=256,
        )
        text = resp.choices[0].message.content.strip()
        # Update history
        self._history.append({"role": "user", "content": user_prompt})
        self._history.append({"role": "assistant", "content": text})
        return text



# ===========================
# Evaluation utilities
# ===========================

@dataclass
class CaseEvaluation:
    case_id: str
    ground_truth: str
    predicted_organisms: List[str]
    predicted_confidences: List[float]  # 0–1
    correct_at_1: bool
    correct_at_3: bool
    correct_at_5: bool
    reciprocal_rank: float
    top1_confidence: float
    num_turns: int


@dataclass
class AggregateMetrics:
    num_cases: int
    accuracy_at_1: float
    accuracy_at_3: float
    accuracy_at_5: float
    mrr: float
    avg_confidence_top1: float
    avg_confidence_correct: float
    avg_num_turns: float


def normalize_organism_name(name: str) -> str:
    """Normalize organism string for comparison (lowercase, strip)."""
    return re.sub(r"\s+", " ", name.strip().lower())


def evaluate_single_case(
    case_id: str,
    ground_truth_organism: str,
    agent_differential: List[AgentDifferentialEntry],
    num_turns: int,
) -> CaseEvaluation:
    gt_norm = normalize_organism_name(ground_truth_organism or "unknown")

    preds_sorted = sorted(
        agent_differential,
        key=lambda x: x.confidence if x.confidence is not None else 0.0,
        reverse=True,
    )
    print(preds_sorted)
    organisms = [p.organism for p in preds_sorted]
    print(organisms)
    confs = [p.confidence for p in preds_sorted]
    print(confs)
    correct_at_1 = False
    correct_at_3 = False
    correct_at_5 = False
    reciprocal_rank = 0.0

    if organisms:
        top1_norm = normalize_organism_name(organisms[0])
        # print(top1_norm) # Removed debug print
        correct_at_1 = (top1_norm.split()[0] in gt_norm) or (gt_norm.split()[0] in top1_norm)
        # print(correct_at_1) # Removed debug print
        top3_norms = [normalize_organism_name(o) for o in organisms[:3]]
        correct_at_3 = any(
            (o.split()[0] in gt_norm) or (gt_norm.split()[0] in o)
            for o in top3_norms
        )
        
        # Top-5
        top5_norms = [normalize_organism_name(o) for o in organisms[:5]]
        correct_at_5 = any(
            (o.split()[0] in gt_norm) or (gt_norm.split()[0] in o)
            for o in top5_norms
        )

        # Reciprocal Rank
        for i, o in enumerate(organisms):
            o_norm = normalize_organism_name(o)
            if (o_norm.split()[0] in gt_norm) or (gt_norm.split()[0] in o_norm):
                reciprocal_rank = 1.0 / (i + 1)
                break
    
    top1_confidence = confs[0] if confs else 0.0

    return CaseEvaluation(
        case_id=case_id,
        ground_truth=ground_truth_organism,
        predicted_organisms=organisms,
        predicted_confidences=confs,
        correct_at_1=correct_at_1,
        correct_at_3=correct_at_3,
        correct_at_5=correct_at_5,
        reciprocal_rank=reciprocal_rank,
        top1_confidence=top1_confidence,
        num_turns=num_turns,
    )


def aggregate_results(results: List[CaseEvaluation]) -> AggregateMetrics:
    n = len(results)
    if n == 0:
        return AggregateMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    acc1 = sum(1 for r in results if r.correct_at_1) / n
    acc3 = sum(1 for r in results if r.correct_at_3) / n
    acc5 = sum(1 for r in results if r.correct_at_5) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    avg_turns = sum(r.num_turns for r in results) / n
    avg_conf_top1 = sum(r.top1_confidence for r in results) / n
    
    correct_confidences = [r.top1_confidence for r in results if r.correct_at_1]
    avg_conf_correct = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0

    return AggregateMetrics(
        num_cases=n,
        accuracy_at_1=acc1,
        accuracy_at_3=acc3,
        accuracy_at_5=acc5,
        mrr=mrr,
        avg_confidence_top1=avg_conf_top1,
        avg_confidence_correct=avg_conf_correct,
        avg_num_turns=avg_turns,
    )


def format_metrics_report(metrics: AggregateMetrics) -> str:
    return (
        "=== BSI Agent Evaluation (Pre-culture) ===\n"
        f"Num cases:         {metrics.num_cases}\n"
        f"Top-1 accuracy:    {metrics.accuracy_at_1:.3f}\n"
        f"Top-3 accuracy:    {metrics.accuracy_at_3:.3f}\n"
        f"Top-5 accuracy:    {metrics.accuracy_at_5:.3f}\n"
        f"MRR:               {metrics.mrr:.3f}\n"
        f"Avg Turn 1 Conf:   {metrics.avg_confidence_top1:.3f}\n"
        f"Avg Conf (Correct):{metrics.avg_confidence_correct:.3f}\n"
        f"Avg #agent turns:  {metrics.avg_num_turns:.2f}\n"
    )


# ===========================
# Dialogue runner
# ===========================

def run_agent_on_case(
    agent: BSIAgent,
    case: Dict[str, Any],
    environment_model: str = "gpt-4o",
    environment_api_key: Optional[str] = None,
    max_turns: int = 8,
    min_agent_turns: int = 3,   # אפשר לשנות
) -> Dict[str, Any]:
    """
    Run a complete pre-culture dialogue between the agent and the HumanEnvironment.

    Guarantees:
    - Roles always alternate user/assistant/user/assistant/...
    - final_response הוא התגובה האחרונה של הסוכן.
    """
    env = HumanEnvironment(case, model=environment_model, api_key=environment_api_key)

    agent.reset()
    initial = env.get_initial_presentation()
    agent.add_environment_message(initial)

    dialogue_history: List[Dict[str, str]] = [
        {"role": "environment", "content": initial}
    ]

    final_response: Optional[AgentResponse] = None
    non_question_streak = 0
    num_agent_turns = 0

    for turn in range(max_turns):
        # כאן מובטח שההודעה האחרונה בהיסטוריה היא user
        response = agent.generate_response()
        dialogue_history.append({"role": "agent", "content": response.raw_text})
        num_agent_turns += 1

        # תנאי עצירה "רך": אם הסוכן נותן final_diagnosis אחרי כמה סיבובים
        if num_agent_turns >= min_agent_turns and response.final_diagnosis and not response.is_question:
            final_response = response
            break

        # תנאי עצירה "קשה": הגענו למקסימום סיבובים
        if num_agent_turns == max_turns:
            final_response = response
            break

        if response.is_question:
            # תשובה קלינית "אמיתית" מהסביבה
            non_question_streak = 0
            env_reply = env.process_query(response.raw_text)
        else:
            # לא שאלה – בכל זאת חייבים user turn כדי לשמור alternation
            non_question_streak += 1
            if non_question_streak >= 3 and num_agent_turns >= min_agent_turns:
                # אחרי כמה סיבובים בלי שאלות – נתייחס לזה כסיום
                final_response = response
                break

            # "ניעור" מהקלינאי, כדי להחזיר את התור לסוכן
            if non_question_streak == 1:
                env_reply = (
                    "I follow your reasoning. Do you need any additional pre-culture information "
                    "or can you further refine your pathogen estimate?"
                )
            else:
                env_reply = (
                    "Understood. Please either ask for specific information or provide your final "
                    "pre-culture assessment so we can move forward."
                )

        # בשתי הסיטואציות – חייבים להוסיף את תגובת ה-environment להיסטוריה,
        # כדי שתמיד יהיה user לפני הקריאה הבאה ל-generate_response
        dialogue_history.append({"role": "environment", "content": env_reply})
        agent.add_environment_message(env_reply)

    # אם לא הוגדר final_response במפורש – נשתמש באחרון
    # If we have no explicit final_response, previous code already sets it.
    if final_response is None:
        last_agent_text = next(
            (d["content"] for d in reversed(dialogue_history) if d["role"] == "agent"),
            ""
        )
        final_response = agent._parse_agent_output(last_agent_text)

    # NEW: if there is still no differential, look backwards for the last estimate block
    if not final_response.differential:
        for d in reversed(dialogue_history):
            if d["role"] == "agent" and "CURRENT PATHOGEN ESTIMATE" in d["content"]:
                final_response = agent._parse_agent_output(d["content"])
                break

    num_agent_turns = sum(1 for d in dialogue_history if d["role"] == "agent")

    return {
        "dialogue": dialogue_history,
        "final_response": final_response,
        "num_turns": num_agent_turns,
    }




# ===========================
# Main CLI
# ===========================

def main():
    parser = argparse.ArgumentParser(description="Pre-culture BSI Agent evaluation")
    parser.add_argument("--model_path", type=str, required=False, help="Path to fine-tuned agent model")
    parser.add_argument("--test_cases", type=str, required=True, help="Path to test cases JSONL")
    parser.add_argument("--max_turns", type=int, default=8, help="Maximum agent turns per case")
    parser.add_argument("--environment_model", type=str, default="gpt-4o", help="OpenAI model for HumanEnvironment")
    parser.add_argument("--environment_api_key", type=str, default=None, help="OpenAI API key (or env OPENAI_API_KEY)")
    parser.add_argument("--max_cases", type=int, default=None, help="Max number of cases to evaluate")
    parser.add_argument("--save_dialogues", type=str, default="./outputs/eval_dialogues.jsonl", help="Optional path to save dialogues JSONL")
    parser.add_argument("--save_metrics", type=str, default="./outputs/eval_metrics.json", help="Optional path to save metrics JSON")
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base HF model (e.g. meta-llama/Meta-Llama-3-8B-Instruct)",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory (e.g. outputs/model/final)",
    )

    args = parser.parse_args()

    # Load test cases
    test_cases: List[Dict[str, Any]] = []
    with open(args.test_cases, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            test_cases.append(json.loads(line))

    if args.max_cases is not None:
        test_cases = test_cases[:args.max_cases]

    print(f"Loaded {len(test_cases)} test cases")

    # Init agent
    agent = BSIAgent(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
    )

    results: List[CaseEvaluation] = []
    dialogues_out = []

    for idx, case in enumerate(tqdm(test_cases, desc="Evaluating cases")):
        case_id = case.get("case_id", f"case_{idx}")
        ground_truth = case.get("organism", "Unknown")

        run_result = run_agent_on_case(
            agent=agent,
            case=case,
            environment_model=args.environment_model,
            environment_api_key=args.environment_api_key,
            max_turns=args.max_turns,
        )

        final_response: AgentResponse = run_result["final_response"]
        num_turns = run_result["num_turns"]

        eval_case = evaluate_single_case(
            case_id=case_id,
            ground_truth_organism=ground_truth,
            agent_differential=final_response.differential,
            num_turns=num_turns,
        )
        results.append(eval_case)

        if args.save_dialogues:
            dialogues_out.append({
                "case_id": case_id,
                "ground_truth": ground_truth,
                "dialogue": run_result["dialogue"],
                "final_response": {
                    "raw_text": final_response.raw_text,
                    "differential": [
                        {"organism": d.organism, "confidence": d.confidence}
                        for d in final_response.differential
                    ],
                },
            })

    # Aggregate and print metrics
    metrics = aggregate_results(results)
    print()
    print(format_metrics_report(metrics))

    # Optionally save metrics
    if args.save_metrics:
        metrics_out_path = Path(args.save_metrics)
        metrics_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out_path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"Metrics saved to: {metrics_out_path}")

    # Optionally save dialogues
    if args.save_dialogues:
        out_path = Path(args.save_dialogues)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for d in dialogues_out:
                f.write(json.dumps(d) + "\n")
        print(f"Dialogues saved to: {out_path}")


if __name__ == "__main__":
    main()
