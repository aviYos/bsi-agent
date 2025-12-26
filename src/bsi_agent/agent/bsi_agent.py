"""BSI Diagnostic Agent implementation."""

import re
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from .prompts import (
    SYSTEM_PROMPT,
    ANTIBIOGRAM_CONTEXT,
    TREATMENT_GUIDELINES_CONTEXT,
    build_agent_prompt,
)


@dataclass
class AgentResponse:
    """Structured response from the BSI agent."""

    raw_text: str
    is_question: bool
    question: Optional[str] = None
    differential: list[dict] = field(default_factory=list)
    final_diagnosis: Optional[str] = None
    confidence: Optional[float] = None
    treatment_recommendation: Optional[str] = None
    reasoning: str = ""


@dataclass
class AgentConfig:
    """Configuration for the BSI agent."""

    model_path: str
    adapter_path: Optional[str] = None
    device: str = "cuda"
    load_in_4bit: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    include_antibiogram: bool = True
    include_guidelines: bool = True


class BSIAgent:
    """
    LLM-based agent for BSI diagnosis.

    This agent interacts with an EHR environment to gather patient data
    and progressively narrows down the likely pathogen.
    """

    def __init__(self, config: AgentConfig):
        """
        Initialize the BSI agent.

        Args:
            config: Agent configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.conversation_history: list[dict] = []

    def load_model(self) -> None:
        """Load the model and tokenizer."""
        print(f"Loading model from {self.config.model_path}...")

        # Quantization config for QLoRA
        if self.config.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        # Load LoRA adapter if specified
        if self.config.adapter_path:
            print(f"Loading adapter from {self.config.adapter_path}...")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.adapter_path,
            )

        self.model.eval()
        print("Model loaded successfully.")

    def reset(self) -> None:
        """Reset conversation history for a new case."""
        self.conversation_history = []

    def add_environment_message(self, content: str) -> None:
        """Add an environment message to history."""
        self.conversation_history.append({
            "role": "environment",
            "content": content,
        })

    def _build_messages(self) -> list[dict]:
        """Build messages list for chat template."""
        messages = [{"role": "system", "content": self._get_system_content()}]

        for turn in self.conversation_history:
            role = "assistant" if turn["role"] == "agent" else "user"
            messages.append({"role": role, "content": turn["content"]})

        return messages

    def _get_system_content(self) -> str:
        """Get full system content with optional contexts."""
        parts = [SYSTEM_PROMPT]

        if self.config.include_antibiogram:
            parts.append(ANTIBIOGRAM_CONTEXT)

        if self.config.include_guidelines:
            parts.append(TREATMENT_GUIDELINES_CONTEXT)

        return "\n\n".join(parts)

    def generate_response(self) -> AgentResponse:
        """
        Generate the next agent response.

        Returns:
            AgentResponse with parsed content
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Build prompt using chat template
        messages = self._build_messages()

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode response (only the new tokens)
        response_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Add to history
        self.conversation_history.append({
            "role": "agent",
            "content": response_text,
        })

        # Parse response
        return self._parse_response(response_text)

    def _parse_response(self, text: str) -> AgentResponse:
        """Parse agent response into structured format."""
        response = AgentResponse(raw_text=text, is_question=False)

        # Check if it's a question
        question_patterns = [
            r"(?:what|are there|is there|can you|could you|do we have|have any)[^?]*\?",
            r"(?:i'?d like to know|i need|please provide)[^.]*[.?]",
        ]

        text_lower = text.lower()
        for pattern in question_patterns:
            match = re.search(pattern, text_lower)
            if match:
                response.is_question = True
                response.question = match.group(0)
                break

        # Extract differential diagnosis
        diff_pattern = r"(\d+)\.\s*([A-Za-z\s]+)\s*[-–]\s*(\d+)%?\s*[-–]?\s*(.*?)(?=\d+\.|$|\n\n)"
        diff_matches = re.findall(diff_pattern, text, re.IGNORECASE)

        for match in diff_matches:
            try:
                response.differential.append({
                    "rank": int(match[0]),
                    "organism": match[1].strip(),
                    "confidence": int(match[2]),
                    "reasoning": match[3].strip(),
                })
            except (ValueError, IndexError):
                continue

        # Extract final diagnosis
        final_patterns = [
            r"(?:final diagnosis|most likely pathogen|confirmed):\s*([A-Za-z\s]+)",
            r"blood culture.*?(?:grew|positive for|identified)\s*([A-Za-z\s]+)",
        ]

        for pattern in final_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                response.final_diagnosis = match.group(1).strip()
                break

        # Extract confidence
        conf_match = re.search(r"confidence:?\s*(\d+)%?", text, re.IGNORECASE)
        if conf_match:
            response.confidence = int(conf_match.group(1)) / 100.0

        # Extract treatment recommendation
        treatment_patterns = [
            r"(?:recommend|suggest|treatment|antibiotic)[^:]*:\s*([^\n]+)",
            r"(?:start|continue|give)\s+([A-Za-z\-]+(?:\s+(?:and|plus|\+)\s+[A-Za-z\-]+)?)",
        ]

        for pattern in treatment_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                response.treatment_recommendation = match.group(1).strip()
                break

        # Extract reasoning (everything after "reasoning:" or "because")
        reasoning_match = re.search(
            r"(?:reasoning|because|rationale)[:\s]+(.+?)(?=\n\n|treatment|$)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            response.reasoning = reasoning_match.group(1).strip()

        return response

    def get_logits_for_organisms(
        self,
        organisms: list[str],
    ) -> dict[str, float]:
        """
        Get model's probability for each organism.

        Useful for calibration evaluation.

        Args:
            organisms: List of organism names to evaluate

        Returns:
            Dict mapping organism name to probability
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        # Build prompt asking for most likely organism
        messages = self._build_messages()
        messages.append({
            "role": "user",
            "content": "What is the single most likely organism? Reply with just the organism name.",
        })

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            probs = torch.softmax(logits, dim=-1)

        # Get probability for each organism's first token
        organism_probs = {}
        for org in organisms:
            org_tokens = self.tokenizer.encode(org, add_special_tokens=False)
            if org_tokens:
                first_token_prob = probs[org_tokens[0]].item()
                organism_probs[org] = first_token_prob

        # Normalize
        total = sum(organism_probs.values())
        if total > 0:
            organism_probs = {k: v / total for k, v in organism_probs.items()}

        return organism_probs


class OpenAIAgent:
    """
    Agent using OpenAI API (GPT-4) for inference.

    Useful for comparison or when local model not available.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        include_antibiogram: bool = True,
        include_guidelines: bool = True,
    ):
        """Initialize OpenAI agent."""
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.include_antibiogram = include_antibiogram
        self.include_guidelines = include_guidelines
        self.conversation_history: list[dict] = []

    def reset(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []

    def add_environment_message(self, content: str) -> None:
        """Add environment message to history."""
        self.conversation_history.append({
            "role": "user",
            "content": f"[ENVIRONMENT]: {content}",
        })

    def generate_response(self) -> AgentResponse:
        """Generate response using OpenAI API."""
        system_parts = [SYSTEM_PROMPT]
        if self.include_antibiogram:
            system_parts.append(ANTIBIOGRAM_CONTEXT)
        if self.include_guidelines:
            system_parts.append(TREATMENT_GUIDELINES_CONTEXT)

        messages = [
            {"role": "system", "content": "\n\n".join(system_parts)},
            *self.conversation_history,
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )

        response_text = response.choices[0].message.content

        self.conversation_history.append({
            "role": "assistant",
            "content": response_text,
        })

        # Use same parsing logic
        agent = BSIAgent.__new__(BSIAgent)
        return agent._parse_response(response_text)
