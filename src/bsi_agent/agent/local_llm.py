# bsi_agent/utils/local_llm.py

from typing import List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalChatModel:
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        torch_dtype=torch.float16,
        max_new_tokens: int = 512,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype if "cuda" in self.device else torch.float32,
            device_map="auto" if "cuda" in self.device else None,
            trust_remote_code=True,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        max_new_tokens = max_new_tokens or self.max_new_tokens

        templ = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if isinstance(templ, torch.Tensor):
            input_ids = templ.to(self.model.device)
            attention_mask = torch.ones_like(input_ids)
        else:
            input_ids = templ["input_ids"].to(self.model.device)
            attention_mask = templ.get("attention_mask", torch.ones_like(input_ids)).to(self.model.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_len = input_ids.shape[1]
        generated_ids = gen_ids[0][input_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return text.strip()


# still in bsi_agent/utils/local_llm.py

import os
from openai import OpenAI


class OpenAIChatBackend:
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content.strip()
