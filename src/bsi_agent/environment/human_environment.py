"""Human-in-the-loop environment simulated with GPT models."""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from openai import OpenAI

from bsi_agent.agent.prompts import ENVIRONMENT_SYSTEM_PROMPT


class HumanEnvironment:
    """Simulate a human clinician presenting the case using an OpenAI model."""

    _LIST_FIELD_LIMITS = {
        "labs": 12,
        "vitals": 12,
        "medications": 10,
        "imaging": 6,
        "procedures": 6,
        "notes": 6,
    }

    _LIST_FIELD_KEYS = {
        "labs": {"lab_name", "valuenum", "value", "valueuom", "charttime", "flag", "priority"},
        "vitals": {"vital_name", "valuenum", "value", "valueuom", "charttime"},
        "medications": {"drug", "dose_val_rx", "dose_unit_rx", "route", "starttime", "stoptime"},
        "imaging": {"imaging_type", "description", "time"},
        "procedures": {"procedure", "time", "notes"},
        "notes": {"timestamp", "author", "category", "summary"},
    }

    def __init__(
        self,
        case: dict,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        self.case = case
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key)
        self._system_prompt = self._build_system_prompt(case)
        self._conversation_history: list[dict[str, str]] = []

    def reset(self) -> None:
        """Reset conversation state."""
        self._conversation_history = []

    def _build_system_prompt(self, case: dict) -> str:
        case_summary = json.dumps(self._compress_case(case), indent=2)
        susceptibilities = (
            json.dumps(self._sanitize(case.get("susceptibilities", {})), indent=2)
            if case.get("susceptibilities")
            else "Pending"
        )
        return ENVIRONMENT_SYSTEM_PROMPT.format(
            case_summary=case_summary,
            organism=case.get("organism", "Unknown"),
            gram_stain=case.get("gram_stain", "Pending"),
            susceptibilities=susceptibilities,
        )

    def _compress_case(self, case: dict) -> dict[str, Any]:
        """Reduce raw case size so prompts stay well below token limits."""
        summary: dict[str, Any] = {}
        for key, value in case.items():
            if isinstance(value, list):
                limit = self._LIST_FIELD_LIMITS.get(key, 0)
                if limit and value:
                    truncated = [
                        self._sanitize(self._select_fields(key, entry))
                        for entry in value[:limit]
                    ]
                    summary[f"{key}_sample"] = truncated
                    if len(value) > limit:
                        summary[f"{key}_omitted_count"] = len(value) - limit
            else:
                summary[key] = self._sanitize(value)

        # Highlight core clinical context up front for easier grounding.
        summary["key_highlights"] = self._build_highlights(case)
        return summary

    def _select_fields(self, key: str, entry: dict[str, Any]) -> dict[str, Any]:
        allowed = self._LIST_FIELD_KEYS.get(key)
        if not allowed:
            return entry
        return {field: entry.get(field) for field in allowed if field in entry}

    def _build_highlights(self, case: dict) -> dict[str, Any]:
        """Extract a concise snapshot of the case for the model."""
        highlights: dict[str, Any] = {}
        demographics = {
            field: case.get(field)
            for field in ("case_id", "age", "gender", "admission_type", "admission_location")
            if case.get(field) is not None
        }
        if demographics:
            highlights["demographics"] = demographics

        highlights["timeline"] = {
            "culture_time": case.get("culture_time"),
            "admit_time": case.get("admit_time"),
        }

        # Provide a quick glance at notable labs/vitals/meds without overwhelming tokens.
        highlights["notable_labs"] = self._collect_unique_entries(
            case.get("labs", []), label_field="lab_name", value_field="valuenum", unit_field="valueuom"
        )
        highlights["notable_vitals"] = self._collect_unique_entries(
            case.get("vitals", []), label_field="vital_name", value_field="valuenum", unit_field="valueuom"
        )
        highlights["active_medications"] = self._collect_unique_entries(
            case.get("medications", []), label_field="drug", value_field="dose_val_rx", unit_field="dose_unit_rx"
        )
        return highlights

    def _collect_unique_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        label_field: str,
        value_field: str,
        unit_field: str,
        max_items: int = 6,
    ) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for entry in entries or []:
            label = entry.get(label_field)
            if not label or label in seen_labels:
                continue
            seen_labels.add(label)
            collected.append({
                "name": label,
                "value": self._sanitize(entry.get(value_field)),
                "unit": self._sanitize(entry.get(unit_field)),
                "time": self._sanitize(entry.get("charttime") or entry.get("storetime") or entry.get("starttime")),
            })
            if len(collected) >= max_items:
                break
        return collected

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, str):
            return value.replace("°", " deg ")
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        if isinstance(value, dict):
            return {key: self._sanitize(val) for key, val in value.items()}
        return value

    def get_initial_presentation(self) -> str:
        """Generate an initial presentation from the clinician."""
        self.reset()
        messages = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    "Provide the initial consultation summary for this case. "
                    "Include demographics, admission context, vital signs, and the most pertinent labs available at time zero."
                ),
            },
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content.strip()
        self._conversation_history.append({"role": "assistant", "content": content})
        return content

    def process_query(self, query: str) -> str:
        """Respond to a consultant question using the simulated clinician."""
        self._conversation_history.append({"role": "user", "content": query})
        messages = [{"role": "system", "content": self._system_prompt}, *self._conversation_history]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content.strip()
        self._conversation_history.append({"role": "assistant", "content": content})
        return content
