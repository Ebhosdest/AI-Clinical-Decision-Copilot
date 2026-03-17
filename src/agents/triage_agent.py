"""Triage agent: classifies urgency and extracts structured patient information."""

import json
import re
from typing import Any

from openai import OpenAI

from src.agents.state import AgentState, PatientInfo

TRIAGE_PROMPT = """You are a clinical triage AI. Analyze the clinical query and extract structured information.

Respond with a single JSON object (no markdown, no code fences):
{
  "urgency": "emergency|urgent|routine|information_request",
  "routing": "fast_track|standard",
  "patient_info": {
    "age": <int or null>,
    "sex": "<male|female|unknown>",
    "symptoms": ["<symptom1>", ...],
    "duration": "<e.g. 2 hours|3 days|chronic|null>",
    "vitals": {},
    "comorbidities": [],
    "medications": [],
    "allergies": []
  },
  "reasoning": "<brief triage rationale>"
}

Urgency definitions:
- emergency: life-threatening (STEMI, stroke, anaphylaxis, sepsis, PE)
- urgent: requires same-day evaluation
- routine: can be managed in standard timeframe
- information_request: general clinical question, no specific patient"""

_EMERGENCY_KEYWORDS = {
    "stemi", "mi", "heart attack", "stroke", "anaphylaxis", "sepsis",
    "pulmonary embolism", "pe", "aortic dissection", "cardiac arrest",
    "respiratory failure", "unconscious", "unresponsive", "chest pain",
    "shortness of breath", "difficulty breathing",
}


def _is_likely_emergency(query: str) -> bool:
    lower = query.lower()
    return any(kw in lower for kw in _EMERGENCY_KEYWORDS)


class TriageAgent:
    """Analyzes queries to determine urgency and extract structured patient data."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Extract JSON from model response, stripping markdown fences if present."""
        content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
        return json.loads(content)

    def run(self, state: AgentState) -> AgentState:
        """Classify urgency and extract patient info; update state in-place."""
        query = state.get("query", "")
        state["current_agent"] = "triage"

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TRIAGE_PROMPT},
                    {"role": "user", "content": f"Clinical query: {query}"},
                ],
                temperature=0.1,
                response_format={"type": "json_object"},
            )
            data = self._parse_response(response.choices[0].message.content or "{}")
            state["urgency"] = data.get("urgency", "routine")
            state["routing"] = data.get("routing", "standard")
            state["patient_info"] = data.get("patient_info", {})

        except Exception as exc:
            # Fallback: keyword-based triage without API call
            state["urgency"] = "emergency" if _is_likely_emergency(query) else "routine"
            state["routing"] = "fast_track" if state["urgency"] == "emergency" else "standard"
            state["patient_info"] = {}
            state["error"] = f"Triage LLM error (fallback used): {exc}"

        return state
