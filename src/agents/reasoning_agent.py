"""Clinical reasoning agent: differential diagnosis with evidence-based reasoning."""

import json
import re

from openai import OpenAI

from src.agents.state import AgentState, DiagnosisEntry

REASONING_PROMPT = """You are a senior clinical physician performing differential diagnosis reasoning.

Given patient information and retrieved clinical guidelines, provide:
1. A ranked differential diagnosis list
2. Reasoning chains for each diagnosis
3. Red flags and urgent findings

Respond with a JSON object (no markdown, no code fences):
{
  "reasoning": "<step-by-step clinical reasoning>",
  "diagnoses": [
    {
      "diagnosis": "<diagnosis name>",
      "likelihood": "high|medium|low",
      "reasoning": "<evidence-based reasoning>",
      "red_flags": ["<flag1>", ...]
    }
  ],
  "urgent_flags": ["<any immediate action required>"],
  "key_investigations": ["<investigation1>", ...]
}

Rules:
- Base all reasoning on the retrieved guideline evidence
- Cite specific guideline recommendations where relevant
- Rank diagnoses from most to least likely
- Explicitly identify any emergency criteria"""


class ReasoningAgent:
    """Performs differential diagnosis reasoning using patient data and evidence."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def run(self, state: AgentState) -> AgentState:
        """Generate differential diagnoses and reasoning chains."""
        state["current_agent"] = "reasoning"
        query = state.get("query", "")
        patient_info = state.get("patient_info", {})
        evidence = state.get("retrieved_evidence", "No evidence retrieved.")

        user_content = (
            f"Patient Query: {query}\n\n"
            f"Patient Information:\n{json.dumps(patient_info, indent=2)}\n\n"
            f"Retrieved Clinical Evidence:\n{evidence}"
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REASONING_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.15,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
            data = json.loads(content)
            state["reasoning"] = data.get("reasoning", "")
            state["diagnoses"] = data.get("diagnoses", [])

        except Exception as exc:
            state["reasoning"] = f"Reasoning error: {exc}"
            state["diagnoses"] = []
            state["error"] = str(exc)

        return state
