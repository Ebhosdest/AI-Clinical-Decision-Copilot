"""LLM response generator with clinical system prompt and streaming support."""

from collections.abc import Iterator
from typing import Optional

from openai import OpenAI

CLINICAL_SYSTEM_PROMPT = """You are a Clinical Decision Support AI assistant. Your role is to provide evidence-based clinical information to healthcare professionals.

CRITICAL SAFETY RULES:
- Only make recommendations supported by the retrieved guidelines context provided.
- Always cite your sources using [Source N] notation referencing the context provided.
- Flag any emergency or life-threatening presentations immediately.
- Never diagnose or prescribe — support clinical decision-making only.
- If the context does not contain sufficient information, explicitly state this limitation.

RESPONSE FORMAT:
Structure every response as follows:

## Clinical Assessment
[Brief summary of the clinical situation]

## Differential Diagnoses
[Ranked list with reasoning, most likely first]

## Recommended Investigations
[Evidence-based workup]

## Management Considerations
[Treatment options per guidelines, with dosing where applicable]

## Sources
[List all cited guideline sources]

## Disclaimer
This information is for qualified healthcare professionals only. Clinical judgement must always be applied. Not a substitute for professional medical advice."""


class Generator:
    """Generates clinical responses using GPT-4o with retrieved guideline context."""

    def __init__(self, api_key: str, model: str = "gpt-4o", temperature: float = 0.2):
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _build_messages(self, query: str, context: str, patient_info: Optional[str] = None) -> list[dict]:
        patient_block = f"\n\nPatient Information:\n{patient_info}" if patient_info else ""
        user_content = (
            f"Clinical Query: {query}"
            f"{patient_block}"
            f"\n\nRetrieved Guideline Context:\n{context}"
            f"\n\nProvide a structured clinical decision support response."
        )
        return [
            {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def generate(self, query: str, context: str, patient_info: Optional[str] = None) -> str:
        """Generate a complete clinical response (non-streaming)."""
        messages = self._build_messages(query, context, patient_info)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content or ""

    def generate_stream(
        self, query: str, context: str, patient_info: Optional[str] = None
    ) -> Iterator[str]:
        """Yield response tokens as they stream from the API."""
        messages = self._build_messages(query, context, patient_info)
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
