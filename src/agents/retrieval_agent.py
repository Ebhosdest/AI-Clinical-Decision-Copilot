"""Evidence retrieval agent: multi-query RAG search with deduplication."""

import json
import re
from typing import Any

from openai import OpenAI

from src.agents.state import AgentState
from src.rag.retriever import Retriever

QUERY_GEN_PROMPT = """You are a clinical information retrieval specialist.
Given patient information, generate 3-5 targeted search queries to retrieve relevant clinical guidelines.

Respond with a JSON array of query strings only (no markdown, no code fences):
["query 1", "query 2", ...]

Focus on:
- Primary diagnosis considerations
- Key symptoms and their differentials
- Relevant investigations
- Treatment guidelines"""


class RetrievalAgent:
    """Generates multiple search queries and retrieves deduplicated evidence."""

    def __init__(self, retriever: Retriever, api_key: str, model: str = "gpt-4o"):
        self._retriever = retriever
        self._client = OpenAI(api_key=api_key)
        self.model = model

    def _generate_queries(self, state: AgentState) -> list[str]:
        """Generate targeted search queries from patient context."""
        patient_info = state.get("patient_info", {})
        query = state.get("query", "")
        context = f"Original query: {query}\nPatient info: {json.dumps(patient_info)}"
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": QUERY_GEN_PROMPT},
                    {"role": "user", "content": context},
                ],
                temperature=0.3,
            )
            content = response.choices[0].message.content or "[]"
            content = re.sub(r"```(?:json)?\s*", "", content).strip().rstrip("`")
            queries = json.loads(content)
            if not isinstance(queries, list):
                raise ValueError("Expected list")
            return [str(q) for q in queries[:5]]
        except Exception:
            return [query]

    def _deduplicate(self, all_results: list[Any]) -> list[Any]:
        """Remove duplicate chunks by document_id + chunk_index."""
        seen: set[str] = set()
        unique = []
        for r in sorted(all_results, key=lambda x: -x.score):
            key = f"{r.chunk.document_id}:{r.chunk.chunk_index}"
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique

    def run(self, state: AgentState) -> AgentState:
        """Execute multi-query retrieval and populate state with evidence."""
        state["current_agent"] = "retrieval"
        queries = self._generate_queries(state)
        state["search_queries"] = queries

        all_results = []
        for q in queries:
            results = self._retriever.retrieve(q)
            all_results.extend(results)

        unique_results = self._deduplicate(all_results)
        top_results = unique_results[: self._retriever.top_k]

        state["retrieved_evidence"] = self._retriever.format_context(top_results)
        state["raw_results"] = [
            {
                "source": r.chunk.source,
                "document_id": r.chunk.document_id,
                "section": r.chunk.section,
                "score": r.score,
            }
            for r in top_results
        ]
        return state
