"""LangGraph workflow definition for the multi-agent clinical pipeline."""

from typing import Literal

from langgraph.graph import END, START, StateGraph

from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.response_agent import ResponseAgent
from src.agents.state import AgentState
from src.agents.triage_agent import TriageAgent
from src.config import Config
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import Retriever


def _route_after_triage(state: AgentState) -> Literal["retrieval", "fast_track_retrieval"]:
    """Route emergencies to fast-track path, all others to standard."""
    if state.get("routing") == "fast_track":
        return "fast_track_retrieval"
    return "retrieval"


def build_graph(config: Config, rag_pipeline: RAGPipeline) -> StateGraph:
    """Construct and compile the LangGraph agent workflow."""
    rag_pipeline._ensure_loaded()
    retriever: Retriever = rag_pipeline._retriever  # type: ignore[assignment]

    triage = TriageAgent(api_key=config.openai_api_key, model=config.chat_model)
    retrieval = RetrievalAgent(
        retriever=retriever,
        api_key=config.openai_api_key,
        model=config.chat_model,
    )
    reasoning = ReasoningAgent(api_key=config.openai_api_key, model=config.chat_model)
    response = ResponseAgent(api_key=config.openai_api_key, model=config.chat_model)

    def triage_node(state: AgentState) -> AgentState:
        return triage.run(state)

    def retrieval_node(state: AgentState) -> AgentState:
        return retrieval.run(state)

    def fast_track_retrieval_node(state: AgentState) -> AgentState:
        """Emergency path: retrieval with original query only (no multi-query expansion)."""
        state["current_agent"] = "fast_track_retrieval"
        query = state.get("query", "")
        results = retriever.retrieve(query)
        state["retrieved_evidence"] = retriever.format_context(results)
        state["raw_results"] = [
            {
                "source": r.chunk.source,
                "document_id": r.chunk.document_id,
                "section": r.chunk.section,
                "score": r.score,
            }
            for r in results
        ]
        return state

    def reasoning_node(state: AgentState) -> AgentState:
        return reasoning.run(state)

    def response_node(state: AgentState) -> AgentState:
        return response.run(state)

    builder = StateGraph(AgentState)

    builder.add_node("triage", triage_node)
    builder.add_node("retrieval", retrieval_node)
    builder.add_node("fast_track_retrieval", fast_track_retrieval_node)
    builder.add_node("reasoning", reasoning_node)
    builder.add_node("response", response_node)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges(
        "triage",
        _route_after_triage,
        {"retrieval": "retrieval", "fast_track_retrieval": "fast_track_retrieval"},
    )
    builder.add_edge("retrieval", "reasoning")
    builder.add_edge("fast_track_retrieval", "reasoning")
    builder.add_edge("reasoning", "response")
    builder.add_edge("response", END)

    return builder.compile()


def run_agent_pipeline(query: str, config: Config, rag_pipeline: RAGPipeline) -> AgentState:
    """Execute the full multi-agent pipeline for a clinical query."""
    graph = build_graph(config, rag_pipeline)
    initial_state: AgentState = {"query": query}
    return graph.invoke(initial_state)
