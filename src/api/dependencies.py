"""FastAPI dependency injection: singleton RAGPipeline and agent graph."""

import logging
from functools import lru_cache
from typing import Optional

from src.config import Config, load_config
from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_rag_pipeline: Optional[RAGPipeline] = None
_compiled_graph = None


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Cached config singleton."""
    return load_config()


def get_rag_pipeline() -> RAGPipeline:
    """Lazy-initialise and return the singleton RAGPipeline."""
    global _rag_pipeline
    if _rag_pipeline is None:
        config = get_config()
        _rag_pipeline = RAGPipeline(config)
        try:
            _rag_pipeline.load_index()
            logger.info("FAISS index loaded successfully.")
        except Exception as exc:
            logger.warning(f"Index not found, run /index/build first: {exc}")
    return _rag_pipeline


def get_agent_graph():
    """Lazy-initialise and return the compiled LangGraph."""
    global _compiled_graph
    if _compiled_graph is None:
        from src.agents.graph import build_graph
        config = get_config()
        pipeline = get_rag_pipeline()
        _compiled_graph = build_graph(config, pipeline)
        logger.info("Agent graph compiled.")
    return _compiled_graph


def reset_pipeline() -> None:
    """Force re-initialisation (called after index rebuild)."""
    global _rag_pipeline, _compiled_graph
    _rag_pipeline = None
    _compiled_graph = None
    get_config.cache_clear()
