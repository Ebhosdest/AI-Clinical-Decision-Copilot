"""FastAPI application for the AI Clinical Decision Copilot."""

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.api.dependencies import get_config, get_rag_pipeline, get_agent_graph, reset_pipeline
from src.api.models import (
    ClinicalQuery,
    ClinicalResponse,
    DiagnosisItem,
    HealthCheck,
    IndexBuildRequest,
    IndexBuildResponse,
)
from src.config import Config
from src.rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Clinical Decision Copilot",
    description="Evidence-based clinical decision support powered by RAG + LangGraph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheck)
async def health_check(config: Config = Depends(get_config)) -> HealthCheck:
    """Return service health and index status."""
    try:
        pipeline = get_rag_pipeline()
        index_loaded = pipeline._vector_store is not None
        index_size = pipeline._vector_store.size if pipeline._vector_store else 0
    except Exception:
        index_loaded = False
        index_size = 0
    return HealthCheck(
        status="healthy",
        index_loaded=index_loaded,
        index_size=index_size,
        model=config.chat_model,
    )


@app.post("/query", response_model=ClinicalResponse)
async def clinical_query(
    request: ClinicalQuery,
    config: Config = Depends(get_config),
) -> ClinicalResponse:
    """Process a clinical query through the RAG or agent pipeline."""
    try:
        if request.use_agents:
            graph = get_agent_graph()
            from src.agents.state import AgentState
            initial: AgentState = {"query": request.query}
            if request.patient_info:
                initial["patient_info"] = request.patient_info  # type: ignore[assignment]
            state = graph.invoke(initial)
            diagnoses = [
                DiagnosisItem(**d) for d in state.get("diagnoses", [])
                if isinstance(d, dict)
            ]
            return ClinicalResponse(
                query=request.query,
                response=state.get("response", ""),
                sources=state.get("sources", []),
                urgency=state.get("urgency"),
                diagnoses=diagnoses,
                num_chunks_retrieved=len(state.get("raw_results", [])),
                agent_path=["triage", "retrieval", "reasoning", "response"],
            )
        else:
            pipeline = get_rag_pipeline()
            result = pipeline.query(request.query)
            return ClinicalResponse(
                query=request.query,
                response=result["response"],
                sources=result["sources"],
                num_chunks_retrieved=result["num_chunks_retrieved"],
            )
    except Exception as exc:
        logger.exception("Query processing failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/query/stream")
async def clinical_query_stream(request: ClinicalQuery) -> StreamingResponse:
    """Stream clinical response tokens as Server-Sent Events."""

    async def token_generator() -> AsyncGenerator[str, None]:
        try:
            pipeline = get_rag_pipeline()
            for token in pipeline.query_stream(request.query):
                event = json.dumps({"event": "token", "data": token})
                yield f"data: {event}\n\n"
            yield f"data: {json.dumps({'event': 'done', 'data': ''})}\n\n"
        except Exception as exc:
            error_event = json.dumps({"event": "error", "data": str(exc)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/index/build", response_model=IndexBuildResponse)
async def build_index(
    request: IndexBuildRequest,
    config: Config = Depends(get_config),
) -> IndexBuildResponse:
    """Rebuild the FAISS index from the guidelines directory."""
    try:
        pipeline_config = config
        if request.guidelines_path:
            from dataclasses import replace
            pipeline_config = replace(config, guidelines_path=request.guidelines_path)

        from src.rag.pipeline import RAGPipeline as _Pipeline
        pipeline = _Pipeline(pipeline_config)
        pipeline.build_index()
        reset_pipeline()

        return IndexBuildResponse(
            status="success",
            message=f"Index built from {pipeline_config.guidelines_path}",
            chunks_indexed=pipeline._vector_store.size if pipeline._vector_store else 0,
        )
    except Exception as exc:
        logger.exception("Index build failed")
        raise HTTPException(status_code=500, detail=str(exc))
