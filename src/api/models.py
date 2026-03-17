"""Pydantic request and response schemas for the clinical API."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class ClinicalQuery(BaseModel):
    query: str = Field(..., min_length=5, description="Clinical question or patient presentation")
    patient_info: Optional[dict[str, Any]] = Field(None, description="Optional structured patient data")
    use_agents: bool = Field(True, description="Use multi-agent pipeline (True) or simple RAG (False)")
    stream: bool = Field(False, description="Enable streaming response")


class DiagnosisItem(BaseModel):
    diagnosis: str
    likelihood: str
    reasoning: str
    red_flags: list[str] = Field(default_factory=list)


class ClinicalResponse(BaseModel):
    query: str
    response: str
    sources: list[str] = Field(default_factory=list)
    urgency: Optional[str] = None
    diagnoses: list[DiagnosisItem] = Field(default_factory=list)
    num_chunks_retrieved: int = 0
    agent_path: Optional[list[str]] = None


class IndexBuildRequest(BaseModel):
    guidelines_path: Optional[str] = None


class IndexBuildResponse(BaseModel):
    status: str
    message: str
    chunks_indexed: int = 0


class HealthCheck(BaseModel):
    status: str
    index_loaded: bool
    index_size: int
    model: str
    version: str = "1.0.0"


class StreamEvent(BaseModel):
    event: str   # token | done | error
    data: str
    metadata: Optional[dict[str, Any]] = None
