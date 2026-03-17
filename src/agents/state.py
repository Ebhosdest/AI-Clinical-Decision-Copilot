"""Agent state definition for the LangGraph clinical workflow."""

from typing import Any, Optional
from typing_extensions import TypedDict


class PatientInfo(TypedDict, total=False):
    age: Optional[int]
    sex: Optional[str]
    symptoms: list[str]
    duration: Optional[str]
    vitals: dict[str, Any]
    comorbidities: list[str]
    medications: list[str]
    allergies: list[str]


class DiagnosisEntry(TypedDict):
    diagnosis: str
    likelihood: str  # high / medium / low
    reasoning: str
    red_flags: list[str]


class AgentState(TypedDict, total=False):
    query: str
    patient_info: PatientInfo
    urgency: str                       # emergency | urgent | routine | information_request
    routing: str                       # fast_track | standard
    search_queries: list[str]
    retrieved_evidence: str            # formatted context string
    raw_results: list[dict]            # raw SearchResult metadata
    reasoning: str
    diagnoses: list[DiagnosisEntry]
    response: str
    sources: list[str]
    current_agent: str
    error: Optional[str]
