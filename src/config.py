"""Centralized configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    score_threshold: float
    log_level: str
    vector_index_path: str
    guidelines_path: str


def load_config() -> Config:
    """Load and validate configuration from environment."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or api_key == "your-openai-api-key-here":
        raise ValueError("OPENAI_API_KEY not set. Configure it in .env")

    return Config(
        openai_api_key=api_key,
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        top_k=int(os.getenv("TOP_K", "5")),
        score_threshold=float(os.getenv("SCORE_THRESHOLD", "0.3")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        vector_index_path=os.getenv("VECTOR_INDEX_PATH", "data/vector_index"),
        guidelines_path=os.getenv("GUIDELINES_PATH", "data/guidelines"),
    )


# Module-level singleton — raises on import if key missing
try:
    config = load_config()
except ValueError:
    config = None  # type: ignore[assignment]
