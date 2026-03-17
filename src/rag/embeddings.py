"""OpenAI embedding service with batch processing."""

import numpy as np
from openai import OpenAI


class EmbeddingService:
    """Generates float32 embeddings via OpenAI text-embedding-3-small."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", batch_size: int = 100):
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.dimensions = 1536

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returning a (N, D) float32 array."""
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self._client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend([e.embedding for e in response.data])
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string, returning a (1, D) float32 array."""
        return self.embed_texts([text])
