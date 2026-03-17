"""FAISS vector store with cosine similarity and persistent storage."""

import json
import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.rag.loader import DocumentChunk


class SearchResult:
    __slots__ = ("chunk", "score")

    def __init__(self, chunk: DocumentChunk, score: float):
        self.chunk = chunk
        self.score = score


class VectorStore:
    """IndexFlatIP-based vector store (L2-normalised → cosine similarity)."""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimensions)
        self._chunks: list[DocumentChunk] = []

    def add(self, embeddings: np.ndarray, chunks: list[DocumentChunk]) -> None:
        """Normalise and add embeddings; store parallel metadata."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embeddings and chunks must have the same length.")
        normalised = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(normalised)
        self._index.add(normalised)
        self._chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[SearchResult]:
        """Return top_k results sorted by cosine similarity (descending)."""
        vec = query_embedding.copy().astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)
        faiss.normalize_L2(vec)
        k = min(top_k, self._index.ntotal)
        if k == 0:
            return []
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append(SearchResult(chunk=self._chunks[idx], score=float(score)))
        return results

    def save(self, directory: str) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        meta = [
            {
                "text": c.text,
                "source": c.source,
                "document_id": c.document_id,
                "section": c.section,
                "chunk_index": c.chunk_index,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str, dimensions: int = 1536) -> "VectorStore":
        """Load FAISS index and metadata from disk."""
        store = cls(dimensions=dimensions)
        store._index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "metadata.json"), "r", encoding="utf-8") as f:
            meta: list[dict[str, Any]] = json.load(f)
        store._chunks = [
            DocumentChunk(
                text=m["text"],
                source=m["source"],
                document_id=m["document_id"],
                section=m["section"],
                chunk_index=m["chunk_index"],
                metadata=m.get("metadata", {}),
            )
            for m in meta
        ]
        return store

    @property
    def size(self) -> int:
        return self._index.ntotal
