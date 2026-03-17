"""RAG orchestrator: index building and end-to-end query execution."""

import sys
from collections.abc import Iterator
from typing import Optional

from src.config import Config, load_config
from src.rag.embeddings import EmbeddingService
from src.rag.generator import Generator
from src.rag.loader import DocumentChunk, load_documents
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore


class RAGPipeline:
    """Orchestrates document indexing and retrieval-augmented generation."""

    def __init__(self, config: Config):
        self.config = config
        self._embedding_service = EmbeddingService(
            api_key=config.openai_api_key,
            model=config.embedding_model,
        )
        self._generator = Generator(
            api_key=config.openai_api_key,
            model=config.chat_model,
        )
        self._vector_store: Optional[VectorStore] = None
        self._retriever: Optional[Retriever] = None

    def build_index(self) -> None:
        """Load documents, embed chunks, and persist the FAISS index."""
        print(f"Loading documents from {self.config.guidelines_path}...")
        chunks: list[DocumentChunk] = load_documents(
            self.config.guidelines_path,
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )
        print(f"Loaded {len(chunks)} chunks. Generating embeddings...")

        texts = [c.text for c in chunks]
        embeddings = self._embedding_service.embed_texts(texts)

        self._vector_store = VectorStore(dimensions=self._embedding_service.dimensions)
        self._vector_store.add(embeddings, chunks)
        self._vector_store.save(self.config.vector_index_path)
        print(f"Index saved to {self.config.vector_index_path} ({self._vector_store.size} vectors).")
        self._init_retriever()

    def load_index(self) -> None:
        """Load an existing FAISS index from disk."""
        self._vector_store = VectorStore.load(
            self.config.vector_index_path,
            dimensions=self._embedding_service.dimensions,
        )
        self._init_retriever()

    def _init_retriever(self) -> None:
        assert self._vector_store is not None
        self._retriever = Retriever(
            vector_store=self._vector_store,
            embedding_service=self._embedding_service,
            top_k=self.config.top_k,
            score_threshold=self.config.score_threshold,
        )

    def _ensure_loaded(self) -> None:
        if self._retriever is None:
            self.load_index()

    def query(self, query: str, patient_info: Optional[str] = None) -> dict:
        """Full retrieve → generate flow. Returns dict with response and sources."""
        self._ensure_loaded()
        assert self._retriever is not None
        results, context = self._retriever.retrieve_and_format(query)
        response = self._generator.generate(query, context, patient_info)
        sources = list({f"{r.chunk.source} ({r.chunk.document_id})" for r in results})
        return {
            "query": query,
            "response": response,
            "sources": sources,
            "num_chunks_retrieved": len(results),
        }

    def query_stream(self, query: str, patient_info: Optional[str] = None) -> Iterator[str]:
        """Streaming version of query(). Yields response tokens."""
        self._ensure_loaded()
        assert self._retriever is not None
        _, context = self._retriever.retrieve_and_format(query)
        yield from self._generator.generate_stream(query, context, patient_info)


def _cli_build(config: Config) -> None:
    pipeline = RAGPipeline(config)
    pipeline.build_index()


def _cli_query(config: Config) -> None:
    query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else input("Enter clinical query: ")
    pipeline = RAGPipeline(config)
    result = pipeline.query(query)
    print("\n" + "=" * 60)
    print(result["response"])
    print("\nSources:", ", ".join(result["sources"]))


if __name__ == "__main__":
    cfg = load_config()
    command = sys.argv[1] if len(sys.argv) > 1 else "query"
    if command == "build":
        _cli_build(cfg)
    else:
        _cli_query(cfg)
