"""Retrieval service: query embedding → FAISS search → formatted context."""

from src.rag.embeddings import EmbeddingService
from src.rag.vector_store import SearchResult, VectorStore


class Retriever:
    """Embeds a query, searches the vector store, and formats context for the LLM."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ):
        self._store = vector_store
        self._embedder = embedding_service
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str) -> list[SearchResult]:
        """Embed query, search FAISS, and filter by score threshold."""
        query_vec = self._embedder.embed_query(query)
        results = self._store.search(query_vec, top_k=self.top_k)
        return [r for r in results if r.score >= self.score_threshold]

    def format_context(self, results: list[SearchResult]) -> str:
        """Format retrieved chunks into a labelled context block for the LLM."""
        if not results:
            return "No relevant clinical guidelines found."

        parts = []
        for i, result in enumerate(results, start=1):
            c = result.chunk
            header = f"[Source {i}: {c.source} | {c.document_id} | Section: {c.section} | Score: {result.score:.3f}]"
            parts.append(f"{header}\n{c.text}")

        return "\n\n---\n\n".join(parts)

    def retrieve_and_format(self, query: str) -> tuple[list[SearchResult], str]:
        """Convenience method returning both raw results and formatted context."""
        results = self.retrieve(query)
        context = self.format_context(results)
        return results, context
