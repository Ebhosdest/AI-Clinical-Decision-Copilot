"""Unit tests for the RAG pipeline (no API key required)."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.rag.loader import DocumentChunk, _chunk_text, _extract_header, _split_into_sections, load_documents
from src.rag.vector_store import VectorStore


# ── Loader tests ──────────────────────────────────────────────────────────────

class TestExtractHeader:
    def test_extracts_source(self):
        text = "SOURCE: American Heart Association\nSome content here."
        assert _extract_header(text, "SOURCE") == "American Heart Association"

    def test_extracts_document_id(self):
        text = "DOCUMENT_ID: AHA-ACS-2023\nContent."
        assert _extract_header(text, "DOCUMENT_ID") == "AHA-ACS-2023"

    def test_returns_empty_when_missing(self):
        assert _extract_header("No header here.", "SOURCE") == ""

    def test_case_insensitive(self):
        text = "source: WHO\nContent."
        assert _extract_header(text, "SOURCE") == "WHO"


class TestSplitIntoSections:
    def test_splits_on_section_markers(self):
        text = "SECTION: Introduction\nIntro text.\nSECTION: Diagnosis\nDiagnosis text."
        sections = _split_into_sections(text)
        assert len(sections) == 2
        assert sections[0][0] == "Introduction"
        assert sections[1][0] == "Diagnosis"

    def test_fallback_to_general(self):
        text = "No section markers here. Just plain text."
        sections = _split_into_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == "General"

    def test_section_text_is_correct(self):
        text = "SECTION: Overview\nThis is overview text.\nSECTION: Treatment\nThis is treatment."
        sections = _split_into_sections(text)
        assert "overview text" in sections[0][1]
        assert "treatment" in sections[1][1]


class TestChunkText:
    def test_short_text_returns_single_chunk(self):
        text = "Short text."
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_creates_multiple_chunks(self):
        text = "A" * 3000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) > 1

    def test_overlap_creates_overlap(self):
        text = "word " * 500  # 2500 chars
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 1100  # Allow slight overshoot at sentence boundaries

    def test_no_empty_chunks(self):
        text = "Hello world. " * 200
        chunks = _chunk_text(text, chunk_size=500, overlap=100)
        assert all(len(c) > 0 for c in chunks)


class TestLoadDocuments:
    def test_loads_txt_files(self, tmp_path: Path):
        doc = tmp_path / "test.txt"
        doc.write_text(
            "SOURCE: Test Source\nDOCUMENT_ID: TEST-001\n"
            "SECTION: Overview\nThis is section content with enough text to form a chunk.",
            encoding="utf-8",
        )
        chunks = load_documents(str(tmp_path), chunk_size=500, overlap=50)
        assert len(chunks) > 0
        assert chunks[0].source == "Test Source"
        assert chunks[0].document_id == "TEST-001"

    def test_metadata_populated(self, tmp_path: Path):
        doc = tmp_path / "sample.txt"
        doc.write_text(
            "SOURCE: AHA\nDOCUMENT_ID: AHA-001\n"
            "SECTION: Clinical\nClinical content here.",
            encoding="utf-8",
        )
        chunks = load_documents(str(tmp_path))
        assert chunks[0].section == "Clinical"
        assert chunks[0].chunk_index == 0
        assert "file" in chunks[0].metadata

    def test_raises_on_missing_directory(self):
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/path")

    def test_multiple_sections(self, tmp_path: Path):
        doc = tmp_path / "multi.txt"
        doc.write_text(
            "SOURCE: WHO\nDOCUMENT_ID: WHO-001\n"
            "SECTION: Section A\nContent A text here.\n"
            "SECTION: Section B\nContent B text here.\n"
            "SECTION: Section C\nContent C text here.\n",
            encoding="utf-8",
        )
        chunks = load_documents(str(tmp_path))
        sections = {c.section for c in chunks}
        assert "Section A" in sections
        assert "Section B" in sections
        assert "Section C" in sections


# ── VectorStore tests ─────────────────────────────────────────────────────────

class TestVectorStore:
    def _make_chunks(self, n: int) -> list[DocumentChunk]:
        return [
            DocumentChunk(
                text=f"Clinical chunk {i}",
                source=f"Source {i}",
                document_id=f"DOC-{i:03d}",
                section="Test Section",
                chunk_index=i,
            )
            for i in range(n)
        ]

    def _make_embeddings(self, n: int, dims: int = 8) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.random((n, dims), dtype=np.float32)

    def test_add_and_search(self):
        store = VectorStore(dimensions=8)
        chunks = self._make_chunks(5)
        embeddings = self._make_embeddings(5)
        store.add(embeddings, chunks)
        assert store.size == 5

        query = self._make_embeddings(1)
        results = store.search(query, top_k=3)
        assert len(results) == 3
        assert all(hasattr(r, "score") for r in results)
        assert all(hasattr(r, "chunk") for r in results)

    def test_scores_between_minus_one_and_one(self):
        store = VectorStore(dimensions=8)
        chunks = self._make_chunks(10)
        embeddings = self._make_embeddings(10)
        store.add(embeddings, chunks)
        query = self._make_embeddings(1)
        results = store.search(query, top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0

    def test_top_k_respected(self):
        store = VectorStore(dimensions=8)
        store.add(self._make_embeddings(20), self._make_chunks(20))
        results = store.search(self._make_embeddings(1), top_k=5)
        assert len(results) == 5

    def test_save_and_load(self, tmp_path: Path):
        store = VectorStore(dimensions=8)
        chunks = self._make_chunks(3)
        store.add(self._make_embeddings(3), chunks)
        store.save(str(tmp_path))

        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "metadata.json").exists()

        loaded = VectorStore.load(str(tmp_path), dimensions=8)
        assert loaded.size == 3
        assert loaded._chunks[0].source == chunks[0].source

    def test_mismatched_lengths_raises(self):
        store = VectorStore(dimensions=8)
        with pytest.raises(ValueError):
            store.add(self._make_embeddings(5), self._make_chunks(3))

    def test_empty_search_returns_empty(self):
        store = VectorStore(dimensions=8)
        results = store.search(self._make_embeddings(1), top_k=5)
        assert results == []
