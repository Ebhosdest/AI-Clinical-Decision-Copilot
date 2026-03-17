"""Retrieval evaluation suite — measures Recall@K and source accuracy.

Runs in two modes:
  - With API key: full embedding-based retrieval evaluation
  - Without API key: structural/metadata validation only
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from src.rag.loader import load_documents
from src.rag.vector_store import VectorStore


@dataclass
class EvalCase:
    query: str
    expected_sources: list[str]       # document IDs expected in top-K results
    expected_keywords: list[str]      # keywords expected in retrieved text
    expected_sections: list[str]      # sections expected to be retrieved
    description: str


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        query="STEMI management primary PCI door to balloon time",
        expected_sources=["AHA-ACS-2023"],
        expected_keywords=["PCI", "90 minutes", "reperfusion", "STEMI"],
        expected_sections=["STEMI Management"],
        description="STEMI primary PCI protocol",
    ),
    EvalCase(
        query="hypertension first line treatment ACE inhibitor ARB calcium channel blocker",
        expected_sources=["WHO-HTN-2023"],
        expected_keywords=["ACE", "ARB", "calcium channel", "thiazide"],
        expected_sections=["Pharmacological Treatment"],
        description="Hypertension first-line pharmacotherapy",
    ),
    EvalCase(
        query="type 2 diabetes metformin first line therapy HbA1c target",
        expected_sources=["ADA-DM-2024"],
        expected_keywords=["metformin", "HbA1c", "first-line", "glucose"],
        expected_sections=["Type 2 Diabetes"],
        description="T2DM metformin initiation",
    ),
    EvalCase(
        query="SGLT2 inhibitor empagliflozin heart failure cardiovascular benefit",
        expected_sources=["ADA-DM-2024"],
        expected_keywords=["SGLT2", "heart failure", "empagliflozin", "cardiovascular"],
        expected_sections=["Cardiovascular Risk Reduction"],
        description="SGLT2 inhibitors in T2DM with HF",
    ),
    EvalCase(
        query="hypertensive emergency labetalol nicardipine MAP reduction",
        expected_sources=["WHO-HTN-2023"],
        expected_keywords=["labetalol", "nicardipine", "MAP", "emergency"],
        expected_sections=["Hypertensive Emergencies"],
        description="Hypertensive emergency management",
    ),
    EvalCase(
        query="dual antiplatelet therapy ticagrelor clopidogrel after acute coronary syndrome",
        expected_sources=["AHA-ACS-2023"],
        expected_keywords=["ticagrelor", "antiplatelet", "P2Y12", "DAPT"],
        expected_sections=["NSTEMI/UA Management", "Pharmacotherapy"],
        description="DAPT selection in ACS",
    ),
    EvalCase(
        query="diabetic kidney disease ACE inhibitor albuminuria eGFR",
        expected_sources=["ADA-DM-2024"],
        expected_keywords=["ACEi", "ARB", "albuminuria", "eGFR", "CKD"],
        expected_sections=["Diabetic Kidney Disease"],
        description="DKD management with ACEi/ARB",
    ),
)


def _keyword_recall(text: str, keywords: list[str]) -> float:
    """Fraction of keywords present in text (case-insensitive)."""
    lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in lower)
    return hits / len(keywords) if keywords else 0.0


def _source_recall(retrieved_doc_ids: list[str], expected: list[str]) -> float:
    """Fraction of expected sources present in retrieved results."""
    hits = sum(1 for src in expected if src in retrieved_doc_ids)
    return hits / len(expected) if expected else 0.0


def _section_recall(retrieved_sections: list[str], expected: list[str]) -> float:
    """Fraction of expected sections present in retrieved results."""
    hits = sum(1 for sec in expected if any(sec.lower() in rs.lower() for rs in retrieved_sections))
    return hits / len(expected) if expected else 0.0


# ── No-API Tests (metadata + structural validation) ───────────────────────────

class TestDocumentStructure:
    """Validate guideline files are well-formed without an API key."""

    GUIDELINES_DIR = "data/guidelines"

    def test_guidelines_directory_exists(self):
        assert Path(self.GUIDELINES_DIR).exists(), f"Missing: {self.GUIDELINES_DIR}"

    def test_all_three_guideline_files_present(self):
        files = list(Path(self.GUIDELINES_DIR).glob("*.txt"))
        assert len(files) >= 3, f"Expected ≥3 guideline files, found {len(files)}"

    def test_aha_acs_guidelines_loads(self):
        chunks = load_documents(self.GUIDELINES_DIR)
        aha_chunks = [c for c in chunks if "AHA-ACS" in c.document_id]
        assert len(aha_chunks) > 0

    def test_who_hypertension_guidelines_loads(self):
        chunks = load_documents(self.GUIDELINES_DIR)
        who_chunks = [c for c in chunks if "WHO-HTN" in c.document_id]
        assert len(who_chunks) > 0

    def test_ada_diabetes_guidelines_loads(self):
        chunks = load_documents(self.GUIDELINES_DIR)
        ada_chunks = [c for c in chunks if "ADA-DM" in c.document_id]
        assert len(ada_chunks) > 0

    def test_all_chunks_have_required_metadata(self):
        chunks = load_documents(self.GUIDELINES_DIR)
        for c in chunks:
            assert c.source, f"Missing source on chunk {c.chunk_index}"
            assert c.document_id, f"Missing document_id on chunk {c.chunk_index}"
            assert c.section, f"Missing section on chunk {c.chunk_index}"
            assert c.text.strip(), f"Empty text on chunk {c.chunk_index}"

    def test_expected_sections_present(self):
        chunks = load_documents(self.GUIDELINES_DIR)
        sections = {c.section for c in chunks}
        expected_sections = [
            "STEMI Management", "NSTEMI/UA Management",
            "Pharmacological Treatment", "Hypertensive Emergencies",
            "Type 2 Diabetes", "Cardiovascular Risk Reduction", "Diabetic Kidney Disease",
        ]
        for sec in expected_sections:
            assert sec in sections, f"Expected section not found: {sec}"

    def test_chunk_text_contains_keywords(self):
        chunks = load_documents(self.GUIDELINES_DIR)
        all_text = " ".join(c.text for c in chunks)
        critical_terms = ["STEMI", "metformin", "hypertension", "ACE", "SGLT2", "ticagrelor"]
        for term in critical_terms:
            assert term.lower() in all_text.lower(), f"Critical term missing from guidelines: {term}"


class TestVectorStoreEvaluation:
    """Keyword-based retrieval evaluation using random embeddings (no API key)."""

    GUIDELINES_DIR = "data/guidelines"

    def _build_mock_store(self, dims: int = 32) -> tuple[VectorStore, list]:
        chunks = load_documents(self.GUIDELINES_DIR)
        rng = np.random.default_rng(42)
        embeddings = rng.random((len(chunks), dims), dtype=np.float32)
        store = VectorStore(dimensions=dims)
        store.add(embeddings, chunks)
        return store, chunks

    def test_store_contains_all_chunks(self):
        chunks = load_documents(self.GUIDELINES_DIR)
        store, _ = self._build_mock_store()
        assert store.size == len(chunks)

    def test_search_returns_correct_count(self):
        store, _ = self._build_mock_store(dims=32)
        rng = np.random.default_rng(0)
        query_vec = rng.random((1, 32), dtype=np.float32)
        results = store.search(query_vec, top_k=5)
        assert len(results) == 5

    def test_keyword_coverage_in_guidelines(self):
        """Verify all eval case keywords appear somewhere in the guidelines."""
        chunks = load_documents(self.GUIDELINES_DIR)
        all_text = " ".join(c.text for c in chunks)
        missing = []
        for case in EVAL_CASES:
            for kw in case.expected_keywords:
                if kw.lower() not in all_text.lower():
                    missing.append(f"Case '{case.description}': keyword '{kw}'")
        assert not missing, f"Keywords not found in guidelines:\n" + "\n".join(missing)

    def test_source_distribution(self):
        """All three guideline sources must be represented."""
        chunks = load_documents(self.GUIDELINES_DIR)
        doc_ids = {c.document_id for c in chunks}
        assert "AHA-ACS-2023" in doc_ids
        assert "WHO-HTN-2023" in doc_ids
        assert "ADA-DM-2024" in doc_ids


# ── Live API Tests (require OPENAI_API_KEY) ───────────────────────────────────

def _api_key_available() -> bool:
    key = os.getenv("OPENAI_API_KEY", "")
    return bool(key) and key != "your-openai-api-key-here"


@pytest.mark.skipif(not _api_key_available(), reason="OPENAI_API_KEY not configured")
class TestLiveRetrieval:
    """Full embedding-based retrieval evaluation (requires API key)."""

    GUIDELINES_DIR = "data/guidelines"
    TOP_K = 5
    MIN_KEYWORD_RECALL = 0.6
    MIN_SOURCE_RECALL = 1.0

    @pytest.fixture(scope="class")
    def rag_components(self):
        from src.config import load_config
        from src.rag.embeddings import EmbeddingService
        from src.rag.retriever import Retriever

        config = load_config()
        chunks = load_documents(self.GUIDELINES_DIR, chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        embedder = EmbeddingService(api_key=config.openai_api_key, model=config.embedding_model)
        texts = [c.text for c in chunks]
        embeddings = embedder.embed_texts(texts)

        store = VectorStore(dimensions=embedder.dimensions)
        store.add(embeddings, chunks)

        retriever = Retriever(store, embedder, top_k=self.TOP_K, score_threshold=0.0)
        return retriever

    def _evaluate_case(self, retriever, case: EvalCase) -> dict:
        results = retriever.retrieve(case.query)
        combined_text = " ".join(r.chunk.text for r in results)
        retrieved_doc_ids = [r.chunk.document_id for r in results]
        retrieved_sections = [r.chunk.section for r in results]

        return {
            "description": case.description,
            "keyword_recall": _keyword_recall(combined_text, case.expected_keywords),
            "source_recall": _source_recall(retrieved_doc_ids, case.expected_sources),
            "section_recall": _section_recall(retrieved_sections, case.expected_sections),
            "num_results": len(results),
        }

    @pytest.mark.parametrize("case", EVAL_CASES, ids=[c.description for c in EVAL_CASES])
    def test_retrieval_eval_case(self, rag_components, case: EvalCase):
        metrics = self._evaluate_case(rag_components, case)
        assert metrics["keyword_recall"] >= self.MIN_KEYWORD_RECALL, (
            f"[{case.description}] Keyword recall {metrics['keyword_recall']:.2f} < {self.MIN_KEYWORD_RECALL}"
        )
        assert metrics["source_recall"] >= self.MIN_SOURCE_RECALL, (
            f"[{case.description}] Source recall {metrics['source_recall']:.2f} < {self.MIN_SOURCE_RECALL}"
        )

    def test_aggregate_recall(self, rag_components):
        """Aggregate Recall@K across all eval cases."""
        keyword_recalls = []
        source_recalls = []
        for case in EVAL_CASES:
            m = self._evaluate_case(rag_components, case)
            keyword_recalls.append(m["keyword_recall"])
            source_recalls.append(m["source_recall"])

        avg_kw = np.mean(keyword_recalls)
        avg_src = np.mean(source_recalls)
        print(f"\nAggregate Keyword Recall@{self.TOP_K}: {avg_kw:.3f}")
        print(f"Aggregate Source Recall@{self.TOP_K}: {avg_src:.3f}")
        assert avg_kw >= 0.5, f"Average keyword recall {avg_kw:.3f} below threshold"
        assert avg_src >= 0.7, f"Average source recall {avg_src:.3f} below threshold"
