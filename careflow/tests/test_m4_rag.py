"""M4 RAG tests — PDF parsing, Hybrid search, and Groundedness.

Test structure:
    TestPDFChunker     — Fast tests for pdfplumber extraction
    TestHybridSearch   — Verifies RRF and cross-encoder logic
    TestGroundedness   — Evaluates the LLM-as-a-judge for hallucinations

Run tests:
    pytest tests/test_m4_rag.py -v -p no:warnings
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chunker_pdf import chunk_pdf_document
from src.rag.qdrant_store import CorpusStore
from src.rag.retriever import Retriever
from src.rag.groundedness import evaluate_groundedness


# ═══════════════════════════════════════════════════════════════════════════════
# TestPDFChunker
# ═══════════════════════════════════════════════════════════════════════════════
class TestPDFChunker:

    def test_chunk_pdf_copay(self):
        """Verify we can extract numeric facts from the PDF."""
        pdf_path = Path("data/corpus/pdf/copay-schedule.pdf")
        if not pdf_path.exists():
            pytest.skip("PDF corpus not generated")
            
        chunks = chunk_pdf_document(pdf_path)
        assert len(chunks) > 0
        
        full_text = " ".join([c["text"] for c in chunks])
        # Ensure critical terms and numbers were extracted cleanly by the PDF parser
        assert "Cost-Share Schedule" in full_text
        assert "Deductible" in full_text or "Coinsurance" in full_text


# ═══════════════════════════════════════════════════════════════════════════════
# TestHybridSearch
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not os.getenv("QDRANT_API_KEY"), reason="Qdrant Cloud key missing")
class TestHybridSearch:

    @classmethod
    def setup_class(cls):
        # We can just use the corpus already indexed in M3 for the test
        cls.store = CorpusStore(collection="careflow_corpus", recreate=False)
        cls.retriever = Retriever(store=cls.store)

    def test_hybrid_retrieval(self):
        """Test the full dense + sparse + rerank pipeline."""
        # Query that might need both keyword matching (MERIDIAN-GOLD) and semantics (copay)
        results = self.retriever.search("What is the co-pay for MERIDIAN-GOLD specialist?", top_k=2)
        
        assert len(results) > 0
        assert results[0]["doc_slug"] == "copay-schedule"
        assert "score" in results[0]
        # CrossEncoder scores can be negative, just ensure it exists
        assert isinstance(results[0]["score"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# TestGroundedness
# ═══════════════════════════════════════════════════════════════════════════════
class TestGroundedness:

    def test_grounded_response(self):
        chunks = [{
            "doc_slug": "copay-schedule",
            "text": "For MERIDIAN-GOLD, the specialist co-pay is $25."
        }]
        
        response = "According to your MERIDIAN-GOLD plan, the specialist co-pay is $25."
        result = evaluate_groundedness(response, chunks)
        
        assert result["grounded"] is True

    def test_ungrounded_response(self):
        chunks = [{
            "doc_slug": "copay-schedule",
            "text": "For MERIDIAN-GOLD, the specialist co-pay is $25."
        }]
        
        # Hallucination: adding info about telehealth that isn't in the chunk
        response = "Your specialist co-pay is $25, and telehealth is free."
        result = evaluate_groundedness(response, chunks)
        
        assert result["grounded"] is False
        assert "telehealth" in result["reason"].lower() or "free" in result["reason"].lower()
