"""M3 tests — Memory and RAG Corpus indexing.

Test structure:
    TestSessionMemory    — Fast in-memory tests (no LLM, no Qdrant)
    TestLongTermMemory   — Real Qdrant tests (uses local embedding model)
    TestCorpusRAG        — Chunking, Indexing, and Retrieval (real Qdrant)

Run tests:
    pytest tests/test_m3_memory.py -v -p no:warnings
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.session import SessionMemory, _total_tokens
from src.memory.long_term import LongTermMemory
from src.rag.chunker import chunk_document, load_all_chunks
from src.rag.qdrant_store import CorpusStore
from src.rag.retriever import Retriever


# ═══════════════════════════════════════════════════════════════════════════════
# TestSessionMemory — Fast, in-memory only
# ═══════════════════════════════════════════════════════════════════════════════
class TestSessionMemory:

    def test_add_turn_and_retrieve(self):
        mem = SessionMemory(recent_keep=4, token_budget=1000)
        mem.add_turn("PT-01", "user", "Hello")
        mem.add_turn("PT-01", "assistant", "Hi there")

        msgs = mem.get_context_messages("PT-01")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert mem.turn_count("PT-01") == 2

    def test_clear_session(self):
        mem = SessionMemory()
        mem.add_turn("PT-02", "user", "Test")
        assert "PT-02" in mem.active_sessions()
        
        mem.clear("PT-02")
        assert "PT-02" not in mem.active_sessions()
        assert len(mem.get_context_messages("PT-02")) == 0

    def test_compression_trigger(self):
        """Test that adding turns exceeding the budget triggers summarization."""
        # Set a tiny budget to force compression quickly
        mem = SessionMemory(recent_keep=2, token_budget=20)
        
        # Add 4 turns (all small)
        for i in range(4):
            mem.add_turn("PT-03", "user" if i % 2 == 0 else "assistant", f"Turn {i} text here " * 5)
        
        # With recent_keep=2 and 4 turns added, it should compress the first 2
        summary = mem.get_summary("PT-03")
        assert len(summary) > 0, "Summary should have been generated"
        
        msgs = mem.get_context_messages("PT-03")
        assert len(msgs) == 3, "1 system summary + 2 kept turns"
        assert msgs[0]["role"] == "system"
        assert "Session memory" in msgs[0]["content"]


# ═══════════════════════════════════════════════════════════════════════════════
# TestLongTermMemory — Requires Qdrant
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not os.getenv("QDRANT_API_KEY"), reason="Qdrant Cloud key missing")
class TestLongTermMemory:

    @pytest.fixture(scope="class")
    def ltm(self):
        # Use a separate test collection to avoid wiping real memory
        return LongTermMemory(collection="test_patient_memory", recreate=True)

    def test_save_and_recall(self, ltm):
        patient_ref = "MHP-P-TEST01"
        
        # Save two distinct memories
        ltm.save_session_summary(
            patient_ref, 
            "Patient is considering ACL surgery for their left knee."
        )
        ltm.save_session_summary(
            patient_ref, 
            "Patient asked about their Meridian Gold copay. It is $25."
        )
        
        # Count should be 2
        assert ltm.count(patient_ref) == 2
        
        # Recall specifically about insurance
        results = ltm.recall(patient_ref, "What is my insurance copay?", top_k=1)
        assert len(results) == 1
        assert "copay" in results[0]["summary"].lower()
        
        # Recall about knee
        results2 = ltm.recall(patient_ref, "orthopedics surgery knee", top_k=1)
        assert len(results2) == 1
        assert "knee" in results2[0]["summary"].lower()

    def test_recall_as_context(self, ltm):
        patient_ref = "MHP-P-TEST01"
        context = ltm.recall_as_context(patient_ref, "copay", top_k=2)
        assert "Long-term memory" in context
        assert "Meridian Gold" in context


# ═══════════════════════════════════════════════════════════════════════════════
# TestCorpusRAG — Requires Qdrant
# ═══════════════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not os.getenv("QDRANT_API_KEY"), reason="Qdrant Cloud key missing")
class TestCorpusRAG:

    @classmethod
    def setup_class(cls):
        """Index the corpus into a test collection before running tests."""
        cls.store = CorpusStore(collection="test_corpus", recreate=True)
        chunks = load_all_chunks()
        cls.store.index_chunks(chunks)
        cls.retriever = Retriever(store=cls.store)

    def test_corpus_size(self):
        count = self.store.count()
        assert count > 10, f"Expected more than 10 chunks, got {count}"

    def test_retrieval_copay_schedule(self):
        results = self.retriever.search("What is the co-pay for MERIDIAN-GOLD specialist?", top_k=1)
        assert len(results) == 1
        chunk = results[0]
        assert chunk["doc_slug"] == "copay-schedule", f"Wrong doc returned: {chunk['doc_slug']}"
        assert chunk["score"] > 0.5

    def test_retrieval_preauth(self):
        results = self.retriever.search("Does an MRI scan require pre-authorization?", top_k=1)
        assert len(results) == 1
        assert results[0]["doc_slug"] == "preauth-matrix"

    def test_retrieval_format_as_context(self):
        context = self.retriever.format_as_context("What is the referral policy?", top_k=2)
        assert "(source: referral-policy /" in context
        assert "[1]" in context
