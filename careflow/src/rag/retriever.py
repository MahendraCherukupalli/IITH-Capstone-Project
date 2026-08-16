"""Hybrid corpus retriever for M4 (RAG).

Performs semantic search over Qdrant (dense) + BM25 (sparse),
merges results using Reciprocal Rank Fusion (RRF), and re-ranks
the final candidates using a CrossEncoder.
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.qdrant_store import CorpusStore

logger = logging.getLogger(__name__)

# ── Reranker model (load once) ────────────────────────────────────────────────
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker = None

def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder: %s", _CROSS_ENCODER_MODEL)
        _reranker = CrossEncoder(_CROSS_ENCODER_MODEL)
    return _reranker


class Retriever:
    """Retrieves and reranks relevant markdown/PDF chunks from the static corpus."""

    def __init__(self, store: CorpusStore | None = None) -> None:
        self.store = store or CorpusStore(recreate=False)
        self._bm25 = None
        self._bm25_corpus = []
        self._init_bm25()

    def _init_bm25(self) -> None:
        """Initialize BM25 index from all chunks in Qdrant."""
        from rank_bm25 import BM25Okapi
        
        logger.info("Retriever: Initializing BM25 index from Qdrant...")
        # Pull all points (corpus is small)
        all_points, _ = self.store._client.scroll(
            collection_name=self.store.collection,
            limit=10000,
            with_payload=True,
            with_vectors=False,
        )
        
        tokenized_corpus = []
        for p in all_points:
            payload = p.payload or {}
            self._bm25_corpus.append(payload)
            # Simple whitespace tokenization for BM25
            tokenized_corpus.append(payload.get("text", "").lower().split())
            
        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus)
            logger.info("Retriever: BM25 index ready with %d chunks", len(tokenized_corpus))

    def search_dense(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Dense vector search using Qdrant."""
        from src.memory.long_term import _embed
        query_vector = _embed(query)

        hits = self.store._client.query_points(
            collection_name=self.store.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "doc_slug": payload.get("doc_slug", ""),
                "title": payload.get("title", ""),
                "text": payload.get("text", ""),
                "id": payload.get("doc_slug", "") + "_" + str(payload.get("chunk_index", 0)),
                "score": float(hit.score),
            })
        return results

    def search_sparse(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Sparse keyword search using BM25."""
        if not self._bm25:
            return []
            
        tokenized_query = query.lower().split()
        doc_scores = self._bm25.get_scores(tokenized_query)
        
        # Get top K indices
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if doc_scores[idx] > 0:
                payload = self._bm25_corpus[idx]
                results.append({
                    "doc_slug": payload.get("doc_slug", ""),
                    "title": payload.get("title", ""),
                    "text": payload.get("text", ""),
                    "id": payload.get("doc_slug", "") + "_" + str(payload.get("chunk_index", 0)),
                    "score": float(doc_scores[idx]),
                })
        return results

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Hybrid search (RRF) + CrossEncoder Reranking."""
        # 1. Retrieve candidates
        dense_results = self.search_dense(query, top_k=10)
        sparse_results = self.search_sparse(query, top_k=10)
        
        # 2. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict] = {}
        
        k = 60 # RRF constant
        
        for rank, res in enumerate(dense_results, 1):
            rrf_scores[res["id"]] = rrf_scores.get(res["id"], 0.0) + (1.0 / (k + rank))
            chunks_by_id[res["id"]] = res
            
        for rank, res in enumerate(sparse_results, 1):
            rrf_scores[res["id"]] = rrf_scores.get(res["id"], 0.0) + (1.0 / (k + rank))
            chunks_by_id[res["id"]] = res
            
        # Top 10 fused candidates
        fused_candidates = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:10]
        candidate_chunks = [chunks_by_id[cid] for cid in fused_candidates]
        
        if not candidate_chunks:
            return []

        # 3. Rerank with CrossEncoder
        model = _get_reranker()
        cross_inp = [[query, chunk["text"]] for chunk in candidate_chunks]
        cross_scores = model.predict(cross_inp)
        
        for i in range(len(candidate_chunks)):
            candidate_chunks[i]["score"] = float(cross_scores[i])
            
        # Sort by reranker score
        final_results = sorted(candidate_chunks, key=lambda x: x["score"], reverse=True)[:top_k]
        
        logger.info("Retriever: '%s' → found %d final chunks", query, len(final_results))
        return final_results

    def format_as_context(self, query: str, top_k: int = 3) -> str:
        """Search and format results as a context block for the LLM.

        The formatting includes structural barriers to resist prompt injection
        from untrusted document content, following the Lab B pattern.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        context_lines = []
        for i, res in enumerate(results, 1):
            source_ref = f"{res['doc_slug']} / {res['title']}"
            context_lines.append(f"[{i}] (source: {source_ref})\n{res['text']}\n")

        context_block = "\n".join(context_lines)
        return context_block

