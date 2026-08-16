"""Qdrant store for the CareFlow corpus (M3).

Indexes markdown chunks into Qdrant Cloud using local sentence-transformers
for embeddings. Collection: careflow_corpus.

Usage (Indexing):
    store = CorpusStore(recreate=True)
    chunks = load_all_chunks()
    store.index_chunks(chunks)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
_QDRANT_KEY = os.getenv("QDRANT_API_KEY")
_COLLECTION = os.getenv("QDRANT_COLLECTION_CORPUS", "careflow_corpus")
_EMBED_DIM  = 384  # all-MiniLM-L6-v2


class CorpusStore:
    """Manages the Qdrant index for the static clinical corpus."""

    def __init__(self, collection: str | None = None, recreate: bool = False) -> None:
        from qdrant_client import QdrantClient
        self.collection = collection or _COLLECTION
        self._client = QdrantClient(
            url=_QDRANT_URL,
            api_key=_QDRANT_KEY,
            timeout=30,
        )
        self._ensure_collection(recreate=recreate)

    def _ensure_collection(self, recreate: bool) -> None:
        """Create the collection if it doesn't exist, optionally wiping it first."""
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        exists = self._client.collection_exists(self.collection)

        if exists and recreate:
            logger.info("CorpusStore: Deleting existing collection '%s'", self.collection)
            self._client.delete_collection(self.collection)
            exists = False

        if not exists:
            logger.info("CorpusStore: Creating collection '%s'", self.collection)
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )
            # Index the doc_slug payload field for faster filtering if needed
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name="doc_slug",
                field_schema=PayloadSchemaType.KEYWORD,
            )
        else:
            logger.info("CorpusStore: Using existing collection '%s'", self.collection)

    def index_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """Embed and upload chunks to Qdrant.

        Uses local sentence-transformers (from memory.long_term) to embed.
        """
        from qdrant_client.models import PointStruct
        # Import the shared embedder to avoid loading the model twice
        from src.memory.long_term import _embed

        if not chunks:
            logger.warning("CorpusStore.index_chunks: No chunks provided")
            return

        logger.info("CorpusStore: Embedding and indexing %d chunks...", len(chunks))

        points = []
        # We use simple sequential IDs for the corpus
        for i, chunk in enumerate(chunks):
            text = chunk["text"]
            vector = _embed(text)
            
            payload = {
                "doc_slug": chunk["doc_slug"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"],
                "text": text,
            }
            
            points.append(
                PointStruct(id=i, vector=vector, payload=payload)
            )

            if len(points) >= 100:
                self._client.upsert(self.collection, points=points)
                points = []

        # Upsert any remainder
        if points:
            self._client.upsert(self.collection, points=points)

        logger.info("CorpusStore: Successfully indexed all chunks.")

    def count(self) -> int:
        """Return the number of vectors in the corpus collection."""
        return self._client.count(collection_name=self.collection, exact=True).count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from src.rag.chunker_pdf import load_all_pdf_chunks
        chunks = load_all_pdf_chunks()
        if not chunks:
            from src.rag.chunker import load_all_chunks
            chunks = load_all_chunks()
    except Exception:
        from src.rag.chunker import load_all_chunks
        chunks = load_all_chunks()

    store = CorpusStore(recreate=True)
    store.index_chunks(chunks)
    print(f"Total chunks in Qdrant: {store.count()}")
