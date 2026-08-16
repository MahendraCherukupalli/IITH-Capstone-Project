"""Long-term patient memory backed by Qdrant Cloud.

Stores and retrieves per-patient session summaries across visits.
Uses sentence-transformers (all-MiniLM-L6-v2) for embeddings — local,
free, no API calls needed for memory operations.

Collection: careflow_patient_memory
Each point:
    id:      auto-incrementing int (hash of patient_ref + timestamp)
    vector:  384-dim from all-MiniLM-L6-v2
    payload: {patient_ref, summary, timestamp, session_count}

Usage:
    ltm = LongTermMemory()
    ltm.save_session_summary("MHP-P-00001", "Patient asked about MERIDIAN-GOLD copay. Copay is $25.")
    memories = ltm.recall("MHP-P-00001", "insurance coverage", top_k=3)
    for m in memories:
        print(m["summary"], m["score"])
"""

from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
_QDRANT_URL  = os.getenv("QDRANT_URL", "http://localhost:6333")
_QDRANT_KEY  = os.getenv("QDRANT_API_KEY")
_COLLECTION  = os.getenv("QDRANT_COLLECTION_MEMORY", "careflow_patient_memory")
_EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, local, fast
_EMBED_DIM   = 384


@lru_cache(maxsize=1)
def _get_embedder():
    """Load the sentence-transformer model once."""
    from sentence_transformers import SentenceTransformer
    logger.info("Loading embedding model: %s", _EMBED_MODEL)
    return SentenceTransformer(_EMBED_MODEL)


def _embed(text: str) -> list[float]:
    model = _get_embedder()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def _make_point_id(patient_ref: str, timestamp: str) -> int:
    """Generate a stable integer ID from patient_ref + timestamp."""
    raw = f"{patient_ref}::{timestamp}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:15], 16) % (2**63)


# ── Long-term memory store ─────────────────────────────────────────────────────
class LongTermMemory:
    """Qdrant-backed long-term memory for patient interactions.

    Args:
        collection: Qdrant collection name (default from env).
        recreate:   If True, drops and recreates the collection on init.
                    Use recreate=True in tests; False in production.
    """

    def __init__(
        self,
        collection: str | None = None,
        recreate: bool = False,
    ) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.collection = collection or _COLLECTION
        self._client = QdrantClient(
            url=_QDRANT_URL,
            api_key=_QDRANT_KEY,
            timeout=15,
        )
        self._ensure_collection(recreate=recreate)

    def _ensure_collection(self, recreate: bool = False) -> None:
        """Create the Qdrant collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams

        exists = self._client.collection_exists(self.collection)

        if exists and recreate:
            self._client.delete_collection(self.collection)
            exists = False

        if not exists:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )
            # Must index patient_ref to allow filtering by it
            from qdrant_client.models import PayloadSchemaType
            self._client.create_payload_index(
                collection_name=self.collection,
                field_name="patient_ref",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("LongTermMemory: created collection '%s'", self.collection)
        else:
            logger.info("LongTermMemory: using existing collection '%s'", self.collection)

    # ── Public API ─────────────────────────────────────────────────────────────
    def save_session_summary(
        self,
        patient_ref: str,
        summary: str,
        extra_metadata: dict | None = None,
    ) -> str:
        """Persist a session summary for a patient.

        Args:
            patient_ref:    Patient reference ID.
            summary:        Text summary of the session (what was discussed).
            extra_metadata: Optional extra fields to store (e.g. urgency, specialty).

        Returns:
            The timestamp string of the saved entry.
        """
        from qdrant_client.models import PointStruct

        timestamp = datetime.now(timezone.utc).isoformat()
        point_id  = _make_point_id(patient_ref, timestamp)
        vector    = _embed(summary)

        payload: dict[str, Any] = {
            "patient_ref":   patient_ref,
            "summary":       summary,
            "timestamp":     timestamp,
        }
        if extra_metadata:
            payload.update(extra_metadata)

        self._client.upsert(
            collection_name=self.collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )

        logger.info(
            "LongTermMemory.save: %s @ %s (%d chars)",
            patient_ref, timestamp, len(summary),
        )
        return timestamp

    def recall(
        self,
        patient_ref: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant past interactions for a patient.

        Args:
            patient_ref: Patient reference ID (filters to only this patient's memories).
            query:       What to search for (e.g. "insurance coverage", "referral status").
            top_k:       Max number of memories to return.

        Returns:
            List of dicts with: summary, timestamp, score (0–1), and any extra metadata.
            Sorted by relevance (highest score first).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        query_vector = _embed(query)

        # Filter to this patient only
        patient_filter = Filter(
            must=[FieldCondition(
                key="patient_ref",
                match=MatchValue(value=patient_ref),
            )]
        )

        hits = self._client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=patient_filter,
            limit=top_k,
            with_payload=True,
        ).points

        results = []
        for hit in hits:
            payload = hit.payload or {}
            results.append({
                "summary":     payload.get("summary", ""),
                "timestamp":   payload.get("timestamp", ""),
                "patient_ref": payload.get("patient_ref", ""),
                "score":       round(float(hit.score), 4),
            })

        logger.info(
            "LongTermMemory.recall: %s / '%s' → %d result(s)",
            patient_ref, query, len(results),
        )
        return results

    def recall_as_context(
        self,
        patient_ref: str,
        query: str,
        top_k: int = 3,
    ) -> str:
        """Recall past interactions formatted as a system message string.

        Ready to inject into the LLM context directly.
        Returns empty string if no past interactions found.
        """
        memories = self.recall(patient_ref, query, top_k=top_k)
        if not memories:
            return ""

        lines = [f"[Long-term memory — past interactions for {patient_ref}]"]
        for i, m in enumerate(memories, 1):
            ts = m["timestamp"][:10]  # date only
            lines.append(f"{i}. ({ts}) {m['summary']}")

        return "\n".join(lines)

    def count(self, patient_ref: str | None = None) -> int:
        """Count stored memories, optionally filtered to one patient."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if patient_ref:
            pfilter = Filter(must=[FieldCondition(
                key="patient_ref", match=MatchValue(value=patient_ref)
            )])
            return self._client.count(
                collection_name=self.collection,
                count_filter=pfilter,
                exact=True,
            ).count
        return self._client.count(collection_name=self.collection, exact=True).count
