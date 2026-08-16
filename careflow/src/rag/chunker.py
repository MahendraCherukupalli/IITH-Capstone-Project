"""Markdown document chunker for RAG (M3).

Reads all .md files from data/corpus/markdown/, extracts the title
from the first # Header, and chunks the text using a sliding window
of 400 tokens with 50 tokens overlap.

Uses tiktoken (cl100k_base) for exact token counts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 400
OVERLAP_TOKENS    = 50
CORPUS_DIR        = Path("data/corpus/markdown")

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _get_title(markdown_text: str) -> str:
    """Extract the first H1 (# Title) as the document title."""
    for line in markdown_text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled Document"


def chunk_document(file_path: Path) -> list[dict[str, Any]]:
    """Read a markdown file and split it into overlapping chunks.

    Returns:
        List of dicts containing chunk text and metadata.
    """
    try:
        text = file_path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.error("Failed to read %s: %s", file_path, exc)
        return []

    if not text:
        return []

    title = _get_title(text)
    doc_slug = file_path.stem

    tokens = _ENCODING.encode(text)
    chunks = []
    chunk_index = 0
    start = 0

    while start < len(tokens):
        end = start + CHUNK_SIZE_TOKENS
        chunk_tokens = tokens[start:end]
        chunk_text = _ENCODING.decode(chunk_tokens).strip()

        chunks.append({
            "doc_slug": doc_slug,
            "title": title,
            "chunk_index": chunk_index,
            "text": chunk_text,
        })

        chunk_index += 1
        # Move forward, minus the overlap
        start = end - OVERLAP_TOKENS

    logger.debug("Chunked %s into %d chunks", doc_slug, len(chunks))
    return chunks


def load_all_chunks(corpus_dir: Path = CORPUS_DIR) -> list[dict[str, Any]]:
    """Load and chunk all .md files in the corpus directory."""
    if not corpus_dir.exists():
        logger.warning("Corpus directory %s does not exist", corpus_dir)
        return []

    all_chunks = []
    files = sorted(corpus_dir.glob("*.md"))

    for f in files:
        file_chunks = chunk_document(f)
        all_chunks.extend(file_chunks)

    logger.info(
        "Loaded %d documents, generated %d total chunks",
        len(files), len(all_chunks)
    )
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = load_all_chunks()
    if chunks:
        print(f"Sample chunk from {chunks[0]['doc_slug']}:")
        print("-" * 40)
        print(chunks[0]["text"][:200] + "...")
        print("-" * 40)
