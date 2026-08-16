"""PDF Document chunker for M4.

Extracts text from the PDF versions of the corpus documents.
Uses pdfplumber to parse text, handling basic formatting.
Returns chunks similarly to the markdown chunker for ingestion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pdfplumber
import tiktoken

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS = 400
OVERLAP_TOKENS    = 50
CORPUS_PDF_DIR    = Path("data/corpus/pdf")

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _get_title(text: str, fallback: str) -> str:
    """Attempt to extract title from the first few lines, or use fallback slug."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines[0]
    return fallback


def chunk_pdf_document(file_path: Path) -> list[dict[str, Any]]:
    """Extract text from PDF and split into overlapping chunks."""
    try:
        full_text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n\n"
        
        full_text = full_text.strip()
    except Exception as exc:
        logger.error("Failed to read %s: %s", file_path, exc)
        return []

    if not full_text:
        return []

    doc_slug = file_path.stem
    title = _get_title(full_text, fallback=doc_slug)

    tokens = _ENCODING.encode(full_text)
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
        start = end - OVERLAP_TOKENS

    logger.debug("Chunked %s into %d chunks (PDF)", doc_slug, len(chunks))
    return chunks


def load_all_pdf_chunks(corpus_dir: Path = CORPUS_PDF_DIR) -> list[dict[str, Any]]:
    """Load and chunk all .pdf files in the corpus directory."""
    if not corpus_dir.exists():
        logger.warning("PDF Corpus directory %s does not exist", corpus_dir)
        return []

    all_chunks = []
    files = sorted(corpus_dir.glob("*.pdf"))

    for f in files:
        file_chunks = chunk_pdf_document(f)
        all_chunks.extend(file_chunks)

    logger.info(
        "Loaded %d PDF documents, generated %d total chunks",
        len(files), len(all_chunks)
    )
    return all_chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    chunks = load_all_pdf_chunks()
    if chunks:
        print(f"Sample chunk from {chunks[0]['doc_slug']} (PDF):")
        print("-" * 40)
        print(chunks[0]["text"][:200] + "...")
        print("-" * 40)
