"""Document store: stores parsed documents and chunks."""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any] = {}
    embedding: list[float] | None = None


class Document(BaseModel):
    doc_id: str
    source_url: str
    title: str = ""
    content: str = ""
    doc_type: str = ""  # paper, repo, webpage, pdf
    chunks: list[Chunk] = []
    metadata: dict[str, Any] = {}


class DocumentStore:
    """In-memory document store with chunking support."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self._documents: dict[str, Document] = {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def add(self, doc: Document, auto_chunk: bool = True) -> Document:
        if auto_chunk and doc.content and not doc.chunks:
            doc.chunks = self._chunk_text(doc.doc_id, doc.content)
        self._documents[doc.doc_id] = doc
        logger.debug("Stored document: %s (%d chunks)", doc.doc_id, len(doc.chunks))
        return doc

    def get(self, doc_id: str) -> Document | None:
        return self._documents.get(doc_id)

    def search_text(self, query: str, limit: int = 10) -> list[Chunk]:
        """Simple text search across all chunks."""
        query_lower = query.lower()
        results = []
        for doc in self._documents.values():
            for chunk in doc.chunks:
                if query_lower in chunk.text.lower():
                    results.append(chunk)
                    if len(results) >= limit:
                        return results
        return results

    def list_documents(self) -> list[Document]:
        return list(self._documents.values())

    def count(self) -> int:
        return len(self._documents)

    def total_chunks(self) -> int:
        return sum(len(d.chunks) for d in self._documents.values())

    def _chunk_text(self, doc_id: str, text: str) -> list[Chunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunk_id = hashlib.md5(f"{doc_id}:{idx}".encode()).hexdigest()[:12]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text,
                metadata={"index": idx, "start": start, "end": min(end, len(text))},
            ))
            start += self.chunk_size - self.chunk_overlap
            idx += 1
        return chunks
