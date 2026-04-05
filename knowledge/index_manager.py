"""Index manager: vector + text search over document chunks."""
from __future__ import annotations

import logging
import math
from collections import Counter

from knowledge.document_store import Chunk, DocumentStore

logger = logging.getLogger(__name__)


class IndexManager:
    """Simple TF-IDF based search index. Replace with pgvector/Qdrant later."""

    def __init__(self, doc_store: DocumentStore) -> None:
        self.doc_store = doc_store
        self._idf: dict[str, float] = {}
        self._chunk_tfs: dict[str, dict[str, float]] = {}
        self._chunk_map: dict[str, Chunk] = {}
        self._built = False

    def build(self) -> None:
        """Build/rebuild the search index from document store."""
        all_chunks = []
        for doc in self.doc_store.list_documents():
            for chunk in doc.chunks:
                all_chunks.append(chunk)
                self._chunk_map[chunk.chunk_id] = chunk

        if not all_chunks:
            self._built = True
            return

        # Compute TF per chunk
        doc_freq: Counter = Counter()
        for chunk in all_chunks:
            tokens = self._tokenize(chunk.text)
            tf = Counter(tokens)
            total = len(tokens) or 1
            self._chunk_tfs[chunk.chunk_id] = {t: c / total for t, c in tf.items()}
            doc_freq.update(set(tokens))

        # Compute IDF (add 1 to numerator to ensure positive values)
        n = len(all_chunks)
        self._idf = {
            term: math.log((n + 1) / (1 + freq)) + 1.0
            for term, freq in doc_freq.items()
        }
        self._built = True
        logger.info("Index built: %d chunks, %d terms", len(all_chunks), len(self._idf))

    def search(self, query: str, limit: int = 10) -> list[tuple[Chunk, float]]:
        """Search for chunks matching query. Returns (chunk, score) pairs."""
        if not self._built:
            self.build()

        query_tokens = self._tokenize(query)
        scores: dict[str, float] = {}

        for chunk_id, tf_map in self._chunk_tfs.items():
            score = 0.0
            for token in query_tokens:
                if token in tf_map:
                    score += tf_map[token] * self._idf.get(token, 0)
            if score > 0:
                scores[chunk_id] = score

        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:limit]

        results = []
        for chunk_id, score in ranked:
            chunk = self._chunk_map.get(chunk_id)
            if chunk:
                results.append((chunk, score))

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [w.lower().strip(".,!?;:()[]{}\"'") for w in text.split() if len(w.strip(".,!?;:()[]{}\"'")) > 2]
