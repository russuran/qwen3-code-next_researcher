from __future__ import annotations

from knowledge.source_registry import SourceRegistry
from knowledge.document_store import DocumentStore, Document
from knowledge.cache_manager import CacheManager
from knowledge.index_manager import IndexManager
from knowledge.change_detector import ChangeDetector


# --- Source Registry ---

def test_source_registry_register():
    reg = SourceRegistry()
    record = reg.register("https://arxiv.org/abs/1234", "arxiv")
    assert record.url == "https://arxiv.org/abs/1234"
    assert reg.count() == 1


def test_source_registry_update_hash():
    reg = SourceRegistry()
    reg.register("https://example.com", "web")
    changed = reg.update_hash("https://example.com", "content v1")
    assert changed is True
    changed = reg.update_hash("https://example.com", "content v1")
    assert changed is False
    changed = reg.update_hash("https://example.com", "content v2")
    assert changed is True


# --- Document Store ---

def test_document_store_add_and_chunk():
    store = DocumentStore(chunk_size=50, chunk_overlap=10)
    doc = Document(doc_id="doc1", source_url="https://example.com", content="word " * 30)
    store.add(doc)
    assert store.count() == 1
    assert store.total_chunks() > 1


def test_document_store_text_search():
    store = DocumentStore(chunk_size=100, chunk_overlap=0)
    store.add(Document(doc_id="d1", source_url="u1", content="Python is great for machine learning"))
    store.add(Document(doc_id="d2", source_url="u2", content="JavaScript runs in browsers"))

    results = store.search_text("machine learning")
    assert len(results) >= 1
    assert "machine learning" in results[0].text.lower()


# --- Cache Manager ---

def test_cache_manager_set_get(tmp_path):
    cache = CacheManager(cache_dir=str(tmp_path / "cache"))
    cache.set("fetch", "url1", {"data": "value"}, ttl_seconds=3600)
    result = cache.get("fetch", "url1")
    assert result == {"data": "value"}


def test_cache_manager_miss(tmp_path):
    cache = CacheManager(cache_dir=str(tmp_path / "cache"))
    result = cache.get("fetch", "nonexistent")
    assert result is None


def test_cache_manager_invalidate(tmp_path):
    cache = CacheManager(cache_dir=str(tmp_path / "cache"))
    cache.set("fetch", "url1", "data")
    cache.invalidate("fetch", "url1")
    assert cache.get("fetch", "url1") is None


# --- Index Manager ---

def test_index_search():
    store = DocumentStore(chunk_size=500, chunk_overlap=0)
    store.add(Document(doc_id="d1", source_url="u1", content="The Transformer architecture revolutionized natural language processing by introducing the attention mechanism which allows models to weigh the importance of different parts of the input sequence"))
    store.add(Document(doc_id="d2", source_url="u2", content="Database optimization involves query performance tuning index creation and proper schema design for PostgreSQL and other relational database management systems"))

    index = IndexManager(store)
    index.build()
    results = index.search("transformer attention", limit=5)
    assert len(results) >= 1
    assert "transformer" in results[0][0].text.lower() or "attention" in results[0][0].text.lower()


# --- Change Detector ---

def test_change_detector():
    reg = SourceRegistry()
    reg.register("https://example.com", "web", content_hash="abc123")
    detector = ChangeDetector(reg)

    result = detector.detect_changes("https://example.com", new_hash="abc123")
    assert result.changed is False

    result = detector.detect_changes("https://example.com", new_hash="def456")
    assert result.changed is True

    result = detector.detect_changes("https://new-url.com", new_hash="xyz")
    assert result.changed is True
    assert result.reason == "new source"
