"""Medical index — mappings, metadata-aware vector store, dataset protection.

Scope
-----
Phase 6 of the medical RAG MVP.  Provides everything needed to build the
bundled medical corpus index (``medibo_default``) while leaving the existing
Potbot retrieval path (RRF, candidate pool, reranker, query rewriting, ES
schema of ``potbot_documents``) untouched.

Dataset protection
------------------
* ``medibo_default`` — the shipped, read-only default medical corpus.
* ``medibo_user``    — user-uploaded / session data, freely writable and
  clearable.

Write and delete operations that target the default index from a protected
(UI/API) path raise :class:`DatasetProtectionError`.  Backend build tooling
opts in explicitly with ``allow_default_write=True``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:
    from elasticsearch import Elasticsearch
except (ImportError, AttributeError, OSError, Exception):  # pragma: no cover
    Elasticsearch = None  # type: ignore[assignment, misc]

import config
from domain.models import Chunk
from ingestion.indexers import ElasticsearchVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index names
# ---------------------------------------------------------------------------

MEDICAL_DEFAULT_INDEX = getattr(config, "MEDICAL_DEFAULT_INDEX", "medibo_default")
MEDICAL_USER_INDEX = getattr(config, "MEDICAL_USER_INDEX", "medibo_user")

#: Indexes that are read-only from the UI/API.
PROTECTED_DEFAULT_INDEXES = frozenset({MEDICAL_DEFAULT_INDEX})

#: Field names carried in ``chunk.metadata`` and flattened into ``_source``.
MEDICAL_METADATA_FIELDS = (
    "title",
    "source",
    "source_url",
    "region",
    "authority",
    "document_type",
    "disease_topic",
    "published_at",
    "updated_at",
    "section",
    "dataset_type",
    "license",
)


class DatasetProtectionError(PermissionError):
    """Raised when a write/delete targets the protected default dataset."""


# ---------------------------------------------------------------------------
# Guard helpers
# ---------------------------------------------------------------------------

def is_protected(index_name: Optional[str]) -> bool:
    """Return True when *index_name* is the read-only default dataset."""
    return index_name in PROTECTED_DEFAULT_INDEXES


def assert_writable_index(index_name: Optional[str], operation: str = "write") -> None:
    """Raise :class:`DatasetProtectionError` for protected *operation*s."""
    if is_protected(index_name):
        raise DatasetProtectionError(
            f"Default medical dataset '{index_name}' is read-only; "
            f"'{operation}' is not permitted from the UI/API path."
        )


def clear_user_data(
    es: Optional[Any] = None,
    index_name: str = MEDICAL_USER_INDEX,
    host: Optional[str] = None,
) -> Dict[str, Any]:
    """Delete the *user* dataset only. The default dataset is never touched.

    Parameters
    ----------
    es:
        Optional injected Elasticsearch client (used by tests).  When ``None``
        a client is created from *host* (or ``config.ELASTICSEARCH_HOST``).
    index_name:
        Must be a user-writable index.  Pointing this at a protected default
        index raises :class:`DatasetProtectionError` before any call reaches
        Elasticsearch.
    """
    assert_writable_index(index_name, operation="delete")

    owns_client = es is None
    if es is None:
        if Elasticsearch is None:
            raise RuntimeError("Elasticsearch client library not available.")
        raw_host = host or config.ELASTICSEARCH_HOST
        if not raw_host.startswith(("http://", "https://")):
            raw_host = f"http://{raw_host}"
        es = Elasticsearch(raw_host)

    try:
        exists = es.indices.exists(index=index_name)
        if exists:
            es.indices.delete(index=index_name)
        return {
            "deleted": bool(exists),
            "index": index_name,
            "protected_default_index": MEDICAL_DEFAULT_INDEX,
            "default_still_present": es.indices.exists(index=MEDICAL_DEFAULT_INDEX),
        }
    finally:
        if owns_client and hasattr(es, "close"):
            try:
                es.close()
            except Exception:  # pragma: no cover
                pass


def user_vector_store(
    host: Optional[str] = None,
    es_client: Optional[Any] = None,
) -> "MedicalVectorStore":
    """Return a store bound to the user dataset (never the default corpus)."""
    return MedicalVectorStore(
        host=host,
        index_name=MEDICAL_USER_INDEX,
        es_client=es_client,
        allow_default_write=False,
    )


# ---------------------------------------------------------------------------
# Mapping
# ---------------------------------------------------------------------------

def build_medical_mapping(dimension: int) -> Dict[str, Any]:
    """Return the full Elasticsearch index body for the medical corpus."""
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                # --- identity / retrieval ---
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "standard"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": True,
                    "similarity": "cosine",
                },
                # --- provenance (file level) ---
                "chunk_index": {"type": "integer"},
                "source_file": {"type": "keyword"},
                "file_name": {"type": "keyword"},
                "file_type": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "modified_date": {"type": "date"},
                # --- medical metadata ---
                "title": {"type": "text"},
                "source": {"type": "keyword"},
                "source_url": {"type": "keyword"},
                "region": {"type": "keyword"},
                "authority": {"type": "text"},
                "document_type": {"type": "keyword"},
                "disease_topic": {"type": "keyword"},
                # Partial dates appear in the corpus ("2015", "2016-03").
                "published_at": {
                    "type": "date",
                    "format": "strict_date_optional_time||yyyy-MM||yyyy",
                },
                "updated_at": {
                    "type": "date",
                    "format": "strict_date_optional_time||yyyy-MM||yyyy",
                },
                "section": {"type": "text"},
                "dataset_type": {"type": "keyword"},
                "license": {"type": "keyword"},
            }
        },
    }


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class MedicalVectorStore(ElasticsearchVectorStore):
    """Elasticsearch store for the medical corpus with dataset protection.

    Parameters
    ----------
    host:
        Elasticsearch URL; defaults to ``config.ELASTICSEARCH_HOST``.
    index_name:
        Defaults to ``medibo_default``.
    es_client:
        Injected Elasticsearch client.  When provided no network connection is
        attempted (hermetic tests pass a fake here).
    allow_default_write:
        Backend/build opt-in that permits writes and ``recreate`` against the
        protected default index.  UI/API paths must leave this ``False``.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        index_name: Optional[str] = None,
        es_client: Optional[Any] = None,
        allow_default_write: bool = False,
    ) -> None:
        target_index = index_name or MEDICAL_DEFAULT_INDEX
        if es_client is not None:
            raw_host = host or config.ELASTICSEARCH_HOST
            if not raw_host.startswith(("http://", "https://")):
                raw_host = f"http://{raw_host}"
            self.host = raw_host
            self.index_name = target_index
            self.es = es_client
        else:
            super().__init__(host=host, index_name=target_index)
        self.allow_default_write = allow_default_write

    # ------------------------------------------------------------------
    # Protection
    # ------------------------------------------------------------------

    def _guard(self, operation: str) -> None:
        """Block *operation* on the default index unless explicitly allowed."""
        if is_protected(self.index_name) and not self.allow_default_write:
            raise DatasetProtectionError(
                f"Default medical dataset '{self.index_name}' is read-only; "
                f"'{operation}' is not permitted from the UI/API path."
            )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def create_index(self, dimension: int, recreate: bool = False) -> None:
        self._guard("create_index(recreate)" if recreate else "create_index")
        if self.es is None:
            logger.error("Cannot create index: Elasticsearch client is unavailable.")
            return
        if self.es.indices.exists(index=self.index_name):
            if recreate:
                logger.warning("Deleting existing index '%s'", self.index_name)
                self.es.indices.delete(index=self.index_name)
            else:
                logger.info("Index '%s' exists. Skipping creation.", self.index_name)
                return
        body = build_medical_mapping(dimension)
        logger.info("Creating index '%s' with dims=%d", self.index_name, dimension)
        self.es.indices.create(index=self.index_name, body=body)
        logger.info("Index '%s' created successfully", self.index_name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @staticmethod
    def build_source(chunk: Chunk) -> Dict[str, Any]:
        """Flatten a :class:`~domain.models.Chunk` into an ES ``_source``."""
        source: Dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "text": chunk.text,
            "embedding": chunk.embedding,
            "chunk_index": chunk.chunk_index,
            "source_file": chunk.source_file,
            "file_name": chunk.file_name,
            "file_type": chunk.file_type,
            "page_number": chunk.page_number,
            "modified_date": chunk.modified_date,
        }
        metadata = chunk.metadata or {}
        for key in MEDICAL_METADATA_FIELDS:
            if key in metadata:
                source[key] = metadata[key]
        # Fall back to dataset_type on the chunk itself (Phase 4 convention).
        if "dataset_type" not in source:
            source["dataset_type"] = "default"
        return source

    def index_chunks(self, chunks: List[Chunk]) -> int:
        self._guard("index_chunks")
        if self.es is None:
            logger.error("Cannot index chunks: Elasticsearch client is unavailable.")
            return 0
        if not chunks:
            return 0

        operations: List[Dict[str, Any]] = []
        for chunk in chunks:
            # es.bulk expects alternating action / source lines.
            operations.append(
                {"index": {"_index": self.index_name, "_id": chunk.chunk_id}}
            )
            operations.append(self.build_source(chunk))

        response = self.es.bulk(operations=operations)
        items = response.get("items", []) if hasattr(response, "get") else []
        success_count = 0
        error_count = 0
        for item in items:
            op = next(iter(item.values())) if item else {}
            if op.get("error"):
                error_count += 1
            else:
                success_count += 1

        if error_count:
            logger.error("Bulk indexing had %d errors", error_count)
        logger.info(
            "Indexed %d/%d chunks into '%s'",
            success_count,
            len(chunks),
            self.index_name,
        )
        return success_count

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        if self.es is None:
            return {"exists": False, "index": self.index_name, "doc_count": 0}
        if not self.es.indices.exists(index=self.index_name):
            return {"exists": False, "index": self.index_name, "doc_count": 0}
        stats = self.es.indices.stats(index=self.index_name)
        primaries = stats["indices"][self.index_name]["primaries"]
        return {
            "exists": True,
            "index": self.index_name,
            "doc_count": primaries["docs"]["count"],
            "store_bytes": primaries.get("store", {}).get("size_in_bytes", 0),
            "dataset_type": (
                "default" if is_protected(self.index_name) else "user"
            ),
        }


__all__ = [
    "MEDICAL_DEFAULT_INDEX",
    "MEDICAL_METADATA_FIELDS",
    "MEDICAL_USER_INDEX",
    "PROTECTED_DEFAULT_INDEXES",
    "DatasetProtectionError",
    "MedicalVectorStore",
    "assert_writable_index",
    "build_medical_mapping",
    "clear_user_data",
    "is_protected",
    "user_vector_store",
]
