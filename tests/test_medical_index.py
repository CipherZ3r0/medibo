"""Unit tests for the Phase 6 medical index: mapping, metadata persistence,
default-dataset protection, and user-data deletion isolation.

All tests are hermetic — Elasticsearch is replaced by an injected fake client,
so no network access is required.
"""

import unittest
from typing import Any, Dict, List

from domain.models import Chunk
from ingestion.medical_index import (
    MEDICAL_DEFAULT_INDEX,
    MEDICAL_METADATA_FIELDS,
    MEDICAL_USER_INDEX,
    DatasetProtectionError,
    MedicalVectorStore,
    assert_writable_index,
    build_medical_mapping,
    clear_user_data,
    is_protected,
    user_vector_store,
)

# ---------------------------------------------------------------------------
# Fake Elasticsearch client
# ---------------------------------------------------------------------------

class FakeIndices:
    def __init__(self, client: "FakeElasticsearch") -> None:
        self._client = client

    def exists(self, index: str) -> bool:
        return index in self._client.indices_data

    def create(self, index: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if index in self._client.indices_data:
            raise RuntimeError(f"index already exists: {index}")
        self._client.indices_data[index] = {"mapping": body, "docs": {}}
        return {"acknowledged": True, "index": index}

    def delete(self, index: str) -> Dict[str, Any]:
        self._client.deleted_indices.append(index)
        self._client.indices_data.pop(index, None)
        return {"acknowledged": True}

    def stats(self, index: str) -> Dict[str, Any]:
        docs = self._client.indices_data[index]["docs"]
        return {
            "indices": {
                index: {
                    "primaries": {
                        "docs": {"count": len(docs)},
                        "store": {"size_in_bytes": sum(
                            384 * 4 for _ in docs
                        )},
                    }
                }
            }
        }

    def refresh(self, index: str) -> Dict[str, Any]:
        self._client.refreshed.append(index)
        return {"acknowledged": True}


class FakeElasticsearch:
    """Minimal stand-in recording every call made against it."""

    def __init__(self, indices: Dict[str, Dict[str, Any]] | None = None) -> None:
        self.indices_data: Dict[str, Dict[str, Any]] = indices or {}
        self.indices = FakeIndices(self)
        self.bulk_calls: List[List[Dict[str, Any]]] = []
        self.deleted_indices: List[str] = []
        self.refreshed: List[str] = []

    def ping(self) -> bool:
        return True

    def bulk(self, operations: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        operations = operations or []
        self.bulk_calls.append(operations)
        items = []
        # operations alternate: action line, source line
        for i in range(0, len(operations), 2):
            action = operations[i]
            source = operations[i + 1]
            op_name, meta = next(iter(action.items()))
            index, doc_id = meta["_index"], meta["_id"]
            if index not in self.indices_data:
                items.append({op_name: {"_index": index, "_id": doc_id,
                                        "status": 404, "error": {"type": "index_not_found"}}})
                continue
            self.indices_data[index]["docs"][doc_id] = source
            items.append({op_name: {"_index": index, "_id": doc_id, "status": 201}})
        return {"errors": any("error" in next(iter(i.values())) for i in items),
                "items": items}

    def get(self, index: str, id: str) -> Dict[str, Any]:
        doc = self.indices_data[index]["docs"][id]
        return {"_index": index, "_id": id, "_source": doc}

    def cat(self) -> None:  # pragma: no cover - not used in tests
        raise NotImplementedError


def make_chunk(**overrides: Any) -> Chunk:
    kwargs: Dict[str, Any] = dict(
        chunk_id="chunk-1",
        doc_id="doc-1",
        text="Paracetamol dosing in adults",
        chunk_index=0,
        source_file="/data/processed/doc.json",
        file_name="doc.json",
        file_type=".json",
        page_number=3,
        embedding=[0.1] * 384,
        metadata={
            "title": "Clinical guidance",
            "source": "moHFW",
            "source_url": "https://example.gov.in/guidance",
            "region": "india",
            "authority": "Ministry of Health and Family Welfare",
            "document_type": "clinical_guideline",
            "disease_topic": "fever",
            "published_at": "2019",
            "updated_at": "2026-10-07",
            "section": "Dosing",
            "dataset_type": "default",
            "license": "Government of India — open licence",
        },
    )
    kwargs.update(overrides)
    return Chunk(**kwargs)


# ---------------------------------------------------------------------------
# 1. Index mapping
# ---------------------------------------------------------------------------

class TestIndexMapping(unittest.TestCase):
    def setUp(self) -> None:
        self.mapping = build_medical_mapping(384)

    def test_settings_and_top_level_keys(self) -> None:
        self.assertEqual(self.mapping["settings"]["number_of_shards"], 1)
        self.assertEqual(self.mapping["settings"]["number_of_replicas"], 0)
        self.assertIn("mappings", self.mapping)

    def test_embedding_field_is_dense_vector(self) -> None:
        emb = self.mapping["mappings"]["properties"]["embedding"]
        self.assertEqual(emb["type"], "dense_vector")
        self.assertEqual(emb["dims"], 384)
        self.assertTrue(emb["index"])
        self.assertEqual(emb["similarity"], "cosine")

    def test_identity_and_provenance_field_types(self) -> None:
        props = self.mapping["mappings"]["properties"]
        expected = {
            "chunk_id": "keyword",
            "doc_id": "keyword",
            "text": "text",
            "chunk_index": "integer",
            "source_file": "keyword",
            "file_name": "keyword",
            "file_type": "keyword",
            "page_number": "integer",
            "modified_date": "date",
        }
        for field, ftype in expected.items():
            self.assertEqual(props[field]["type"], ftype, field)

    def test_all_required_metadata_fields_present(self) -> None:
        props = self.mapping["mappings"]["properties"]
        required = {
            "title": "text",
            "source": "keyword",
            "source_url": "keyword",
            "region": "keyword",
            "authority": "text",
            "document_type": "keyword",
            "disease_topic": "keyword",
            "published_at": "date",
            "updated_at": "date",
            "page_number": "integer",
            "section": "text",
            "dataset_type": "keyword",
        }
        for field, ftype in required.items():
            self.assertIn(field, props, f"missing mapping for {field}")
            self.assertEqual(props[field]["type"], ftype, field)

    def test_partial_dates_supported(self) -> None:
        fmt = self.mapping["mappings"]["properties"]["published_at"]["format"]
        self.assertIn("yyyy-MM", fmt)
        self.assertIn("yyyy", fmt)

    def test_default_index_names(self) -> None:
        self.assertEqual(MEDICAL_DEFAULT_INDEX, "medibo_default")
        self.assertEqual(MEDICAL_USER_INDEX, "medibo_user")
        self.assertTrue(is_protected(MEDICAL_DEFAULT_INDEX))
        self.assertFalse(is_protected(MEDICAL_USER_INDEX))


# ---------------------------------------------------------------------------
# 2. Metadata persistence (round-trip through a fake ES client)
# ---------------------------------------------------------------------------

class TestMetadataPersistence(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeElasticsearch()
        self.store = MedicalVectorStore(
            index_name=MEDICAL_DEFAULT_INDEX,
            es_client=self.fake,
            allow_default_write=True,
        )
        self.store.create_index(384, recreate=True)
        self.chunk = make_chunk()

    def test_index_created_with_medical_mapping(self) -> None:
        body = self.fake.indices_data[MEDICAL_DEFAULT_INDEX]["mapping"]
        self.assertEqual(
            body["mappings"]["properties"]["embedding"]["dims"], 384
        )

    def test_chunk_source_contains_all_metadata(self) -> None:
        n = self.store.index_chunks([self.chunk])
        self.assertEqual(n, 1)
        stored = self.fake.indices_data[MEDICAL_DEFAULT_INDEX]["docs"][
            self.chunk.chunk_id
        ]
        for field in MEDICAL_METADATA_FIELDS:
            self.assertIn(field, stored, f"missing metadata field {field}")
        self.assertEqual(stored["title"], "Clinical guidance")
        self.assertEqual(stored["source"], "moHFW")
        self.assertEqual(stored["source_url"], "https://example.gov.in/guidance")
        self.assertEqual(stored["region"], "india")
        self.assertEqual(
            stored["authority"], "Ministry of Health and Family Welfare"
        )
        self.assertEqual(stored["document_type"], "clinical_guideline")
        self.assertEqual(stored["disease_topic"], "fever")
        self.assertEqual(stored["published_at"], "2019")
        self.assertEqual(stored["updated_at"], "2026-10-07")
        self.assertEqual(stored["section"], "Dosing")
        self.assertEqual(stored["dataset_type"], "default")
        self.assertEqual(stored["page_number"], 3)
        self.assertEqual(len(stored["embedding"]), 384)

    def test_round_trip_via_get(self) -> None:
        self.store.index_chunks([self.chunk])
        doc = self.fake.get(MEDICAL_DEFAULT_INDEX, self.chunk.chunk_id)
        source = doc["_source"]
        self.assertEqual(source["chunk_id"], self.chunk.chunk_id)
        self.assertEqual(source["text"], self.chunk.text)
        self.assertEqual(source["license"], "Government of India — open licence")

    def test_missing_metadata_defaults_dataset_type(self) -> None:
        bare = make_chunk(metadata={})
        self.store.index_chunks([bare])
        stored = self.fake.indices_data[MEDICAL_DEFAULT_INDEX]["docs"][
            bare.chunk_id
        ]
        self.assertEqual(stored["dataset_type"], "default")
        for field in MEDICAL_METADATA_FIELDS:
            if field == "dataset_type":
                continue
            self.assertNotIn(field, stored)

    def test_multiple_chunks_counted_correctly(self) -> None:
        chunks = [make_chunk(chunk_id=f"c{i}") for i in range(4)]
        self.assertEqual(self.store.index_chunks(chunks), 4)
        stats = self.store.get_stats()
        self.assertEqual(stats["doc_count"], 4)
        self.assertEqual(stats["dataset_type"], "default")

    def test_build_source_from_real_corpus_row(self) -> None:
        """Round-trip a real Phase 4 chunk row end-to-end."""
        row: Dict[str, Any] = {
            "chunk_id": "real-chunk",
            "doc_id": "icmr-doc",
            "text": "text",
            "chunk_index": 2,
            "source_file": "/x/processed/icmr-doc.json",
            "file_name": "icmr-doc.json",
            "file_type": ".json",
            "page_number": 7,
            "section": "Sepsis",
            "dataset_type": "default",
            "title": "Treatment Guidelines",
            "source": "icmr",
            "source_url": "https://www.icmr.gov.in/x.pdf",
            "region": "india",
            "authority": "ICMR",
            "document_type": "treatment_guideline",
            "disease_topic": "sepsis",
            "published_at": "2019",
            "updated_at": None,
            "license": "ICMR copyright",
        }
        metadata = {k: row[k] for k in MEDICAL_METADATA_FIELDS if k in row}
        chunk = Chunk(
            chunk_id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            chunk_index=row["chunk_index"],
            source_file=row["source_file"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            page_number=row["page_number"],
            embedding=[0.2] * 384,
            metadata=metadata,
        )
        self.store.index_chunks([chunk])
        stored = self.fake.indices_data[MEDICAL_DEFAULT_INDEX]["docs"]["real-chunk"]
        self.assertEqual(stored["section"], "Sepsis")
        self.assertIsNone(stored["updated_at"])
        self.assertEqual(stored["dataset_type"], "default")
        # Non-metadata bookkeeping field is preserved too
        self.assertEqual(stored["file_name"], "icmr-doc.json")


# ---------------------------------------------------------------------------
# 3. Default dataset protection
# ---------------------------------------------------------------------------

class TestDefaultDatasetProtection(unittest.TestCase):
    def test_default_is_protected(self) -> None:
        self.assertTrue(is_protected(MEDICAL_DEFAULT_INDEX))
        with self.assertRaises(DatasetProtectionError):
            assert_writable_index(MEDICAL_DEFAULT_INDEX, "write")
        with self.assertRaises(DatasetProtectionError):
            assert_writable_index(MEDICAL_DEFAULT_INDEX, "delete")
        # user index is writable
        assert_writable_index(MEDICAL_USER_INDEX, "write")

    def test_protection_error_is_permission_error(self) -> None:
        self.assertTrue(
            issubclass(DatasetProtectionError, PermissionError)
        )

    def test_ui_path_cannot_write_default_index(self) -> None:
        fake = FakeElasticsearch()
        store = MedicalVectorStore(
            index_name=MEDICAL_DEFAULT_INDEX, es_client=fake
        )  # allow_default_write=False (UI/API default)
        with self.assertRaises(DatasetProtectionError):
            store.index_chunks([make_chunk()])
        with self.assertRaises(DatasetProtectionError):
            store.create_index(384, recreate=True)
        self.assertEqual(fake.bulk_calls, [])
        self.assertNotIn(MEDICAL_DEFAULT_INDEX, fake.indices_data)

    def test_ui_path_cannot_recreate_default_index(self) -> None:
        fake = FakeElasticsearch(
            indices={MEDICAL_DEFAULT_INDEX: {"mapping": {}, "docs": {}}}
        )
        store = MedicalVectorStore(
            index_name=MEDICAL_DEFAULT_INDEX, es_client=fake
        )
        with self.assertRaises(DatasetProtectionError):
            store.create_index(384, recreate=True)
        # the pre-existing index must survive untouched
        self.assertIn(MEDICAL_DEFAULT_INDEX, fake.indices_data)

    def test_backend_opt_in_allows_default_write(self) -> None:
        fake = FakeElasticsearch()
        store = MedicalVectorStore(
            index_name=MEDICAL_DEFAULT_INDEX,
            es_client=fake,
            allow_default_write=True,
        )
        store.create_index(384, recreate=True)
        store.index_chunks([make_chunk()])
        self.assertEqual(
            len(fake.indices_data[MEDICAL_DEFAULT_INDEX]["docs"]), 1
        )

    def test_user_store_is_bound_to_user_index(self) -> None:
        fake = FakeElasticsearch()
        store = user_vector_store(es_client=fake)
        self.assertEqual(store.index_name, MEDICAL_USER_INDEX)
        self.assertFalse(store.allow_default_write)
        store.create_index(384, recreate=False)
        store.index_chunks([make_chunk()])
        self.assertIn(MEDICAL_USER_INDEX, fake.indices_data)
        self.assertNotIn(MEDICAL_DEFAULT_INDEX, fake.indices_data)

    def test_default_write_guard_message_mentions_read_only(self) -> None:
        fake = FakeElasticsearch()
        store = MedicalVectorStore(
            index_name=MEDICAL_DEFAULT_INDEX, es_client=fake
        )
        with self.assertRaises(DatasetProtectionError) as ctx:
            store.index_chunks([make_chunk()])
        self.assertIn("read-only", str(ctx.exception))


# ---------------------------------------------------------------------------
# 4. User deletion isolation
# ---------------------------------------------------------------------------

class TestUserDeletionIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeElasticsearch(
            indices={
                MEDICAL_DEFAULT_INDEX: {
                    "mapping": {},
                    "docs": {"keep-1": {"text": "default"}},
                },
                MEDICAL_USER_INDEX: {
                    "mapping": {},
                    "docs": {"user-1": {"text": "user"}},
                },
            }
        )

    def test_clear_user_data_deletes_only_user_index(self) -> None:
        result = clear_user_data(es=self.fake)
        self.assertTrue(result["deleted"])
        self.assertEqual(result["index"], MEDICAL_USER_INDEX)
        self.assertNotIn(MEDICAL_USER_INDEX, self.fake.indices_data)
        self.assertIn(MEDICAL_DEFAULT_INDEX, self.fake.indices_data)
        self.assertTrue(result["default_still_present"])
        self.assertEqual(
            self.fake.indices_data[MEDICAL_DEFAULT_INDEX]["docs"]["keep-1"]["text"],
            "default",
        )

    def test_clear_user_data_never_deletes_default(self) -> None:
        with self.assertRaises(DatasetProtectionError):
            clear_user_data(es=self.fake, index_name=MEDICAL_DEFAULT_INDEX)
        # nothing was deleted at all
        self.assertEqual(self.fake.deleted_indices, [])
        self.assertIn(MEDICAL_DEFAULT_INDEX, self.fake.indices_data)
        self.assertIn(MEDICAL_USER_INDEX, self.fake.indices_data)

    def test_clear_user_data_on_missing_user_index_is_safe(self) -> None:
        self.fake.indices_data.pop(MEDICAL_USER_INDEX)
        result = clear_user_data(es=self.fake)
        self.assertFalse(result["deleted"])
        self.assertTrue(result["default_still_present"])

    def test_clear_user_data_returns_default_index_name(self) -> None:
        with self.assertRaises(DatasetProtectionError) as ctx:
            clear_user_data(es=self.fake, index_name=MEDICAL_DEFAULT_INDEX)
        self.assertIn(MEDICAL_DEFAULT_INDEX, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
