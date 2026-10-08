"""Phase 6.6 — application wiring: retrieval reads medibo_default + medibo_user,
uploads land in medibo_user, "Clear my data" removes user data only.

All tests are hermetic: Elasticsearch is replaced by an in-memory fake that
supports the handful of APIs the application path uses (search, count, bulk,
index management). RRF fusion, the candidate pool, the reranker, and the
embedding model are not touched by these tests — they exercise the existing
pipeline with the index set swapped.
"""

import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import config
from domain.models import Chunk
from ingestion.indexers import ElasticsearchVectorStore
from ingestion.medical_index import (
    MEDICAL_DEFAULT_INDEX,
    MEDICAL_USER_INDEX,
    DatasetProtectionError,
    MedicalVectorStore,
    assert_writable_index,
    clear_user_data,
    dataset_counts,
    is_protected,
    retrieval_store,
    user_vector_store,
)
from rag.pipeline import RAGPipeline, default_retrieval_store
from rag.prompt_builders import TemplatePromptBuilder
from rag.query_rewriters import NoOpQueryRewriter
from rag.rerankers import NoOpReranker
from rag.retrievers import HybridSearchStrategy

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeIndices:
    def __init__(self, client: "FakeElasticsearch") -> None:
        self._client = client

    def exists(self, index: str) -> bool:
        return index in self._client.indices_data

    def create(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self._client.indices_data.setdefault(index, {"mapping": body, "docs": {}})
        return {"acknowledged": True, "index": index}

    def delete(self, index: str) -> dict[str, Any]:
        self._client.deleted_indices.append(index)
        self._client.indices_data.pop(index, None)
        return {"acknowledged": True}

    def stats(self, index: str) -> dict[str, Any]:
        return {
            "indices": {
                index: {
                    "primaries": {
                        "docs": {
                            "count": len(
                                self._client.indices_data.get(index, {}).get("docs", {})
                            )
                        }
                    }
                }
            }
        }


class FakeElasticsearch:
    """Enough Elasticsearch for the application read/write path."""

    def __init__(self, indices: dict[str, dict[str, Any]] | None = None) -> None:
        self.indices_data: dict[str, dict[str, Any]] = indices or {}
        self.indices = FakeIndices(self)
        self.searched_indexes: list[str] = []
        self.deleted_indices: list[str] = []
        self.bulk_calls: list[list[dict[str, Any]]] = []

    def ping(self) -> bool:
        return True

    def close(self) -> None:  # called by dataset_counts() when it owns a client
        pass

    def count(self, index: str) -> dict[str, Any]:
        total = 0
        for name in str(index).split(","):
            name = name.strip()
            if name not in self.indices_data:
                raise RuntimeError(f"index_not_found_exception: {name}")
            total += len(self.indices_data[name]["docs"])
        return {"count": total}

    def bulk(self, operations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        operations = operations or []
        self.bulk_calls.append(operations)
        items = []
        for i in range(0, len(operations), 2):
            action, source = operations[i], operations[i + 1]
            op_name, meta = next(iter(action.items()))
            index, doc_id = meta["_index"], meta["_id"]
            if index not in self.indices_data:
                items.append({op_name: {"_index": index, "_id": doc_id, "status": 404,
                                        "error": {"type": "index_not_found"}}})
                continue
            self.indices_data[index]["docs"][doc_id] = source
            items.append({op_name: {"_index": index, "_id": doc_id, "status": 201}})
        return {"errors": False, "items": items}

    def search(self, index: str | None = None, body: dict[str, Any] | None = None,
               **kwargs: Any) -> dict[str, Any]:
        body = body or {}
        self.searched_indexes.append(index)
        requested = [i.strip() for i in str(index).split(",") if i.strip()]
        missing = [i for i in requested if i not in self.indices_data]
        if missing and not kwargs.get("ignore_unavailable"):
            raise RuntimeError(f"index_not_found_exception: {missing}")

        size = int(body.get("size", 10))
        knn = body.get("knn")
        match_text = ""
        query = body.get("query") or {}
        if "match" in query:
            match_text = query["match"]["text"]["query"]

        hits: list[dict[str, Any]] = []
        for name in requested:
            for doc_id, src in self.indices_data.get(name, {}).get("docs", {}).items():
                if knn is not None:
                    score = _cosine(
                        knn.get("query_vector") or [], src.get("embedding") or []
                    )
                else:
                    score = _token_overlap(match_text, src.get("text", ""))
                if score <= 0:
                    continue
                hits.append({
                    "_index": name,
                    "_id": doc_id,
                    "_score": float(score),
                    "_source": src,
                })
        hits.sort(key=lambda h: h["_score"], reverse=True)
        return {"hits": {"hits": hits[:size]}}


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _token_overlap(query: str, text: str) -> float:
    q = {t for t in query.lower().split() if t.isalpha()}
    t = {t for t in text.lower().split() if t.isalpha()}
    return float(len(q & t))


class FakeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        return [1.0] * 384


class FakeLLM:
    def generate(self, messages, model=None):
        return {
            "answer": "stubbed answer",
            "model": model or "stub",
            "response_time_ms": 1,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def medical_chunk(chunk_id: str, text: str, **metadata: Any) -> Chunk:
    meta = {
        "title": "Dengue clinical guidance",
        "source": "mohfw",
        "source_url": "https://mohfw.gov.in/dengue",
        "section": "Symptoms",
        "region": "india",
        "authority": "MoHFW",
        "document_type": "clinical_guideline",
        "disease_topic": "dengue",
        "published_at": "2019",
        "license": "Open Government Licence",
        "dataset_type": "default",
    }
    meta.update(metadata)
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        text=text,
        chunk_index=0,
        source_file=f"/data/{chunk_id}.json",
        file_name=f"{chunk_id}.json",
        file_type=".json",
        page_number=1,
        embedding=[1.0] * 384,
        metadata=meta,
    )


DEFAULT_DOC_IDS = ["default-dengue", "default-malaria"]
USER_DOC_IDS = ["user-dengue-note"]


def seed_default(fake: FakeElasticsearch) -> None:
    fake.indices_data[MEDICAL_DEFAULT_INDEX] = {"mapping": {}, "docs": {}}
    store = MedicalVectorStore(
        index_name=MEDICAL_DEFAULT_INDEX, es_client=fake, allow_default_write=True
    )
    store.index_chunks([
        medical_chunk(
            "default-dengue",
            "Common symptoms of dengue fever include high fever, severe headache, "
            "pain behind the eyes, joint pain, nausea and rash.",
        ),
        medical_chunk(
            "default-malaria",
            "Malaria typically presents with fever, chills and sweating episodes.",
            title="Malaria guidance",
            source="icmr",
            disease_topic="malaria",
        ),
    ])


def seed_user(fake: FakeElasticsearch) -> None:
    fake.indices_data.setdefault(MEDICAL_USER_INDEX, {"mapping": {}, "docs": {}})
    store = user_vector_store(es_client=fake)
    store.index_chunks([
        medical_chunk(
            "user-dengue-note",
            "My clinic note: dengue fever cases rising, watch for warning signs.",
            dataset_type="user",
            title="Clinic note",
            source="my_upload",
        )
    ])


def app_pipeline(store: ElasticsearchVectorStore) -> RAGPipeline:
    """RAGPipeline wired exactly like the Streamlit app, minus the network."""
    return RAGPipeline(
        search_strategy=HybridSearchStrategy(embedder=FakeEmbedder(), vector_store=store),
        vector_store=store,
        reranker=NoOpReranker(),
        query_rewriter=NoOpQueryRewriter(),
        prompt_builder=TemplatePromptBuilder(),
        llm_provider=FakeLLM(),
        repository=MagicMock(),
    )


# ---------------------------------------------------------------------------
# 1. Default data is always included in retrieval
# ---------------------------------------------------------------------------

class TestDefaultDataInRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeElasticsearch()
        seed_default(self.fake)  # medibo_user does not exist yet
        self.store = ElasticsearchVectorStore(
            index_name=MEDICAL_DEFAULT_INDEX,
            search_indexes=config.MEDICAL_RETRIEVAL_INDEXES,
            es_client=self.fake,
        )

    def test_read_path_targets_default_and_user_indexes(self) -> None:
        self.assertEqual(
            config.MEDICAL_RETRIEVAL_INDEXES,
            f"{MEDICAL_DEFAULT_INDEX},{MEDICAL_USER_INDEX}",
        )
        self.assertEqual(self.store.search_indexes, config.MEDICAL_RETRIEVAL_INDEXES)

    def test_default_documents_are_returned(self) -> None:
        response = app_pipeline(self.store).query(
            "What are common symptoms of dengue?",
            use_reranking=False,
            use_query_rewriting=False,
            save_to_db=False,
        )
        ids = [d.chunk_id for d in response.retrieved_docs]
        self.assertIn("default-dengue", ids)

    def test_retrieval_survives_a_missing_user_index(self) -> None:
        self.assertNotIn(MEDICAL_USER_INDEX, self.fake.indices_data)
        response = app_pipeline(self.store).query(
            "What are common symptoms of dengue?",
            use_reranking=False,
            use_query_rewriting=False,
            save_to_db=False,
        )
        self.assertTrue(response.retrieved_docs)

    def test_medical_metadata_survives_retrieval(self) -> None:
        response = app_pipeline(self.store).query(
            "What are common symptoms of dengue?",
            use_reranking=False,
            use_query_rewriting=False,
            save_to_db=False,
        )
        doc = next(d for d in response.retrieved_docs if d.chunk_id == "default-dengue")
        self.assertEqual(doc.metadata.get("source"), "mohfw")
        self.assertEqual(doc.metadata.get("title"), "Dengue clinical guidance")
        self.assertEqual(doc.metadata.get("section"), "Symptoms")
        self.assertEqual(doc.metadata.get("source_url"), "https://mohfw.gov.in/dengue")
        self.assertEqual(doc.page_number, 1)

    def test_never_queries_the_legacy_potbot_index(self) -> None:
        app_pipeline(self.store).query(
            "What are common symptoms of dengue?",
            use_reranking=False,
            use_query_rewriting=False,
            save_to_db=False,
        )
        self.assertTrue(self.fake.searched_indexes)
        for index_expr in self.fake.searched_indexes:
            self.assertNotIn("potbot_documents", index_expr)


# ---------------------------------------------------------------------------
# 2. User data is included when present
# ---------------------------------------------------------------------------

class TestUserDataInRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeElasticsearch()
        seed_default(self.fake)
        seed_user(self.fake)
        self.store = ElasticsearchVectorStore(
            index_name=MEDICAL_DEFAULT_INDEX,
            search_indexes=config.MEDICAL_RETRIEVAL_INDEXES,
            es_client=self.fake,
        )

    def test_both_datasets_are_searched(self) -> None:
        response = app_pipeline(self.store).query(
            "What are common symptoms of dengue?",
            use_reranking=False,
            use_query_rewriting=False,
            save_to_db=False,
        )
        ids = {d.chunk_id for d in response.retrieved_docs}
        self.assertIn("user-dengue-note", ids)
        self.assertTrue(set(DEFAULT_DOC_IDS) & ids)

    def test_user_results_are_tagged_as_user_data(self) -> None:
        response = app_pipeline(self.store).query(
            "What are common symptoms of dengue?",
            use_reranking=False,
            use_query_rewriting=False,
            save_to_db=False,
        )
        user_doc = next(d for d in response.retrieved_docs if d.chunk_id == "user-dengue-note")
        self.assertEqual(user_doc.metadata.get("dataset_type"), "user")
        self.assertEqual(user_doc.metadata.get("source"), "my_upload")


# ---------------------------------------------------------------------------
# 3. + 4. Clear my data: user data only, default data survives
# ---------------------------------------------------------------------------

class TestClearMyDataIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeElasticsearch()
        seed_default(self.fake)
        seed_user(self.fake)
        self.store = ElasticsearchVectorStore(
            index_name=MEDICAL_DEFAULT_INDEX,
            search_indexes=config.MEDICAL_RETRIEVAL_INDEXES,
            es_client=self.fake,
        )

    def test_clear_removes_only_the_user_dataset(self) -> None:
        result = clear_user_data(es=self.fake)
        self.assertTrue(result["deleted"])
        self.assertEqual(result["index"], MEDICAL_USER_INDEX)
        self.assertNotIn(MEDICAL_USER_INDEX, self.fake.indices_data)
        self.assertIn(MEDICAL_DEFAULT_INDEX, self.fake.indices_data)

    def test_default_data_remains_after_clearing(self) -> None:
        clear_user_data(es=self.fake)
        default_docs = self.fake.indices_data[MEDICAL_DEFAULT_INDEX]["docs"]
        self.assertEqual(sorted(default_docs), sorted(DEFAULT_DOC_IDS))

    def test_retrieval_still_returns_default_data_after_clearing(self) -> None:
        clear_user_data(es=self.fake)
        response = app_pipeline(self.store).query(
            "What are common symptoms of dengue?",
            use_reranking=False,
            use_query_rewriting=False,
            save_to_db=False,
        )
        ids = {d.chunk_id for d in response.retrieved_docs}
        self.assertIn("default-dengue", ids)
        self.assertNotIn("user-dengue-note", ids)

    def test_clearing_twice_is_safe(self) -> None:
        clear_user_data(es=self.fake)
        result = clear_user_data(es=self.fake)
        self.assertFalse(result["deleted"])
        self.assertIn(MEDICAL_DEFAULT_INDEX, self.fake.indices_data)


# ---------------------------------------------------------------------------
# 5. UI/API cannot delete or write the default dataset
# ---------------------------------------------------------------------------

class TestDefaultDatasetIsReadOnly(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeElasticsearch()
        seed_default(self.fake)

    def test_clear_cannot_target_the_default_dataset(self) -> None:
        with self.assertRaises(DatasetProtectionError):
            clear_user_data(es=self.fake, index_name=MEDICAL_DEFAULT_INDEX)
        self.assertIn(MEDICAL_DEFAULT_INDEX, self.fake.indices_data)
        self.assertEqual(self.fake.deleted_indices, [])

    def test_assert_writable_rejects_default_delete(self) -> None:
        with self.assertRaises(DatasetProtectionError):
            assert_writable_index(MEDICAL_DEFAULT_INDEX, operation="delete")
        assert_writable_index(MEDICAL_USER_INDEX, operation="delete")

    def test_app_store_cannot_write_the_default_dataset(self) -> None:
        store = retrieval_store(es_client=self.fake)
        self.assertFalse(store.allow_default_write)
        self.assertTrue(is_protected(store.index_name))
        bulk_calls_before = len(self.fake.bulk_calls)
        with self.assertRaises(DatasetProtectionError):
            store.index_chunks([medical_chunk("x", "text")])
        with self.assertRaises(DatasetProtectionError):
            store.create_index(384, recreate=True)
        # nothing new reached Elasticsearch and the seeded docs are untouched
        self.assertEqual(len(self.fake.bulk_calls), bulk_calls_before)
        self.assertEqual(
            sorted(self.fake.indices_data[MEDICAL_DEFAULT_INDEX]["docs"]),
            sorted(DEFAULT_DOC_IDS),
        )

    def test_ingestion_default_target_is_the_user_dataset(self) -> None:
        fake_store = MagicMock()
        with patch("ingestion.pipeline.user_vector_store", return_value=fake_store) as factory:
            from ingestion.pipeline import IngestionPipeline

            embedder = MagicMock()
            embedder.stream_embed.side_effect = lambda chunks, **kw: iter(chunks)
            embedder.get_dimension.return_value = 384
            pipeline = IngestionPipeline(embedder=embedder)
        factory.assert_called_once()
        self.assertIs(pipeline.vector_store, fake_store)
        self.assertEqual(config.MEDICAL_UPLOAD_INDEX, MEDICAL_USER_INDEX)
        self.assertFalse(is_protected(config.MEDICAL_UPLOAD_INDEX))


# ---------------------------------------------------------------------------
# 6. Counts distinguish default and user data
# ---------------------------------------------------------------------------

class TestDatasetCounts(unittest.TestCase):
    def setUp(self) -> None:
        self.fake = FakeElasticsearch()
        seed_default(self.fake)
        seed_user(self.fake)

    def test_counts_are_reported_separately(self) -> None:
        counts = dataset_counts(es=self.fake)
        self.assertEqual(counts["default"], 2)
        self.assertEqual(counts["user"], 1)
        self.assertEqual(counts["default_index"], MEDICAL_DEFAULT_INDEX)
        self.assertEqual(counts["user_index"], MEDICAL_USER_INDEX)

    def test_counts_update_when_user_data_is_cleared(self) -> None:
        before = dataset_counts(es=self.fake)
        clear_user_data(es=self.fake)
        after = dataset_counts(es=self.fake)
        self.assertEqual(after["user"], 0)
        self.assertEqual(after["default"], before["default"])

    def test_counts_zero_when_indexes_absent(self) -> None:
        counts = dataset_counts(es=FakeElasticsearch())
        self.assertEqual(counts["default"], 0)
        self.assertEqual(counts["user"], 0)

    def test_counts_match_stored_documents(self) -> None:
        counts = dataset_counts(es=self.fake)
        stored_default = len(
            self.fake.indices_data[MEDICAL_DEFAULT_INDEX]["docs"]
        )
        self.assertEqual(counts["default"], stored_default)


# ---------------------------------------------------------------------------
# Store factories used by the application
# ---------------------------------------------------------------------------

class TestStoreFactories(unittest.TestCase):
    def test_app_read_path_factory_is_the_medical_read_only_store(self) -> None:
        sentinel = object()
        with patch("rag.pipeline.retrieval_store", return_value=sentinel) as factory:
            self.assertIs(default_retrieval_store(), sentinel)
        factory.assert_called_once_with()

    def test_retrieval_store_is_read_only_on_default_index(self) -> None:
        store = retrieval_store(es_client=FakeElasticsearch())
        self.assertEqual(store.index_name, MEDICAL_DEFAULT_INDEX)
        self.assertEqual(store.search_indexes, config.MEDICAL_RETRIEVAL_INDEXES)
        self.assertFalse(store.allow_default_write)
        with self.assertRaises(DatasetProtectionError):
            store.index_chunks([medical_chunk("x", "y")])

    def test_user_store_targets_user_index(self) -> None:
        store = user_vector_store(es_client=FakeElasticsearch())
        self.assertEqual(store.index_name, MEDICAL_USER_INDEX)
        self.assertFalse(is_protected(store.index_name))
        self.assertEqual(store.search_indexes, MEDICAL_USER_INDEX)

    def test_strategy_methods_share_the_medical_read_path(self) -> None:
        fake = FakeElasticsearch()
        seed_default(fake)
        store = retrieval_store(es_client=fake)
        pipeline = app_pipeline(store)

        with patch("rag.pipeline.VectorSearchStrategy") as vector_cls, \
                patch("rag.pipeline.TextSearchStrategy") as text_cls:
            vector_cls.return_value = MagicMock()
            text_cls.return_value = MagicMock()
            self.assertIs(pipeline._get_strategy("vector"), vector_cls.return_value)
            self.assertIs(pipeline._get_strategy("text"), text_cls.return_value)
            # strategies are cached, so a repeat call does not rebuild them
            self.assertIs(pipeline._get_strategy("vector"), vector_cls.return_value)

        vector_cls.assert_called_once_with(vector_store=store)
        text_cls.assert_called_once_with(vector_store=store)

    def test_unknown_method_still_raises(self) -> None:
        pipeline = app_pipeline(retrieval_store(es_client=FakeElasticsearch()))
        with self.assertRaises(ValueError):
            pipeline._get_strategy("semantic")


if __name__ == "__main__":
    unittest.main()
