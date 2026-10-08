"""Phase 7 — trusted evidence gate.

Covers the five required behaviours:

1. strong trusted evidence → grounded response
2. only user evidence      → clearly marked as user-provided
3. no sufficient evidence  → explicit untrusted fallback
4. default data remains trusted
5. fallback never claims unsupported source attribution

plus the isolation of the decision policy (requirement 6) and the retrieval
read path (requirement 1).  No Elasticsearch or network access is required.
"""

import unittest
from unittest.mock import MagicMock, patch

import config
from domain.models import SearchResult
from rag.evidence import (
    SOURCE_TRUSTED,
    SOURCE_UNKNOWN,
    SOURCE_USER,
    TRUST_LEVEL_NONE,
    TRUST_LEVEL_TRUSTED,
    TRUST_LEVEL_USER_ONLY,
    UNTRUSTED_ANSWER_WARNING,
    EvidenceGate,
    EvidencePolicy,
    classify_source,
)
from rag.pipeline import RAGPipeline
from rag.prompt_builders import TemplatePromptBuilder
from rag.query_rewriters import NoOpQueryRewriter
from rag.rerankers import NoOpReranker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_doc(
    chunk_id,
    text,
    rerank=None,
    dataset_type="default",
    file_name="medlineplus-dengue.json",
    extra_meta=None,
):
    meta = {"source": "medlineplus", "title": "Dengue"}
    if dataset_type is not None:
        meta["dataset_type"] = dataset_type
    if extra_meta:
        meta.update(extra_meta)
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        file_name=file_name,
        source_file=f"/data/{file_name}",
        file_type=".json",
        page_number=1,
        score=1.0,
        rerank_score=rerank,
        metadata=meta,
    )


class StubStrategy:
    """Search strategy returning a fixed result list (no ES involved)."""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return list(self.results)


class RecordingLLM:
    """LLM provider that records every prompt it is handed."""

    def __init__(self, answer="stub answer"):
        self.answer = answer
        self.prompts = []

    def generate(self, messages, model=None):
        self.prompts.append(messages)
        return {
            "answer": self.answer,
            "model": "stub-model",
            "response_time_ms": 1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }


def build_pipeline(results, llm=None, gate=None, reranker=None):
    return RAGPipeline(
        search_strategy=StubStrategy(results),
        reranker=reranker or NoOpReranker(),
        query_rewriter=NoOpQueryRewriter(),
        prompt_builder=TemplatePromptBuilder(),
        llm_provider=llm or RecordingLLM(),
        repository=MagicMock(),
        evidence_gate=gate,
    )


def run_query(pipeline, query="What are common symptoms of dengue?", rerank_top_n=3):
    return pipeline.query(
        query,
        retrieval_method="hybrid",
        use_reranking=True,
        use_query_rewriting=False,
        rerank_top_n=rerank_top_n,
        save_to_db=False,
    )


TRUSTED_DOCS = [
    make_doc("d1", "Dengue causes high fever, rash and severe headache.", rerank=7.4),
    make_doc("d2", "Warning signs include severe abdominal pain and bleeding.", rerank=6.2),
    make_doc("d3", "Seek medical care if fever lasts more than a week.", rerank=5.0),
]

USER_DOCS = [
    make_doc(
        "u1",
        "My clinic note: watch platelet counts in dengue patients.",
        rerank=5.2,
        dataset_type="user",
        file_name="my-dengue-clinic-note.txt",
    ),
]

WEAK_DOCS = [
    make_doc("w1", "Hypertension management overview.", rerank=-8.1),
    make_doc("w2", "Antimicrobial prescribing guidance.", rerank=-7.6),
    make_doc("w3", "Snakebite first aid summary.", rerank=-9.4),
]


# ---------------------------------------------------------------------------
# 1. Strong trusted evidence → grounded response
# ---------------------------------------------------------------------------

class TestStrongTrustedEvidence(unittest.TestCase):
    def setUp(self):
        self.llm = RecordingLLM(answer="Dengue typically causes fever, rash and headache.")
        self.pipeline = build_pipeline(TRUSTED_DOCS, llm=self.llm)
        self.response = run_query(self.pipeline)

    def test_decision_is_sufficient_and_trusted(self):
        evidence = self.response.evidence
        self.assertTrue(evidence.sufficient)
        self.assertEqual(evidence.trust_level, TRUST_LEVEL_TRUSTED)
        self.assertTrue(evidence.has_trusted_evidence)
        self.assertIsNone(evidence.warning)

    def test_answer_is_not_marked_as_untrusted(self):
        self.assertIsNone(self.response.evidence_warning)
        self.assertNotIn("No trusted source found", self.response.answer)
        self.assertEqual(self.response.answer, "Dengue typically causes fever, rash and headache.")

    def test_context_prompt_carries_the_evidence(self):
        system, user = self.llm.prompts[0]
        self.assertIn("Dengue causes high fever", user["content"])
        self.assertIn("medlineplus-dengue.json", user["content"])
        self.assertIn("Source type: Trusted medical source", user["content"])
        self.assertNotIn("user-provided upload", user["content"])

    def test_all_trusted_docs_are_presented_as_sources(self):
        evidence = self.response.evidence
        self.assertEqual(len(evidence.trusted_docs), 3)
        self.assertEqual(evidence.user_docs, ())
        self.assertEqual(len(evidence.supporting_docs), 3)

    def test_retrieved_docs_unchanged_by_the_gate(self):
        self.assertEqual(
            [d.chunk_id for d in self.response.retrieved_docs], ["d1", "d2", "d3"]
        )


# ---------------------------------------------------------------------------
# 2. Only user evidence → clearly marked as user-provided
# ---------------------------------------------------------------------------

class TestUserOnlyEvidence(unittest.TestCase):
    def setUp(self):
        self.llm = RecordingLLM(answer="Watch platelet counts closely.")
        self.pipeline = build_pipeline(USER_DOCS, llm=self.llm)
        self.response = run_query(self.pipeline)

    def test_decision_is_sufficient_but_user_only(self):
        evidence = self.response.evidence
        self.assertTrue(evidence.sufficient)
        self.assertEqual(evidence.trust_level, TRUST_LEVEL_USER_ONLY)
        self.assertFalse(evidence.has_trusted_evidence)
        self.assertIsNone(evidence.warning)

    def test_user_docs_are_classified_as_user_provided(self):
        evidence = self.response.evidence
        self.assertEqual(evidence.trusted_docs, ())
        self.assertEqual(len(evidence.user_docs), 1)
        self.assertEqual(classify_source(evidence.user_docs[0]), SOURCE_USER)

    def test_answer_is_grounded_not_fallback(self):
        self.assertNotIn("No trusted source found", self.response.answer)
        self.assertIsNone(self.response.evidence_warning)

    def test_prompt_labels_the_upload_as_user_provided(self):
        _, user = self.llm.prompts[0]
        self.assertIn("Source type: User-provided upload", user["content"])
        self.assertIn("my-dengue-clinic-note.txt", user["content"])


# ---------------------------------------------------------------------------
# 3. No sufficient evidence → explicit untrusted fallback
# ---------------------------------------------------------------------------

class TestUntrustedFallback(unittest.TestCase):
    def setUp(self):
        self.llm = RecordingLLM(
            answer="General-knowledge answer about an unrelated topic."
        )
        self.pipeline = build_pipeline(WEAK_DOCS, llm=self.llm)
        self.response = run_query(self.pipeline, query="Who wrote Pride and Prejudice?")

    def test_decision_is_insufficient(self):
        evidence = self.response.evidence
        self.assertFalse(evidence.sufficient)
        self.assertEqual(evidence.trust_level, TRUST_LEVEL_NONE)
        self.assertEqual(evidence.supporting_docs, ())

    def test_response_carries_the_required_warning(self):
        self.assertEqual(self.response.evidence_warning, UNTRUSTED_ANSWER_WARNING)
        self.assertTrue(self.response.answer.startswith(UNTRUSTED_ANSWER_WARNING))
        self.assertIn("General-knowledge answer", self.response.answer)

    def test_prompt_has_no_context_documents(self):
        system, user = self.llm.prompts[0]
        self.assertEqual(system["content"], TemplatePromptBuilder.FALLBACK_SYSTEM_PROMPT)
        self.assertNotIn("Context Document", user["content"])
        self.assertNotIn("medlineplus-dengue.json", user["content"])
        self.assertNotIn("Hypertension management overview", user["content"])
        self.assertIn("Who wrote Pride and Prejudice?", user["content"])

    def test_weak_documents_are_not_offered_as_evidence(self):
        self.assertEqual(self.response.evidence.trusted_docs, ())
        self.assertEqual(self.response.evidence.user_docs, ())
        self.assertEqual(self.response.evidence.supporting_docs, ())


# ---------------------------------------------------------------------------
# 4. Default data remains trusted
# ---------------------------------------------------------------------------

class TestDefaultDataRemainsTrusted(unittest.TestCase):
    def test_default_dataset_type_is_trusted(self):
        self.assertEqual(classify_source(make_doc("a", "t", dataset_type="default")), SOURCE_TRUSTED)

    def test_missing_dataset_type_is_unknown_and_never_trusted(self):
        # Phase 8: provenance is authoritative. A chunk without the field has
        # unknown origin and must never be classified as trusted.
        self.assertEqual(
            classify_source(make_doc("a", "t", dataset_type=None)), SOURCE_UNKNOWN
        )

    def test_unknown_dataset_type_value_is_never_trusted(self):
        self.assertEqual(
            classify_source(make_doc("a", "t", dataset_type="legacy-import")), SOURCE_UNKNOWN
        )

    def test_user_dataset_type_is_never_trusted(self):
        self.assertEqual(
            classify_source(make_doc("a", "t", dataset_type="user")), SOURCE_USER
        )

    def test_mixed_evidence_keeps_the_partition(self):
        docs = TRUSTED_DOCS + USER_DOCS
        # rerank_top_n=4 so the user chunk survives re-ranking and reaches the gate
        response = run_query(build_pipeline(docs, llm=RecordingLLM()), rerank_top_n=4)
        evidence = response.evidence

        self.assertTrue(evidence.sufficient)
        self.assertEqual(evidence.trust_level, TRUST_LEVEL_TRUSTED)
        self.assertEqual(
            sorted(d.chunk_id for d in evidence.trusted_docs), ["d1", "d2", "d3"]
        )
        self.assertEqual([d.chunk_id for d in evidence.user_docs], ["u1"])
        self.assertEqual(len(evidence.supporting_docs), 4)
        # every supporting doc is attributed to exactly one bucket
        self.assertEqual(
            len(evidence.trusted_docs) + len(evidence.user_docs),
            len(evidence.supporting_docs),
        )

    def test_user_docs_never_leak_into_the_trusted_bucket(self):
        docs = USER_DOCS + TRUSTED_DOCS
        response = run_query(build_pipeline(docs, llm=RecordingLLM()))
        trusted_names = [d.file_name for d in response.evidence.trusted_docs]
        self.assertTrue(all("my-dengue-clinic-note" not in n for n in trusted_names))


# ---------------------------------------------------------------------------
# 5. Fallback never claims unsupported source attribution
# ---------------------------------------------------------------------------

class TestFallbackNeverClaimsSources(unittest.TestCase):
    def setUp(self):
        self.llm = RecordingLLM(answer="I do not have that in the knowledge base.")
        self.pipeline = build_pipeline(WEAK_DOCS, llm=self.llm)
        self.response = run_query(self.pipeline, query="What is the airspeed velocity of a swallow?")

    def test_no_docs_are_attributed_as_sources(self):
        evidence = self.response.evidence
        self.assertEqual(evidence.trusted_docs, ())
        self.assertEqual(evidence.user_docs, ())
        self.assertEqual(evidence.supporting_docs, ())

    def test_fallback_prompt_forbids_citations(self):
        system, _ = self.llm.prompts[0]
        self.assertIn("Do NOT cite", system["content"])
        self.assertIn("knowledge base", system["content"])

    def test_answer_is_marked_and_untouched_by_retrieved_text(self):
        self.assertTrue(self.response.answer.startswith(UNTRUSTED_ANSWER_WARNING))
        self.assertNotIn("Hypertension management overview", self.response.answer)

    def test_ui_source_buckets_are_empty(self):
        # The Streamlit app builds its source lists from the decision — both
        # buckets must be empty so nothing is rendered as a source.
        evidence = self.response.evidence
        self.assertFalse(list(evidence.trusted_docs))
        self.assertFalse(list(evidence.user_docs))

    def test_retrieval_still_recorded_for_diagnostics(self):
        # Retrieved chunks stay on the response for evaluation, they are just
        # never presented as evidence.
        self.assertEqual(len(self.response.retrieved_docs), 3)
        self.assertFalse(self.response.evidence.sufficient)


# ---------------------------------------------------------------------------
# 6. Policy isolation / evaluability
# ---------------------------------------------------------------------------

class TestEvidencePolicyIsolation(unittest.TestCase):
    def test_default_policy_matches_config(self):
        policy = EvidencePolicy.from_config(config)
        self.assertEqual(policy.name, config.EVIDENCE_POLICY_NAME)
        self.assertEqual(policy.min_relevant_docs, config.EVIDENCE_MIN_RELEVANT_DOCS)
        self.assertEqual(policy.min_rerank_score, config.EVIDENCE_MIN_RERANK_SCORE)
        self.assertEqual(policy.min_similarity, config.EVIDENCE_MIN_SIMILARITY)

    def test_threshold_change_flips_decision_without_touching_retrieval(self):
        gate_loose = EvidenceGate(policy=EvidencePolicy(name="loose", min_rerank_score=0.5))
        gate_strict = EvidenceGate(policy=EvidencePolicy(name="strict", min_rerank_score=100.0))

        docs = [make_doc("a", "text", rerank=5.0)]
        loose = gate_loose.evaluate("q", docs)
        strict = gate_strict.evaluate("q", docs)

        self.assertTrue(loose.sufficient)
        self.assertFalse(strict.sufficient)
        # retrieval output itself is untouched
        self.assertEqual(docs[0].rerank_score, 5.0)
        self.assertEqual(loose.policy_name, "loose")
        self.assertEqual(strict.policy_name, "strict")

    def test_min_relevant_docs_gate(self):
        docs = [
            make_doc("a", "text", rerank=5.0),
            make_doc("b", "text", rerank=-2.0),
        ]
        gate = EvidenceGate(policy=EvidencePolicy(min_relevant_docs=2))
        decision = gate.evaluate("q", docs)
        self.assertFalse(decision.sufficient)
        self.assertEqual(len(decision.supporting_docs), 1)

    def test_config_override_is_honoured(self):
        with patch.object(config, "EVIDENCE_MIN_RERANK_SCORE", 42.0), \
             patch.object(config, "EVIDENCE_POLICY_NAME", "sweep-42"):
            policy = EvidencePolicy.from_config(config)
        self.assertEqual(policy.min_rerank_score, 42.0)
        self.assertEqual(policy.name, "sweep-42")

    def test_decision_exposes_scores_for_offline_evaluation(self):
        gate = EvidenceGate(policy=EvidencePolicy())
        decision = gate.evaluate("q", [make_doc("a", "text", rerank=5.0)])
        snapshot = decision.as_dict()

        self.assertEqual(snapshot["policy_name"], "evidence-v1")
        self.assertTrue(snapshot["sufficient"])
        self.assertEqual(snapshot["scores"][0]["signal"], "rerank")
        self.assertEqual(snapshot["scores"][0]["value"], 5.0)
        self.assertEqual(snapshot["scores"][0]["threshold"], 1.0)
        self.assertTrue(snapshot["scores"][0]["relevant"])

    def test_pipeline_builds_gate_from_config(self):
        pipeline = build_pipeline(TRUSTED_DOCS)
        gate = pipeline._get_evidence_gate()
        self.assertIsInstance(gate, EvidenceGate)
        self.assertEqual(gate.policy.name, config.EVIDENCE_POLICY_NAME)

    def test_pipeline_accepts_injected_gate(self):
        injected = EvidenceGate(policy=EvidencePolicy(name="injected"))
        pipeline = build_pipeline(TRUSTED_DOCS, gate=injected)
        self.assertIs(pipeline._get_evidence_gate(), injected)
        self.assertEqual(run_query(pipeline).evidence.policy_name, "injected")


class TestSimilarityFallback(unittest.TestCase):
    """Re-ranking disabled → cosine similarity is the relevance signal."""

    def test_similarity_signal_used_when_rerank_scores_absent(self):
        gate = EvidenceGate(
            policy=EvidencePolicy(min_similarity=0.6),
            similarity_fn=lambda q, t: 0.9,
        )
        docs = [make_doc("a", "text", rerank=None)]
        decision = gate.evaluate("q", docs)
        self.assertTrue(decision.sufficient)
        self.assertEqual(decision.scores[0].signal, "similarity")

    def test_low_similarity_marks_evidence_insufficient(self):
        gate = EvidenceGate(
            policy=EvidencePolicy(min_similarity=0.6),
            similarity_fn=lambda q, t: 0.1,
        )
        decision = gate.evaluate("q", [make_doc("a", "text", rerank=None)])
        self.assertFalse(decision.sufficient)
        self.assertEqual(decision.trust_level, TRUST_LEVEL_NONE)

    def test_no_signal_marks_evidence_insufficient(self):
        gate = EvidenceGate(policy=EvidencePolicy())
        decision = gate.evaluate("q", [make_doc("a", "text", rerank=None)])
        self.assertFalse(decision.sufficient)
        self.assertEqual(decision.scores[0].signal, "none")
        self.assertIn("no relevance signal", decision.reason)

    def test_broken_similarity_fn_does_not_crash_the_gate(self):
        def boom(q, t):
            raise RuntimeError("model unavailable")

        gate = EvidenceGate(policy=EvidencePolicy(), similarity_fn=boom)
        decision = gate.evaluate("q", [make_doc("a", "text", rerank=None)])
        self.assertFalse(decision.sufficient)
        self.assertIsNone(decision.scores[0].value)

    def test_pipeline_fallback_uses_strategy_embedder(self):
        class FakeEmbedder:
            def embed_text(self, text):
                return [1.0, 0.0]

        from rag.retrievers import VectorSearchStrategy

        strategy = VectorSearchStrategy(embedder=FakeEmbedder(), vector_store=MagicMock())
        pipeline = RAGPipeline(
            search_strategy=strategy,
            reranker=NoOpReranker(),
            query_rewriter=NoOpQueryRewriter(),
            prompt_builder=TemplatePromptBuilder(),
            llm_provider=RecordingLLM(),
            repository=MagicMock(),
        )
        sim = pipeline._make_similarity_fn()
        self.assertAlmostEqual(sim("q", "text"), 1.0)


class TestAdoptedProductionPrompt(unittest.TestCase):
    """Phase 9 finalize: variant_structured_brief is the production prompt.

    The default pipeline builder must emit the adopted grounded prompt, the
    provenance context and source labels must reach the model, and the
    no-evidence path must still use the unchanged fallback prompt.
    """

    def _pipeline(self, docs, llm):
        return RAGPipeline(
            search_strategy=StubStrategy(docs),
            reranker=NoOpReranker(),
            query_rewriter=NoOpQueryRewriter(),
            llm_provider=llm,
            repository=MagicMock(),
        )

    def test_pipeline_default_builder_is_the_adopted_variant(self):
        llm = RecordingLLM()
        pipeline = self._pipeline(TRUSTED_DOCS, llm)
        self.assertEqual(pipeline.prompt_builder.variant, "structured_brief")
        response = run_query(pipeline)
        self.assertTrue(response.evidence.sufficient)
        system, user = llm.prompts[-1][0]["content"], llm.prompts[-1][1]["content"]
        self.assertTrue(
            system.startswith(
                TemplatePromptBuilder.PROMPT_VARIANTS["structured_brief"]["system"]
            )
        )
        self.assertTrue(system.endswith(TemplatePromptBuilder.GROUNDING_RULE))
        self.assertIn("Issued by:", user)
        self.assertIn("Trusted medical source (Medibo knowledge base)", user)
        # citations still built from the supporting chunks (Phase 8 preserved)
        self.assertTrue(response.citations)

    def test_fallback_path_still_uses_unchanged_fallback_prompt(self):
        llm = RecordingLLM()
        run_query(self._pipeline([], llm))
        self.assertEqual(
            llm.prompts[-1][0]["content"],
            TemplatePromptBuilder.FALLBACK_SYSTEM_PROMPT,
        )


class TestEmptyRetrieval(unittest.TestCase):
    def test_no_results_means_insufficient_evidence(self):
        response = run_query(build_pipeline([], llm=RecordingLLM()))
        self.assertFalse(response.evidence.sufficient)
        self.assertEqual(response.evidence.reason, "no documents retrieved")
        self.assertTrue(response.answer.startswith(UNTRUSTED_ANSWER_WARNING))


# ---------------------------------------------------------------------------
# 1. Read path: medibo_default always, medibo_user when present
# ---------------------------------------------------------------------------

class TestEvidenceReadPath(unittest.TestCase):
    def test_default_index_is_always_in_the_read_set(self):
        self.assertIn("medibo_default", config.MEDICAL_RETRIEVAL_INDEXES)

    def test_user_index_is_read_together_with_the_default(self):
        self.assertIn("medibo_user", config.MEDICAL_RETRIEVAL_INDEXES)

    def test_store_searches_both_indexes_in_one_query(self):
        from ingestion.medical_index import retrieval_store

        store = retrieval_store()
        self.assertEqual(store.search_indexes, "medibo_default,medibo_user")
        # a missing medibo_user must not break retrieval of medibo_default
        self.assertTrue(store._search_kwargs().get("ignore_unavailable"))

    def test_upload_index_is_separate_from_the_read_set(self):
        self.assertEqual(config.MEDICAL_UPLOAD_INDEX, "medibo_user")


if __name__ == "__main__":
    unittest.main()
