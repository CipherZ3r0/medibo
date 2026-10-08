"""Phase 8 — citations / provenance and strict trust classification.

Covers the required scenarios:

1. trusted citation (``dataset_type="default"``)
2. user citation (``dataset_type="user"``)
3. unknown / missing ``dataset_type`` → never trusted, never cited
4. page / section preservation
5. source URL preservation
6. untrusted fallback has no citations

No Elasticsearch or network access is required.
"""

import unittest
from unittest.mock import MagicMock

from domain.models import SearchResult
from rag.citations import (
    Citation,
    build_citations,
    citations_as_dicts,
    citations_markdown,
)
from rag.evidence import (
    SOURCE_TRUSTED,
    SOURCE_UNKNOWN,
    SOURCE_USER,
    TRUST_LEVEL_NONE,
    TRUST_LEVEL_TRUSTED,
    TRUST_LEVEL_USER_ONLY,
    UNTRUSTED_ANSWER_WARNING,
    USER_ONLY_BANNER,
    EvidenceGate,
    EvidencePolicy,
    classify_source,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TRUSTED_META = {
    "dataset_type": "default",
    "source": "medlineplus",
    "title": "Dengue — Symptoms and Signs",
    "source_url": "https://medlineplus.gov/dengue.html",
    "section": "Symptoms",
    "authority": "U.S. National Library of Medicine",
    "region": "us",
    "document_type": "health_topic",
    "disease_topic": "Dengue",
}


def make_doc(
    chunk_id="c1",
    text="text",
    rerank=5.0,
    dataset_type="default",
    file_name="medlineplus-dengue.json",
    page_number=4,
    metadata=None,
):
    meta = {"source": "medlineplus", "title": "Dengue"}
    if metadata:
        meta.update(metadata)
    if dataset_type is not None:
        meta["dataset_type"] = dataset_type
    return SearchResult(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        text=text,
        file_name=file_name,
        source_file=f"/data/{file_name}",
        file_type=".json",
        page_number=page_number,
        score=1.0,
        rerank_score=rerank,
        metadata=meta,
    )


class RecordingLLM:
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


class StubStrategy:
    def __init__(self, results):
        self.results = results

    def search(self, query, top_k):
        return list(self.results)


def run_query(docs, query="What are the symptoms of dengue?"):
    """Run one pipeline query over *docs* and return the RAGResponse."""
    from rag.pipeline import RAGPipeline
    from rag.prompt_builders import TemplatePromptBuilder
    from rag.query_rewriters import NoOpQueryRewriter
    from rag.rerankers import NoOpReranker

    pipeline = RAGPipeline(
        search_strategy=StubStrategy(docs),
        reranker=NoOpReranker(),
        query_rewriter=NoOpQueryRewriter(),
        prompt_builder=TemplatePromptBuilder(),
        llm_provider=RecordingLLM(),
        repository=MagicMock(),
    )
    return pipeline.query(
        query,
        retrieval_method="hybrid",
        use_reranking=True,
        use_query_rewriting=False,
        rerank_top_n=len(docs) or 1,
        save_to_db=False,
    )


# ---------------------------------------------------------------------------
# 1. Trusted citation
# ---------------------------------------------------------------------------

class TestTrustedCitation(unittest.TestCase):
    def setUp(self):
        self.doc = make_doc(metadata=TRUSTED_META)
        self.citations = build_citations([self.doc])
        self.citation = self.citations[0]

    def test_trusted_chunk_produces_a_trusted_citation(self):
        self.assertEqual(len(self.citations), 1)
        self.assertEqual(self.citation.trust, SOURCE_TRUSTED)
        self.assertEqual(self.citation.trust_tag, "trusted")
        self.assertEqual(classify_source(self.doc), SOURCE_TRUSTED)

    def test_citation_preserves_the_provenance_set(self):
        self.assertEqual(self.citation.source, "medlineplus")
        self.assertEqual(self.citation.title, "Dengue — Symptoms and Signs")
        self.assertEqual(self.citation.authority, "U.S. National Library of Medicine")
        self.assertEqual(self.citation.region, "us")
        self.assertEqual(self.citation.document_type, "health_topic")
        self.assertEqual(self.citation.disease_topic, "Dengue")
        self.assertEqual(self.citation.dataset_type, "default")

    def test_label_contains_title_page_and_url(self):
        label = self.citation.label
        self.assertIn("Dengue — Symptoms and Signs", label)
        self.assertIn("p. 4", label)
        self.assertIn("https://medlineplus.gov/dengue.html", label)
        self.assertIn("via medlineplus", label)

    def test_numbering_follows_supporting_order(self):
        docs = [make_doc("a"), make_doc("b"), make_doc("c")]
        indices = [c.index for c in build_citations(docs)]
        self.assertEqual(indices, [1, 2, 3])

    def test_ground_response_carries_the_citation(self):
        response = run_query([self.doc])
        self.assertTrue(response.evidence.sufficient)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].trust, SOURCE_TRUSTED)
        self.assertEqual(response.citations[0].chunk_id, self.doc.chunk_id)


# ---------------------------------------------------------------------------
# 2. User citation
# ---------------------------------------------------------------------------

class TestUserCitation(unittest.TestCase):
    def setUp(self):
        self.doc = make_doc(
            file_name="my-dengue-clinic-note.txt",
            dataset_type="user",
            page_number=None,
            metadata={"section": "Clinic note"},
        )
        self.citation = build_citations([self.doc])[0]

    def test_user_chunk_is_cited_as_user_upload(self):
        self.assertEqual(self.citation.trust, SOURCE_USER)
        self.assertEqual(self.citation.trust_tag, "user upload")
        self.assertEqual(classify_source(self.doc), SOURCE_USER)

    def test_user_citation_keeps_the_file_name_and_section(self):
        self.assertEqual(self.citation.file_name, "my-dengue-clinic-note.txt")
        self.assertEqual(self.citation.section, "Clinic note")
        self.assertEqual(self.citation.locator, "Clinic note")
        self.assertIn("my-dengue-clinic-note.txt", self.citation.label)
        self.assertIn("Clinic note", self.citation.label)

    def test_user_only_response_is_sufficient_and_marked(self):
        response = run_query([self.doc])
        self.assertTrue(response.evidence.sufficient)
        self.assertEqual(response.evidence.trust_level, TRUST_LEVEL_USER_ONLY)
        self.assertIsNone(response.evidence_warning)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].trust, SOURCE_USER)
        self.assertEqual(USER_ONLY_BANNER, "Based only on your uploaded data. This "
                           "information has not been independently verified by Medibo.")

    def test_user_citation_never_lands_in_the_trusted_bucket(self):
        response = run_query([self.doc])
        self.assertEqual(response.evidence.trusted_docs, ())
        self.assertEqual(
            [c.trust for c in response.citations],
            [SOURCE_USER],
        )


# ---------------------------------------------------------------------------
# 3. Unknown / missing dataset_type
# ---------------------------------------------------------------------------

class TestUnknownProvenance(unittest.TestCase):
    def test_missing_dataset_type_is_unknown(self):
        self.assertEqual(classify_source(make_doc(dataset_type=None)), SOURCE_UNKNOWN)

    def test_blank_and_exotic_dataset_types_are_unknown(self):
        for value in ("", "   ", "legacy-import", "DEFAULT-OLD", "User upload"):
            # only exact "default" / "user" (case-insensitive) classify
            expected = (
                SOURCE_TRUSTED if value.strip().lower() == "default"
                else SOURCE_USER if value.strip().lower() == "user"
                else SOURCE_UNKNOWN
            )
            self.assertEqual(classify_source(make_doc(dataset_type=value)), expected, value)

    def test_unknown_chunk_is_never_turned_into_a_citation(self):
        docs = [make_doc("u1", dataset_type=None), make_doc("d1", dataset_type="default")]
        citations = build_citations(docs)
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].chunk_id, "d1")
        self.assertEqual(citations[0].trust, SOURCE_TRUSTED)

    def test_unknown_only_evidence_is_insufficient(self):
        docs = [make_doc("u1", dataset_type=None), make_doc("u2", dataset_type="")]

        # gate level
        decision = EvidenceGate(policy=EvidencePolicy()).evaluate("q", docs)
        self.assertFalse(decision.sufficient)
        self.assertEqual(decision.trust_level, TRUST_LEVEL_NONE)
        self.assertEqual(decision.supporting_docs, ())
        self.assertEqual(decision.trusted_docs, ())
        self.assertEqual(decision.user_docs, ())
        self.assertEqual(len(decision.unknown_docs), 2)
        self.assertEqual(decision.warning, UNTRUSTED_ANSWER_WARNING)
        self.assertIn("unknown/untrusted provenance", decision.reason)

        # end to end: fallback answer, no citations
        response = run_query(docs, query="Who wrote Pride and Prejudice?")
        self.assertFalse(response.evidence.sufficient)
        self.assertEqual(response.citations, ())
        self.assertTrue(response.answer.startswith(UNTRUSTED_ANSWER_WARNING))

    def test_unknown_chunk_never_counts_as_supporting_evidence(self):
        docs = [make_doc("u1", dataset_type=None, rerank=9.0)]
        gate = EvidenceGate(policy=EvidencePolicy(min_relevant_docs=1))
        decision = gate.evaluate("q", docs)
        self.assertEqual(len([s for s in decision.scores if s.relevant]), 1)
        self.assertEqual(decision.supporting_docs, ())

    def test_mixed_unknown_and_trusted_keeps_unknown_out_of_support(self):
        docs = [
            make_doc("u1", dataset_type=None, rerank=9.0),
            make_doc("d1", dataset_type="default", rerank=5.0),
        ]
        decision = EvidenceGate(policy=EvidencePolicy()).evaluate("q", docs)
        self.assertTrue(decision.sufficient)
        self.assertEqual(decision.trust_level, TRUST_LEVEL_TRUSTED)
        self.assertEqual([d.chunk_id for d in decision.supporting_docs], ["d1"])
        self.assertEqual([d.chunk_id for d in decision.unknown_docs], ["u1"])

    def test_unknown_docs_are_labelled_in_the_prompt(self):
        from rag.prompt_builders import TemplatePromptBuilder

        docs = [make_doc("u1", dataset_type=None)]
        messages = TemplatePromptBuilder().build_prompt("q", docs, style="detailed")
        self.assertIn("Unknown provenance", messages[1]["content"])


# ---------------------------------------------------------------------------
# 4. Page / section preservation
# ---------------------------------------------------------------------------

class TestPageAndSectionPreservation(unittest.TestCase):
    def test_page_number_is_preserved(self):
        citation = build_citations([make_doc(page_number=12)])[0]
        self.assertEqual(citation.page_number, 12)
        self.assertEqual(citation.locator, "p. 12")

    def test_section_is_preserved_and_used_when_there_is_no_page(self):
        doc = make_doc(page_number=None, metadata={"section": "Treatment"})
        citation = build_citations([doc])[0]
        self.assertIsNone(citation.page_number)
        self.assertEqual(citation.section, "Treatment")
        self.assertEqual(citation.locator, "Treatment")
        self.assertIn("Treatment", citation.label)

    def test_page_wins_over_section_for_the_locator(self):
        doc = make_doc(page_number=7, metadata={"section": "Treatment"})
        citation = build_citations([doc])[0]
        self.assertEqual(citation.section, "Treatment")
        self.assertEqual(citation.locator, "p. 7")

    def test_page_and_section_survive_serialisation(self):
        doc = make_doc(page_number=3, metadata={"section": "Prevention"})
        payload = citations_as_dicts(build_citations([doc]))
        self.assertEqual(payload[0]["page_number"], 3)
        self.assertEqual(payload[0]["section"], "Prevention")
        restored = Citation(**payload[0])
        self.assertEqual(restored.label, build_citations([doc])[0].label)


# ---------------------------------------------------------------------------
# 5. Source URL preservation
# ---------------------------------------------------------------------------

class TestSourceUrlPreservation(unittest.TestCase):
    def test_source_url_is_preserved(self):
        doc = make_doc(metadata=TRUSTED_META)
        citation = build_citations([doc])[0]
        self.assertEqual(citation.source_url, "https://medlineplus.gov/dengue.html")
        self.assertIn("https://medlineplus.gov/dengue.html", citation.label)

    def test_missing_source_url_is_none_and_absent_from_the_label(self):
        citation = build_citations([make_doc(metadata={"source_url": None})])[0]
        self.assertIsNone(citation.source_url)
        self.assertNotIn("http", citation.label)

    def test_url_survives_serialisation(self):
        payload = citations_as_dicts(build_citations([make_doc(metadata=TRUSTED_META)]))
        self.assertEqual(payload[0]["source_url"], "https://medlineplus.gov/dengue.html")


# ---------------------------------------------------------------------------
# 6. Untrusted fallback has no citations
# ---------------------------------------------------------------------------

class TestFallbackHasNoTrustedCitations(unittest.TestCase):
    def setUp(self):
        weak = [
            make_doc("w1", "Hypertension management overview.", rerank=-8.1),
            make_doc("w2", "Antimicrobial prescribing guidance.", rerank=-7.6),
        ]
        self.response = run_query(weak, query="Who wrote Pride and Prejudice?")

    def test_fallback_response_has_no_citations(self):
        self.assertFalse(self.response.evidence.sufficient)
        self.assertEqual(self.response.citations, ())

    def test_fallback_has_no_trusted_or_user_chunks(self):
        evidence = self.response.evidence
        self.assertEqual(evidence.trusted_docs, ())
        self.assertEqual(evidence.user_docs, ())
        self.assertEqual(evidence.supporting_docs, ())

    def test_fallback_carries_the_required_warning(self):
        self.assertEqual(self.response.evidence_warning, UNTRUSTED_ANSWER_WARNING)
        self.assertTrue(self.response.answer.startswith(UNTRUSTED_ANSWER_WARNING))
        self.assertEqual(citations_markdown(self.response.citations), "")

    def test_citation_markdown_renders_trust_tags(self):
        citations = build_citations([make_doc(metadata=TRUSTED_META)])
        markdown = citations_markdown(citations)
        self.assertIn("**Citations**", markdown)
        self.assertIn("1.", markdown)
        self.assertIn("`[trusted]`", markdown)
        user_md = citations_markdown(
            build_citations([make_doc(dataset_type="user", file_name="my-note.txt")])
        )
        self.assertIn("`[user upload]`", user_md)


if __name__ == "__main__":
    unittest.main()
