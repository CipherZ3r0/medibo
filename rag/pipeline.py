"""
RAG Pipeline Orchestrator — Clean Facade & Dependency Injection architecture.
"""

import logging
from typing import Dict, Optional

import config
from domain.models import RAGResponse
from ingestion.embedders import SentenceTransformerEmbedder
from ingestion.medical_index import retrieval_store
from rag.citations import build_citations
from rag.evidence import (
    EvidenceDecision,
    EvidenceGate,
    EvidencePolicy,
    UNTRUSTED_ANSWER_WARNING,
    cosine_similarity,
)
from rag.retrievers import (
    BaseSearchStrategy,
    HybridSearchStrategy,
    SearchStrategyFactory,
    TextSearchStrategy,
    VectorSearchStrategy,
)
from rag.rerankers import BaseReranker, CrossEncoderReranker, NoOpReranker
from rag.query_rewriters import BaseQueryRewriter, LLMQueryRewriter, NoOpQueryRewriter
from rag.prompt_builders import BasePromptBuilder, TemplatePromptBuilder
from rag.llm_providers import BaseLLMProvider, create_llm_provider
from app.database import BaseDatabaseRepository, PostgresDatabaseRepository

logger = logging.getLogger(__name__)


def default_retrieval_store():
    """Store bound to the application read path: ``medibo_default`` + ``medibo_user``.

    Kept as a factory function so tests can substitute a store backed by a
    fake Elasticsearch client without touching the pipeline.
    """
    return retrieval_store()


class RAGPipeline:
    """Facade orchestrating Query Rewriting, Search Retrieval, Document Re-ranking, Prompt Construction, and LLM Generation."""

    def __init__(
        self,
        search_strategy: Optional[BaseSearchStrategy] = None,
        reranker: Optional[BaseReranker] = None,
        query_rewriter: Optional[BaseQueryRewriter] = None,
        prompt_builder: Optional[BasePromptBuilder] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        repository: Optional[BaseDatabaseRepository] = None,
        vector_store=None,
        evidence_gate: Optional[EvidenceGate] = None,
    ):
        #: Read path shared by every search method (lazily created when the
        #: caller does not inject one): default medical corpus + user data.
        self.vector_store = vector_store
        self.search_strategy = search_strategy or HybridSearchStrategy(
            vector_store=self._get_vector_store()
        )
        self.reranker = reranker or CrossEncoderReranker()
        self.query_rewriter = query_rewriter or LLMQueryRewriter()
        #: Phase 9 adopted grounded prompt: the builder's default is now
        #: variant_structured_brief; the fallback prompt, evidence gate,
        #: retrieval and citations are unchanged.
        self.prompt_builder = prompt_builder or TemplatePromptBuilder()
        self.llm_provider = llm_provider or create_llm_provider()
        self.repository = repository or PostgresDatabaseRepository()
        #: Phase 7 trusted-evidence gate (policy + optional similarity fallback).
        self.evidence_gate = evidence_gate
        self._strategies: Dict[str, BaseSearchStrategy] = {
            "hybrid": self.search_strategy,
        }

    # ------------------------------------------------------------------
    # Search strategy selection
    # ------------------------------------------------------------------

    def _get_vector_store(self):
        """Return the shared medical read-path store, creating it on first use."""
        if self.vector_store is None:
            self.vector_store = default_retrieval_store()
        return self.vector_store

    def _get_strategy(self, retrieval_method: str) -> BaseSearchStrategy:
        """Resolve a search strategy bound to the medical read path.

        Same strategy classes (vector / BM25 / hybrid RRF) as before — only
        the index set they query differs. Unknown names still raise through
        :class:`SearchStrategyFactory`.
        """
        name = (retrieval_method or "hybrid").lower()
        if name in self._strategies:
            return self._strategies[name]
        if name == "vector":
            strategy: BaseSearchStrategy = VectorSearchStrategy(
                vector_store=self._get_vector_store()
            )
        elif name == "text":
            strategy = TextSearchStrategy(vector_store=self._get_vector_store())
        else:
            strategy = SearchStrategyFactory.get_strategy(name)
        self._strategies[name] = strategy
        return strategy

    # ------------------------------------------------------------------
    # Trusted evidence gate (Phase 7)
    # ------------------------------------------------------------------

    def _make_similarity_fn(self):
        """Build the fallback relevance scorer used when re-ranking is off.

        Uses the embedding model already bound to the search strategy so the
        fallback measures the same similarity the index was built with.  The
        model is only instantiated if the fallback is actually consulted.
        """
        embedder = getattr(
            getattr(self.search_strategy, "vector_strategy", None), "embedder", None
        )
        if embedder is None:
            embedder = getattr(self.search_strategy, "embedder", None)

        def _similarity(query: str, text: str) -> float:
            nonlocal embedder
            if embedder is None:
                embedder = SentenceTransformerEmbedder()
            return cosine_similarity(embedder.embed_text(query), embedder.embed_text(text))

        return _similarity

    def _get_evidence_gate(self) -> EvidenceGate:
        """Return the evidence gate, creating it lazily on first use.

        The gate is a pure decision layer: it reads the scores produced by
        retrieval/re-ranking and never modifies them.
        """
        if self.evidence_gate is None:
            self.evidence_gate = EvidenceGate(
                policy=EvidencePolicy.from_config(config),
                similarity_fn=self._make_similarity_fn(),
            )
        return self.evidence_gate

    def query(
        self,
        user_query: str,
        retrieval_method: str = "hybrid",
        use_reranking: bool = True,
        use_query_rewriting: bool = config.USE_QUERY_REWRITING,
        prompt_style: str = "detailed",
        llm_provider_name: str = None,
        llm_model: str = None,
        top_k: int = config.TOP_K_RETRIEVAL,
        rerank_top_n: int = config.RERANK_TOP_N,
        save_to_db: bool = True,
    ) -> RAGResponse:
        """Execute complete RAG flow for a user query."""

        # 1. Select search strategy (same classes as before; the read path
        #    queries medibo_default + medibo_user — see default_retrieval_store)
        strategy = self._get_strategy(retrieval_method)

        # Get the active LLM provider for this query
        active_llm_provider = create_llm_provider(llm_provider_name) if llm_provider_name else self.llm_provider

        # 2. Query Rewriting (optional)
        if use_query_rewriting:
            rewriter = LLMQueryRewriter(llm_provider=active_llm_provider, model=llm_model)
        else:
            rewriter = NoOpQueryRewriter()
        
        search_q = rewriter.rewrite(user_query)

        # 3. Retrieval
        fetch_k = config.RETRIEVAL_CANDIDATE_POOL if use_reranking else top_k
        raw_results = strategy.search(search_q, top_k=fetch_k)

        # 4. Re-ranking (optional)
        reranker_svc = self.reranker if use_reranking else NoOpReranker()
        final_results = reranker_svc.rerank(search_q, raw_results, top_n=rerank_top_n)

        # 5. Trusted evidence gate (Phase 7): is there enough relevant
        #    evidence for a knowledge-base grounded answer? Pure decision —
        #    scores and ranking are left untouched.
        decision: EvidenceDecision = self._get_evidence_gate().evaluate(
            search_q, final_results
        )

        # 6. Prompt Construction + citations (Phase 8)
        if decision.sufficient:
            messages = self.prompt_builder.build_prompt(
                user_query, list(decision.supporting_docs), style=prompt_style
            )
            # Citations are derived from the *supporting* chunks only, so the
            # answer can cite provenance it actually used.
            citations = build_citations(decision.supporting_docs)
        else:
            # No sufficient evidence → general-knowledge prompt with no context
            # documents, so the model cannot invent source attributions.
            messages = self.prompt_builder.build_fallback_prompt(
                user_query, style=prompt_style
            )
            citations = ()

        # 7. LLM Generation
        gen_result = active_llm_provider.generate(messages, model=llm_model)

        answer = gen_result["answer"]
        if not decision.sufficient:
            # Mark the response itself so the warning travels with it through
            # the UI, the chat history and the database.
            answer = f"{UNTRUSTED_ANSWER_WARNING}\n\n{answer}"

        # 8. Assemble RAGResponse domain object
        rag_response = RAGResponse(
            answer=answer,
            query=user_query,
            rewritten_query=search_q if use_query_rewriting else None,
            retrieved_docs=final_results,
            model=gen_result["model"],
            prompt_style=prompt_style,
            retrieval_method=retrieval_method,
            response_time_ms=gen_result["response_time_ms"],
            prompt_tokens=gen_result["prompt_tokens"],
            completion_tokens=gen_result["completion_tokens"],
            total_tokens=gen_result["total_tokens"],
            evidence=decision,
            evidence_warning=decision.warning,
            citations=citations,
        )

        # 9. Save interaction to Database Repository
        if save_to_db and self.repository:
            try:
                conv_id = self.repository.save_conversation(
                    rag_response,
                    reranking_used=use_reranking,
                    query_rewriting_used=use_query_rewriting,
                )
                rag_response.conversation_id = conv_id
            except Exception as e:
                logger.warning(f"Failed to persist conversation record: {e}")

        return rag_response
