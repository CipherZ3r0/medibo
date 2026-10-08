"""Trusted evidence gate — decide whether retrieval actually supports an answer.

Phase 7.  The gate sits **after** retrieval and re-ranking and **before**
prompt construction, and answers one question: *is there enough relevant
evidence to answer this query from the knowledge base?*

Design constraints
------------------
* Retrieval, RRF, the candidate pool, the re-ranker, the embedding model and
  the Elasticsearch mappings are untouched — the gate only **reads** the
  scores those stages already produce.
* The decision is a pure value object (:class:`EvidenceDecision`) built from
  an explicit, serialisable policy (:class:`EvidencePolicy`) so the threshold
  can be swept/evaluated offline later without touching pipeline code.
 * Provenance is authoritative: ``dataset_type == "default"`` → *trusted*,
   ``dataset_type == "user"`` → *user-provided* (never presented as curated),
   anything else / missing → *unknown* (never trusted, never used to ground).

Relevance signals (first one available wins)
---------------------------------------------
``rerank``
    Cross-encoder logit produced by ``CrossEncoderReranker``.  Calibrated on
    this corpus: relevant queries score ``+4.8 … +7.4``, unrelated queries
    ``-7.6 … -11.2`` → default threshold ``1.0``.
``similarity``
    Cosine similarity between the query and the chunk text (same embedding
    model as indexing) — used when re-ranking is disabled.  Calibrated:
    relevant ``0.68 … 0.75``, unrelated ``0.19 … 0.32`` → default threshold
    ``0.60``.
``none``
    No signal available → the document never counts as evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from domain.models import SearchResult

logger = logging.getLogger(__name__)

#: Exact string the UI must show when the answer is *not* knowledge-base grounded.
UNTRUSTED_ANSWER_WARNING = (
    "⚠ No trusted source found in Medibo's knowledge base. "
    "This answer is generated from the model's general knowledge."
)

#: Shown when evidence exists but comes exclusively from the user's uploads.
USER_ONLY_BANNER = (
    "Based only on your uploaded data. "
    "This information has not been independently verified by Medibo."
)

TRUST_LEVEL_TRUSTED = "trusted"
TRUST_LEVEL_USER_ONLY = "user_only"
TRUST_LEVEL_NONE = "none"

SOURCE_TRUSTED = "trusted"
SOURCE_USER = "user"
#: Provenance is missing or is neither "default" nor "user" — never trusted.
SOURCE_UNKNOWN = "unknown"

#: ``dataset_type`` values that mark a document as part of the curated corpus
#: or as a personal upload.  Anything else is untrusted.
TRUSTED_DATASET_TYPE = "default"
USER_DATASET_TYPE = "user"


def classify_dataset_type(value: Any) -> str:
    """Map a raw ``dataset_type`` value onto a trust class.

    * ``"default"`` → ``"trusted"``
    * ``"user"``    → ``"user"``
    * anything else / missing / blank → ``"unknown"`` (never trusted)
    """
    dataset_type = str(value or "").strip().lower()
    if dataset_type == TRUSTED_DATASET_TYPE:
        return SOURCE_TRUSTED
    if dataset_type == USER_DATASET_TYPE:
        return SOURCE_USER
    return SOURCE_UNKNOWN


def classify_source(result: SearchResult) -> str:
    """Return the trust class of a retrieved chunk: ``trusted`` / ``user`` / ``unknown``.

    Phase 8: classification is strict and provenance-driven.  Only documents
    explicitly tagged ``dataset_type == "default"`` (written at index time by
    :func:`ingestion.medical_index.build_source`) are trusted; documents tagged
    ``dataset_type == "user"`` are user-provided; a missing, blank or unknown
    value is **unknown** and must never be presented as trusted.
    """
    meta = getattr(result, "metadata", None) or {}
    return classify_dataset_type(meta.get("dataset_type"))


@dataclass(frozen=True)
class EvidencePolicy:
    """Thresholds and labels that drive the sufficiency decision.

    Kept as a small immutable object (with a stable ``name``) so an offline
    evaluation can sweep thresholds and record which policy produced a
    decision without re-running retrieval.
    """

    name: str = "evidence-v1"
    min_relevant_docs: int = 1
    min_rerank_score: float = 1.0
    min_similarity: float = 0.60

    @classmethod
    def from_config(cls, cfg: Any = None) -> "EvidencePolicy":
        """Build a policy from environment-backed config values."""
        if cfg is None:
            import config as cfg  # local import keeps this module dependency-light

        def _get(name: str, default: Any, cast):
            raw = getattr(cfg, name, None)
            if raw is None:
                return default
            try:
                return cast(raw)
            except (TypeError, ValueError):
                logger.warning("Invalid EVIDENCE config %s=%r — using %r", name, raw, default)
                return default

        return cls(
            name=_get("EVIDENCE_POLICY_NAME", "evidence-v1", str),
            min_relevant_docs=_get("EVIDENCE_MIN_RELEVANT_DOCS", 1, int),
            min_rerank_score=_get("EVIDENCE_MIN_RERANK_SCORE", 1.0, float),
            min_similarity=_get("EVIDENCE_MIN_SIMILARITY", 0.60, float),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "min_relevant_docs": self.min_relevant_docs,
            "min_rerank_score": self.min_rerank_score,
            "min_similarity": self.min_similarity,
        }


@dataclass(frozen=True)
class EvidenceScore:
    """Per-document scoring record — the raw material for policy evaluation."""

    chunk_id: str
    file_name: str
    source: str                 # "trusted" | "user" | "unknown"
    signal: str                 # "rerank" | "similarity" | "none"
    value: Optional[float]
    threshold: Optional[float]
    relevant: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "file_name": self.file_name,
            "source": self.source,
            "signal": self.signal,
            "value": self.value,
            "threshold": self.threshold,
            "relevant": self.relevant,
        }


@dataclass(frozen=True)
class EvidenceDecision:
    """Result of the gate: sufficiency, trust split and the scores behind it.

    ``supporting_docs`` (and therefore the grounded prompt and its citations)
    contains only *trusted* and *user-provided* chunks — relevant chunks with
    missing/unknown provenance land in ``unknown_docs`` and can never ground
    an answer or be cited as trusted.
    """

    sufficient: bool
    trust_level: str
    supporting_docs: Tuple[SearchResult, ...] = ()
    trusted_docs: Tuple[SearchResult, ...] = ()
    user_docs: Tuple[SearchResult, ...] = ()
    unknown_docs: Tuple[SearchResult, ...] = ()
    scores: Tuple[EvidenceScore, ...] = ()
    reason: str = ""
    policy_name: str = ""
    warning: Optional[str] = None

    @property
    def has_trusted_evidence(self) -> bool:
        return bool(self.trusted_docs)

    @property
    def has_user_evidence(self) -> bool:
        return bool(self.user_docs)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-friendly snapshot (useful when evaluating policies offline)."""
        return {
            "sufficient": self.sufficient,
            "trust_level": self.trust_level,
            "reason": self.reason,
            "policy_name": self.policy_name,
            "warning": self.warning,
            "trusted_chunk_ids": [d.chunk_id for d in self.trusted_docs],
            "user_chunk_ids": [d.chunk_id for d in self.user_docs],
            "unknown_chunk_ids": [d.chunk_id for d in self.unknown_docs],
            "scores": [s.as_dict() for s in self.scores],
        }


class EvidenceGate:
    """Pure decision function over already-retrieved, already-ranked results.

    ``similarity_fn(query, text) -> float`` is optional; it is only consulted
    for documents that carry no re-ranker score (re-ranking disabled).
    """

    def __init__(
        self,
        policy: Optional[EvidencePolicy] = None,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.policy = policy or EvidencePolicy()
        self.similarity_fn = similarity_fn

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_one(
        self, query: str, result: SearchResult, similarity_fn: Optional[Callable[[str, str], float]]
    ) -> EvidenceScore:
        source = classify_source(result)
        rerank_score = getattr(result, "rerank_score", None)

        if rerank_score is not None:
            value = float(rerank_score)
            threshold = self.policy.min_rerank_score
            signal = "rerank"
        elif similarity_fn is not None:
            threshold = self.policy.min_similarity
            signal = "similarity"
            try:
                value = float(similarity_fn(query, result.text or ""))
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("similarity_fn failed for %s: %s", result.chunk_id, exc)
                value = None
        else:
            value = None
            threshold = None
            signal = "none"

        relevant = value is not None and threshold is not None and value >= threshold
        return EvidenceScore(
            chunk_id=result.chunk_id,
            file_name=result.file_name,
            source=source,
            signal=signal,
            value=value,
            threshold=threshold,
            relevant=relevant,
        )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def evaluate(
        self,
        query: str,
        results: Sequence[SearchResult],
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> EvidenceDecision:
        """Decide whether *results* contain sufficient evidence for *query*."""
        sim_fn = similarity_fn or self.similarity_fn
        candidates = list(results or [])

        if not candidates:
            return EvidenceDecision(
                sufficient=False,
                trust_level=TRUST_LEVEL_NONE,
                scores=(),
                reason="no documents retrieved",
                policy_name=self.policy.name,
                warning=UNTRUSTED_ANSWER_WARNING,
            )

        scores: List[EvidenceScore] = []
        relevant_results: List[SearchResult] = []
        for result in candidates:
            score = self._score_one(query, result, sim_fn)
            scores.append(score)
            if score.relevant:
                relevant_results.append(result)

        # Phase 8: provenance is authoritative.  Only "default" (trusted) and
        # "user" (user-provided) chunks may ground an answer; chunks whose
        # dataset_type is missing/unknown are recorded but never counted.
        trusted = [r for r in relevant_results if classify_source(r) == SOURCE_TRUSTED]
        user = [r for r in relevant_results if classify_source(r) == SOURCE_USER]
        unknown = [r for r in relevant_results if classify_source(r) == SOURCE_UNKNOWN]
        supporting = trusted + user

        sufficient = len(supporting) >= self.policy.min_relevant_docs
        unknown_note = (
            f"; {len(unknown)} relevant document(s) had unknown provenance "
            f"and were not used as evidence"
            if unknown
            else ""
        )
        if sufficient:
            trust_level = TRUST_LEVEL_TRUSTED if trusted else TRUST_LEVEL_USER_ONLY
            warning = None
            reason = (
                f"{len(supporting)}/{len(candidates)} retrieved documents met the "
                f"relevance threshold (policy '{self.policy.name}'){unknown_note}"
            )
        else:
            trust_level = TRUST_LEVEL_NONE
            warning = UNTRUSTED_ANSWER_WARNING
            best = next((s for s in scores if s.value is not None), None)
            if best is None:
                reason = "no relevance signal available for any retrieved document"
            elif not relevant_results:
                reason = (
                    f"0/{len(candidates)} retrieved documents met the relevance "
                    f"threshold (best {best.signal}={best.value:.3f} < {best.threshold})"
                    f"{unknown_note}"
                )
            else:
                # Relevant chunks exist but all of them are unknown provenance.
                reason = (
                    f"{len(relevant_results)}/{len(candidates)} retrieved documents were "
                    f"relevant but had unknown/untrusted provenance, so none of them "
                    f"could ground the answer"
                )

        logger.info(
            "Evidence gate: sufficient=%s trust=%s reason=%s",
            sufficient,
            trust_level,
            reason,
        )
        return EvidenceDecision(
            sufficient=sufficient,
            trust_level=trust_level,
            supporting_docs=tuple(supporting),
            trusted_docs=tuple(trusted),
            user_docs=tuple(user),
            unknown_docs=tuple(unknown),
            scores=tuple(scores),
            reason=reason,
            policy_name=self.policy.name,
            warning=warning,
        )


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity over two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


__all__ = [
    "UNTRUSTED_ANSWER_WARNING",
    "USER_ONLY_BANNER",
    "TRUST_LEVEL_TRUSTED",
    "TRUST_LEVEL_USER_ONLY",
    "TRUST_LEVEL_NONE",
    "SOURCE_TRUSTED",
    "SOURCE_USER",
    "SOURCE_UNKNOWN",
    "TRUSTED_DATASET_TYPE",
    "USER_DATASET_TYPE",
    "EvidencePolicy",
    "EvidenceScore",
    "EvidenceDecision",
    "EvidenceGate",
    "classify_dataset_type",
    "classify_source",
    "cosine_similarity",
]
