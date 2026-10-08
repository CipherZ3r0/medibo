"""Citations / provenance for grounded answers.

Phase 8.  Citations are derived **from the evidence decision's supporting
chunks** — never from raw retrieval — so an answer can only ever cite chunks
that actually passed the trust/relevance gate.

Design constraints
------------------
* Pure, separate from retrieval: this module only reads
  :class:`~domain.models.SearchResult` metadata.
* A citation preserves the full provenance set the corpus stores:
  ``source``, ``title``, ``source_url``, ``page_number``, ``section``,
  ``authority``, ``region``, ``document_type``, ``disease_topic`` and
  ``dataset_type``.
* ``dataset_type`` drives the trust class (``default`` → trusted,
  ``user`` → user-provided, missing/unknown → unknown).  Unknown-provenance
  chunks are never converted into citations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from domain.models import SearchResult
from rag.evidence import SOURCE_UNKNOWN, SOURCE_USER, classify_source

#: Provenance keys carried from chunk metadata into every citation.
PROVENANCE_FIELDS: Tuple[str, ...] = (
    "source",
    "title",
    "source_url",
    "page_number",
    "section",
    "authority",
    "region",
    "document_type",
    "disease_topic",
    "dataset_type",
)

TRUST_TAG = {
    "trusted": "trusted",
    "user": "user upload",
    "unknown": "unverified",
}


@dataclass(frozen=True)
class Citation:
    """One supporting chunk, with provenance, as shown next to the answer."""

    index: int
    chunk_id: str
    trust: str                      # "trusted" | "user" | "unknown"
    file_name: str
    source: Optional[str] = None
    title: Optional[str] = None
    source_url: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    authority: Optional[str] = None
    region: Optional[str] = None
    document_type: Optional[str] = None
    disease_topic: Optional[str] = None
    dataset_type: Optional[str] = None

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Human name of the cited chunk.

        Curated chunks lead with their ``title``; user uploads lead with the
        uploaded file name (that is the only identity the user recognises).
        """
        if self.trust == SOURCE_USER:
            order = (self.file_name, self.title, self.source)
        else:
            order = (self.title, self.file_name, self.source)
        for candidate in order:
            if candidate:
                return str(candidate)
        return "Unknown source"

    @property
    def locator(self) -> str:
        """``p. 4`` / section name, whichever the chunk actually carries."""
        if self.page_number:
            return f"p. {self.page_number}"
        if self.section:
            return str(self.section)
        return ""

    @property
    def label(self) -> str:
        """Minimum viable citation text: source/title + page or section + URL."""
        parts = [self.name]
        if self.source and str(self.source) != self.name:
            parts.append(f"via {self.source}")
        if self.locator:
            parts.append(self.locator)
        if self.source_url:
            parts.append(str(self.source_url))
        return " — ".join(parts)

    @property
    def trust_tag(self) -> str:
        return TRUST_TAG.get(self.trust, self.trust)

    def as_dict(self) -> Dict[str, Any]:
        """Flat dict (message/DB persistence, UI rendering)."""
        return asdict(self)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_search_result(cls, index: int, result: SearchResult) -> "Citation":
        meta = getattr(result, "metadata", None) or {}
        values: Dict[str, Any] = {}
        for key in PROVENANCE_FIELDS:
            value = meta.get(key)
            if value is not None and str(value).strip() != "":
                values[key] = value
        page_number = result.page_number or values.pop("page_number", None)
        return cls(
            index=index,
            chunk_id=result.chunk_id,
            trust=classify_source(result),
            file_name=result.file_name or "",
            page_number=page_number,
            **values,
        )


def build_citations(results: Sequence[SearchResult]) -> Tuple[Citation, ...]:
    """Build numbered citations from *supporting* chunks.

    Chunks with unknown provenance are dropped defensively: they must never
    appear as citations, even if a caller passes raw retrieval output.
    """
    citations: List[Citation] = []
    for result in results or []:
        if classify_source(result) == SOURCE_UNKNOWN:
            continue
        citations.append(Citation.from_search_result(len(citations) + 1, result))
    return tuple(citations)


def citations_as_dicts(citations: Iterable[Citation]) -> List[Dict[str, Any]]:
    """Serialise citations for session state / persistence."""
    return [c.as_dict() for c in citations or ()]


def citations_markdown(citations: Sequence[Citation]) -> str:
    """Render citations as a markdown numbered list (used by the UI)."""
    if not citations:
        return ""
    lines = ["**Citations**"]
    for c in citations or ():
        lines.append(f"{c.index}. {c.label}  `[{c.trust_tag}]`")
    return "\n".join(lines)


__all__ = [
    "PROVENANCE_FIELDS",
    "Citation",
    "build_citations",
    "citations_as_dicts",
    "citations_markdown",
]
