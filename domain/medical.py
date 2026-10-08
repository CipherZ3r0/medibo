"""
Medical Corpus Domain Models — metadata schema and raw/indexed text separation.

Phase 1 of the medical RAG MVP.  This module defines:

* :class:`MedicalDocumentMetadata` — the fixed metadata schema attached to every
  medical document and (later) to every indexed chunk.
* :class:`MedicalDocument` — a document that carries its **raw text** (verbatim
  extraction, never indexed) and its **indexed text** (cleaned, searchable text)
  as separate fields, persisted in separate directories.
* Manifest helpers — loading and validating ``data/medical/source_manifest.json``,
  the curated list of official sources (MedlinePlus, ICMR, MoHFW Clinical
  Establishments, NCDC).

Deliberately stdlib-only so it imports cleanly in minimal test environments and
does not pull ingestion, Elasticsearch or embedding dependencies.

Nothing here changes retrieval, RRF fusion, the candidate pool, the reranker or
the embedding model — that work is scheduled for Phase 2.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple

__all__ = [
    "PROJECT_ROOT",
    "MEDICAL_DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "SOURCE_MANIFEST_PATH",
    "REGIONS",
    "DOCUMENT_TYPES",
    "SCHEMA_VERSION",
    "DATE_PATTERN",
    "DOC_ID_PATTERN",
    "OFFICIAL_HOSTS",
    "MedicalCorpusError",
    "MedicalManifestError",
    "MedicalDocumentMetadata",
    "MedicalDocument",
    "load_manifest",
    "validate_manifest",
    "assert_valid_manifest",
    "iter_manifest_documents",
    "metadata_from_manifest",
    "manifest_source_ids",
    "raw_document_dir",
    "raw_text_path",
    "raw_original_path",
    "processed_path",
]


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDICAL_DATA_DIR = PROJECT_ROOT / "data" / "medical"
RAW_DIR = MEDICAL_DATA_DIR / "raw"
PROCESSED_DIR = MEDICAL_DATA_DIR / "processed"
SOURCE_MANIFEST_PATH = MEDICAL_DATA_DIR / "source_manifest.json"

SCHEMA_VERSION = 1
MANIFEST_TYPE = "medical_source_manifest"

# Geographic scope of a source's content.
REGIONS = frozenset({"global", "india"})

# Controlled vocabulary for the ``document_type`` metadata field.
DOCUMENT_TYPES = frozenset(
    {
        "health_topic_dataset",
        "clinical_guideline",
        "treatment_guideline",
        "standard_treatment_guideline",
        "public_health_advisory",
        "public_health_guideline",
        "patient_faq",
    }
)

# ISO-8601 partial dates: YYYY, YYYY-MM or YYYY-MM-DD (null when unavailable).
DATE_PATTERN = re.compile(r"^\d{4}(-(0[1-9]|1[0-2])(-(0[1-9]|[12]\d|3[01]))?)?$")

DOC_ID_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

# Domains that may appear in manifest URLs (keeps the corpus on official sites).
OFFICIAL_HOSTS = (
    "medlineplus.gov",
    "icmr.gov.in",
    "clinicalestablishments.mohfw.gov.in",
    "ncdc.mohfw.gov.in",
)

REQUIRED_MANIFEST_FIELDS = ("schema_version", "manifest_type", "sources")
REQUIRED_SOURCE_FIELDS = (
    "source",
    "name",
    "region",
    "authority",
    "landing_url",
    "license",
    "license_notes",
    "documents",
)
REQUIRED_DOCUMENT_FIELDS = (
    "doc_id",
    "title",
    "url",
    "document_type",
    "topic",
    "published_at",
    "updated_at",
    "format",
    "selection_reason",
)


class MedicalCorpusError(ValueError):
    """Raised when medical corpus metadata or document structure is invalid."""


class MedicalManifestError(MedicalCorpusError):
    """Raised when the source manifest fails validation."""


# ---------------------------------------------------------------------------
# Metadata schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MedicalDocumentMetadata:
    """Fixed metadata schema for every medical document and chunk.

    Field names mirror the Phase 1 specification exactly:

    ==================  =========================================================
    Field               Meaning
    ==================  =========================================================
    source              Source id from the manifest (e.g. ``icmr``).
    source_url          Canonical URL the document was obtained from.
    region              Geographic scope: ``global`` or ``india``.
    authority           Issuing body (e.g. "Indian Council of Medical Research").
    document_type       One of :data:`DOCUMENT_TYPES`.
    disease_topic       Disease / symptom / topic the document covers.
    published_at        Publication date (``YYYY``, ``YYYY-MM``, ``YYYY-MM-DD``).
    updated_at          Last update date, same formats, ``None`` if unknown.
    section             Section heading (or logical section) of the passage.
    page                1-based page number for PDF-derived passages.
    license             Short licence / usage statement for the source.
    ==================  =========================================================

    Dates are ``None`` when the source does not state them — the manifest only
    carries dates that are actually available.
    """

    source: str
    source_url: str
    region: str
    authority: str
    document_type: str
    disease_topic: str
    published_at: Optional[str] = None
    updated_at: Optional[str] = None
    section: Optional[str] = None
    page: Optional[int] = None
    license: str = ""

    FIELD_NAMES: ClassVar[Tuple[str, ...]] = (
        "source",
        "source_url",
        "region",
        "authority",
        "document_type",
        "disease_topic",
        "published_at",
        "updated_at",
        "section",
        "page",
        "license",
    )

    def validate(self) -> None:
        """Raise :class:`MedicalCorpusError` if any field is malformed."""
        for name in ("source", "source_url", "authority", "disease_topic"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise MedicalCorpusError(f"{name} must be a non-empty string")
        if not self.source_url.lower().startswith("https://"):
            raise MedicalCorpusError(f"source_url must be https: {self.source_url}")
        if self.region not in REGIONS:
            raise MedicalCorpusError(
                f"region must be one of {sorted(REGIONS)}, got {self.region!r}"
            )
        if self.document_type not in DOCUMENT_TYPES:
            raise MedicalCorpusError(
                f"document_type must be one of {sorted(DOCUMENT_TYPES)}, "
                f"got {self.document_type!r}"
            )
        for name in ("published_at", "updated_at"):
            value = getattr(self, name)
            if value is not None and not DATE_PATTERN.match(value):
                raise MedicalCorpusError(
                    f"{name} must be YYYY, YYYY-MM, YYYY-MM-DD or None, got {value!r}"
                )
        if self.section is not None and not isinstance(self.section, str):
            raise MedicalCorpusError("section must be a string or None")
        if self.page is not None and (not isinstance(self.page, int) or self.page < 1):
            raise MedicalCorpusError("page must be a positive integer or None")
        if not isinstance(self.license, str) or not self.license.strip():
            raise MedicalCorpusError("license must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELD_NAMES}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedicalDocumentMetadata":
        if not isinstance(data, dict):
            raise MedicalCorpusError("metadata must be a mapping")
        missing = [name for name in cls.FIELD_NAMES if name not in data]
        if missing:
            raise MedicalCorpusError(f"metadata missing fields: {missing}")
        metadata = cls(**{name: data[name] for name in cls.FIELD_NAMES})
        metadata.validate()
        return metadata


# ---------------------------------------------------------------------------
# Document model — raw text vs. indexed text
# ---------------------------------------------------------------------------


@dataclass
class MedicalDocument:
    """A medical corpus document with strictly separated text representations.

    ``raw_text`` is the verbatim extraction of the source document (headers,
    footers, disclaimers and all).  It is only ever written to ``raw/``.

    ``indexed_text`` is the cleaned, section-aware text that is chunked,
    embedded and indexed.  It is only ever written to ``processed/``.

    The two are separate fields and are persisted in separate directories, so
    raw document text can never leak into the searchable index by accident.
    """

    doc_id: str
    title: str
    metadata: MedicalDocumentMetadata
    raw_text: Optional[str] = None
    indexed_text: Optional[str] = None

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`MedicalCorpusError` if the document is malformed."""
        if not DOC_ID_PATTERN.match(self.doc_id or ""):
            raise MedicalCorpusError(
                f"doc_id must be a lowercase slug, got {self.doc_id!r}"
            )
        if not isinstance(self.title, str) or not self.title.strip():
            raise MedicalCorpusError("title must be a non-empty string")
        self.metadata.validate()
        for name in ("raw_text", "indexed_text"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise MedicalCorpusError(f"{name} must be a string or None")
        if self.indexed_text is not None and not self.indexed_text.strip():
            raise MedicalCorpusError("indexed_text must not be empty when provided")

    # -- path helpers (raw/ vs processed/ stay disjoint) -------------------

    @property
    def raw_dir(self) -> Path:
        return raw_document_dir(self.metadata.source, self.doc_id)

    @property
    def raw_text_file(self) -> Path:
        return raw_text_path(self.metadata.source, self.doc_id)

    def raw_original_file(self, suffix: str) -> Path:
        return raw_original_path(self.metadata.source, self.doc_id, suffix)

    @property
    def processed_file(self) -> Path:
        return processed_path(self.doc_id)

    # -- persistence -------------------------------------------------------

    def write_raw_text(self, base_dir: Optional[Path] = None) -> Path:
        """Persist verbatim text under ``raw/<source>/<doc_id>/raw_text.txt``."""
        if self.raw_text is None:
            raise MedicalCorpusError(f"{self.doc_id}: no raw_text to write")
        target = (base_dir or RAW_DIR) / self.metadata.source / self.doc_id / "raw_text.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.raw_text, encoding="utf-8")
        return target

    def to_processed_dict(self) -> Dict[str, Any]:
        """Build the ``processed/`` payload.

        The raw text is referenced by path only — it is never inlined, so
        raw text and searchable text stay physically separate.
        """
        if self.indexed_text is None:
            raise MedicalCorpusError(f"{self.doc_id}: no indexed_text to process")
        payload: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "doc_id": self.doc_id,
            "title": self.title,
            "metadata": self.metadata.to_dict(),
            "raw_text_file": _relative_to_project(self.raw_text_file),
            "indexed_text": self.indexed_text,
        }
        original = next(
            (p for p in sorted(self.raw_dir.glob("original.*")) if p.is_file()), None
        )
        if original is not None:
            payload["original_file"] = _relative_to_project(original)
        return payload

    def write_processed(self, base_dir: Optional[Path] = None) -> Path:
        """Write ``processed/<doc_id>.json`` (metadata + indexed text only)."""
        self.validate()
        target = (base_dir or PROCESSED_DIR) / f"{self.doc_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_processed_dict()
        if "raw_text" in payload:
            raise MedicalCorpusError(
                f"{self.doc_id}: raw text must not enter processed output"
            )
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    @classmethod
    def from_processed_dict(cls, payload: Dict[str, Any]) -> "MedicalDocument":
        """Rehydrate a document from its ``processed/`` payload.

        ``raw_text`` is intentionally left empty — read it back from
        ``raw_text_file`` with :meth:`load_raw_text` when needed.
        """
        if not isinstance(payload, dict):
            raise MedicalCorpusError("processed payload must be a mapping")
        for key in ("schema_version", "doc_id", "title", "metadata", "indexed_text"):
            if key not in payload:
                raise MedicalCorpusError(f"processed payload missing {key!r}")
        if payload["schema_version"] != SCHEMA_VERSION:
            raise MedicalCorpusError(
                f"unsupported schema_version {payload['schema_version']!r}"
            )
        if "raw_text" in payload:
            raise MedicalCorpusError("processed payload must not contain raw_text")
        doc = cls(
            doc_id=payload["doc_id"],
            title=payload["title"],
            metadata=MedicalDocumentMetadata.from_dict(payload["metadata"]),
            indexed_text=payload["indexed_text"],
        )
        doc.validate()
        return doc

    @classmethod
    def load_processed(cls, doc_id: str, base_dir: Optional[Path] = None) -> "MedicalDocument":
        path = (base_dir or PROCESSED_DIR) / f"{doc_id}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_processed_dict(payload)

    def load_raw_text(self) -> Optional[str]:
        """Read the verbatim text back from ``raw/`` (not from ``processed/``)."""
        if not self.raw_text_file.is_file():
            return None
        self.raw_text = self.raw_text_file.read_text(encoding="utf-8")
        return self.raw_text


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def raw_document_dir(source: str, doc_id: str) -> Path:
    """``data/medical/raw/<source>/<doc_id>/`` — verbatim artifacts live here."""
    return RAW_DIR / source / doc_id


def raw_text_path(source: str, doc_id: str) -> Path:
    """``data/medical/raw/<source>/<doc_id>/raw_text.txt`` — verbatim text."""
    return raw_document_dir(source, doc_id) / "raw_text.txt"


def raw_original_path(source: str, doc_id: str, suffix: str) -> Path:
    """``data/medical/raw/<source>/<doc_id>/original.<ext>`` — download bytes."""
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return raw_document_dir(source, doc_id) / f"original{suffix}"


def processed_path(doc_id: str) -> Path:
    """``data/medical/processed/<doc_id>.json`` — searchable/indexed text."""
    return PROCESSED_DIR / f"{doc_id}.json"


def _relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def load_manifest(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load and validate ``data/medical/source_manifest.json``."""
    manifest_path = Path(path) if path else SOURCE_MANIFEST_PATH
    if not manifest_path.is_file():
        raise MedicalManifestError(f"manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MedicalManifestError(f"manifest is not valid JSON: {exc}") from exc
    assert_valid_manifest(manifest)
    return manifest


def validate_manifest(manifest: Any) -> List[str]:
    """Return a list of validation problems (empty list means valid)."""
    errors: List[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be a JSON object"]

    for key in REQUIRED_MANIFEST_FIELDS:
        if key not in manifest:
            errors.append(f"manifest missing {key!r}")
    if errors:
        return errors

    if manifest["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if manifest["manifest_type"] != MANIFEST_TYPE:
        errors.append(f"manifest_type must be {MANIFEST_TYPE!r}")

    sources = manifest["sources"]
    if not isinstance(sources, list) or not sources:
        errors.append("sources must be a non-empty list")
        return errors

    seen_doc_ids: set = set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            errors.append(f"sources[{index}] must be an object")
            continue
        label = source.get("source", f"sources[{index}]")
        for key in REQUIRED_SOURCE_FIELDS:
            if key not in source:
                errors.append(f"{label}: missing {key!r}")
        if any(key not in source for key in REQUIRED_SOURCE_FIELDS):
            continue

        if source["region"] not in REGIONS:
            errors.append(f"{label}: invalid region {source['region']!r}")
        for key in ("source", "name", "authority", "license", "license_notes"):
            if not str(source[key]).strip():
                errors.append(f"{label}: {key} must not be empty")
        if not str(source["landing_url"]).lower().startswith("https://"):
            errors.append(f"{label}: landing_url must be https")

        documents = source["documents"]
        if not isinstance(documents, list) or not documents:
            errors.append(f"{label}: documents must be a non-empty list")
            continue

        for d_index, doc in enumerate(documents):
            if not isinstance(doc, dict):
                errors.append(f"{label}.documents[{d_index}] must be an object")
                continue
            d_label = doc.get("doc_id", f"{label}.documents[{d_index}]")
            for key in REQUIRED_DOCUMENT_FIELDS:
                if key not in doc:
                    errors.append(f"{d_label}: missing {key!r}")
            if any(key not in doc for key in REQUIRED_DOCUMENT_FIELDS):
                continue

            doc_id = doc["doc_id"]
            if not DOC_ID_PATTERN.match(doc_id):
                errors.append(f"{d_label}: doc_id must be a lowercase slug")
            if doc_id in seen_doc_ids:
                errors.append(f"{d_label}: duplicate doc_id")
            seen_doc_ids.add(doc_id)

            for key in ("title", "topic", "format", "selection_reason"):
                if not str(doc[key]).strip():
                    errors.append(f"{d_label}: {key} must not be empty")
            if doc["document_type"] not in DOCUMENT_TYPES:
                errors.append(
                    f"{d_label}: invalid document_type {doc['document_type']!r}"
                )
            for key in ("url", "source_page"):
                value = doc.get(key)
                if value is None:
                    continue
                if not str(value).lower().startswith("https://"):
                    errors.append(f"{d_label}: {key} must be https")
                elif not any(host in str(value).lower() for host in OFFICIAL_HOSTS):
                    errors.append(f"{d_label}: {key} is not an official host: {value}")
            for key in ("published_at", "updated_at"):
                value = doc[key]
                if value is not None and not DATE_PATTERN.match(str(value)):
                    errors.append(f"{d_label}: {key} has invalid date {value!r}")
    return errors


def assert_valid_manifest(manifest: Any) -> None:
    """Raise :class:`MedicalManifestError` if :func:`validate_manifest` fails."""
    errors = validate_manifest(manifest)
    if errors:
        raise MedicalManifestError(
            "invalid source manifest:\n- " + "\n- ".join(errors)
        )


def iter_manifest_documents(
    manifest: Dict[str, Any],
) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Yield ``(source_block, document_block)`` pairs from a valid manifest."""
    assert_valid_manifest(manifest)
    for source in manifest["sources"]:
        for doc in source["documents"]:
            yield source, doc


def metadata_from_manifest(
    source: Dict[str, Any], doc: Dict[str, Any]
) -> MedicalDocumentMetadata:
    """Build validated metadata for one manifest document entry."""
    metadata = MedicalDocumentMetadata(
        source=source["source"],
        source_url=doc["url"],
        region=source["region"],
        authority=source["authority"],
        document_type=doc["document_type"],
        disease_topic=doc["topic"],
        published_at=doc.get("published_at"),
        updated_at=doc.get("updated_at"),
        section=None,
        page=None,
        license=source["license"],
    )
    metadata.validate()
    return metadata


def manifest_source_ids(manifest: Dict[str, Any]) -> List[str]:
    """Return the ordered list of source ids in the manifest."""
    assert_valid_manifest(manifest)
    return [source["source"] for source in manifest["sources"]]
