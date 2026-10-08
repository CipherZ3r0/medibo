#!/usr/bin/env python3
"""
Fetch the Phase 2 medical corpus from data/medical/source_manifest.json.

For every document in the manifest this script writes into
``data/medical/raw/<source>/<doc_id>/``:

* ``original.<ext>``   — the byte-identical download (PDF or XML)
* ``raw_text.txt``     — verbatim extracted text (never indexed directly)
* ``metadata.json``    — manifest source + document metadata attached to the doc

MedlinePlus specifics
---------------------
The Health Topics XML is downloaded whole and preserved as ``original.xml``.
Only the curated topics listed in the manifest (``topic_titles``, English
records) are kept; each one gets ``topics/<slug>/raw_text.txt`` plus a
``metadata.json`` sidecar, and ``raw_text.txt`` at the document root is the
concatenation of the selected topics.

PDF specifics
-------------
Files are validated as real PDFs (header + openable by PyMuPDF), text is
extracted with PyMuPDF, and documents with no usable text are rejected and
logged instead of being silently kept.

Nothing here touches retrieval, RRF, the candidate pool, the reranker or the
embedding model, and no source or document outside the manifest is fetched.

Usage::

    python scripts/fetch_medical_corpus.py
    python scripts/fetch_medical_corpus.py --force
    python scripts/fetch_medical_corpus.py --doc-id ncdc-leptospirosis-national-guidelines
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import requests

try:
    import fitz  # PyMuPDF — same API as ingestion/loaders.py
except ImportError:  # pragma: no cover - only hit without PyMuPDF installed
    fitz = None

from domain.medical import (
    MEDICAL_DATA_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    SCHEMA_VERSION,
    MedicalCorpusError,
    iter_manifest_documents,
    load_manifest,
    metadata_from_manifest,
)

logger = logging.getLogger("fetch_medical_corpus")

DEFAULT_REPORT_PATH = MEDICAL_DATA_DIR / "fetch_report.json"
DEFAULT_TIMEOUT = 180
# Minimum characters of extracted text for a PDF to be considered usable.
MIN_EXTRACTED_CHARS = 200
USER_AGENT = "medibo-medical-corpus-fetch/1.0 (research corpus; official sources only)"

STATUS_DOWNLOADED = "downloaded"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


class FetchError(Exception):
    """A document could not be downloaded, validated or parsed."""


# ---------------------------------------------------------------------------
# Result / report models
# ---------------------------------------------------------------------------


@dataclass
class FetchResult:
    """Outcome of fetching one manifest document."""

    source: str
    doc_id: str
    url: str
    status: str
    bytes: int = 0
    extracted_chars: int = 0
    file: Optional[str] = None
    raw_text_file: Optional[str] = None
    topics: int = 0
    error: Optional[str] = None


def build_report(results: Sequence[FetchResult], manifest_path: Optional[Path] = None) -> Dict[str, Any]:
    """Build the fetch report payload (downloaded / failed / skipped / sizes)."""
    manifest_path = manifest_path or (MEDICAL_DATA_DIR / "source_manifest.json")
    counts = {STATUS_DOWNLOADED: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest": _relative_to_project(Path(manifest_path)),
        "totals": {
            "documents": len(results),
            "downloaded": counts.get(STATUS_DOWNLOADED, 0),
            "failed": counts.get(STATUS_FAILED, 0),
            "skipped": counts.get(STATUS_SKIPPED, 0),
            "raw_bytes": sum(r.bytes for r in results),
            "extracted_chars": sum(r.extracted_chars for r in results),
            "topics": sum(r.topics for r in results),
        },
        "results": [vars(result) for result in results],
    }


def write_report(report: Dict[str, Any], path: Optional[Path] = None) -> Path:
    report_path = Path(path) if path else DEFAULT_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path


def format_report(report: Dict[str, Any]) -> str:
    """Render a plain-text table of the fetch report."""
    rows = [("#", "status", "source", "doc_id", "bytes", "chars", "error")]
    for index, result in enumerate(report["results"], start=1):
        rows.append(
            (
                str(index),
                result["status"],
                result["source"],
                result["doc_id"],
                str(result["bytes"]),
                str(result["extracted_chars"]),
                (result["error"] or "")[:60],
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    totals = report["totals"]
    lines.append("")
    lines.append(
        "documents={documents} downloaded={downloaded} failed={failed} "
        "skipped={skipped} raw_bytes={raw_bytes} extracted_chars={extracted_chars} "
        "topics={topics}".format(**totals)
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class HttpClient:
    """Minimal HTTP client so tests can inject a fake transport."""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def get(self, url: str) -> Tuple[int, bytes]:
        response = requests.get(
            url,
            timeout=self.timeout,
            headers={"User-Agent": USER_AGENT},
            stream=True,
        )
        try:
            chunks: List[bytes] = []
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    chunks.append(chunk)
            return response.status_code, b"".join(chunks)
        finally:
            response.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relative_to_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def topic_slug(title: str) -> str:
    """``Nausea and Vomiting`` -> ``nausea-and-vomiting``."""
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def html_to_text(html: str) -> str:
    """Convert the MedlinePlus ``full-summary`` HTML fragment to plain text."""
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
    text = re.sub(r"</(p|div|li|ul|ol|h[1-6]|tr|table|section)\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<li[^>]*>", "\n- ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_pdf_bytes(data: bytes, doc_id: str = "?") -> None:
    """Raise :class:`FetchError` unless *data* is a real PDF file."""
    if not data:
        raise FetchError(f"{doc_id}: empty response body")
    if b"%PDF-" not in data[:1024]:
        raise FetchError(f"{doc_id}: response is not a PDF (missing %PDF- header)")


def extract_pdf_text(path: Path) -> str:
    """Extract page text with PyMuPDF, joining pages with page markers."""
    if fitz is None:
        raise FetchError("PyMuPDF (fitz) is not installed")
    try:
        with fitz.open(path) as document:
            if document.page_count == 0:
                raise FetchError(f"{path.name}: PDF has no pages")
            pages = []
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    pages.append(f"[page {page_number}]\n{page_text}")
            return "\n\n".join(pages).strip()
    except FetchError:
        raise
    except Exception as exc:  # corrupt / unreadable PDF
        raise FetchError(f"{path.name}: cannot read PDF ({exc})") from exc


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------


def build_metadata_payload(
    source_block: Dict[str, Any],
    doc_block: Dict[str, Any],
    *,
    original_file: Path,
    data: bytes,
    text_chars: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Manifest metadata attached to every fetched document."""
    manifest_source = {k: v for k, v in source_block.items() if k != "documents"}
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "doc_id": doc_block["doc_id"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "original_file": _relative_to_project(original_file),
        "bytes": len(data),
        "sha256": sha256_of(data),
        "text_chars": text_chars,
        # Fixed Phase 1 metadata schema, derived from the manifest.
        "metadata": metadata_from_manifest(source_block, doc_block).to_dict(),
        # Raw manifest blocks, kept verbatim for provenance.
        "manifest_source": manifest_source,
        "manifest_document": doc_block,
    }
    if extra:
        payload.update(extra)
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# PDF documents
# ---------------------------------------------------------------------------


def fetch_pdf_document(
    source_block: Dict[str, Any],
    doc_block: Dict[str, Any],
    http: Any,
    raw_root: Path = RAW_DIR,
    force: bool = False,
) -> FetchResult:
    """Download, validate and text-extract one manifest PDF."""
    source_id = source_block["source"]
    doc_id = doc_block["doc_id"]
    url = doc_block["url"]
    doc_dir = raw_root / source_id / doc_id
    original = doc_dir / f"original.{doc_block.get('format', 'pdf')}"
    raw_text = doc_dir / "raw_text.txt"
    metadata_path = doc_dir / "metadata.json"

    if not force and original.is_file() and raw_text.is_file() and metadata_path.is_file():
        text = raw_text.read_text(encoding="utf-8")
        logger.info("skip (already fetched): %s", doc_id)
        return FetchResult(
            source=source_id,
            doc_id=doc_id,
            url=url,
            status=STATUS_SKIPPED,
            bytes=original.stat().st_size,
            extracted_chars=len(text),
            file=_relative_to_project(original),
            raw_text_file=_relative_to_project(raw_text),
        )

    result = FetchResult(source=source_id, doc_id=doc_id, url=url, status=STATUS_FAILED)
    try:
        status_code, data = http.get(url)
        if status_code != 200:
            raise FetchError(f"{doc_id}: HTTP {status_code} for {url}")
        validate_pdf_bytes(data, doc_id)

        doc_dir.mkdir(parents=True, exist_ok=True)
        original.write_bytes(data)
        result.bytes = len(data)
        result.file = _relative_to_project(original)

        text = extract_pdf_text(original)
        if len(text) < MIN_EXTRACTED_CHARS:
            raise FetchError(
                f"{doc_id}: no usable text extracted ({len(text)} chars < {MIN_EXTRACTED_CHARS})"
            )

        raw_text.write_text(text, encoding="utf-8")
        _write_json(
            metadata_path,
            build_metadata_payload(
                source_block,
                doc_block,
                original_file=original,
                data=data,
                text_chars=len(text),
            ),
        )
        result.status = STATUS_DOWNLOADED
        result.extracted_chars = len(text)
        result.raw_text_file = _relative_to_project(raw_text)
        logger.info("fetched %s (%d bytes, %d chars)", doc_id, len(data), len(text))
    except (FetchError, MedicalCorpusError, requests.RequestException, OSError) as exc:
        result.status = STATUS_FAILED
        result.error = str(exc)
        logger.error("failed %s: %s", doc_id, exc)
    return result


# ---------------------------------------------------------------------------
# MedlinePlus health topic dataset
# ---------------------------------------------------------------------------


@dataclass
class HealthTopic:
    """One English MedlinePlus health topic selected by the manifest."""

    title: str
    url: str
    topic_id: str
    date_created: Optional[str]
    also_called: List[str]
    summary_html: str

    @property
    def slug(self) -> str:
        return topic_slug(self.title)

    @property
    def summary_text(self) -> str:
        return html_to_text(self.summary_html)

    def as_text(self) -> str:
        lines = [self.title, self.url]
        if self.date_created:
            lines.append(f"MedlinePlus topic id: {self.topic_id} (created {self.date_created})")
        if self.also_called:
            lines.append("Also called: " + "; ".join(self.also_called))
        lines.append("")
        lines.append(self.summary_text)
        return "\n".join(lines).strip() + "\n"


def parse_selected_topics(xml_bytes: bytes, titles: Sequence[str]) -> List[HealthTopic]:
    """Parse the Health Topics XML and keep only curated English topics.

    Raises :class:`FetchError` if a curated title from the manifest is absent,
    so the corpus can never silently drift from the manifest.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise FetchError(f"MedlinePlus XML is not well-formed: {exc}") from exc

    if root.tag != "health-topics":
        raise FetchError(f"unexpected XML root element: {root.tag!r}")

    english: Dict[str, ET.Element] = {}
    for node in root.findall("health-topic"):
        if node.get("language") != "English":
            continue
        english.setdefault(node.get("title") or "", node)

    missing = [title for title in titles if title not in english]
    if missing:
        raise FetchError(f"curated topics missing from XML: {missing}")

    topics: List[HealthTopic] = []
    for title in titles:
        node = english[title]
        summary_node = node.find("full-summary")
        topics.append(
            HealthTopic(
                title=title,
                url=node.get("url") or "",
                topic_id=node.get("id") or "",
                date_created=node.get("date-created"),
                also_called=[
                    (child.text or "").strip()
                    for child in node.findall("also-called")
                    if (child.text or "").strip()
                ],
                summary_html=(summary_node.text or "") if summary_node is not None else "",
            )
        )
    return topics


def fetch_health_topic_dataset(
    source_block: Dict[str, Any],
    doc_block: Dict[str, Any],
    http: Any,
    raw_root: Path = RAW_DIR,
    force: bool = False,
) -> FetchResult:
    """Download the MedlinePlus XML, keep curated topics, write raw text."""
    source_id = source_block["source"]
    doc_id = doc_block["doc_id"]
    url = doc_block["url"]
    titles: Sequence[str] = doc_block.get("topic_titles") or []
    if not titles:
        raise MedicalCorpusError(f"{doc_id}: manifest entry has no topic_titles")

    doc_dir = raw_root / source_id / doc_id
    original = doc_dir / f"original.{doc_block.get('format', 'xml')}"
    raw_text = doc_dir / "raw_text.txt"
    metadata_path = doc_dir / "metadata.json"
    topics_dir = doc_dir / "topics"

    def topic_files() -> List[Path]:
        return sorted(topics_dir.glob("*/raw_text.txt")) if topics_dir.is_dir() else []

    if (
        not force
        and original.is_file()
        and raw_text.is_file()
        and metadata_path.is_file()
        and len(topic_files()) == len(titles)
    ):
        text = raw_text.read_text(encoding="utf-8")
        logger.info("skip (already fetched): %s", doc_id)
        return FetchResult(
            source=source_id,
            doc_id=doc_id,
            url=url,
            status=STATUS_SKIPPED,
            bytes=original.stat().st_size,
            extracted_chars=len(text),
            file=_relative_to_project(original),
            raw_text_file=_relative_to_project(raw_text),
            topics=len(topic_files()),
        )

    result = FetchResult(source=source_id, doc_id=doc_id, url=url, status=STATUS_FAILED)
    try:
        status_code, data = http.get(url)
        if status_code != 200:
            raise FetchError(f"{doc_id}: HTTP {status_code} for {url}")
        if not data.strip():
            raise FetchError(f"{doc_id}: empty response body")

        topics = parse_selected_topics(data, titles)  # validates XML + curated titles

        doc_dir.mkdir(parents=True, exist_ok=True)
        original.write_bytes(data)  # original XML preserved byte-for-byte
        result.bytes = len(data)
        result.file = _relative_to_project(original)

        if topics_dir.exists():
            for stale in topics_dir.glob("*/raw_text.txt"):
                stale.unlink()
        pieces: List[str] = []
        for topic in topics:
            topic_dir = topics_dir / topic.slug
            topic_dir.mkdir(parents=True, exist_ok=True)
            text = topic.as_text()
            (topic_dir / "raw_text.txt").write_text(text, encoding="utf-8")
            _write_json(
                topic_dir / "metadata.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "doc_id": f"{doc_id}--{topic.slug}",
                    "parent_doc_id": doc_id,
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "text_chars": len(text),
                    "metadata": {
                        **metadata_from_manifest(source_block, doc_block).to_dict(),
                        "source_url": topic.url,
                        "disease_topic": topic.title,
                    },
                    "manifest_document": doc_block,
                    "topic": {
                        "title": topic.title,
                        "url": topic.url,
                        "id": topic.topic_id,
                        "date_created": topic.date_created,
                        "also_called": topic.also_called,
                    },
                },
            )
            pieces.append(text)

        combined = ("\n\n" + ("=" * 72) + "\n\n").join(pieces)
        raw_text.write_text(combined, encoding="utf-8")
        _write_json(
            metadata_path,
            build_metadata_payload(
                source_block,
                doc_block,
                original_file=original,
                data=data,
                text_chars=len(combined),
                extra={"topics": [topic.title for topic in topics]},
            ),
        )
        result.status = STATUS_DOWNLOADED
        result.extracted_chars = len(combined)
        result.raw_text_file = _relative_to_project(raw_text)
        result.topics = len(topics)
        logger.info(
            "fetched %s (%d bytes, %d topics, %d chars)",
            doc_id,
            len(data),
            len(topics),
            len(combined),
        )
    except (FetchError, MedicalCorpusError, requests.RequestException, OSError) as exc:
        result.status = STATUS_FAILED
        result.error = str(exc)
        logger.error("failed %s: %s", doc_id, exc)
    return result


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def fetch_document(
    source_block: Dict[str, Any],
    doc_block: Dict[str, Any],
    http: Any,
    raw_root: Path = RAW_DIR,
    force: bool = False,
) -> FetchResult:
    """Dispatch one manifest document to the right fetcher."""
    if doc_block.get("format") == "xml" or doc_block.get("document_type") == "health_topic_dataset":
        return fetch_health_topic_dataset(source_block, doc_block, http, raw_root, force)
    return fetch_pdf_document(source_block, doc_block, http, raw_root, force)


def fetch_corpus(
    manifest: Dict[str, Any],
    http: Any,
    raw_root: Path = RAW_DIR,
    force: bool = False,
    doc_ids: Optional[Iterable[str]] = None,
) -> List[FetchResult]:
    """Fetch every (or the selected) manifest document — nothing else."""
    wanted = set(doc_ids) if doc_ids else None
    results: List[FetchResult] = []
    for source_block, doc_block in iter_manifest_documents(manifest):
        if wanted is not None and doc_block["doc_id"] not in wanted:
            continue
        results.append(fetch_document(source_block, doc_block, http, raw_root, force))
    return results


def run_fetch(
    manifest_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
    raw_root: Path = RAW_DIR,
    force: bool = False,
    doc_ids: Optional[Iterable[str]] = None,
    http: Any = None,
) -> Dict[str, Any]:
    """Fetch the corpus, write the report file and return the report payload."""
    manifest = load_manifest(manifest_path)
    results = fetch_corpus(
        manifest,
        http=http or HttpClient(),
        raw_root=raw_root,
        force=force,
        doc_ids=doc_ids,
    )
    report = build_report(results, manifest_path=manifest_path)
    write_report(report, report_path)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch the Medibo medical corpus")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to source_manifest.json (default: data/medical/source_manifest.json)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Where to write the fetch report JSON (default: data/medical/fetch_report.json)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_DIR,
        help="Raw corpus root (default: data/medical/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download documents that were already fetched",
    )
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        default=None,
        help="Fetch only this doc_id (repeatable)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if fitz is None:
        logger.error("PyMuPDF (fitz) is not installed — pip install PyMuPDF")
        return 2
    try:
        report = run_fetch(
            manifest_path=args.manifest,
            report_path=args.report,
            raw_root=args.raw_root,
            force=args.force,
            doc_ids=args.doc_ids,
        )
    except MedicalManifestError as exc:
        logger.error("%s", exc)
        return 2

    print(format_report(report))
    print(f"report: {_relative_to_project(Path(args.report or DEFAULT_REPORT_PATH))}")
    return 1 if report["totals"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
