"""
Unit tests for the Phase 2 medical corpus fetcher (scripts/fetch_medical_corpus.py).

No network access: HTTP is replaced by a fake transport serving generated PDFs
and a small MedlinePlus XML fixture.  Covers download layout, PDF validation,
text extraction, rejection of unusable documents, curated-topic filtering,
manifest metadata attachment and the fetch report.
"""

import json
import tempfile
import unittest
from pathlib import Path

from domain import medical as med
from scripts import fetch_medical_corpus as fetch

try:
    import fitz
except ImportError:  # pragma: no cover
    fitz = None


SAMPLE_PDF_TEXT = (
    "Standard Treatment Guidelines Hypertension. Blood pressure is the force of "
    "blood against the walls of the arteries. Seek medical attention immediately "
    "if you have chest pain, trouble breathing, or weakness on one side of the body."
)


def make_pdf(text: str = SAMPLE_PDF_TEXT) -> bytes:
    """Build a small in-memory PDF (optionally blank) with PyMuPDF."""
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_textbox(page.rect, text)
    data = document.tobytes()
    document.close()
    return data


def make_xml(titles, extra_topics=()) -> bytes:
    """Build a MedlinePlus-style Health Topics XML fixture.

    *titles* become English records; *extra_topics* are appended verbatim
    (used for Spanish records and non-curated topics).
    """
    nodes = []
    for index, title in enumerate(titles, start=1):
        nodes.append(
            f'<health-topic meta-desc="desc" title="{title}" '
            f'url="https://medlineplus.gov/{fetch.topic_slug(title)}.html" '
            f'id="{index}" language="English" date-created="01/01/2010">'
            f"<also-called>Alias of {title}</also-called>"
            f"<full-summary>&lt;p&gt;{title} is explained here. "
            f"&lt;a href=\"https://medlineplus.gov/x.html\"&gt;See care steps&lt;/a&gt;.&lt;/p&gt;"
            f"</full-summary></health-topic>"
        )
    nodes.extend(extra_topics)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<health-topics total="{len(nodes)}" date-generated="10/07/2026 02:30:36">'
        + "".join(nodes)
        + "</health-topics>"
    )
    return xml.encode("utf-8")


class FakeHttp:
    """Transport double: maps a URL to (status_code, bytes) or a callable."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def get(self, url):
        self.calls.append(url)
        response = self.responses.get(url)
        if response is None:
            return 404, b""
        if callable(response):
            response = response(url)
        return response


def make_manifest(documents, source="icmr"):
    """Valid minimal manifest containing exactly *documents*."""
    return {
        "schema_version": med.SCHEMA_VERSION,
        "manifest_type": "medical_source_manifest",
        "sources": [
            {
                "source": source,
                "name": "Fixture Source",
                "region": "india",
                "authority": "Test Authority",
                "landing_url": "https://www.icmr.gov.in/guidelines",
                "license": "Test licence",
                "license_notes": "Test usage notes",
                "documents": documents,
            }
        ],
    }


def make_pdf_doc(doc_id="icmr-test-guideline", url=None):
    return {
        "doc_id": doc_id,
        "title": "Test Clinical Guideline",
        "url": url or "https://www.icmr.gov.in/guidelines/test.pdf",
        "document_type": "clinical_guideline",
        "topic": "Test disease",
        "published_at": "2018",
        "updated_at": None,
        "format": "pdf",
        "selection_reason": "fixture",
        "notes": "fixture",
    }


def make_xml_doc(titles, doc_id="medlineplus-health-topics-xml"):
    return {
        "doc_id": doc_id,
        "title": "MedlinePlus Health Topic XML",
        "url": "https://medlineplus.gov/xml/mplus_topics_2026-10-07.xml",
        "document_type": "health_topic_dataset",
        "topic": "All English health topics",
        "published_at": "2026-10-07",
        "updated_at": "2026-10-07",
        "format": "xml",
        "selection_reason": "fixture",
        "topic_titles": list(titles),
        "language_filter": "English",
    }


@unittest.skipIf(fitz is None, "PyMuPDF not installed")
class TestFetchHelpers(unittest.TestCase):

    def test_topic_slug(self):
        self.assertEqual(fetch.topic_slug("Nausea and Vomiting"), "nausea-and-vomiting")
        self.assertEqual(fetch.topic_slug("Diabetes Type 2"), "diabetes-type-2")
        self.assertEqual(fetch.topic_slug("H1N1 Flu (Swine Flu)"), "h1n1-flu-swine-flu")

    def test_html_to_text_strips_tags_and_unescapes(self):
        html = '<p>Care &amp; advice</p><ul><li>Rest</li><li>Fluids</li></ul>'
        text = fetch.html_to_text(html)
        self.assertNotIn("<", text)
        self.assertIn("Care & advice", text)
        self.assertIn("- Rest", text)
        self.assertIn("- Fluids", text)

    def test_validate_pdf_bytes_accepts_pdf_header(self):
        fetch.validate_pdf_bytes(make_pdf())

    def test_validate_pdf_bytes_rejects_html_and_empty(self):
        with self.assertRaises(fetch.FetchError):
            fetch.validate_pdf_bytes(b"<html>not a pdf</html>", "doc")
        with self.assertRaises(fetch.FetchError):
            fetch.validate_pdf_bytes(b"", "doc")

    def test_extract_pdf_text_reads_inserted_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "original.pdf"
            path.write_bytes(make_pdf("Hypertension quick reference guide"))
            text = fetch.extract_pdf_text(path)
        self.assertIn("Hypertension quick reference guide", text)
        self.assertIn("[page 1]", text)

    def test_extract_pdf_text_is_empty_for_blank_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "original.pdf"
            path.write_bytes(make_pdf(""))
            self.assertEqual(fetch.extract_pdf_text(path), "")

    def test_extract_pdf_text_rejects_non_pdf_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "original.pdf"
            path.write_bytes(b"plain text")
            with self.assertRaises(fetch.FetchError):
                fetch.extract_pdf_text(path)


@unittest.skipIf(fitz is None, "PyMuPDF not installed")
class TestPdfFetch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_root = Path(self.tmp.name)
        self.manifest = make_manifest([make_pdf_doc()])
        self.url = self.manifest["sources"][0]["documents"][0]["url"]
        self.pdf = make_pdf()
        self.http = FakeHttp({self.url: (200, self.pdf)})

    def _fetch(self, force=False):
        return fetch.fetch_corpus(
            self.manifest, http=self.http, raw_root=self.raw_root, force=force
        )

    def test_downloads_original_raw_text_and_metadata(self):
        result = self._fetch()[0]
        self.assertEqual(result.status, fetch.STATUS_DOWNLOADED)

        doc_dir = self.raw_root / "icmr" / "icmr-test-guideline"
        original = doc_dir / "original.pdf"
        raw_text = doc_dir / "raw_text.txt"
        metadata = doc_dir / "metadata.json"

        self.assertTrue(original.is_file())
        self.assertEqual(original.read_bytes(), self.pdf)
        self.assertTrue(raw_text.is_file())
        self.assertIn("Hypertension", raw_text.read_text(encoding="utf-8"))
        self.assertTrue(metadata.is_file())

        self.assertEqual(result.bytes, original.stat().st_size)
        self.assertEqual(result.extracted_chars, len(raw_text.read_text(encoding="utf-8")))
        self.assertTrue(result.raw_text_file.endswith("raw_text.txt"))

    def test_metadata_sidecar_attaches_manifest_source_and_document(self):
        self._fetch()
        payload = json.loads(
            (self.raw_root / "icmr" / "icmr-test-guideline" / "metadata.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(payload["schema_version"], med.SCHEMA_VERSION)
        self.assertEqual(payload["manifest_source"]["source"], "icmr")
        self.assertEqual(payload["manifest_source"]["authority"], "Test Authority")
        self.assertEqual(payload["manifest_source"]["license"], "Test licence")
        self.assertEqual(payload["manifest_document"]["doc_id"], "icmr-test-guideline")
        # Fixed Phase 1 metadata schema is attached too.
        metadata = payload["metadata"]
        for field_name in med.MedicalDocumentMetadata.FIELD_NAMES:
            self.assertIn(field_name, metadata)
        self.assertEqual(metadata["source"], "icmr")
        self.assertEqual(metadata["source_url"], self.url)
        self.assertEqual(metadata["document_type"], "clinical_guideline")
        self.assertTrue(payload["sha256"])

    def test_rejects_response_that_is_not_a_pdf(self):
        self.http.responses[self.url] = (200, b"<html>maintenance page</html>")
        result = self._fetch()[0]
        self.assertEqual(result.status, fetch.STATUS_FAILED)
        self.assertIn("not a PDF", result.error)
        doc_dir = self.raw_root / "icmr" / "icmr-test-guideline"
        self.assertFalse((doc_dir / "raw_text.txt").exists())
        self.assertFalse((doc_dir / "original.pdf").exists())

    def test_rejects_pdf_without_usable_text(self):
        self.http.responses[self.url] = (200, make_pdf(""))
        result = self._fetch()[0]
        self.assertEqual(result.status, fetch.STATUS_FAILED)
        self.assertIn("no usable text", result.error)
        doc_dir = self.raw_root / "icmr" / "icmr-test-guideline"
        self.assertFalse((doc_dir / "raw_text.txt").exists())
        # The original is kept for debugging, but never indexed.
        self.assertTrue((doc_dir / "original.pdf").exists())

    def test_rejects_http_error_status(self):
        self.http.responses[self.url] = (404, b"")
        result = self._fetch()[0]
        self.assertEqual(result.status, fetch.STATUS_FAILED)
        self.assertIn("HTTP 404", result.error)

    def test_second_run_skips_and_force_redownloads(self):
        first = self._fetch()[0]
        self.assertEqual(first.status, fetch.STATUS_DOWNLOADED)
        calls_after_first = len(self.http.calls)

        second = self._fetch()[0]
        self.assertEqual(second.status, fetch.STATUS_SKIPPED)
        self.assertEqual(len(self.http.calls), calls_after_first)

        third = self._fetch(force=True)
        self.assertEqual(third[0].status, fetch.STATUS_DOWNLOADED)
        self.assertEqual(len(self.http.calls), calls_after_first + 1)

    def test_failed_document_is_retried_on_next_run(self):
        self.http.responses[self.url] = (200, b"junk")
        self.assertEqual(self._fetch()[0].status, fetch.STATUS_FAILED)
        self.http.responses[self.url] = (200, self.pdf)
        self.assertEqual(self._fetch()[0].status, fetch.STATUS_DOWNLOADED)

    def test_fetches_only_manifest_documents(self):
        self._fetch()
        source_dirs = sorted(p.name for p in self.raw_root.iterdir() if p.is_dir())
        doc_dirs = sorted(
            p.name for p in (self.raw_root / "icmr").iterdir() if p.is_dir()
        )
        self.assertEqual(source_dirs, ["icmr"])
        self.assertEqual(doc_dirs, ["icmr-test-guideline"])


@unittest.skipIf(fitz is None, "PyMuPDF not installed")
class TestMedlinePlusFetch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_root = Path(self.tmp.name)
        self.titles = ["Fever", "Malaria"]
        self.manifest = make_manifest(
            [make_xml_doc(self.titles)], source="medlineplus"
        )
        self.manifest["sources"][0].update(
            landing_url="https://medlineplus.gov/xml.html"
        )
        self.url = self.manifest["sources"][0]["documents"][0]["url"]
        self.xml = make_xml(
            self.titles,
            extra_topics=[
                '<health-topic title="Fever" url="https://medlineplus.gov/spanish/fever.html" '
                'id="90" language="Spanish"><full-summary>Resumen en espanol</full-summary></health-topic>',
                '<health-topic title="Not Curated" url="https://medlineplus.gov/other.html" '
                'id="91" language="English"><full-summary>&lt;p&gt;Ignore me.&lt;/p&gt;</full-summary></health-topic>',
            ],
        )
        self.http = FakeHttp({self.url: (200, self.xml)})

    def _fetch(self, force=False):
        return fetch.fetch_corpus(
            self.manifest, http=self.http, raw_root=self.raw_root, force=force
        )

    def test_preserves_original_xml_and_writes_root_raw_text(self):
        result = self._fetch()[0]
        self.assertEqual(result.status, fetch.STATUS_DOWNLOADED)
        doc_dir = self.raw_root / "medlineplus" / "medlineplus-health-topics-xml"
        self.assertEqual((doc_dir / "original.xml").read_bytes(), self.xml)
        combined = (doc_dir / "raw_text.txt").read_text(encoding="utf-8")
        self.assertIn("Fever", combined)
        self.assertIn("Malaria", combined)
        self.assertEqual(result.extracted_chars, len(combined))
        self.assertEqual(result.topics, len(self.titles))

    def test_generates_raw_text_per_selected_topic(self):
        self._fetch()
        topics_dir = (
            self.raw_root
            / "medlineplus"
            / "medlineplus-health-topics-xml"
            / "topics"
        )
        produced = sorted(p.name for p in topics_dir.iterdir() if p.is_dir())
        self.assertEqual(produced, ["fever", "malaria"])
        fever = (topics_dir / "fever" / "raw_text.txt").read_text(encoding="utf-8")
        self.assertIn("Fever", fever)
        self.assertIn("https://medlineplus.gov/fever.html", fever)
        self.assertIn("Also called: Alias of Fever", fever)
        self.assertNotIn("<", fever)

    def test_keeps_only_curated_english_topics(self):
        self._fetch()
        topics_dir = (
            self.raw_root
            / "medlineplus"
            / "medlineplus-health-topics-xml"
            / "topics"
        )
        slugs = {p.name for p in topics_dir.iterdir() if p.is_dir()}
        self.assertEqual(slugs, {"fever", "malaria"})
        combined = (
            self.raw_root
            / "medlineplus"
            / "medlineplus-health-topics-xml"
            / "raw_text.txt"
        ).read_text(encoding="utf-8")
        self.assertNotIn("Not Curated", combined)
        self.assertNotIn("Resumen en espanol", combined)

    def test_per_topic_metadata_sidecar(self):
        self._fetch()
        payload = json.loads(
            (
                self.raw_root
                / "medlineplus"
                / "medlineplus-health-topics-xml"
                / "topics"
                / "malaria"
                / "metadata.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            payload["doc_id"], "medlineplus-health-topics-xml--malaria"
        )
        self.assertEqual(payload["parent_doc_id"], "medlineplus-health-topics-xml")
        self.assertEqual(payload["metadata"]["disease_topic"], "Malaria")
        self.assertEqual(
            payload["metadata"]["source_url"],
            "https://medlineplus.gov/malaria.html",
        )
        self.assertEqual(payload["metadata"]["source"], "medlineplus")
        self.assertEqual(payload["manifest_document"]["format"], "xml")

    def test_missing_curated_topic_fails_loudly(self):
        self.manifest["sources"][0]["documents"][0]["topic_titles"] = [
            "Fever",
            "Topic That Does Not Exist",
        ]
        result = self._fetch()[0]
        self.assertEqual(result.status, fetch.STATUS_FAILED)
        self.assertIn("missing from XML", result.error)

    def test_malformed_xml_fails(self):
        self.http.responses[self.url] = (200, b"<health-topics><broken></health-topics>")
        result = self._fetch()[0]
        self.assertEqual(result.status, fetch.STATUS_FAILED)
        self.assertIn("not well-formed", result.error)

    def test_second_run_skips(self):
        self.assertEqual(self._fetch()[0].status, fetch.STATUS_DOWNLOADED)
        result = self._fetch()[0]
        self.assertEqual(result.status, fetch.STATUS_SKIPPED)
        self.assertEqual(result.topics, len(self.titles))


@unittest.skipIf(fitz is None, "PyMuPDF not installed")
class TestReportAndOrchestration(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_root = Path(self.tmp.name) / "raw"
        self.manifest = med.load_manifest()
        pdf = make_pdf()
        xml = make_xml(self.manifest["sources"][0]["documents"][0]["topic_titles"])
        responses = {}
        for _, doc in med.iter_manifest_documents(self.manifest):
            responses[doc["url"]] = (200, xml if doc["format"] == "xml" else pdf)
        self.http = FakeHttp(responses)

    def test_full_manifest_fetch_creates_exactly_manifest_documents(self):
        results = fetch.fetch_corpus(
            self.manifest, http=self.http, raw_root=self.raw_root
        )
        self.assertEqual(len(results), 16)
        self.assertEqual(
            {r.status for r in results}, {fetch.STATUS_DOWNLOADED}
        )

        expected_sources = set(med.manifest_source_ids(self.manifest))
        actual_sources = {p.name for p in self.raw_root.iterdir() if p.is_dir()}
        self.assertEqual(actual_sources, expected_sources)

        expected_docs = {
            doc["doc_id"] for _, doc in med.iter_manifest_documents(self.manifest)
        }
        actual_docs = {
            p.name
            for source_dir in self.raw_root.iterdir()
            if source_dir.is_dir()
            for p in source_dir.iterdir()
            if p.is_dir()
        }
        self.assertEqual(actual_docs, expected_docs)

        for doc_id in expected_docs:
            doc_dirs = [p for p in self.raw_root.rglob(doc_id) if p.is_dir()]
            doc_dir = doc_dirs[0]
            self.assertTrue(list(doc_dir.glob("original.*")), doc_id)
            self.assertTrue((doc_dir / "raw_text.txt").is_file(), doc_id)
            self.assertTrue((doc_dir / "metadata.json").is_file(), doc_id)

    def test_medlineplus_topics_extracted_for_full_manifest(self):
        fetch.fetch_corpus(self.manifest, http=self.http, raw_root=self.raw_root)
        topics_dir = (
            self.raw_root
            / "medlineplus"
            / "medlineplus-health-topics-xml"
            / "topics"
        )
        titles = self.manifest["sources"][0]["documents"][0]["topic_titles"]
        self.assertEqual(len(list(topics_dir.glob("*/raw_text.txt"))), len(titles))

    def test_report_structure_and_totals(self):
        report = fetch.run_fetch(
            raw_root=self.raw_root,
            report_path=Path(self.tmp.name) / "report.json",
            http=self.http,
        )
        self.assertEqual(report["totals"]["documents"], 16)
        self.assertEqual(report["totals"]["downloaded"], 16)
        self.assertEqual(report["totals"]["failed"], 0)
        self.assertEqual(report["totals"]["skipped"], 0)
        self.assertGreater(report["totals"]["raw_bytes"], 0)
        self.assertGreater(report["totals"]["extracted_chars"], 0)
        self.assertGreater(report["totals"]["topics"], 0)

        for result in report["results"]:
            for key in (
                "source",
                "doc_id",
                "url",
                "status",
                "bytes",
                "extracted_chars",
                "error",
            ):
                self.assertIn(key, result)
            self.assertIn(result["status"], ("downloaded", "failed", "skipped"))

        written = json.loads(
            (Path(self.tmp.name) / "report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(written["totals"], report["totals"])

    def test_report_counts_failures(self):
        bad_url = "https://www.icmr.gov.in/guidelines/test.pdf"
        manifest_path = Path(self.tmp.name) / "manifest.json"
        manifest_path.write_text(
            json.dumps(make_manifest([make_pdf_doc(url=bad_url)])), encoding="utf-8"
        )
        http = FakeHttp({bad_url: (200, b"garbage")})
        report = fetch.run_fetch(
            manifest_path=manifest_path,
            raw_root=self.raw_root,
            report_path=Path(self.tmp.name) / "failed_report.json",
            http=http,
        )
        self.assertEqual(report["totals"]["failed"], 1)
        self.assertEqual(report["totals"]["downloaded"], 0)
        self.assertIn("failed", fetch.format_report(report))

    def test_format_report_prints_per_document_rows(self):
        report = fetch.run_fetch(
            raw_root=self.raw_root,
            report_path=Path(self.tmp.name) / "fmt.json",
            http=self.http,
            doc_ids=["icmr-type-2-diabetes-guideline"],
        )
        text = fetch.format_report(report)
        self.assertIn("icmr-type-2-diabetes-guideline", text)
        self.assertIn("downloaded", text)
        self.assertIn("extracted_chars=", text)

    def test_doc_id_filter_fetches_single_document(self):
        results = fetch.fetch_corpus(
            self.manifest,
            http=self.http,
            raw_root=self.raw_root,
            doc_ids=["ncdc-hypertension-faq-general-public"],
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].doc_id, "ncdc-hypertension-faq-general-public")
        self.assertEqual(len(self.http.calls), 1)
        source_dirs = {p.name for p in self.raw_root.iterdir() if p.is_dir()}
        self.assertEqual(source_dirs, {"ncdc"})


if __name__ == "__main__":
    unittest.main()
