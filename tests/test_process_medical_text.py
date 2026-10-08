"""
Unit tests for Phase 3 medical text processing.
"""
import json
import tempfile
import unittest
from pathlib import Path

from domain import medical as med
from scripts import process_medical_text as proc
from scripts import fetch_medical_corpus as fetch

try:
    import fitz
except ImportError:
    fitz = None


SAMPLE_PDF_TEXT = "Guideline for care.\n\n[page 1]\nIntroduction to treatment.\n\n[page 2]\nNext steps: rest and fluids. Follow up.\n"


def make_pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    if text:
        page.insert_textbox(page.rect, text)
    data = document.tobytes()
    document.close()
    return data


class TestProcessingHelpers(unittest.TestCase):
    def test_clean_pdf_text_preserves_markers_and_skips_pure_numbers(self):
        raw = "[page 1]\nHEADER\n123\nIntroduction\n"
        cleaned = proc.clean_pdf_text(raw)
        self.assertIn("[page 1]", cleaned)
        self.assertIn("HEADER", cleaned)
        self.assertNotIn("123", cleaned)
        self.assertIn("Introduction", cleaned)

    def test_build_indexed_text_preserves_page_boundaries(self):
        text = "[page 1]\nA\n[page 2]\nB"
        idx = proc.build_indexed_text(text)
        self.assertIn("[page 1]", idx)
        self.assertIn("[page 2]", idx)

    def test_find_and_split_page_markers(self):
        text = "[page 1]\nX\n[page 2]\nY"
        pages = proc.find_page_boundaries(text)
        self.assertEqual(pages, [(1, "[page 1]"), (2, "[page 2]")])
        segs = proc.split_by_page_markers(text)
        self.assertEqual(len(segs), 2)


@unittest.skipIf(fitz is None, "PyMuPDF not installed")
class TestProcessCorpus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.raw_root = Path(self.tmp.name)
        # create minimal PDF doc
        pdf_doc = {
            "doc_id": "test-pdf",
            "title": "Test PDF",
            "url": "https://www.icmr.gov.in/test.pdf",
            "document_type": "clinical_guideline",
            "topic": "Test",
            "published_at": "2020",
            "updated_at": None,
            "format": "pdf",
            "selection_reason": "t",
        }
        manifest = {
            "schema_version": med.SCHEMA_VERSION,
            "manifest_type": "medical_source_manifest",
            "sources": [
                {
                    "source": "icmr",
                    "name": "ICMR",
                    "region": "india",
                    "authority": "ICMR",
                    "landing_url": "https://www.icmr.gov.in",
                    "license": "Public domain",
                    "license_notes": "usage",
                    "documents": [pdf_doc],
                }
            ],
        }
        self.manifest = manifest
        # write raw
        raw_dir = self.raw_root / "icmr" / "test-pdf"
        raw_dir.mkdir(parents=True)
        raw_dir.joinpath("original.pdf").write_bytes(make_pdf(SAMPLE_PDF_TEXT))
        raw_dir.joinpath("raw_text.txt").write_text(SAMPLE_PDF_TEXT)
        raw_dir.joinpath("metadata.json").write_text(json.dumps({
            "doc_id": "test-pdf",
            "sha256": "abc",
            "bytes": len(make_pdf(SAMPLE_PDF_TEXT)),
            "text_chars": len(SAMPLE_PDF_TEXT),
        }))
        # monkeypatch
        self.old_raw = proc.RAW_DIR
        self.old_proc = proc.PROCESSED_DIR
        proc.RAW_DIR = self.raw_root
        proc.PROCESSED_DIR = Path(self.tmp.name) / "processed"
        proc.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        proc.RAW_DIR = self.old_raw
        proc.PROCESSED_DIR = self.old_proc

    def test_processed_output_schema_and_separation(self):
        res = proc.process_pdf_document("icmr", "test-pdf", self.manifest)
        out = proc.PROCESSED_DIR / "test-pdf.json"
        self.assertTrue(out.exists())
        self.assertNotEqual(proc.RAW_DIR, proc.PROCESSED_DIR)
        req = ['doc_id', 'title', 'indexed_text', 'source', 'manifest_document', 'pages', 'sections', 'raw_text_path']
        for k in req:
            self.assertIn(k, res)
        self.assertIn('[page 1]', res['indexed_text'])
        self.assertIn('[page 2]', res['indexed_text'])
        self.assertTrue(res['raw_text_path'].endswith('data/medical/raw/icmr/test-pdf/raw_text.txt') or res['raw_text_path'].endswith('icmr/test-pdf/raw_text.txt'))
        self.assertIn('icmr/test-pdf/raw_text.txt', res['raw_text_path'])


if __name__ == "__main__":
    unittest.main()
