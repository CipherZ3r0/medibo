"""
Unit tests for the Phase 1 medical corpus foundation:

* ``data/medical/source_manifest.json`` structure and validation rules
* ``domain/medical.py`` metadata schema (``MedicalDocumentMetadata``)
* raw vs. indexed text separation (``MedicalDocument``, ``raw/`` vs ``processed/``)
* ``data/medical/`` directory layout

Stdlib-only on purpose: these tests must run without Elasticsearch, embedding,
Groq or pandas installed.
"""

import json
import tempfile
import unittest
from pathlib import Path

from domain import medical as med
from domain.medical import (
    MedicalCorpusError,
    MedicalDocument,
    MedicalDocumentMetadata,
    MedicalManifestError,
)


SPEC_METADATA_FIELDS = (
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

EXPECTED_SOURCES = {
    "medlineplus",
    "icmr",
    "mohfw_clinical_establishments",
    "ncdc",
}


def _sample_metadata(**overrides):
    values = dict(
        source="icmr",
        source_url="https://www.icmr.gov.in/guidelines/example.pdf",
        region="india",
        authority="Indian Council of Medical Research (ICMR)",
        document_type="clinical_guideline",
        disease_topic="Type 2 diabetes mellitus",
        published_at="2018",
        updated_at=None,
        section=None,
        page=None,
        license="Copyright ICMR / Government of India",
    )
    values.update(overrides)
    return MedicalDocumentMetadata(**values)


class TestMedicalSourceManifest(unittest.TestCase):
    """Structure and validation of data/medical/source_manifest.json."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = med.load_manifest()
        cls.raw_manifest = json.loads(
            med.SOURCE_MANIFEST_PATH.read_text(encoding="utf-8")
        )

    # -- file / top level --------------------------------------------------

    def test_manifest_file_exists(self):
        self.assertTrue(med.SOURCE_MANIFEST_PATH.is_file())

    def test_manifest_validates_without_errors(self):
        self.assertEqual(med.validate_manifest(self.manifest), [])

    def test_manifest_top_level_schema(self):
        self.assertEqual(self.raw_manifest["schema_version"], med.SCHEMA_VERSION)
        self.assertEqual(self.raw_manifest["manifest_type"], "medical_source_manifest")
        self.assertIn("sources", self.raw_manifest)
        self.assertIsInstance(self.raw_manifest["sources"], list)
        self.assertTrue(self.raw_manifest["scope"].strip())

    def test_load_manifest_missing_file_raises(self):
        with self.assertRaises(MedicalManifestError):
            med.load_manifest(Path("/nonexistent/source_manifest.json"))

    # -- sources ------------------------------------------------------------

    def test_all_four_official_sources_present(self):
        self.assertEqual(set(med.manifest_source_ids(self.manifest)), EXPECTED_SOURCES)

    def test_source_required_fields_present(self):
        for source in self.manifest["sources"]:
            for key in med.REQUIRED_SOURCE_FIELDS:
                self.assertIn(key, source, f"{source.get('source')}: missing {key}")
            self.assertTrue(str(source["source"]).strip())
            self.assertTrue(str(source["authority"]).strip())
            self.assertTrue(str(source["name"]).strip())

    def test_source_region_values_are_controlled(self):
        for source in self.manifest["sources"]:
            self.assertIn(source["region"], med.REGIONS)

    def test_source_license_and_usage_notes_present(self):
        for source in self.manifest["sources"]:
            self.assertTrue(source["license"].strip(), source["source"])
            self.assertTrue(source["license_notes"].strip(), source["source"])
            self.assertTrue(
                source["landing_url"].lower().startswith("https://"), source["source"]
            )

    # -- documents ----------------------------------------------------------

    def test_document_required_fields_present(self):
        for _, doc in med.iter_manifest_documents(self.manifest):
            for key in med.REQUIRED_DOCUMENT_FIELDS:
                self.assertIn(key, doc, f"{doc.get('doc_id')}: missing {key}")
            self.assertTrue(doc["title"].strip())
            self.assertTrue(doc["topic"].strip())
            self.assertTrue(doc["selection_reason"].strip())

    def test_doc_ids_are_unique_lowercase_slugs(self):
        seen = []
        for _, doc in med.iter_manifest_documents(self.manifest):
            self.assertRegex(doc["doc_id"], med.DOC_ID_PATTERN.pattern)
            seen.append(doc["doc_id"])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(len(seen), 16)

    def test_document_types_are_in_controlled_vocabulary(self):
        for _, doc in med.iter_manifest_documents(self.manifest):
            self.assertIn(doc["document_type"], med.DOCUMENT_TYPES)

    def test_dates_are_iso_partial_or_none(self):
        for _, doc in med.iter_manifest_documents(self.manifest):
            for key in ("published_at", "updated_at"):
                value = doc[key]
                if value is not None:
                    self.assertRegex(str(value), med.DATE_PATTERN.pattern)

    def test_document_urls_are_https_official_hosts(self):
        for _, doc in med.iter_manifest_documents(self.manifest):
            self.assertTrue(doc["url"].lower().startswith("https://"), doc["doc_id"])
            self.assertTrue(
                any(host in doc["url"].lower() for host in med.OFFICIAL_HOSTS),
                f"{doc['doc_id']}: unexpected host in {doc['url']}",
            )

    def test_medlineplus_entry_has_language_filter_and_topic_subset(self):
        source = next(
            s for s in self.manifest["sources"] if s["source"] == "medlineplus"
        )
        doc = source["documents"][0]
        self.assertEqual(doc["language_filter"], "English")
        self.assertTrue(doc["dtd_url"].startswith("https://medlineplus.gov/"))
        topics = doc["topic_titles"]
        self.assertGreaterEqual(len(topics), 10)
        self.assertEqual(len(topics), len(set(topics)))
        self.assertTrue(all(t.strip() for t in topics))
        self.assertRegex(doc["url"], r"mplus_topics_\d{4}-\d{2}-\d{2}\.xml$")
        self.assertEqual(doc["published_at"], "2026-10-07")

    def test_every_document_maps_to_metadata_schema(self):
        for source, doc in med.iter_manifest_documents(self.manifest):
            metadata = med.metadata_from_manifest(source, doc)
            metadata.validate()
            self.assertEqual(metadata.source, source["source"])
            self.assertEqual(metadata.source_url, doc["url"])
            self.assertEqual(metadata.region, source["region"])
            self.assertEqual(metadata.authority, source["authority"])
            self.assertEqual(metadata.document_type, doc["document_type"])
            self.assertEqual(metadata.disease_topic, doc["topic"])
            self.assertEqual(metadata.license, source["license"])
            self.assertEqual(metadata.page, None)
            self.assertEqual(metadata.section, None)

    # -- validator detects broken manifests ---------------------------------

    def _mutated(self, mutate):
        manifest = json.loads(json.dumps(self.raw_manifest))
        mutate(manifest)
        return med.validate_manifest(manifest)

    def test_validator_detects_missing_document_url(self):
        errors = self._mutated(
            lambda m: m["sources"][1]["documents"][0].pop("url")
        )
        self.assertTrue(any("missing 'url'" in e for e in errors), errors)

    def test_validator_detects_invalid_region(self):
        errors = self._mutated(lambda m: m["sources"][1].update(region="mars"))
        self.assertTrue(any("invalid region" in e for e in errors), errors)

    def test_validator_detects_invalid_document_type(self):
        errors = self._mutated(
            lambda m: m["sources"][1]["documents"][0].update(
                document_type="blog_post"
            )
        )
        self.assertTrue(any("invalid document_type" in e for e in errors), errors)

    def test_validator_detects_duplicate_doc_id(self):
        def mutate(m):
            docs = m["sources"][1]["documents"]
            docs[1]["doc_id"] = docs[0]["doc_id"]

        errors = self._mutated(mutate)
        self.assertTrue(any("duplicate doc_id" in e for e in errors), errors)

    def test_validator_detects_non_official_url(self):
        errors = self._mutated(
            lambda m: m["sources"][1]["documents"][0].update(
                url="https://example.com/guideline.pdf"
            )
        )
        self.assertTrue(any("not an official host" in e for e in errors), errors)

    def test_validator_detects_bad_date(self):
        errors = self._mutated(
            lambda m: m["sources"][1]["documents"][0].update(published_at="March 2016")
        )
        self.assertTrue(any("invalid date" in e for e in errors), errors)

    def test_validator_detects_missing_license_notes(self):
        def mutate(m):
            m["sources"][3].pop("license_notes")

        errors = self._mutated(mutate)
        self.assertTrue(any("missing 'license_notes'" in e for e in errors), errors)

    def test_assert_valid_manifest_raises(self):
        with self.assertRaises(MedicalManifestError):
            med.assert_valid_manifest({"schema_version": med.SCHEMA_VERSION})


class TestMedicalMetadataSchema(unittest.TestCase):
    """MedicalDocumentMetadata field names, defaults and validation."""

    def test_field_names_match_phase1_specification(self):
        from dataclasses import fields

        self.assertEqual(MedicalDocumentMetadata.FIELD_NAMES, SPEC_METADATA_FIELDS)
        self.assertEqual(
            tuple(f.name for f in fields(MedicalDocumentMetadata)),
            SPEC_METADATA_FIELDS,
        )

    def test_valid_metadata_round_trips(self):
        metadata = _sample_metadata(section="Treatment", page=12)
        self.assertEqual(MedicalDocumentMetadata.from_dict(metadata.to_dict()), metadata)

    def test_optional_dates_section_page_default_to_none(self):
        metadata = MedicalDocumentMetadata(
            source="icmr",
            source_url="https://www.icmr.gov.in/guidelines/example.pdf",
            region="india",
            authority="Indian Council of Medical Research (ICMR)",
            document_type="clinical_guideline",
            disease_topic="Type 2 diabetes mellitus",
            license="Copyright ICMR / Government of India",
        )
        metadata.validate()
        self.assertIsNone(metadata.published_at)
        self.assertIsNone(metadata.updated_at)
        self.assertIsNone(metadata.section)
        self.assertIsNone(metadata.page)

    def test_from_dict_missing_field_raises(self):
        data = _sample_metadata().to_dict()
        data.pop("disease_topic")
        with self.assertRaises(MedicalCorpusError):
            MedicalDocumentMetadata.from_dict(data)

    def test_validate_rejects_bad_region(self):
        with self.assertRaises(MedicalCorpusError):
            _sample_metadata(region="us").validate()

    def test_validate_rejects_bad_document_type(self):
        with self.assertRaises(MedicalCorpusError):
            _sample_metadata(document_type="tweet").validate()

    def test_validate_rejects_bad_dates(self):
        for bad in ("March 2016", "2016-13", "2016-03-32", "16"):
            with self.assertRaises(MedicalCorpusError):
                _sample_metadata(published_at=bad).validate()

    def test_validate_accepts_partial_iso_dates(self):
        for good in ("2016", "2016-03", "2016-03-01"):
            _sample_metadata(published_at=good).validate()

    def test_validate_rejects_non_https_source_url(self):
        with self.assertRaises(MedicalCorpusError):
            _sample_metadata(source_url="http://www.icmr.gov.in/x.pdf").validate()

    def test_validate_rejects_invalid_page(self):
        for bad in (0, -1, "12"):
            with self.assertRaises(MedicalCorpusError):
                _sample_metadata(page=bad).validate()

    def test_validate_rejects_empty_license(self):
        with self.assertRaises(MedicalCorpusError):
            _sample_metadata(license="  ").validate()

    def test_validate_rejects_empty_authority(self):
        with self.assertRaises(MedicalCorpusError):
            _sample_metadata(authority="").validate()


class TestMedicalDocumentStructure(unittest.TestCase):
    """raw/ vs processed/ separation and document persistence."""

    @classmethod
    def setUpClass(cls):
        cls.manifest = med.load_manifest()

    def test_raw_and_processed_directories_exist(self):
        self.assertTrue(med.RAW_DIR.is_dir())
        self.assertTrue(med.PROCESSED_DIR.is_dir())

    def test_one_raw_subdirectory_per_manifest_source(self):
        for source in self.manifest["sources"]:
            path = med.RAW_DIR / source["source"]
            self.assertTrue(path.is_dir(), f"missing {path}")
            self.assertTrue(path.is_dir() and any(path.iterdir()), f"empty {path}")

    def test_raw_and_processed_roots_are_distinct(self):
        self.assertNotEqual(med.RAW_DIR, med.PROCESSED_DIR)
        self.assertEqual(med.RAW_DIR.name, "raw")
        self.assertEqual(med.PROCESSED_DIR.name, "processed")

    def test_raw_and_processed_paths_never_collide(self):
        for _, doc in med.iter_manifest_documents(self.manifest):
            raw_text = med.raw_text_path("icmr", doc["doc_id"])
            original = med.raw_original_path("icmr", doc["doc_id"], ".pdf")
            processed = med.processed_path(doc["doc_id"])
            self.assertNotEqual(raw_text, processed)
            self.assertNotEqual(original, processed)
            self.assertTrue(str(raw_text).startswith(str(med.RAW_DIR)))
            self.assertTrue(str(processed).startswith(str(med.PROCESSED_DIR)))
            self.assertEqual(processed.suffix, ".json")
            self.assertEqual(processed.stem, doc["doc_id"])
            self.assertEqual(raw_text.name, "raw_text.txt")

    def _sample_document(self):
        return MedicalDocument(
            doc_id="icmr-type-2-diabetes-guideline",
            title="Guidelines for Management of Type 2 Diabetes",
            metadata=_sample_metadata(),
            raw_text="RAW verbatim text with page footers and disclaimers.",
            indexed_text="INDEXED cleaned text prepared for chunking and embedding.",
        )

    def test_raw_and_indexed_text_are_separate_attributes(self):
        doc = self._sample_document()
        doc.validate()
        fields = list(MedicalDocument.__dataclass_fields__)
        self.assertIn("raw_text", fields)
        self.assertIn("indexed_text", fields)
        self.assertNotEqual(doc.raw_text, doc.indexed_text)
        # Mutating one must not affect the other.
        doc.raw_text = "changed"
        self.assertNotEqual(doc.raw_text, doc.indexed_text)

    def test_write_raw_text_and_write_processed_are_separate_files(self):
        doc = self._sample_document()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_file = doc.write_raw_text(base_dir=root / "raw")
            processed_file = doc.write_processed(base_dir=root / "processed")

            self.assertTrue(raw_file.is_file())
            self.assertTrue(processed_file.is_file())
            self.assertNotEqual(raw_file, processed_file)
            self.assertEqual(raw_file.read_text(encoding="utf-8"), doc.raw_text)
            self.assertEqual(raw_file.name, "raw_text.txt")

            payload = json.loads(processed_file.read_text(encoding="utf-8"))
            self.assertNotIn("raw_text", payload)
            self.assertNotIn(doc.raw_text, json.dumps(payload))
            self.assertEqual(payload["indexed_text"], doc.indexed_text)
            self.assertEqual(payload["metadata"]["source"], "icmr")

    def test_processed_payload_references_raw_text_by_path_only(self):
        doc = self._sample_document()
        payload = doc.to_processed_dict()
        self.assertIn("raw_text_file", payload)
        self.assertTrue(payload["raw_text_file"].endswith("raw_text.txt"))
        self.assertNotIn("raw_text", payload)

    def test_processed_round_trip_preserves_metadata_and_indexed_text(self):
        doc = self._sample_document()
        doc.metadata = _sample_metadata(section="Management", page=42)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            doc.write_processed(base_dir=base)
            reloaded = MedicalDocument.load_processed(doc.doc_id, base_dir=base)

        self.assertEqual(reloaded.doc_id, doc.doc_id)
        self.assertEqual(reloaded.title, doc.title)
        self.assertEqual(reloaded.indexed_text, doc.indexed_text)
        self.assertEqual(reloaded.metadata, doc.metadata)
        self.assertEqual(reloaded.metadata.section, "Management")
        self.assertEqual(reloaded.metadata.page, 42)
        self.assertIsNone(reloaded.raw_text)

    def test_from_processed_dict_rejects_embedded_raw_text(self):
        doc = self._sample_document()
        payload = doc.to_processed_dict()
        payload["raw_text"] = doc.raw_text
        with self.assertRaises(MedicalCorpusError):
            MedicalDocument.from_processed_dict(payload)

    def test_from_processed_dict_rejects_wrong_schema_version(self):
        payload = self._sample_document().to_processed_dict()
        payload["schema_version"] = 99
        with self.assertRaises(MedicalCorpusError):
            MedicalDocument.from_processed_dict(payload)

    def test_validate_rejects_bad_doc_id(self):
        doc = self._sample_document()
        doc.doc_id = "Not A Slug"
        with self.assertRaises(MedicalCorpusError):
            doc.validate()

    def test_validate_rejects_empty_indexed_text(self):
        doc = self._sample_document()
        doc.indexed_text = "   "
        with self.assertRaises(MedicalCorpusError):
            doc.validate()

    def test_validate_rejects_missing_title(self):
        doc = self._sample_document()
        doc.title = ""
        with self.assertRaises(MedicalCorpusError):
            doc.validate()

    def test_write_raw_text_requires_raw_text(self):
        doc = self._sample_document()
        doc.raw_text = None
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MedicalCorpusError):
                doc.write_raw_text(base_dir=Path(tmp))

    def test_processed_output_is_json_per_document(self):
        for _, entry in med.iter_manifest_documents(self.manifest):
            path = med.processed_path(entry["doc_id"])
            self.assertEqual(path.parent, med.PROCESSED_DIR)
            self.assertTrue(path.name.startswith(entry["doc_id"]))


if __name__ == "__main__":
    unittest.main()
