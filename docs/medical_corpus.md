# Medical Corpus — Phases 1–2 (Foundation + Fetching)

Phase 1 of the Medibo medical RAG MVP built **only the corpus foundation**:
curated official sources, a source manifest, a fixed metadata schema, on-disk
layout, tests and documentation. Phase 2 added fetching: downloading every
manifest document and extracting verbatim raw text. Retrieval, RRF fusion, the
candidate pool, the reranker and the embedding model are **unchanged**.

## Goals

- Diseases, symptoms, basic/normal care, warning signs and
  "when to seek medical attention" material.
- Small curated set of useful Indian documents plus one global
  plain-language health source.
- No massive datasets (no PMC, UMLS, SNOMED CT, PrimeKG, PrimeKG successors)
  in this phase.

## Directory structure

```
data/medical/
├── source_manifest.json        # curated list of official sources + documents
├── raw/                        # verbatim artifacts — never indexed directly
│   ├── medlineplus/
│   ├── icmr/
│   ├── mohfw_clinical_establishments/
│   └── ncdc/
└── processed/                  # searchable/indexed text + metadata
    └── <doc_id>.json           # one JSON document per corpus document
```

Per document, downloads and verbatim text live under
`raw/<source>/<doc_id>/`:

```
data/medical/raw/<source>/<doc_id>/
├── original.<ext>              # byte-identical download (git-ignored)
└── raw_text.txt                # verbatim extracted text (git-ignored)
```

`raw/` is regenerable from the manifest, so downloads are git-ignored; only
the manifest, the processed output and `.gitkeep` markers are tracked.

## Official sources

| source id | publisher / authority | region | landing page | licence / usage |
|---|---|---|---|---|
| `medlineplus` | U.S. National Library of Medicine (NLM), NIH, HHS | global | https://medlineplus.gov/xml.html | Public domain (U.S. Government work); attribution to MedlinePlus.gov requested. Some linked third-party content (e.g. A.D.A.M. articles) is copyrighted and must be excluded. |
| `icmr` | Indian Council of Medical Research (ICMR), Dept. of Health Research, MoHFW | india | https://www.icmr.gov.in/guidelines | Copyright ICMR / Government of India — non-commercial use with attribution; verify ICMR terms before redistribution. |
| `mohfw_clinical_establishments` | DGHS / National Council for Clinical Establishments, MoHFW | india | https://clinicalestablishments.mohfw.gov.in/en/standard-treatment-guidelines | Government of India Standard Treatment Guidelines — non-commercial educational/research use with attribution. |
| `ncdc` | National Centre for Disease Control (NCDC), DGHS, MoHFW | india | https://ncdc.mohfw.gov.in/includes/Resource_Library/ | Government of India public-health material — non-commercial use with attribution; verify NCDC/MoHFW terms. |

### Selected documents (16)

| doc_id | source | type | topic | published |
|---|---|---|---|---|
| `medlineplus-health-topics-xml` | medlineplus | health_topic_dataset | All English health topics; 26-topic curated subset | 2026-10-07 (daily file) |
| `icmr-type-2-diabetes-guideline` | icmr | clinical_guideline | Type 2 diabetes — care, complications | 2018 |
| `icmr-common-syndromes-antimicrobial-2019` | icmr | treatment_guideline | Common infectious syndromes (fever, respiratory, diarrhoea) | 2019 |
| `icmr-rickettsial-diseases-india` | icmr | clinical_guideline | Rickettsial diseases — fever with rash | 2015 |
| `icmr-malaria-drug-policy-north-east` | icmr | treatment_guideline | P. falciparum malaria drug policy (North East) | — |
| `mohfw-stg-hypertension-full` | mohfw_clinical_establishments | standard_treatment_guideline | Hypertension in adults | 2016-03 |
| `mohfw-stg-hypertension-quick-reference` | mohfw_clinical_establishments | standard_treatment_guideline | Hypertension quick reference | 2016-05 |
| `mohfw-stg-snakebite-full` | mohfw_clinical_establishments | standard_treatment_guideline | Snakebite — first aid, ASV, referral | 2016-06 |
| `mohfw-stg-snakebite-quick-reference` | mohfw_clinical_establishments | standard_treatment_guideline | Snakebite quick reference | 2016-07 |
| `mohfw-stg-major-trauma-full` | mohfw_clinical_establishments | standard_treatment_guideline | Major trauma — escalation/red flags | — |
| `mohfw-stg-child-respiratory-infection-quick-reference` | mohfw_clinical_establishments | standard_treatment_guideline | Child respiratory infections — danger signs | 2016-07 |
| `mohfw-stg-rabies-prophylaxis` | mohfw_clinical_establishments | standard_treatment_guideline | Rabies — bite first aid, PEP | — |
| `ncdc-leptospirosis-national-guidelines` | ncdc | clinical_guideline | Leptospirosis — diagnosis, case management, prevention | 2015 |
| `ncdc-heatwave-public-health-advisory-2024` | ncdc | public_health_advisory | Heatwave — prevention do's and don'ts | 2024 |
| `ncdc-emergency-cooling-heat-illness-2024` | ncdc | public_health_guideline | Severe heat illness — emergency cooling | 2024-03 |
| `ncdc-hypertension-faq-general-public` | ncdc | patient_faq | High blood pressure — patient FAQ | — |

Every selected PDF was checked for machine-extractable text; image-only scans
were rejected (for example the 52-page NCDC rabies PDF yields no text layer).

The MedlinePlus entry stores the date-stamped Health Topic XML URL
(`mplus_topics_YYYY-MM-DD.xml`, published Tuesday–Saturday) plus the DTD URL,
an `English` language filter and this curated topic subset for Phase 2:
Dengue, Malaria, Chikungunya, Rabies, Tuberculosis, Cholera, High Blood
Pressure, Diabetes Type 2, Fever, Cough, Breathing Problems, Pneumonia,
Diarrhea, Dehydration, Nausea and Vomiting, Headache, Rashes, Fainting,
Seizures, Heat Illness, First Aid, Stroke, Heart Attack, Jaundice,
Encephalitis, Flu.

## Source manifest

`data/medical/source_manifest.json` (schema version `1`, type
`medical_source_manifest`) has one block per source and one entry per
selected document.

Source block:

| field | description |
|---|---|
| `source` | Stable source id (`medlineplus`, `icmr`, …) used in paths and metadata |
| `name` | Human-readable source name |
| `region` | `global` \| `india` |
| `authority` | Issuing body (organisation, ministry, country) |
| `landing_url` | Official index page the documents were selected from |
| `license` | Short licence / usage statement |
| `license_notes` | Attribution, redistribution and exclusion notes |
| `documents` | List of document entries |

Document entry:

| field | description |
|---|---|
| `doc_id` | Unique lowercase slug; also the `processed/<doc_id>.json` file name |
| `title` | Document title as published |
| `url` | Direct HTTPS download URL on an official host |
| `document_type` | Controlled vocabulary (below) |
| `topic` | Disease / symptom / topic covered |
| `published_at` | `YYYY`, `YYYY-MM`, `YYYY-MM-DD`, or `null` when not stated |
| `updated_at` | Same formats, `null` when not stated |
| `format` | `pdf` \| `xml` |
| `source_page` | Optional page that lists the document |
| `selection_reason` | Why it is in the MVP corpus |
| `notes` | Extraction/date caveats |

Validation rules (`domain/medical.validate_manifest`): required fields present,
`region` and `document_type` in their controlled vocabularies, `doc_id` unique
slug, dates in partial ISO-8601 form, all URLs HTTPS on one of the four
official hosts, licence and usage notes non-empty.

## Metadata schema

`domain/medical.py::MedicalDocumentMetadata` is the fixed schema attached to
every medical document (and, in Phase 2, to every indexed chunk):

| field | type | notes |
|---|---|---|
| `source` | `str` | Manifest source id |
| `source_url` | `str` | HTTPS URL the document came from |
| `region` | `str` | `global` \| `india` |
| `authority` | `str` | Issuing body |
| `document_type` | `str` | `health_topic_dataset`, `clinical_guideline`, `treatment_guideline`, `standard_treatment_guideline`, `public_health_advisory`, `public_health_guideline`, `patient_faq` |
| `disease_topic` | `str` | Disease / symptom / topic |
| `published_at` | `str \| None` | `YYYY`, `YYYY-MM`, `YYYY-MM-DD` |
| `updated_at` | `str \| None` | Same formats |
| `section` | `str \| None` | Section heading of the passage |
| `page` | `int \| None` | 1-based page for PDF-derived passages |
| `license` | `str` | Licence inherited from the source block |

`MedicalDocument` adds `doc_id`, `title`, `raw_text` and `indexed_text`.

## Raw text vs. indexed text

The two representations are separate fields and separate files:

| | raw text | indexed text |
|---|---|---|
| Field | `MedicalDocument.raw_text` | `MedicalDocument.indexed_text` |
| Location | `data/medical/raw/<source>/<doc_id>/raw_text.txt` | `data/medical/processed/<doc_id>.json` → `indexed_text` |
| Content | Verbatim extraction (headers, footers, disclaimers) | Cleaned, section-aware text intended for chunking/embedding |
| Indexed? | No — never written to `processed/` | Yes |

`MedicalDocument.to_processed_dict()` stores the raw text **by path only**
(`raw_text_file`), so raw text cannot leak into the searchable output;
`MedicalDocument.from_processed_dict()` rejects payloads that contain a
`raw_text` key.

## What is reused from Potbot (unchanged)

- `ingestion/loaders.py`, `chunkers.py`, `embedders.py`, `indexers.py`,
  `pipeline.py` — ingestion pipeline as-is.
- Elasticsearch index, hybrid retrieval, RRF fusion, candidate pool,
  reranker, prompt builders and LLM providers — untouched.
- `config.py` / `ingestion/config.py` — no new retrieval knobs in Phase 1.
- Test layout: plain `unittest` modules under `tests/`.

New code is limited to `domain/medical.py` (pure dataclasses + manifest
helpers, stdlib-only), `scripts/fetch_medical_corpus.py` (Phase 2 downloads,
PyMuPDF extraction, MedlinePlus XML parsing, fetch report) and their tests.

## Tests

```bash
python -m pytest tests/test_medical_corpus.py -q      # manifest + metadata schema
python -m pytest tests/test_fetch_medical_corpus.py -q  # fetch / extraction (no network)
```

Covers: manifest existence and schema, all four sources, required fields,
controlled vocabularies, unique `doc_id` slugs, date formats, official-host
URL checks, licence/usage notes, validator detection of broken manifests,
metadata field names/defaults/validation, directory layout, raw/processed path
disjointness, and raw-vs-indexed persistence round-trips — plus, for Phase 2:
download layout, `original.*`/`raw_text.txt`/`metadata.json` contents,
manifest metadata attachment, non-PDF and no-text rejection, skip/force
behaviour, curated+English MedlinePlus topic filtering, per-topic sidecars,
missing-title and malformed-XML failures, and report structure/totals.

## Fetching the corpus (Phase 2)

`scripts/fetch_medical_corpus.py` reads `source_manifest.json`, downloads every
listed document and writes the layout above. It never invents sources or
documents: only `doc_id`s present in the manifest are created.

```bash
python -m scripts.fetch_medical_corpus                                        # fetch + write report
python -m scripts.fetch_medical_corpus --force                                # re-download everything
python -m scripts.fetch_medical_corpus --doc-id ncdc-hypertension-faq-general-public
```

Per document:

- `original.<ext>` — byte-identical response body (PDF or XML).
- `raw_text.txt` — extracted text; PDFs go through PyMuPDF with `[page N]`
  page markers so page metadata survives into Phase 3.
- `metadata.json` — the fixed Phase 1 metadata schema (`metadata` key) plus
  `manifest_source` and `manifest_document` (the manifest block that selected
  the document), `sha256` of the original bytes and the raw text character
  count.

MedlinePlus XML is handled separately: `original.xml` plus a root
`raw_text.txt` concatenating the curated topics, and one
`topics/<slug>/raw_text.txt` + `metadata.json` per selected English topic
(`metadata.disease_topic` carries the topic title, `doc_id` is
`<parent_doc_id>--<slug>`). Spanish records and non-curated topics are
discarded; a curated title missing from the XML fails the document.

Documents with a non-PDF body or with fewer than 200 characters of extracted
text are marked `failed` (no `raw_text.txt` is produced, so they are never
indexed and are retried on the next run). Already-fetched documents are
`skipped` unless `--force` is given.

The run writes `data/medical/fetch_report.json` — per-document
`source`/`doc_id`/`status`/`bytes`/`extracted_chars`/`error` rows plus totals
(`documents`, `downloaded`, `failed`, `skipped`, `raw_bytes`,
`extracted_chars`, `topics`). Exit code is `1` when any document failed.

Fetched artifacts stay git-ignored (`data/medical/raw/**`); the report is
tracked.

## Phase 2 prerequisites

1. ~~Fetch script~~ — done: `scripts/fetch_medical_corpus.py` (Phase 2).
2. A cleaning step that turns `raw_text` into `indexed_text`
   (strip headers/footers, split sections, keep page/section metadata).
3. A `MedicalDocument` loader hook so `IngestionPipeline` can consume
   `data/medical/processed/*.json` without changing the existing loaders.
4. Chunk → embed → index while carrying the metadata schema into
   Elasticsearch chunk metadata for source-backed answers.
5. Only then: retrieval/prompt tuning for medical queries and a medical
   ground-truth set (kept separate from the existing office corpus eval).
