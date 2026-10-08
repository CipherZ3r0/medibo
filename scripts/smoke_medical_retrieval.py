#!/usr/bin/env python3
"""PHASE 6.5 — Medical retrieval smoke test (read-only, temporary).

Runs the *existing* Medibo retrieval pipeline unchanged against the
``medibo_default`` index for a fixed set of queries:

    query -> medibo_default -> hybrid retrieval -> RRF -> existing reranker

Nothing in production code is imported-and-mutated: this script only wires the
already-existing components (``HybridSearchStrategy`` = vector + BM25 + RRF,
``CrossEncoderReranker``) to a store bound to ``medibo_default`` instead of
``potbot_documents``. No writes are issued (the store is constructed without
``allow_default_write``, so writes would raise ``DatasetProtectionError``).

The full ``_source`` for each returned chunk_id is then fetched read-only from
Elasticsearch so the report can show which medical metadata fields survive
retrieval on the ``SearchResult`` object itself versus which only exist in the
index.

Usage::

    python scripts/smoke_medical_retrieval.py
    python scripts/smoke_medical_retrieval.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from ingestion.embedders import SentenceTransformerEmbedder
from ingestion.medical_index import MedicalVectorStore
from rag.rerankers import CrossEncoderReranker
from rag.retrievers import HybridSearchStrategy

INDEX_NAME = "medibo_default"

QUERIES = [
    "What are common symptoms of dengue?",
    "What should I do for a fever?",
    "What can help with cough and cold?",
    "What are the symptoms and basic management of hypertension?",
    "What are the basic care recommendations for type 2 diabetes?",
    "What should be done immediately after a snakebite?",
    "What should I do after a possible rabies exposure?",
    "What should I do for heat exhaustion or heat stroke?",
    "What are common symptoms of malaria?",
    "What is the treatment for condition XYZ that is not in the medical corpus?",
]

# Smoke-test oracle: which corpus documents *should* surface (checked against
# doc_id / file_name / title / disease_topic). This is a test expectation, not
# retrieval logic.
ORACLE: list[dict[str, Any]] = [
    {"label": "dengue", "match": ["dengue"], "india": False},
    {"label": "fever", "match": ["fever", "first-aid", "common-syndromes"], "india": False},
    {"label": "cough/cold", "match": ["cough", "respiratory", "breathing"], "india": False},
    {"label": "hypertension", "match": ["hypertension", "high-blood-pressure"], "india": True},
    {"label": "type 2 diabetes", "match": ["type-2-diabetes", "diabetes"], "india": True},
    {"label": "snakebite", "match": ["snakebite"], "india": True},
    {"label": "rabies", "match": ["rabies"], "india": True},
    {"label": "heat illness", "match": ["heat", "emergency-cooling", "heatwave"], "india": True},
    {"label": "malaria", "match": ["malaria"], "india": False},
    {"label": "out-of-corpus (control)", "match": ["zzz-no-match-zzz"], "india": False},
]

# Metadata fields requested by the report (page_number lives on SearchResult).
METADATA_FIELDS = [
    "title",
    "source",
    "source_url",
    "region",
    "authority",
    "document_type",
    "disease_topic",
    "published_at",
    "updated_at",
    "section",
    "dataset_type",
]


def preview(text: str, limit: int = 150) -> str:
    flat = " ".join((text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def fetch_index_sources(
    store: MedicalVectorStore, chunk_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Read-only ``es.get`` per chunk_id — no search, no writes."""
    out: dict[str, dict[str, Any]] = {}
    if store.es is None:
        return out
    for cid in chunk_ids:
        try:
            doc = store.es.get(index=store.index_name, id=cid)
        except Exception as exc:  # pragma: no cover - diagnostic only
            out[cid] = {"_error": str(exc)[:200]}
            continue
        out[cid] = dict(doc.get("_source", {}))
    return out


def match_score(oracle: dict[str, Any], source: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(source.get(k, "") or "")
        for k in ("doc_id", "file_name", "title", "disease_topic", "section")
    ).lower()
    return any(token in haystack for token in oracle["match"])


def run() -> dict[str, Any]:
    print(f"index           : {INDEX_NAME}")
    print(f"embedding model : {config.EMBEDDING_MODEL} (device=cpu)")
    print(f"strategy        : hybrid (vector + BM25) + RRF "
          f"v={config.VECTOR_RRF_WEIGHT} t={config.TEXT_RRF_WEIGHT} k={config.RRF_K}")
    print(f"candidate pool  : {config.RETRIEVAL_CANDIDATE_POOL} "
          f"(production default; fetch_k = pool*3 per leg)")
    print("reranker        : cross-encoder/ms-marco-MiniLM-L-6-v2 -> top 5")
    print(f"query rewriting : {config.USE_QUERY_REWRITING} (production default)")
    print()

    store = MedicalVectorStore(index_name=INDEX_NAME)  # read-only: no write opt-in
    stats = store.get_stats()
    if not stats.get("exists"):
        raise SystemExit(f"Index '{INDEX_NAME}' does not exist. Aborting.")
    print(f"docs in index   : {stats['doc_count']}\n")

    embedder = SentenceTransformerEmbedder(
        model_name=config.EMBEDDING_MODEL, device="cpu"
    )
    strategy = HybridSearchStrategy(embedder=embedder, vector_store=store)
    reranker = CrossEncoderReranker()

    report: dict[str, Any] = {
        "index": INDEX_NAME,
        "doc_count": stats["doc_count"],
        "candidate_pool": config.RETRIEVAL_CANDIDATE_POOL,
        "queries": [],
    }

    carried_field_hits = 0
    carried_field_total = 0
    value_mismatches: list[dict[str, Any]] = []
    relevant_pass = 0
    india_pass = 0
    india_checks = 0
    control_max_rerank: float | None = None
    relevant_reranks: list[float] = []

    for i, query in enumerate(QUERIES):
        oracle = ORACLE[i]
        t0 = time.perf_counter()
        fused = strategy.search(query, top_k=config.RETRIEVAL_CANDIDATE_POOL)
        reranked = reranker.rerank(query, fused, top_n=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        sources = fetch_index_sources(store, [r.chunk_id for r in reranked])

        bar = "=" * 78
        print(bar)
        print(f"[{i + 1}/10] {query}")
        print(f"    candidate_pool={len(fused)}  top={len(reranked)}  "
              f"latency={elapsed_ms:.0f}ms  oracle={oracle['label']}")
        print(bar)

        rows = []
        matches = 0
        india_hits = 0
        for rank, res in enumerate(reranked, start=1):
            idx_src = sources.get(res.chunk_id, {})
            carried = [f for f in METADATA_FIELDS if f in res.metadata]
            carried_field_hits += len(carried)
            carried_field_total += len(METADATA_FIELDS)
            # Exactness: every carried value must equal the stored _source value.
            for f in carried:
                if res.metadata.get(f) != idx_src.get(f):
                    value_mismatches.append(
                        {"chunk_id": res.chunk_id, "field": f,
                         "carried": res.metadata.get(f), "index": idx_src.get(f)}
                    )

            if match_score(oracle, idx_src):
                matches += 1
            if str(idx_src.get("region", "")).lower() == "india":
                india_hits += 1

            print(
                f"  #{rank}  rrf={res.rrf_score if res.rrf_score is None else round(res.rrf_score, 5)}"
                f"  rerank={None if res.rerank_score is None else round(res.rerank_score, 4)}"
                f"  page={res.page_number}"
            )
            print(f"      source        : {idx_src.get('source')}")
            print(f"      title         : {preview(str(idx_src.get('title')), 110)}")
            print(f"      disease_topic : {preview(str(idx_src.get('disease_topic')), 90)}")
            print(f"      region        : {idx_src.get('region')}")
            print(f"      authority     : {preview(str(idx_src.get('authority')), 90)}")
            print(f"      section       : {preview(str(idx_src.get('section')), 70)}")
            print(f"      source_url    : {idx_src.get('source_url')}")
            print(f"      file_name     : {res.file_name}")
            print(f"      preview       : {preview(res.text)}")
            if res.metadata:
                print(f"      carried meta  : {sorted(res.metadata)}")
            else:
                print("      carried meta  : (empty on SearchResult)")
            rows.append(
                {
                    "rank": rank,
                    "chunk_id": res.chunk_id,
                    "doc_id": res.doc_id,
                    "rrf_score": res.rrf_score,
                    "rerank_score": res.rerank_score,
                    "page_number": res.page_number,
                    "file_name": res.file_name,
                    "carried_metadata": {k: res.metadata.get(k) for k in METADATA_FIELDS},
                    "index_source": {
                        k: idx_src.get(k) for k in [*METADATA_FIELDS, "chunk_id", "doc_id"]
                    },
                    "preview": preview(res.text),
                }
            )

        # --- per-query validation ---
        notes = []
        if i < 9:
            ok = matches > 0
            relevant_pass += 1 if ok else 0
            notes.append(f"relevant_in_top5={'PASS' if ok else 'FAIL'} ({matches}/5)")
            for r in reranked:
                if r.rerank_score is not None:
                    relevant_reranks.append(float(r.rerank_score))
        else:
            control_max = max(
                (float(r.rerank_score) for r in reranked if r.rerank_score is not None),
                default=None,
            )
            control_max_rerank = control_max
            mentions_xyz = any(
                "xyz" in (r.text or "").lower() for r in reranked
            )
            notes.append(
                f"corpus_mentions_XYZ={mentions_xyz}  max_rerank={None if control_max is None else round(control_max, 4)}"
            )
            notes.append(
                "control: no document should claim to cover 'condition XYZ'"
            )

        if oracle["india"]:
            india_checks += 1
            ok = india_hits > 0
            india_pass += 1 if ok else 0
            notes.append(f"indian_source_in_top5={'PASS' if ok else 'FAIL'} ({india_hits}/5)")

        print(f"  validation: {' | '.join(notes)}")
        print()

        report["queries"].append(
            {
                "query": query,
                "oracle": oracle["label"],
                "latency_ms": round(elapsed_ms, 1),
                "results": rows,
                "notes": notes,
            }
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    mean_rel = (
        sum(relevant_reranks) / len(relevant_reranks) if relevant_reranks else None
    )
    metadata_complete = carried_field_hits == carried_field_total

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"queries 1-9 relevant doc in top-5 : {relevant_pass}/9")
    print(f"Indian-source checks (q4-q8)      : {india_pass}/{india_checks}")
    print(f"control query max rerank score    : "
          f"{None if control_max_rerank is None else round(control_max_rerank, 4)}")
    print(f"mean rerank score (queries 1-9)   : "
          f"{None if mean_rel is None else round(mean_rel, 4)}")
    print(f"metadata carried on SearchResult  : "
          f"{carried_field_hits}/{carried_field_total} field slots "
          f"({'COMPLETE' if metadata_complete else 'INCOMPLETE'})")
    print(f"carried values match index source : "
          f"{'YES' if not value_mismatches else f'NO ({len(value_mismatches)} mismatches)'}")
    if not metadata_complete:
        print("  -> medical metadata (title/source/region/authority/disease_topic/")
        print("     section/source_url/...) is stored in the index and fetched in")
        print("     this report via read-only es.get(), but is NOT populated onto")
        print("     SearchResult by ingestion/indexers.py::_parse_search_hits.")
    print()

    report["summary"] = {
        "relevant_pass": relevant_pass,
        "india_pass": india_pass,
        "india_checks": india_checks,
        "control_max_rerank": control_max_rerank,
        "mean_rerank_queries_1_9": mean_rel,
        "metadata_carried_hits": carried_field_hits,
        "metadata_carried_total": carried_field_total,
        "metadata_complete": metadata_complete,
        "metadata_value_mismatches": value_mismatches,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Optional path to dump the full report JSON")
    args = parser.parse_args(argv)
    report = run()
    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
