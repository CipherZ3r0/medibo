"""Build / rebuild the medical corpus index (Phase 6).

Embeds ``data/medical/chunks/medical_chunks.json`` with
``all-MiniLM-L6-v2`` and writes every chunk — including its full medical
metadata — into the ``medibo_default`` Elasticsearch index.  Also ensures the
separate ``medibo_user`` index exists so user data never shares storage with
the default corpus.

The existing ``potbot_documents`` index and every retrieval behaviour
(RRF / candidate pool / reranker / query rewriting) are untouched.

Usage::

    python scripts/build_medical_index.py                # full rebuild
    python scripts/build_medical_index.py --no-recreate  # additive only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from domain.models import Chunk
from ingestion.embedders import SentenceTransformerEmbedder
from ingestion.medical_index import (
    MEDICAL_DEFAULT_INDEX,
    MEDICAL_METADATA_FIELDS,
    MEDICAL_USER_INDEX,
    MedicalVectorStore,
)

CHUNKS_PATH = ROOT / "data" / "medical" / "chunks" / "medical_chunks.json"
REPORT_PATH = ROOT / "data" / "medical" / "index_build_report.json"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BULK_SIZE = 200


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_chunk_rows(path: Path = CHUNKS_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of chunks in {path}")
    return rows


def to_domain_chunks(rows: List[Dict[str, Any]]) -> List[Chunk]:
    """Convert raw chunk rows (Phase 4 output) into domain Chunks."""
    chunks: List[Chunk] = []
    for row in rows:
        metadata = {k: row.get(k) for k in MEDICAL_METADATA_FIELDS if k in row}
        chunks.append(
            Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["doc_id"],
                text=row["text"],
                chunk_index=row.get("chunk_index", 0),
                source_file=row.get("source_file", ""),
                file_name=row.get("file_name", ""),
                file_type=row.get("file_type", ""),
                page_number=row.get("page_number"),
                modified_date=row.get("modified_date"),
                embedding=None,
                metadata=metadata,
            )
        )
    return chunks


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_medical_index(
    chunks_path: Path = CHUNKS_PATH,
    index_name: str = MEDICAL_DEFAULT_INDEX,
    user_index_name: str = MEDICAL_USER_INDEX,
    recreate: bool = True,
    device: str = "cpu",
    report_path: Path = REPORT_PATH,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Embed the medical corpus and (re)build *index_name* + user index."""
    rows = load_chunk_rows(chunks_path)
    chunks = to_domain_chunks(rows)

    # --- embed (CPU; GPU probe recorded in data/medical/gpu_benchmark_results.json)
    embedder = SentenceTransformerEmbedder(
        model_name=EMBEDDING_MODEL, device=device
    )
    load_t0 = time.perf_counter()
    dimension = embedder.get_dimension()  # forces model load
    load_s = time.perf_counter() - load_t0

    embed_t0 = time.perf_counter()
    vectors = embedder.embed_batch([c.text for c in chunks], batch_size=batch_size)
    embed_s = time.perf_counter() - embed_t0
    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    # --- store
    store = MedicalVectorStore(index_name=index_name, allow_default_write=True)
    if store.es is None:
        raise RuntimeError(
            "Elasticsearch is unavailable; cannot build the medical index."
        )

    store.create_index(dimension, recreate=recreate)
    indexed = store.stream_index(chunks, bulk_size=BULK_SIZE)
    store.es.indices.refresh(index=index_name)
    stats = store.get_stats()

    # --- separate, empty user dataset (never shares storage with default)
    user_store = MedicalVectorStore(index_name=user_index_name)
    user_store.create_index(dimension, recreate=False)
    user_stats = user_store.get_stats()

    cat_rows = cast(
        List[Dict[str, Any]], store.es.cat.indices(format="json", h="index")
    )
    existing_indices = sorted(str(row["index"]) for row in cat_rows)

    report: Dict[str, Any] = {
        "generated_at": _now_iso(),
        "index": index_name,
        "user_index": user_index_name,
        "existing_indices": existing_indices,
        "chunks_source": str(chunks_path),
        "chunks_in_source": len(chunks),
        "chunks_indexed": indexed,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_device": device,
        "dimension": dimension,
        "model_load_time_s": round(load_s, 3),
        "embed_time_s": round(embed_s, 3),
        "embed_throughput_chunks_s": (
            round(len(chunks) / embed_s, 2) if embed_s > 0 else None
        ),
        "recreated": recreate,
        "index_exists": stats.get("exists", False),
        "doc_count": stats.get("doc_count", 0),
        "store_bytes": stats.get("store_bytes", 0),
        "store_size_mb": round(stats.get("store_bytes", 0) / (1024 * 1024), 3),
        "user_index_doc_count": user_stats.get("doc_count", 0),
        "metadata_fields": list(MEDICAL_METADATA_FIELDS),
        "dataset_type": "default",
        "retrieval_untouched": [
            "potbot_documents",
            "RETRIEVAL_CANDIDATE_POOL",
            "VECTOR_RRF_WEIGHT / TEXT_RRF_WEIGHT / RRF_K",
            "reranker",
            "query rewriting",
        ],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", default=str(CHUNKS_PATH))
    parser.add_argument("--index", default=MEDICAL_DEFAULT_INDEX)
    parser.add_argument("--user-index", default=MEDICAL_USER_INDEX)
    parser.add_argument("--report", default=str(REPORT_PATH))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Do not delete an existing index first (additive indexing).",
    )
    args = parser.parse_args(argv)

    report = build_medical_index(
        chunks_path=Path(args.chunks),
        index_name=args.index,
        user_index_name=args.user_index,
        recreate=not args.no_recreate,
        device=args.device,
        report_path=Path(args.report),
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
