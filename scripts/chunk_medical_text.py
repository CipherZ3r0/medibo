"""
Medical chunking utilities (Phase 4). 
Creates deterministic chunk output for processed medical documents.
Preserves metadata and page/section info.
"""
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestion.chunkers import RecursiveCharacterChunker
from domain.models import Document

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "medical" / "processed"
CHUNKS_DIR = Path(__file__).resolve().parents[1] / "data" / "medical" / "chunks"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class MedicalChunk:
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    source_file: str
    file_name: str
    file_type: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    dataset_type: str = "default"
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["metadata"] is None:
            d["metadata"] = {}
        # flatten key metadata
        return {
            "chunk_id": d["chunk_id"],
            "doc_id": d["doc_id"],
            "text": d["text"],
            "chunk_index": d["chunk_index"],
            "source_file": d["source_file"],
            "file_name": d["file_name"],
            "file_type": d["file_type"],
            "page_number": d["page_number"],
            "section": d["section"],
            "dataset_type": "default",
            **self.flatten_meta(d["metadata"], include_medical=True),
        }

    @staticmethod
    def flatten_meta(meta: Dict[str, Any], include_medical: bool = True) -> Dict[str, Any]:
        out = dict(meta) if meta else {}
        # also pull from parent context passed in metadata? handle below
        return out


def generate_chunk_id_unique(doc_id: str, page: Optional[int], chunk_index: int, salt: str = "") -> str:
    raw = f"med::{doc_id}::p{page or 0}::c{chunk_index}::{salt}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def chunk_medical_document(proc_path: Path, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    with open(proc_path, 'r', encoding='utf-8') as f:
        pd = json.load(f)
    doc_id = pd['doc_id']
    title = pd['title']
    indexed_text = pd['indexed_text']
    source = pd['source']
    manifest = pd['manifest_document']
    pages = pd.get('pages', [])
    sections = pd.get('sections', [])
    # build section map by page range
    # create synthetic doc for chunker? but chunker uses Document; easier to chunk by page segments to preserve page
    chunks_out = []
    # split by page markers
    from scripts import process_medical_text as pmt
    segs = pmt.split_by_page_markers(indexed_text) if re.search(r'\[page\s+\d+\]', indexed_text, flags=re.IGNORECASE) else [(pages[0] if pages else 1, indexed_text)]
    global_idx = 0
    for pnum, segtext in segs:
        # find best section heading for this page
        sec_heading = None
        for s in sections:
            sp = s.get('start_page')
            ep = s.get('end_page')
            if sp is not None and ep is not None:
                if sp <= pnum <= ep:
                    sec_heading = s.get('heading')
                    break
        # chunk this segment
        cc = RecursiveCharacterChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        d = Document(
            text=segtext,
            source_file=str(proc_path),
            file_name=proc_path.name,
            file_type='.json',
            page_number=pnum,
        )
        chs = cc.chunk_document(d)
        for c in chs:
            chunk_id = generate_chunk_id_unique(doc_id, pnum, c.chunk_index, salt=f"{global_idx}")
            global_idx += 1
            md = {
                'title': title,
                'source': source['source'],
                'source_url': manifest.get('url'),
                'region': source.get('region'),
                'authority': source.get('authority'),
                'document_type': manifest.get('document_type'),
                'disease_topic': manifest.get('topic') or source.get('disease_topic'),
                'published_at': manifest.get('published_at'),
                'updated_at': manifest.get('updated_at'),
                'license': source.get('license'),
            }
            chunks_out.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'text': c.text,
                'chunk_index': c.chunk_index,
                'source_file': str(proc_path),
                'file_name': proc_path.name,
                'file_type': '.json',
                'page_number': pnum,
                'section': sec_heading,
                'dataset_type': 'default',
                **md
            })
    return chunks_out


def chunk_all(chunk_size: int = 800, chunk_overlap: int = 150) -> Dict[str, Any]:
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    processed_files = sorted(PROCESSED_DIR.glob('*.json'))
    for pf in processed_files:
        try:
            chs = chunk_medical_document(pf, chunk_size, chunk_overlap)
            results.extend(chs)
        except Exception as e:
            results.append({'error': str(e), 'file': str(pf)})
    # write per-doc chunk files? or one combined? req says "deterministic processed-chunk output, e.g. data/medical/chunks/"
    # write combined by default
    out = CHUNKS_DIR / 'medical_chunks.json'
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    # also write summary
    summary = {
        'generated_at': _now_iso(),
        'total_chunks': len(results),
        'files': len(processed_files),
        'output': str(out),
    }
    (CHUNKS_DIR / 'chunk_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return {'results': results, 'summary': summary}


def run() -> Dict[str, Any]:
    return chunk_all()


if __name__ == '__main__':
    r = run()
    print(json.dumps(r['summary'], indent=2))
