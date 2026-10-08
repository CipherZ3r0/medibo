"""
Process medical raw text into indexed_text for chunking/indexing (Phase 3).
Preserves raw files. Creates data/medical/processed/*.json for each logical
document with indexed_text, citations, metadata. Preserves [page N] boundaries.
"""
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from domain import medical as med

RAW_DIR = med.RAW_DIR
PROCESSED_DIR = Path(med.PROJECT_ROOT) / "data" / "medical" / "processed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _clean_whitespace(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def clean_pdf_text(text: str) -> str:
    if not text:
        return text
    lines = text.splitlines()
    out: List[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            out.append("")
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        out.append(line)
    cleaned = "\n".join(out)
    cleaned = _clean_whitespace(cleaned)
    return cleaned


def clean_plain_text(text: str) -> str:
    return _clean_whitespace(text)


def find_page_boundaries(text: str) -> List[Tuple[int, str]]:
    pages = []
    for m in re.finditer(r"\[page\s+(\d+)\]", text, flags=re.IGNORECASE):
        pages.append((int(m.group(1)), m.group(0)))
    return pages


def split_by_page_markers(text: str) -> List[Tuple[int, str]]:
    if not text:
        return []
    chunks = re.split(r"(\[page\s+\d+\])", text, flags=re.IGNORECASE)
    segments: List[Tuple[int, str]] = []
    k = 0
    while k < len(chunks):
        c = chunks[k]
        if re.match(r"\[page\s+\d+\]", c, flags=re.IGNORECASE):
            m = re.search(r"(\d+)", c)
            pnum = int(m.group(1)) if m else 1
            seg = c
            if k + 1 < len(chunks):
                seg += chunks[k + 1]
                k += 2
            else:
                k += 1
            segments.append((pnum, seg))
            continue
        k += 1
    if not segments:
        return [(1, text)]
    return segments


def extract_sections(text: str) -> List[Dict[str, Any]]:
    sections = []
    lines = text.splitlines()
    current = {"heading": None, "start_page": 1, "end_page": 1, "text": []}
    current_page = 1
    for line in lines:
        m = re.search(r"\[page\s+(\d+)\]", line, flags=re.IGNORECASE)
        if m:
            current_page = int(m.group(1))
        s = line.strip()
        if s and len(s) < 120 and (s.isupper() or s.endswith(":") or re.match(r"^[A-Z][A-Za-z\s]+$", s)):
            if current["text"] or current["heading"] is not None:
                current["end_page"] = current_page
                sections.append(current)
                current = {"heading": s, "start_page": current_page, "end_page": current_page, "text": []}
                continue
        current["text"].append(line)
        current["end_page"] = current_page
    if current["text"] or current["heading"] is not None:
        sections.append(current)
    out = []
    for sec in sections:
        out.append({
            "heading": sec["heading"],
            "start_page": sec["start_page"],
            "end_page": sec["end_page"],
            "text": "\n".join(sec["text"]).strip(),
        })
    return out


def build_indexed_text(text: str) -> str:
    cleaned = clean_pdf_text(text) if re.search(r"\[page\s+\d+\]", text, flags=re.IGNORECASE) else clean_plain_text(text)
    return cleaned

@dataclass
class ProcessedDoc:
    doc_id: str
    title: str
    indexed_text: str
    source: Dict[str, Any]
    manifest_document: Dict[str, Any]
    pages: List[int]
    sections: List[Dict[str, Any]]
    raw_text_path: str
    original_path: Optional[str] = None
    bytes: Optional[int] = None
    sha256: Optional[str] = None
    processed_at: str = ""
    problems: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["problems"] is None:
            d["problems"] = []
        return d


def _get_manifest_doc(manifest: Dict[str, Any], source_id: str, doc_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    for src, doc in med.iter_manifest_documents(manifest):
        if src["source"] == source_id and doc["doc_id"] == doc_id:
            return src, doc
    raise ValueError(f"manifest entry not found: {source_id}/{doc_id}")


def process_pdf_document(source_id: str, doc_id: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    raw_dir = RAW_DIR / source_id / doc_id
    raw_text_path = raw_dir / "raw_text.txt"
    if not raw_text_path.exists():
        raise FileNotFoundError(f"missing raw_text: {raw_text_path}")
    raw_text = raw_text_path.read_text(encoding="utf-8", errors="replace")
    indexed_text = build_indexed_text(raw_text)
    pages = [p for p, _ in find_page_boundaries(indexed_text)] or [1]
    sections = extract_sections(indexed_text)
    src, mdoc = _get_manifest_doc(manifest, source_id, doc_id)
    orig = None
    for ext in (".pdf", ".xml"):
        cand = raw_dir / f"original{ext}"
        if cand.exists():
            try:
                orig = str(cand.relative_to(med.PROJECT_ROOT))
            except Exception:
                orig = str(cand)
            break
    meta_path = raw_dir / "metadata.json"
    sha256 = None
    bytes_cnt = None
    if meta_path.exists():
        try:
            mm = json.loads(meta_path.read_text(encoding="utf-8"))
            sha256 = mm.get("sha256")
            bytes_cnt = mm.get("bytes")
        except Exception:
            pass
    problems: List[str] = []
    if len(indexed_text) < 200:
        problems.append("indexed_text_too_short")
    try:
        raw_text_rel = str(raw_text_path.relative_to(med.PROJECT_ROOT))
    except Exception:
        raw_text_rel = str(raw_text_path)
    pd = ProcessedDoc(
        doc_id=doc_id,
        title=mdoc.get("title", doc_id),
        indexed_text=indexed_text,
        source=src,
        manifest_document=mdoc,
        pages=pages,
        sections=sections,
        raw_text_path=raw_text_rel,
        original_path=orig,
        bytes=bytes_cnt,
        sha256=sha256,
        processed_at=_now_iso(),
        problems=problems or None,
    )
    out_path = PROCESSED_DIR / f"{doc_id}.json"
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(pd.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return pd.to_dict()


def process_medlineplus(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = RAW_DIR / "medlineplus" / "medlineplus-health-topics-xml"
    topics_dir = raw / "topics"
    results = []
    src, mdoc_parent = _get_manifest_doc(manifest, "medlineplus", "medlineplus-health-topics-xml")
    for topic_dir in sorted(topics_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        raw_t = topic_dir / "raw_text.txt"
        if not raw_t.exists():
            continue
        meta_t = topic_dir / "metadata.json"
        title = topic_dir.name.replace("-", " ").title()
        mdoc_child = dict(mdoc_parent)
        if meta_t.exists():
            try:
                mt = json.loads(meta_t.read_text(encoding="utf-8"))
                title = mt["metadata"].get("disease_topic") or title
                mdoc_child = mt.get("manifest_document", mdoc_child)
            except Exception:
                pass
        text = raw_t.read_text(encoding="utf-8", errors="replace")
        indexed_text = clean_plain_text(text)
        pages = [1]
        sections = [{"heading": title, "start_page": 1, "end_page": 1, "text": indexed_text}]
        problems = []
        if len(indexed_text) < 50:
            problems.append("indexed_text_too_short")
        doc_id = f"medlineplus-health-topics-xml--{topic_dir.name}"
        pd = ProcessedDoc(
            doc_id=doc_id,
            title=title,
            indexed_text=indexed_text,
            source=dict(src),
            manifest_document=dict(mdoc_child),
            pages=pages,
            sections=sections,
            raw_text_path=str(raw_t.relative_to(med.PROJECT_ROOT)),
            processed_at=_now_iso(),
            problems=problems or None,
        )
        out = PROCESSED_DIR / f"{doc_id}.json"
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pd.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(pd.to_dict())
    return results


def process_corpus(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if manifest is None:
        manifest = med.load_manifest()
    results: List[Dict[str, Any]] = []
    for src, mdoc in med.iter_manifest_documents(manifest):
        sid = src["source"]
        did = mdoc["doc_id"]
        if sid == "medlineplus":
            continue
        try:
            res = process_pdf_document(sid, did, manifest)
            results.append(res)
        except Exception as e:
            results.append({"doc_id": did, "source": sid, "error": str(e), "status": "failed"})
    try:
        results.extend(process_medlineplus(manifest))
    except Exception as e:
        results.append({"doc_id": "medlineplus-health-topics-xml", "error": str(e), "status": "failed"})
    return results


def run_process() -> Dict[str, Any]:
    manifest = med.load_manifest()
    processed = process_corpus(manifest)
    ok = [p for p in processed if p.get("status") != "failed"]
    failed = [p for p in processed if p.get("status") == "failed"]
    total_chars = sum(len(p.get("indexed_text", "")) for p in ok)
    report = {
        "processed_at": _now_iso(),
        "manifest": "data/medical/source_manifest.json",
        "count": len(ok),
        "failed": len(failed),
        "total_indexed_chars": total_chars,
        "processed": ok,
        "failures": failed,
    }
    out = Path(med.PROJECT_ROOT) / "data" / "medical" / "processing_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = run_process()
    print(json.dumps({"count": r["count"], "failed": r["failed"], "total_indexed_chars": r["total_indexed_chars"]}, indent=2))
