"""Retrieval filters: filename hints, dedupe, API line extraction."""

from __future__ import annotations

import re
from typing import Any, List, Optional

from langchain_core.documents import Document

from src.retrieval.bm25_index import content_hash

_API_LINE = re.compile(r"(?:^|\s)(\d{6,})\s*[-–—]\s*(.+?)\s*$", re.MULTILINE)
_QUERY_NOISE = re.compile(r"[有多少个接口文档？?的\s]+")
_QUERY_FILLER = {
    "简述",
    "介绍",
    "说明",
    "总结",
    "查询",
    "请问",
    "帮我",
    "帮忙",
    "一下",
    "什么",
    "怎么",
    "如何",
    "请",
    "讲讲",
    "说说",
    "看看",
    "描述",
}


def _token_set(text: str) -> set[str]:
    import jieba

    return {t.strip() for t in jieba.lcut(text or "") if len(t.strip()) >= 2}


def extract_filename_hint(question: str, known_filenames: List[str]) -> Optional[str]:
    q = (question or "").strip()
    if not q or not known_filenames:
        return None
    for fn in sorted(known_filenames, key=len, reverse=True):
        if fn in q:
            return fn
        stem = fn.rsplit(".", 1)[0]
        if len(stem) >= 4 and (stem in q or q in stem):
            return fn
        for frag in _QUERY_NOISE.split(q):
            frag = frag.strip()
            if len(frag) >= 3 and frag in stem:
                return fn
        for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", q):
            if len(token) >= 4 and token in stem:
                return fn

    # Fuzzy: jieba token overlap against filename stems (e.g. 股票标签 → 搜索股票标签逻辑.md)
    q_tokens = _token_set(q) - _QUERY_FILLER
    if len(q_tokens) < 2:
        return None
    best_fn: Optional[str] = None
    best_score = 0.0
    for fn in known_filenames:
        stem = fn.rsplit(".", 1)[0]
        stem_tokens = _token_set(stem)
        if len(stem_tokens) < 2:
            continue
        overlap = q_tokens & stem_tokens
        if len(overlap) < 2:
            continue
        score = len(overlap) / len(stem_tokens)
        if score > best_score or (
            score == best_score
            and best_fn is not None
            and len(stem) > len(best_fn.rsplit(".", 1)[0])
        ):
            best_score = score
            best_fn = fn
    if best_fn is not None and best_score >= 0.5:
        return best_fn
    return None


def extract_api_codes(text: str) -> List[str]:
    return re.findall(r"\d{6,}", text or "")


def find_filename_by_api_codes(
    codes: List[str], vectorstore: Any, known_filenames: List[str]
) -> Optional[str]:
    if not codes:
        return None
    try:
        raw = vectorstore.get(include=["documents", "metadatas"])
    except Exception:
        return None
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    for content, meta in zip(documents, metadatas):
        if not content or not meta:
            continue
        fname = meta.get("filename")
        if fname not in known_filenames:
            continue
        if any(code in content for code in codes):
            return fname
    return None


def find_filename_by_title_overlap(
    text: str, vectorstore: Any, known_filenames: List[str], min_len: int = 4
) -> Optional[str]:
    """Match distinctive title lines from the question against stored documents."""
    candidates = []
    for line in re.split(r"[\n\r]+", text or ""):
        line = line.strip()
        if len(line) < min_len:
            continue
        if re.match(r"^\d{6,}\s*[-–—]", line):
            continue
        candidates.append(line)
    if not candidates:
        return None

    try:
        raw = vectorstore.get(include=["documents", "metadatas"])
    except Exception:
        return None
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    for content, meta in zip(documents, metadatas):
        if not content or not meta:
            continue
        fname = meta.get("filename")
        if fname not in known_filenames:
            continue
        for phrase in candidates:
            if phrase in content:
                return fname
    return None


def resolve_target_filename(
    question: str,
    standalone_question: str,
    known_filenames: List[str],
    retrieved_docs: List[Document],
    vectorstore: Any = None,
) -> Optional[str]:
    combined = f"{question}\n{standalone_question}"
    for text in (question, standalone_question, combined):
        hint = extract_filename_hint(text, known_filenames)
        if hint:
            return hint

    codes = extract_api_codes(combined)
    if codes:
        if vectorstore is not None:
            by_code = find_filename_by_api_codes(codes, vectorstore, known_filenames)
            if by_code:
                return by_code
        for doc in retrieved_docs:
            if any(code in doc.page_content for code in codes):
                return doc.metadata.get("filename")

    if vectorstore is not None:
        by_title = find_filename_by_title_overlap(combined, vectorstore, known_filenames)
        if by_title:
            return by_title

    if len(known_filenames) == 1:
        return known_filenames[0]

    return None


def dedupe_documents(docs: List[Document]) -> List[Document]:
    seen: set[str] = set()
    out: List[Document] = []
    for doc in docs:
        h = content_hash(doc.page_content or "")
        if h in seen:
            continue
        seen.add(h)
        out.append(doc)
    return out


def filter_documents_by_filename(
    docs: List[Document], filename: str
) -> List[Document]:
    if not filename:
        return docs
    return [d for d in docs if d.metadata.get("filename") == filename]


def extract_api_lines(text: str) -> List[str]:
    """Return normalized API lines like '2076134-产品协议业务协议查询'."""
    lines: List[str] = []
    for match in _API_LINE.finditer(text or ""):
        code, name = match.group(1), match.group(2).strip()
        lines.append(f"{code}-{name}")
    return lines


def extract_api_lines_from_docs(docs: List[Document]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for doc in docs:
        for line in extract_api_lines(doc.page_content):
            if line not in seen:
                seen.add(line)
                ordered.append(line)
    return ordered


def filenames_with_api_lines(vectorstore: Any, known_filenames: List[str]) -> List[str]:
    try:
        raw = vectorstore.get(include=["documents", "metadatas"])
    except Exception:
        return []
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    found: set[str] = set()
    for content, meta in zip(documents, metadatas):
        if not content or not meta:
            continue
        fname = meta.get("filename")
        if fname not in known_filenames:
            continue
        if extract_api_lines(content):
            found.add(fname)
    return sorted(found)
