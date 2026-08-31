"""Hybrid retrieval: dense vector + BM25 fused with RRF."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document

from src.retrieval.bm25_index import chunk_key


def _ensure_id(doc: Document) -> str:
    meta = doc.metadata or {}
    if meta.get("_id"):
        return str(meta["_id"])
    key = chunk_key(None, meta, doc.page_content)
    doc.metadata = dict(meta)
    doc.metadata["_id"] = key
    return key


def rrf_fuse(rank_lists: Iterable[List[Document]], rrf_k: int = 60) -> List[Document]:
    scores: Dict[str, float] = {}
    keep: Dict[str, Document] = {}
    for docs in rank_lists:
        for rank, doc in enumerate(docs):
            doc_id = _ensure_id(doc)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            keep[doc_id] = doc
    ordered = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return [keep[i] for i in ordered]


def hybrid_search(
    query: str,
    vectorstore: Any,
    bm25_index: Any,
    fetch_k: int = 10,
    rrf_k: int = 60,
) -> List[Document]:
    vector_docs = vectorstore.similarity_search(query, k=fetch_k)
    for d in vector_docs:
        _ensure_id(d)
    bm25_docs = bm25_index.search(query, k=fetch_k) if bm25_index is not None else []
    if not bm25_docs:
        return vector_docs
    if not vector_docs:
        return bm25_docs
    return rrf_fuse([vector_docs, bm25_docs], rrf_k=rrf_k)
