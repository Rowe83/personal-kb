from __future__ import annotations

import hashlib
from typing import Any, List, Optional

import jieba
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


def chunk_key(
    doc_id: Optional[str], metadata: Optional[dict], content: str
) -> str:
    if doc_id:
        return str(doc_id)
    meta = metadata or {}
    digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:16]
    return f"{meta.get('filename','')}|{meta.get('page','')}|{digest}"


def content_hash(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


class BM25Index:
    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._docs: List[Document] = []

    @classmethod
    def from_vectorstore(cls, vectorstore: Any) -> "BM25Index":
        index = cls()
        index.rebuild_from_vectorstore(vectorstore)
        return index

    def rebuild_from_vectorstore(self, vectorstore: Any) -> None:
        raw = vectorstore.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []

        self._docs = []
        tokenized = []
        for i, content in enumerate(documents):
            if content is None:
                continue
            meta = dict(metadatas[i] or {})
            doc_id = ids[i] if i < len(ids) else None
            meta["_id"] = chunk_key(doc_id, meta, content)
            meta["_content_hash"] = content_hash(content)
            self._docs.append(Document(page_content=content, metadata=meta))
            tokenized.append(list(jieba.lcut(content)))

        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, k: int) -> List[Document]:
        if not self._bm25 or not self._docs or k <= 0:
            return []
        tokens = list(jieba.lcut(query))
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: List[Document] = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                doc_freq = self._bm25.doc_freqs[i]
                if not any(doc_freq.get(t, 0) > 0 for t in tokens):
                    continue
            out.append(self._docs[i])
        return out
