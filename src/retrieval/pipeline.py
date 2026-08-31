"""Lightweight retrieval pipeline: Hybrid only (no multi-query / rerank)."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.documents import Document

from src.retrieval.filters import dedupe_documents, filter_documents_by_filename
from src.retrieval.hybrid import hybrid_search


class RetrievalPipeline:
    def __init__(
        self,
        vectorstore: Any,
        bm25_index: Any,
        hybrid_fetch_k: int = 10,
        rrf_k: int = 60,
    ) -> None:
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.hybrid_fetch_k = hybrid_fetch_k
        self.rrf_k = rrf_k

    def refresh_bm25(self) -> None:
        if self.bm25_index is not None:
            self.bm25_index.rebuild_from_vectorstore(self.vectorstore)

    def retrieve(
        self,
        standalone_question: str,
        top_k: int = 3,
        filename_hint: Optional[str] = None,
    ) -> List[Document]:
        fused = hybrid_search(
            standalone_question,
            self.vectorstore,
            self.bm25_index,
            fetch_k=self.hybrid_fetch_k,
            rrf_k=self.rrf_k,
        )
        fused = dedupe_documents(fused)
        if filename_hint:
            filtered = filter_documents_by_filename(fused, filename_hint)
            if filtered:
                fused = filtered
        return fused[:top_k]
