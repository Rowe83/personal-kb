"""Retrieval pipeline: optional Multi-Query + Hybrid + optional Cross-Encoder."""

from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.documents import Document

from src.retrieval.filters import dedupe_documents, filter_documents_by_filename
from src.retrieval.hybrid import hybrid_search
from src.retrieval.multi_query import generate_multi_queries
from src.retrieval.rerank import DEFAULT_RERANK_MODEL, rerank


class RetrievalPipeline:
    def __init__(
        self,
        vectorstore: Any,
        bm25_index: Any,
        hybrid_fetch_k: int = 10,
        rrf_k: int = 60,
        llm: Any = None,
        multi_query_n: int = 3,
        enable_multi_query: bool = True,
        enable_rerank: bool = True,
        rerank_model: str = DEFAULT_RERANK_MODEL,
    ) -> None:
        self.vectorstore = vectorstore
        self.bm25_index = bm25_index
        self.hybrid_fetch_k = hybrid_fetch_k
        self.rrf_k = rrf_k
        self.llm = llm
        self.multi_query_n = multi_query_n
        self.enable_multi_query = enable_multi_query
        self.enable_rerank = enable_rerank
        self.rerank_model = rerank_model

    def refresh_bm25(self) -> None:
        if self.bm25_index is not None:
            self.bm25_index.rebuild_from_vectorstore(self.vectorstore)

    def retrieve(
        self,
        standalone_question: str,
        top_k: int = 3,
        filename_hint: Optional[str] = None,
    ) -> List[Document]:
        if self.enable_multi_query:
            queries = generate_multi_queries(
                standalone_question, self.llm, n=self.multi_query_n
            )
        else:
            q = (standalone_question or "").strip()
            queries = [q] if q else []

        merged: List[Document] = []
        for query in queries:
            fused = hybrid_search(
                query,
                self.vectorstore,
                self.bm25_index,
                fetch_k=self.hybrid_fetch_k,
                rrf_k=self.rrf_k,
            )
            merged.extend(fused)

        docs = dedupe_documents(merged)
        if filename_hint:
            filtered = filter_documents_by_filename(docs, filename_hint)
            if filtered:
                docs = filtered

        if self.enable_rerank:
            return rerank(
                standalone_question,
                docs,
                top_k,
                model_name=self.rerank_model,
            )
        return docs[:top_k]
