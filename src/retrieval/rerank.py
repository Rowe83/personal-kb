"""Local Cross-Encoder reranking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder_cache: Dict[str, Any] = {}


def _get_encoder(model_name: str) -> Any:
    if model_name not in _encoder_cache:
        from sentence_transformers import CrossEncoder

        _encoder_cache[model_name] = CrossEncoder(model_name)
    return _encoder_cache[model_name]


def rerank(
    query: str,
    docs: List[Document],
    top_k: int,
    model_name: str = DEFAULT_RERANK_MODEL,
    cross_encoder: Optional[Any] = None,
) -> List[Document]:
    if not docs:
        return []
    k = max(0, top_k)
    if k == 0:
        return []
    try:
        model = cross_encoder if cross_encoder is not None else _get_encoder(model_name)
        pairs = [(query, d.page_content or "") for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(docs, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [doc for doc, _ in ranked[:k]]
    except Exception:
        return docs[:k]
