"""Retrieval package — import submodules directly to avoid heavy eager loads."""

__all__ = [
    "RetrievalPipeline",
    "is_catalog_question",
    "format_catalog_answer",
    "BM25Index",
]


def __getattr__(name: str):
    if name == "RetrievalPipeline":
        from src.retrieval.pipeline import RetrievalPipeline

        return RetrievalPipeline
    if name == "is_catalog_question":
        from src.retrieval.catalog import is_catalog_question

        return is_catalog_question
    if name == "format_catalog_answer":
        from src.retrieval.catalog import format_catalog_answer

        return format_catalog_answer
    if name == "BM25Index":
        from src.retrieval.bm25_index import BM25Index

        return BM25Index
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
