from langchain_core.documents import Document
from src.retrieval.bm25_index import BM25Index


class FakeVS:
    def get(self, include=None):
        return {
            "ids": ["a", "b"],
            "documents": [
                "展示产品协议业务协议 2076134-产品协议业务协议查询",
                "理财首页宽屏 rem 适配方案",
            ],
            "metadatas": [
                {"filename": "原生场外开放式基金接口整理.txt", "page": 1},
                {"filename": "技术方案.pdf", "page": 1},
            ],
        }


def test_bm25_hits_api_code():
    index = BM25Index.from_vectorstore(FakeVS())
    hits = index.search("2076134", k=1)
    assert len(hits) == 1
    assert hits[0].metadata["filename"] == "原生场外开放式基金接口整理.txt"


def test_bm25_empty_corpus_returns_empty():
    class EmptyVS:
        def get(self, include=None):
            return {"ids": [], "documents": [], "metadatas": []}

    index = BM25Index.from_vectorstore(EmptyVS())
    assert index.search("anything", k=3) == []
