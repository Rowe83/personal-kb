# RAG Retrieval Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为个人知识库接入 Hybrid（向量+BM25）+ Multi-Query + 极简 Cross-Encoder 重排，并修复同名重复入库与目录类问题答错。

**Architecture:** 新增 `src/retrieval/` 管线；`KnowledgeBaseService` 改为调用 `RetrievalPipeline.retrieve`；上传按 filename 先删后写；目录意图短路到 `list_documents()`。

**Tech Stack:** Chroma、rank-bm25、jieba、sentence-transformers CrossEncoder、现有 DeepSeek LLM

**Spec:** `docs/superpowers/specs/2026-08-26-rag-retrieval-optimization-design.md`

## Global Constraints

- 不上 HyDE；不用 Cohere / 云端 Rerank
- 不改 FastAPI / Streamlit 对外 API 契约
- `query_multi_turn` / `query_multi_turn_stream` 返回字段不变
- 默认：`MULTI_QUERY_N=3`，`HYBRID_FETCH_K=10`，`RRF_K=60`，`RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- 依赖钉版本：`rank-bm25==0.2.2`，`jieba==0.42.1`，`sentence-transformers==5.1.2`（避免 6.x 过大变动；若安装冲突可升到环境已解析的兼容版，但须写入 requirements）
- Git commit 若遇 `unknown option trailer`，使用 `/usr/local/bin/git commit`

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/config.py` | 检索相关配置 |
| `requirements.txt` | 新增三依赖 |
| `src/retrieval/__init__.py` | 导出 `RetrievalPipeline` |
| `src/retrieval/bm25_index.py` | BM25 索引构建/刷新/检索 |
| `src/retrieval/hybrid.py` | 向量+BM25+RRF |
| `src/retrieval/multi_query.py` | Multi-Query 生成 |
| `src/retrieval/rerank.py` | Cross-Encoder 重排 |
| `src/retrieval/pipeline.py` | 编排 `retrieve` |
| `src/services.py` | 接线、去重上传、目录短路 |
| `tests/test_bm25_index.py` 等 | 各模块单测 |
| `README.md` | 文档更新 |

---

### Task 1: Config + dependencies

**Files:**
- Modify: `src/config.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `settings.MULTI_QUERY_N`, `HYBRID_FETCH_K`, `RRF_K`, `RERANK_MODEL`, `ENABLE_MULTI_QUERY`, `ENABLE_RERANK`（均可被 env 覆盖）

- [ ] **Step 1: 更新 `requirements.txt`**

在文件末尾追加：

```
rank-bm25==0.2.2
jieba==0.42.1
sentence-transformers==5.1.2
```

- [ ] **Step 2: 扩展 `src/config.py`**

在 `Settings` 类中增加（保持现有字段不变）：

```python
    # RAG retrieval
    MULTI_QUERY_N: int = int(os.getenv("MULTI_QUERY_N", "3"))
    HYBRID_FETCH_K: int = int(os.getenv("HYBRID_FETCH_K", "10"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    RERANK_MODEL: str = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    ENABLE_MULTI_QUERY: bool = os.getenv("ENABLE_MULTI_QUERY", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "true").lower() in (
        "1",
        "true",
        "yes",
    )
```

- [ ] **Step 3: 安装依赖**

```bash
python -m pip install rank-bm25==0.2.2 jieba==0.42.1 sentence-transformers==5.1.2
python -c "import rank_bm25, jieba, sentence_transformers; print('ok')"
```

Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add src/config.py requirements.txt
/usr/local/bin/git commit -m "chore: add retrieval config and hybrid/rerank deps"
```

---

### Task 2: BM25 index

**Files:**
- Create: `src/retrieval/__init__.py`
- Create: `src/retrieval/bm25_index.py`
- Create: `tests/test_bm25_index.py`

**Interfaces:**
- Consumes: Chroma-like object with `.get(include=...)`
- Produces:
  - `class BM25Index`
  - `BM25Index.from_vectorstore(vectorstore) -> BM25Index`
  - `BM25Index.rebuild_from_vectorstore(vectorstore) -> None`
  - `BM25Index.search(query: str, k: int) -> List[Document]`（Document 的 metadata 含 `filename`/`page`，并写入 `_id`）
  - `chunk_key(doc_id, metadata, content) -> str`（导出供 hybrid 使用）

- [ ] **Step 1: 写失败测试 `tests/test_bm25_index.py`**

```python
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
```

- [ ] **Step 2: 跑测确认失败**

```bash
python -m pytest tests/test_bm25_index.py -v
```

Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 `src/retrieval/bm25_index.py` 与 `__init__.py`**

`src/retrieval/__init__.py`：

```python
"""Retrieval pipeline package."""
```

`src/retrieval/bm25_index.py`：

```python
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
                continue
            out.append(self._docs[i])
        return out
```

- [ ] **Step 4: 跑测通过**

```bash
python -m pytest tests/test_bm25_index.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retrieval tests/test_bm25_index.py
/usr/local/bin/git commit -m "feat(retrieval): add BM25 index with jieba tokenization"
```

---

### Task 3: Hybrid RRF

**Files:**
- Create: `src/retrieval/hybrid.py`
- Create: `tests/test_hybrid.py`

**Interfaces:**
- Consumes: `BM25Index.search`；vectorstore `.similarity_search(query, k)`
- Produces: `hybrid_search(query, vectorstore, bm25_index, fetch_k, rrf_k) -> List[Document]`

- [ ] **Step 1: 写失败测试**

```python
from langchain_core.documents import Document
from src.retrieval.hybrid import hybrid_search, rrf_fuse


def test_rrf_fuse_prefers_shared_high_ranks():
    d1 = Document(page_content="a", metadata={"_id": "1"})
    d2 = Document(page_content="b", metadata={"_id": "2"})
    d3 = Document(page_content="c", metadata={"_id": "3"})
    fused = rrf_fuse([[d1, d2], [d2, d3]], rrf_k=60)
    assert fused[0].metadata["_id"] == "2"


class FakeVS:
    def similarity_search(self, query, k=10):
        return [
            Document(page_content="pdf rem", metadata={"filename": "p.pdf", "page": 1, "_id": "pdf"}),
            Document(page_content="other", metadata={"filename": "p.pdf", "page": 2, "_id": "pdf2"}),
        ]


class FakeBM25:
    def search(self, query, k):
        return [
            Document(
                page_content="2076134 协议",
                metadata={"filename": "fund.txt", "page": 1, "_id": "txt"},
            )
        ]


def test_hybrid_includes_bm25_hit():
    docs = hybrid_search("2076134", FakeVS(), FakeBM25(), fetch_k=2, rrf_k=60)
    ids = [d.metadata["_id"] for d in docs]
    assert "txt" in ids
```

- [ ] **Step 2: 跑测确认失败**

```bash
python -m pytest tests/test_hybrid.py -v
```

- [ ] **Step 3: 实现 `src/retrieval/hybrid.py`**

```python
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
```

- [ ] **Step 4: 跑测通过并 commit**

```bash
python -m pytest tests/test_hybrid.py tests/test_bm25_index.py -v
git add src/retrieval/hybrid.py tests/test_hybrid.py
/usr/local/bin/git commit -m "feat(retrieval): add hybrid vector+BM25 RRF fusion"
```

---

### Task 4: Multi-Query

**Files:**
- Create: `src/retrieval/multi_query.py`
- Create: `tests/test_multi_query.py`

**Interfaces:**
- Produces: `generate_multi_queries(llm, question: str, n: int) -> List[str]`  
  - 成功：含原问 + 最多 n 条改写，去重保序  
  - LLM 抛错：仅返回 `[question]`

- [ ] **Step 1: 写失败测试**

```python
from unittest.mock import MagicMock
from src.retrieval.multi_query import generate_multi_queries


def test_multi_query_merges_and_dedupes():
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(
        content="产品协议查询\n2076134 接口\n产品协议查询"
    )
    out = generate_multi_queries(llm, "协议签署有哪些接口", n=3)
    assert out[0] == "协议签署有哪些接口"
    assert "产品协议查询" in out
    assert "2076134 接口" in out
    assert len(out) == len(set(out))


def test_multi_query_falls_back_on_error():
    llm = MagicMock()
    llm.invoke.side_effect = RuntimeError("boom")
    assert generate_multi_queries(llm, "原问题", n=3) == ["原问题"]
```

- [ ] **Step 2: 跑测确认失败 → 实现 → 通过**

`src/retrieval/multi_query.py`：

```python
from __future__ import annotations

from typing import Any, List


_PROMPT = (
    "你是检索查询改写助手。请针对用户问题生成 {n} 条不同的检索查询，"
    "分别侧重：同义改写、关键词/实体、接口号或文件名。"
    "每行一条，不要编号，不要解释。\n\n用户问题：{question}"
)


def generate_multi_queries(llm: Any, question: str, n: int = 3) -> List[str]:
    queries = [question.strip()]
    if n <= 0:
        return queries
    try:
        resp = llm.invoke(_PROMPT.format(n=n, question=question))
        text = resp.content if hasattr(resp, "content") else str(resp)
        for line in text.splitlines():
            q = line.strip().lstrip("0123456789.-、)） ").strip()
            if q and q not in queries:
                queries.append(q)
            if len(queries) >= n + 1:
                break
    except Exception:
        return [question.strip()]
    return queries
```

- [ ] **Step 3: Commit**

```bash
git add src/retrieval/multi_query.py tests/test_multi_query.py
/usr/local/bin/git commit -m "feat(retrieval): add multi-query rewrite with fallback"
```

---

### Task 5: Cross-Encoder rerank

**Files:**
- Create: `src/retrieval/rerank.py`
- Create: `tests/test_rerank.py`

**Interfaces:**
- Produces: `rerank_documents(query, docs, top_k, model_name, cross_encoder=None) -> List[Document]`
  - 可注入 fake `cross_encoder`（有 `.predict(pairs)`）便于单测
  - predict 失败：返回 `docs[:top_k]`

- [ ] **Step 1: 写失败测试**

```python
from langchain_core.documents import Document
from src.retrieval.rerank import rerank_documents


class FakeCE:
    def predict(self, pairs):
        # higher score for doc containing 2076134
        scores = []
        for _, text in pairs:
            scores.append(1.0 if "2076134" in text else 0.1)
        return scores


def test_rerank_orders_by_score():
    docs = [
        Document(page_content="宽屏 rem", metadata={"_id": "pdf"}),
        Document(page_content="2076134 协议查询", metadata={"_id": "txt"}),
    ]
    out = rerank_documents("2076134", docs, top_k=1, cross_encoder=FakeCE())
    assert out[0].metadata["_id"] == "txt"


def test_rerank_falls_back_on_error():
    class Boom:
        def predict(self, pairs):
            raise RuntimeError("no model")

    docs = [Document(page_content="a", metadata={"_id": "1"})]
    out = rerank_documents("q", docs, top_k=1, cross_encoder=Boom())
    assert out[0].metadata["_id"] == "1"
```

- [ ] **Step 2: 实现 `src/retrieval/rerank.py`**

```python
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.documents import Document

_cached_ce = None


def _load_cross_encoder(model_name: str) -> Any:
    global _cached_ce
    if _cached_ce is None or getattr(_cached_ce, "model_name", None) != model_name:
        from sentence_transformers import CrossEncoder

        _cached_ce = CrossEncoder(model_name)
        _cached_ce.model_name = model_name
    return _cached_ce


def rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    cross_encoder: Optional[Any] = None,
) -> List[Document]:
    if not docs or top_k <= 0:
        return []
    try:
        ce = cross_encoder or _load_cross_encoder(model_name)
        pairs = [(query, d.page_content) for d in docs]
        scores = ce.predict(pairs)
        ranked = sorted(
            zip(docs, scores), key=lambda x: float(x[1]), reverse=True
        )
        return [d for d, _ in ranked[:top_k]]
    except Exception:
        return docs[:top_k]
```

- [ ] **Step 3: 跑测 + commit**

```bash
python -m pytest tests/test_rerank.py -v
git add src/retrieval/rerank.py tests/test_rerank.py
/usr/local/bin/git commit -m "feat(retrieval): add minimal cross-encoder reranker"
```

---

### Task 6: Retrieval pipeline

**Files:**
- Create: `src/retrieval/pipeline.py`
- Modify: `src/retrieval/__init__.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `class RetrievalPipeline`
  - `RetrievalPipeline.retrieve(standalone_question: str, top_k: int) -> List[Document]`
  - `RetrievalPipeline.refresh_bm25() -> None`
  - 构造注入：`vectorstore`, `llm`, `bm25_index`, 以及 settings 字段；可选 `cross_encoder` 供测试

- [ ] **Step 1: 写失败测试（全 mock，不下载模型）**

```python
from langchain_core.documents import Document
from src.retrieval.pipeline import RetrievalPipeline


class FakeVS:
    def similarity_search(self, query, k=10):
        return [
            Document(page_content="pdf", metadata={"filename": "a.pdf", "page": 1, "_id": "pdf"}),
            Document(page_content="2076134", metadata={"filename": "f.txt", "page": 1, "_id": "txt"}),
        ]


class FakeBM25:
    def search(self, query, k):
        return [
            Document(page_content="2076134", metadata={"filename": "f.txt", "page": 1, "_id": "txt"})
        ]

    def rebuild_from_vectorstore(self, vs):
        pass


class FakeLLM:
    def invoke(self, prompt):
        return type("R", (), {"content": "基金协议接口\n2076134"})()


class FakeCE:
    def predict(self, pairs):
        return [0.2 if "pdf" in t else 0.9 for _, t in pairs]


def test_pipeline_returns_top_k_preferring_txt():
    pipe = RetrievalPipeline(
        vectorstore=FakeVS(),
        llm=FakeLLM(),
        bm25_index=FakeBM25(),
        multi_query_n=2,
        hybrid_fetch_k=2,
        rrf_k=60,
        rerank_model="dummy",
        enable_multi_query=True,
        enable_rerank=True,
        cross_encoder=FakeCE(),
    )
    docs = pipe.retrieve("原生场外接口", top_k=1)
    assert len(docs) == 1
    assert docs[0].metadata["_id"] == "txt"
```

- [ ] **Step 2: 实现 `pipeline.py`**

```python
from __future__ import annotations

from typing import Any, List, Optional

from langchain_core.documents import Document

from src.retrieval.hybrid import hybrid_search, rrf_fuse
from src.retrieval.multi_query import generate_multi_queries
from src.retrieval.rerank import rerank_documents


class RetrievalPipeline:
    def __init__(
        self,
        vectorstore: Any,
        llm: Any,
        bm25_index: Any,
        multi_query_n: int = 3,
        hybrid_fetch_k: int = 10,
        rrf_k: int = 60,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_multi_query: bool = True,
        enable_rerank: bool = True,
        cross_encoder: Optional[Any] = None,
    ) -> None:
        self.vectorstore = vectorstore
        self.llm = llm
        self.bm25_index = bm25_index
        self.multi_query_n = multi_query_n
        self.hybrid_fetch_k = hybrid_fetch_k
        self.rrf_k = rrf_k
        self.rerank_model = rerank_model
        self.enable_multi_query = enable_multi_query
        self.enable_rerank = enable_rerank
        self.cross_encoder = cross_encoder

    def refresh_bm25(self) -> None:
        if self.bm25_index is not None:
            self.bm25_index.rebuild_from_vectorstore(self.vectorstore)

    def retrieve(self, standalone_question: str, top_k: int = 3) -> List[Document]:
        if self.enable_multi_query:
            queries = generate_multi_queries(
                self.llm, standalone_question, n=self.multi_query_n
            )
        else:
            queries = [standalone_question]

        rank_lists: List[List[Document]] = []
        for q in queries:
            rank_lists.append(
                hybrid_search(
                    q,
                    self.vectorstore,
                    self.bm25_index,
                    fetch_k=self.hybrid_fetch_k,
                    rrf_k=self.rrf_k,
                )
            )
        fused = rrf_fuse(rank_lists, rrf_k=self.rrf_k) if len(rank_lists) > 1 else (
            rank_lists[0] if rank_lists else []
        )

        if self.enable_rerank:
            return rerank_documents(
                standalone_question,
                fused,
                top_k=top_k,
                model_name=self.rerank_model,
                cross_encoder=self.cross_encoder,
            )
        return fused[:top_k]
```

更新 `__init__.py`：

```python
from src.retrieval.pipeline import RetrievalPipeline

__all__ = ["RetrievalPipeline"]
```

- [ ] **Step 3: 跑测 + commit**

```bash
python -m pytest tests/test_pipeline.py tests/test_hybrid.py tests/test_multi_query.py tests/test_rerank.py -v
git add src/retrieval/pipeline.py src/retrieval/__init__.py tests/test_pipeline.py
/usr/local/bin/git commit -m "feat(retrieval): wire multi-query hybrid rerank pipeline"
```

---

### Task 7: Wire services — dedupe, catalog intent, pipeline

**Files:**
- Modify: `src/services.py`
- Create: `tests/test_catalog_intent.py`
- Create: `tests/test_add_documents_dedupe.py`（可用 Fake vectorstore，不必真连 Chroma）

**Interfaces:**
- Produces:
  - `is_catalog_question(question: str) -> bool`
  - `format_catalog_answer(docs: List[Dict]) -> str`
  - `KnowledgeBaseService.dedupe_vectorstore() -> int`（删除重复条数）
  - `add_documents`：同名先删再加，并 `pipeline.refresh_bm25()`
  - `_prepare_rag_context`：用 `self.retrieval_pipeline.retrieve`
  - `query_multi_turn` / `stream`：目录意图短路

- [ ] **Step 1: 目录意图测试**

```python
from src.services import is_catalog_question, format_catalog_answer


def test_catalog_intent_positive():
    assert is_catalog_question("一共有几个文档")
    assert is_catalog_question("知识库里有哪些文件")
    assert is_catalog_question("文档列表")


def test_catalog_intent_negative():
    assert not is_catalog_question("2076134 是什么接口")
    assert not is_catalog_question("原生场外开放式基金接口整理里有什么")


def test_format_catalog_answer():
    text = format_catalog_answer(
        [
            {"filename": "a.txt", "chunk_count": 1},
            {"filename": "b.pdf", "chunk_count": 2},
        ]
    )
    assert "2" in text
    assert "a.txt" in text
    assert "b.pdf" in text
```

- [ ] **Step 2: 实现目录辅助函数（可放在 `services.py` 顶部模块级）**

```python
import re

_CATALOG_PATTERNS = [
    r"几个文档",
    r"多少(?:个)?文档",
    r"有哪些文件",
    r"有哪些文档",
    r"文档列表",
    r"文件列表",
    r"上传了什么",
]


def is_catalog_question(question: str) -> bool:
    q = (question or "").strip()
    return any(re.search(p, q) for p in _CATALOG_PATTERNS)


def format_catalog_answer(docs: List[Dict]) -> str:
    if not docs:
        return "知识库中当前没有已入库文档。"
    lines = [f"知识库中共有 {len(docs)} 个文档："]
    for i, d in enumerate(docs, 1):
        lines.append(f"{i}. {d['filename']}（{d['chunk_count']} 个分块）")
    return "\n".join(lines)
```

- [ ] **Step 3: 在 `KnowledgeBaseService.__init__` 末尾初始化 pipeline + BM25，并调用 `dedupe_vectorstore`**

```python
from src.retrieval.bm25_index import BM25Index
from src.retrieval.pipeline import RetrievalPipeline

# inside __init__, after chains:
self.bm25_index = BM25Index.from_vectorstore(self.vectorstore)
self.retrieval_pipeline = RetrievalPipeline(
    vectorstore=self.vectorstore,
    llm=self.llm,
    bm25_index=self.bm25_index,
    multi_query_n=settings.MULTI_QUERY_N,
    hybrid_fetch_k=settings.HYBRID_FETCH_K,
    rrf_k=settings.RRF_K,
    rerank_model=settings.RERANK_MODEL,
    enable_multi_query=settings.ENABLE_MULTI_QUERY,
    enable_rerank=settings.ENABLE_RERANK,
)
self.dedupe_vectorstore()
```

实现：

```python
    def dedupe_vectorstore(self) -> int:
        raw = self.vectorstore.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        seen = {}
        to_delete = []
        from src.retrieval.bm25_index import content_hash

        for i, content in enumerate(documents):
            if content is None:
                continue
            h = content_hash(content)
            if h in seen:
                to_delete.append(ids[i])
            else:
                seen[h] = ids[i]
        if to_delete:
            self.vectorstore.delete(ids=to_delete)
            self.bm25_index.rebuild_from_vectorstore(self.vectorstore)
        return len(to_delete)
```

- [ ] **Step 4: 改 `add_documents` 同名先删**

在 `add_documents` 切片前：

```python
        raw = self.vectorstore.get(include=["metadatas"])
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        old_ids = [
            ids[i]
            for i, m in enumerate(metadatas)
            if m and m.get("filename") == filename
        ]
        if old_ids:
            self.vectorstore.delete(ids=old_ids)

        # ... existing split + add_documents ...

        self.bm25_index.rebuild_from_vectorstore(self.vectorstore)
        return len(chunks)
```

- [ ] **Step 5: 改 `_prepare_rag_context` 使用 pipeline**

将：

```python
        retrieval = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retrieval.invoke(standalone_question)
```

替换为：

```python
        retrieved_docs = self.retrieval_pipeline.retrieve(
            standalone_question, top_k=top_k
        )
```

- [ ] **Step 6: 目录短路接入 query 方法**

在 `query_multi_turn` / `query_multi_turn_stream` 开头：

```python
        if is_catalog_question(question):
            answer = format_catalog_answer(self.list_documents())
            # stream: return iter([answer]), []
            # non-stream: return dict with answer, sources=[], standalone_question=question
```

`query_multi_turn_stream` 返回 `iter([answer]), []`。  
`query_multi_turn` 返回完整 dict，`sources=[]`。

- [ ] **Step 7: 跑测**

```bash
python -m pytest tests/ -v
```

Expected: 全部 PASS（含原 `test_query_stream.py`；若因全局 `kb_service` 初始化过重，保持现有 `__new__` 测法，目录测试只测纯函数）

- [ ] **Step 8: Commit**

```bash
git add src/services.py tests/test_catalog_intent.py
/usr/local/bin/git commit -m "feat(rag): wire hybrid pipeline, dedupe uploads, catalog intent"
```

---

### Task 8: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 在功能特性增加**

```markdown
- Hybrid 检索：向量 + BM25（RRF 融合）+ Cross-Encoder 重排
- Multi-Query 查询改写（可配置关闭）
- 同名文档重新上传会覆盖旧向量；支持目录类问题直接列出已入库文件
```

- [ ] **Step 2: 在配置示例增加可选 env**

```env
MULTI_QUERY_N=3
HYBRID_FETCH_K=10
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENABLE_MULTI_QUERY=true
ENABLE_RERANK=true
```

- [ ] **Step 3: 注意事项补充**

```markdown
- 首次启用重排会下载 Cross-Encoder 模型，需可访问 Hugging Face（或已有本地缓存）
- 同名文件再次上传会删除该文件旧 chunk 后重建索引
```

- [ ] **Step 4: Commit**

```bash
git add README.md
/usr/local/bin/git commit -m "docs: document hybrid retrieval and catalog intent"
```

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| BM25 + jieba | Task 2 |
| Hybrid RRF | Task 3 |
| Multi-Query + 降级 | Task 4 |
| Cross-Encoder + 降级 | Task 5 |
| Pipeline 编排 | Task 6 |
| 接 services / 去重上传 / dedupe / 目录短路 | Task 7 |
| config + deps | Task 1 |
| README | Task 8 |
| 不上 HyDE / 不改 API 契约 | Global + Task 7 |

## Self-Review Notes

- 无 TBD；接口名在任务间一致（`retrieve` / `hybrid_search` / `generate_multi_queries` / `rerank_documents`）
- 单测均可用 Fake，避免 CI 必须下载 CE 模型；真实 CE 加载仅在运行时路径
- `sentence-transformers==5.1.2` 若与当前 torch 冲突，允许在 Task 1 改为环境可解析的最近兼容钉版本并更新 requirements
