# Streamlit Cache & Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Streamlit 直连 `KnowledgeBaseService`，用 `@st.cache_resource` 缓存 Heavy 对象，并用真流式 LLM + `st.write_stream` 实现打字机问答。

**Architecture:** 在 `KnowledgeBaseService` 抽出共享的检索准备逻辑，新增 `query_multi_turn_stream` 返回 `(token_iterator, sources)`；`streamlit_app.py` 去掉 HTTP，经 `cache_resource` 获取 service，上传/列表/问答全部直调；FastAPI 与 `query_multi_turn` 保持不变。

**Tech Stack:** Streamlit 1.62.0、LangChain、Chroma、现有 DeepSeek / 智谱配置

**Spec:** `docs/superpowers/specs/2026-08-26-streamlit-cache-streaming-design.md`

## Global Constraints

- 本次不修改 `src/main.py`
- 不引入 `asyncio` / `astream` / FastAPI SSE
- 不改变 `query_multi_turn` 的返回字典契约
- Streamlit 钉版本：`streamlit==1.62.0`
- 上传单文件上限与 FastAPI 一致：20MB
- 日常建议 Streamlit 与 FastAPI 不同时写同一 `chroma_db/`

---

## File Structure

| 文件 | 职责 |
|------|------|
| `src/services.py` | 抽出 `_prepare_rag_context`；新增 `query_multi_turn_stream`；保留 `query_multi_turn` |
| `tests/test_query_stream.py` | 流式方法的单元测试（mock LLM / retriever） |
| `streamlit_app.py` | `cache_resource`、直连上传/列表、`write_stream` 问答 |
| `requirements.txt` | 增加 `streamlit==1.62.0` |
| `README.md` | Streamlit 启动与并存说明 |

---

### Task 1: 服务层流式查询 API

**Files:**
- Modify: `src/services.py`
- Create: `tests/test_query_stream.py`
- Create: `tests/__init__.py`（空文件，便于发现）

**Interfaces:**
- Consumes: 现有 `rephrase_chain`、`qa_chain`、`vectorstore`、`_convert_chat_history`
- Produces:
  - `_prepare_rag_context(question, history, top_k) -> dict` with keys `chat_history`, `standalone_question`, `context_str`, `sources`, `retrieved_docs`
  - `query_multi_turn_stream(question: str, history: List[Dict[str, str]], top_k: int = 3) -> Tuple[Iterator[str], List[Dict]]`
  - `query_multi_turn(...)` 仍返回含 `question` / `standalone_question` / `answer` / `sources` 的 dict

- [ ] **Step 1: 创建测试文件与失败用例**

创建 `tests/__init__.py`（空）与 `tests/test_query_stream.py`：

```python
from typing import Iterator, List, Dict, Tuple
from unittest.mock import MagicMock, patch
import pytest

from langchain_core.documents import Document


def test_query_multi_turn_stream_yields_tokens_and_sources():
    from src.services import KnowledgeBaseService

    service = KnowledgeBaseService.__new__(KnowledgeBaseService)
    service._convert_chat_history = MagicMock(return_value=[])
    service.rephrase_chain = MagicMock()

    doc = Document(
        page_content="LangChain 是一个 LLM 应用框架。",
        metadata={"filename": "demo.txt", "page": 1},
    )
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]
    service.vectorstore = MagicMock()
    service.vectorstore.as_retriever.return_value = retriever

    def fake_stream(_inputs):
        yield "你好"
        yield "世界"

    service.qa_chain = MagicMock()
    service.qa_chain.stream.side_effect = fake_stream

    token_iter, sources = service.query_multi_turn_stream(
        question="什么是 LangChain？", history=[], top_k=3
    )

    assert isinstance(token_iter, Iterator) or hasattr(token_iter, "__iter__")
    assert "".join(list(token_iter)) == "你好世界"
    assert len(sources) == 1
    assert sources[0]["filename"] == "demo.txt"
    assert sources[0]["page"] == 1
    assert "LangChain" in sources[0]["content_snippet"]
    service.qa_chain.stream.assert_called_once()
    service.rephrase_chain.invoke.assert_not_called()


def test_query_multi_turn_stream_rephrases_when_history_exists():
    from src.services import KnowledgeBaseService
    from langchain_core.messages import HumanMessage, AIMessage

    service = KnowledgeBaseService.__new__(KnowledgeBaseService)
    history = [
        {"role": "user", "content": "介绍一下知识库"},
        {"role": "assistant", "content": "这是个人 RAG"},
    ]
    chat_history = [
        HumanMessage(content="介绍一下知识库"),
        AIMessage(content="这是个人 RAG"),
    ]
    service._convert_chat_history = MagicMock(return_value=chat_history)
    service.rephrase_chain = MagicMock()
    service.rephrase_chain.invoke.return_value = "个人知识库支持什么格式？"

    retriever = MagicMock()
    retriever.invoke.return_value = []
    service.vectorstore = MagicMock()
    service.vectorstore.as_retriever.return_value = retriever
    service.qa_chain = MagicMock()
    service.qa_chain.stream.return_value = iter(["无相关内容"])

    token_iter, sources = service.query_multi_turn_stream(
        question="它支持什么格式？", history=history, top_k=2
    )
    assert list(token_iter) == ["无相关内容"]
    assert sources == []
    service.rephrase_chain.invoke.assert_called_once()
    retriever.invoke.assert_called_once_with("个人知识库支持什么格式？")
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```bash
cd /Users/luoh/workspace/GitHub/personal-kb
python -m pytest tests/test_query_stream.py -v
```

Expected: FAIL（`query_multi_turn_stream` 不存在，或 import/属性错误）

- [ ] **Step 3: 在 `src/services.py` 实现共享准备逻辑与流式方法**

在文件顶部 typing 中确保有 `Iterator`、`Tuple`（若尚无则补充）：

```python
from typing import List, Dict, Iterator, Tuple
```

在 `KnowledgeBaseService` 中，将 `query_multi_turn` 的「重写 + 检索 + sources」抽成私有方法，并让两个公开方法共用。在 `query_multi_turn` **之前**插入：

```python
    def _prepare_rag_context(
        self, question: str, history: List[Dict[str, str]], top_k: int = 3
    ) -> Dict:
        """重写追问、检索文档并构建 context / sources（同步）"""
        chat_history = self._convert_chat_history(history)

        if chat_history:
            standalone_question = self.rephrase_chain.invoke(
                {"chat_history": chat_history, "question": question}
            )
        else:
            standalone_question = question

        retrieval = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retrieval.invoke(standalone_question)

        context_str = "\n\n".join(
            [
                f"出处 {doc.metadata['filename']} 第 {doc.metadata['page']} 页: {doc.page_content}"
                for doc in retrieved_docs
            ]
        )

        sources = []
        seen = set()
        for doc in retrieved_docs:
            fname = doc.metadata.get("filename", "未知")
            page = doc.metadata.get("page", 1)
            key = f"{fname}-{page}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "filename": fname,
                        "page": page,
                        "content_snippet": doc.page_content[:120] + "...",
                    }
                )

        return {
            "chat_history": chat_history,
            "standalone_question": standalone_question,
            "context_str": context_str,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
        }

    def query_multi_turn_stream(
        self, question: str, history: List[Dict[str, str]], top_k: int = 3
    ) -> Tuple[Iterator[str], List[Dict]]:
        """检索同步完成后，流式 yield LLM token；同时返回 sources。"""
        prepared = self._prepare_rag_context(question, history, top_k)

        token_iter = self.qa_chain.stream(
            {
                "context": prepared["context_str"],
                "chat_history": prepared["chat_history"],
                "question": question,
            }
        )
        return token_iter, prepared["sources"]
```

将现有 `query_multi_turn` 体改写为调用 `_prepare_rag_context` + `qa_chain.invoke`（返回字段名与原先完全一致）：

```python
    def query_multi_turn(
        self, question: str, history: List[Dict[str, str]], top_k: int = 3
    ) -> Dict:
        """检索并生成回答与出处引用"""
        prepared = self._prepare_rag_context(question, history, top_k)
        answer = self.qa_chain.invoke(
            {
                "context": prepared["context_str"],
                "chat_history": prepared["chat_history"],
                "question": question,
            }
        )
        return {
            "question": question,
            "standalone_question": prepared["standalone_question"],
            "answer": answer,
            "sources": prepared["sources"],
        }
```

注意：模块末尾的 `kb_service = KnowledgeBaseService()` 保持不变（FastAPI 仍用）。

- [ ] **Step 4: 再跑测试确认通过**

Run:

```bash
python -m pytest tests/test_query_stream.py -v
```

Expected: PASS（2 passed）

若缺少 pytest：

```bash
pip install pytest
python -m pytest tests/test_query_stream.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/services.py tests/__init__.py tests/test_query_stream.py
git commit -m "feat(services): add streaming multi-turn query API"
```

---

### Task 2: Streamlit 直连 + cache_resource + write_stream

**Files:**
- Modify: `streamlit_app.py`（整体替换为直连实现）
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `KnowledgeBaseService.add_documents` / `list_documents` / `query_multi_turn_stream`；`settings.UPLOAD_DIR`
- Produces: 可独立运行的 Streamlit UI（无需 FastAPI）

- [ ] **Step 1: 将 `streamlit` 写入依赖**

在 `requirements.txt` 末尾追加一行：

```
streamlit==1.62.0
```

安装：

```bash
pip install streamlit==1.62.0
```

- [ ] **Step 2: 重写 `streamlit_app.py`**

用下面完整内容替换 `streamlit_app.py`（保留现有 CSS / 布局风格，去掉 `requests`）：

```python
import os
import streamlit as st

from src.config import settings

# =====================================================================
# 1. 页面配置与自定义 CSS
# =====================================================================
st.set_page_config(
    page_title="AI 个人知识库",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #f8f9fa;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    .source-box {
        background-color: #f1f3f5;
        border-left: 4px solid #4c6ef5;
        padding: 8px 12px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #495057;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


@st.cache_resource
def get_kb_service():
    from src.services import KnowledgeBaseService

    return KnowledgeBaseService()


def render_sources(sources):
    if not sources:
        return
    with st.expander("📌 查看参考引用来源"):
        for idx, src in enumerate(sources, start=1):
            st.markdown(
                f"<div class='source-box'>"
                f"<b>来源 {idx}:</b> {src['filename']} (第 {src['page']} 页)<br>"
                f"<i>\"{src['content_snippet']}\"</i>"
                f"</div>",
                unsafe_allow_html=True,
            )


kb_service = get_kb_service()

# =====================================================================
# 2. Session State
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 你好！我是你的 **AI 个人知识库助手**。请在左侧上传文档，然后向我提问吧！",
            "sources": [],
        }
    ]

# =====================================================================
# 3. 侧边栏：上传、列表、设置
# =====================================================================
with st.sidebar:
    st.title("📚 知识库管理")
    st.markdown("---")

    st.subheader("📤 上传新文档")
    uploaded_file = st.file_uploader("选择 PDF 或 TXT 文件", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("🚀 开始解析并向量化", use_container_width=True):
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > MAX_FILE_SIZE:
                st.error("文件大小超过限制（最大 20MB）")
            else:
                file_path = os.path.join(settings.UPLOAD_DIR, uploaded_file.name)
                with st.spinner("文档上传与切片处理中..."):
                    try:
                        with open(file_path, "wb") as buffer:
                            buffer.write(file_bytes)
                        chunks_created = kb_service.add_documents(
                            file_path, uploaded_file.name
                        )
                        st.success(
                            f"✅ 上传成功！创建了 {chunks_created} 个文档分块。"
                        )
                        st.rerun()
                    except ValueError as ve:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        st.error(f"❌ 上传失败: {ve}")
                    except Exception as e:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        st.error(f"⚠️ 处理失败: {e}")

    st.markdown("---")

    st.subheader("📑 已入库文档")
    try:
        docs_data = kb_service.list_documents()
        if not docs_data:
            st.info("暂无已入库文档，请先上传。")
        else:
            for doc in docs_data:
                st.text(f"📄 {doc['filename']} ({doc['chunk_count']} 块)")
    except Exception as e:
        st.warning(f"⚠️ 无法读取文档列表: {e}")

    st.markdown("---")

    st.subheader("⚙️ 检索参数与设置")
    top_k = st.slider("检索匹配块数量 (Top-K)", min_value=1, max_value=5, value=3)

    if st.button("🧹 清空对话历史", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "对话已重置。请随时向我提问！",
                "sources": [],
            }
        ]
        st.rerun()

# =====================================================================
# 4. 主界面：历史 + 流式问答
# =====================================================================
st.title("🤖 个人知识库智能问答")
st.caption("基于 LangChain + DeepSeek + Chroma + Streamlit 驱动")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if user_input := st.chat_input("请针对知识库文档提出你的问题..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ["user", "assistant"]
    ]

    with st.chat_message("assistant"):
        try:
            with st.spinner("正在检索知识库..."):
                token_iter, sources = kb_service.query_multi_turn_stream(
                    question=user_input,
                    history=history_payload,
                    top_k=top_k,
                )
            answer = st.write_stream(token_iter)
            render_sources(sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer if answer is not None else "",
                    "sources": sources,
                }
            )
        except Exception as e:
            st.error(f"⚠️ 问答失败: {e}")
```

- [ ] **Step 3: 语法检查**

Run:

```bash
python -m py_compile streamlit_app.py src/services.py
```

Expected: 无输出、exit code 0

- [ ] **Step 4: 手动冒烟（有 API Key 时）**

Run:

```bash
streamlit run streamlit_app.py
```

检查清单：

1. 页面可打开，侧边栏能列出已有文档（或显示暂无文档）
2. 提问后先出现「正在检索知识库…」，随后答案逐 token 打出
3. 有命中时 expander 显示来源
4. 上传一个小 TXT 后列表更新，再提问能引用
5. 二次提问时进程不再明显卡在「初始化模型」（Heavy 对象已缓存）

若本机暂时无法开浏览器，至少完成 Step 3，并在 commit message body 注明「UI 冒烟待本地验证」。

- [ ] **Step 5: Commit**

```bash
git add streamlit_app.py requirements.txt
git commit -m "feat(streamlit): cache KB service and stream answers"
```

---

### Task 3: README 补充 Streamlit 用法

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 2 的启动方式与架构决策
- Produces: 文档说明 Streamlit 直连、FastAPI 并存、Chroma 勿双写

- [ ] **Step 1: 更新 README 关键段落与结构**

将「基于 **LangChain + Chroma + FastAPI** 的个人 RAG 知识库后端」改为同时涵盖 UI，例如：

```markdown
基于 **LangChain + Chroma + FastAPI / Streamlit** 的个人 RAG 知识库：上传 PDF/TXT，向量化入库，支持多轮对话问答（API 或 Streamlit UI），并返回引用来源。
```

在「功能特性」增加两条：

```markdown
- Streamlit UI：`@st.cache_resource` 缓存 Heavy 对象，LLM 真流式打字机输出
- FastAPI 与 Streamlit 可并存；日常请避免两边同时写入同一 `chroma_db/`
```

在「技术栈」表增加一行：

```markdown
| Streamlit | 本地 Web UI（直连 KnowledgeBaseService） |
```

在「项目结构」补充：

```markdown
├── streamlit_app.py   # Streamlit UI（直连服务层）
├── docs/              # 设计与实现计划
```

- [ ] **Step 2: 在「启动服务」后增加 Streamlit 小节**

紧接 `python -m src.main` 说明之后插入：

```markdown
### 5. 启动 Streamlit UI（推荐日常使用）

```bash
streamlit run streamlit_app.py
```

Streamlit 进程内直连 `KnowledgeBaseService`，**不需要**先启动 FastAPI。

若同时需要 HTTP API，可另开终端运行 `python -m src.main`。请勿让 Streamlit 与 FastAPI 同时对同一 `chroma_db/` 做写入（上传/向量化），以免索引冲突。
```

（原「注意事项」可追加一句重申 Chroma 双写风险。）

- [ ] **Step 3: 目视检查 README 代码块闭合、路径正确**

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document Streamlit direct mode and cache/streaming"
```

---

## Spec Coverage Checklist

| Spec 要求 | 对应 Task |
|-----------|-----------|
| `@st.cache_resource` 缓存 `KnowledgeBaseService` | Task 2 |
| 混合真流式（检索同步 + LLM.stream + write_stream） | Task 1 + 2 |
| Streamlit 去掉 HTTP / requests | Task 2 |
| 不改 `src/main.py` / 保留 `query_multi_turn` | Task 1（重构内部，契约不变） |
| 上传/列表直连 service，失败删临时文件 | Task 2 |
| 流式异常不写入半截 assistant 消息 | Task 2（except 分支不 append） |
| `requirements.txt` 钉 streamlit | Task 2 |
| README 直连与并存说明 | Task 3 |
| 不做 asyncio / SSE / 假流式 | 全局约束 |

## Self-Review Notes

- 无 TBD/TODO 占位
- `query_multi_turn_stream` 签名在 Task 1 / Task 2 一致：`Tuple[Iterator[str], List[Dict]]`
- `st.write_stream` 返回完整字符串后再写入 `session_state`，与 spec 一致
