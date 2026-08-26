# Streamlit 性能优化：缓存与流式输出

**日期:** 2026-08-26  
**状态:** 已确认设计  
**范围:** Streamlit 直连知识库 + `@st.cache_resource` + 真流式打字机

## 目标

1. 使用 `@st.cache_resource` 缓存 Heavy 对象（`KnowledgeBaseService`），避免每次 Streamlit rerun 重复初始化 Embeddings / LLM / Chroma。
2. 问答链路采用混合真流式：检索与追问重写同步完成，最终 LLM 生成用 token 流 + `st.write_stream` 打字机效果。
3. Streamlit 不再依赖 FastAPI HTTP；本次不改 `src/main.py`。

## 非目标

- FastAPI SSE / 流式 endpoint
- `asyncio` / `astream`
- 假流式（完整答案再按字符拆分）
- 修改现有 `query_multi_turn` 的对外返回契约
- 为 Chroma 跨进程写入加分布式锁

## 架构

```
streamlit_app.py
  └─ @st.cache_resource → get_kb_service()
         └─ KnowledgeBaseService（Embeddings / LLM / Chroma / chains）
              ├─ add_documents / list_documents（上传与列表，同步）
              ├─ query_multi_turn（保留，供 FastAPI）
              └─ query_multi_turn_stream（新增：检索同步 + LLM.stream）

src/main.py  ← 本次不改，继续调用 query_multi_turn
```

要点：

- Streamlit 进程内直连 service，去掉 `requests` 与 `API_BASE_URL`。
- Heavy 对象只初始化一次，跨 rerun 复用。
- FastAPI 与 Streamlit 可并存；各自进程各有一份 service 实例，共享同一 Chroma 持久化目录。
- 日常使用建议只开一边，避免同时写同一 Chroma 目录。

## 组件设计

### 1. `get_kb_service()`（`streamlit_app.py`）

```python
@st.cache_resource
def get_kb_service():
    from src.services import KnowledgeBaseService
    return KnowledgeBaseService()
```

不直接缓存模块级全局 `kb_service`，避免与 FastAPI 进程生命周期纠缠；Streamlit 侧自行 `cache_resource` 一份。

### 2. `query_multi_turn_stream`（`src/services.py`）

与现有 `query_multi_turn` 对齐，仅将最终生成改为流式：

1. `_convert_chat_history(history)`
2. 若有历史：`rephrase_chain.invoke` → `standalone_question`（同步）
3. 向量检索 `top_k`（同步）
4. 拼接 `context`，构建去重后的 `sources`
5. 对 `qa_chain` 使用 `.stream(...)`，yield 每个 token 字符串

**返回约定:**

```python
Tuple[Iterator[str], List[Dict]]  # (token_iter, sources)
```

调用方先拿到 `sources`，再消费 `token_iter`；流结束后将完整 answer 写入 session state。

保留 `query_multi_turn` 不变，供 FastAPI `/query` 使用。

### 3. Streamlit 问答路径

1. 用户输入 → 追加到 `st.session_state.messages`
2. 展示「检索中…」状态（spinner / status）
3. 调用 `query_multi_turn_stream`（完成重写 + 检索 + 得到 sources 与 token 迭代器）
4. `st.write_stream(token_iter)` 打字机输出
5. 若有 sources，渲染 expander
6. 将完整 `answer` + `sources` 写入 `session_state.messages`

### 4. 上传与文档列表

- 上传：临时落盘到 `settings.UPLOAD_DIR` → `kb_service.add_documents`（逻辑与 FastAPI 一致）
- 列表：`kb_service.list_documents()`
- 移除对后端 HTTP 的依赖

## 错误处理

| 场景 | 行为 |
|------|------|
| 上传空文件 / 非法格式 / 解析失败 | `st.error` 展示 `ValueError`；临时文件失败时删除 |
| 检索或流式生成异常 | `st.error`；不把半截 assistant 消息写入 `session_state` |
| `write_stream` 中途失败 | 本轮不 append assistant 消息 |
| 空库 / 无命中 | 由现有 prompt 约束流式输出「知识库中未找到相关内容」；sources 可为空 |
| 需重建 Heavy 对象 | 重启 Streamlit 进程（`cache_resource` 随进程生命周期） |

## 依赖与运维

- `requirements.txt` 增加 `streamlit`（钉版本，风格与现有依赖一致）
- 启动：`streamlit run streamlit_app.py`（不要求先启动 FastAPI）
- FastAPI 仍可单独 `uvicorn` 供 API 客户端使用
- README 简短补充：Streamlit 直连说明、与 FastAPI 并存及「勿同时写同一 Chroma」注意点

## 测试要点

- 冷启动后首次问答应完成 service 初始化；再次提问不应重复加载 Heavy 对象（观察启动耗时 / 日志）
- 多轮追问：重写后检索正确，流式回答可见
- 上传 PDF/TXT 后列表立即可见；问答能引用新文档
- 故意断网 / 无效 API Key：错误提示清晰，历史消息不被污染
- FastAPI `/query` 在未改动的前提下仍可用（回归）

## 实现顺序建议

1. `services.py`：新增 `query_multi_turn_stream`，复用现有重写/检索/sources 逻辑
2. `streamlit_app.py`：`cache_resource` + 直连上传/列表/流式问答
3. `requirements.txt` + README 更新
