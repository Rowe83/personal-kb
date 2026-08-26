# RAG 检索策略优化设计

**日期:** 2026-08-26  
**状态:** 已确认设计  
**范围:** Hybrid Search（向量 + BM25）+ 极简 Cross-Encoder 重排 + Multi-Query；上传去重与目录类问题短路

## 目标

1. 用 **Hybrid Search**（Chroma 向量 + 内存 BM25）提升接口号、文件名、短 TXT 的召回。
2. 用 **极简 Cross-Encoder** 对融合候选重排序，截断为最终 `top_k`。
3. 用 **Multi-Query**（不上 HyDE）扩展查询后再融合召回。
4. **同名上传去重** + 清理已有重复 chunk，避免 top_k 被重复 PDF 占满。
5. **目录类问题**（「几个文档 / 有哪些文件」）走 `list_documents()`，不依赖纯 RAG。

## 非目标

- HyDE
- Cohere / 云端 Rerank API
- 更换向量库（仍用 Chroma）
- 改动 Streamlit / FastAPI 对外 API 契约
- 新增检索调试 UI

## 已确认决策

| 项 | 选择 |
|----|------|
| 架构 | 模块化 `src/retrieval/` 管线（方案 1） |
| Rerank | 极简 Cross-Encoder（`sentence-transformers`） |
| 查询改写 | 仅 Multi-Query |
| 附带修复 | 上传去重 + 清重复 + 目录意图短路 |

## 架构

```
用户问题
  ├─ [目录意图?] → list_documents() → 直接回答（不走向量）
  └─ 多轮 rephrase → standalone_q
         ↓
    Multi-Query（LLM 生成 N 条改写，与原问合并去重）
         ↓
    对每条 query 做 Hybrid 召回
         ├─ 向量：Chroma similarity（每路 fetch_k）
         └─ 关键词：内存 BM25（jieba + rank_bm25）
         ↓
    RRF 融合 + 按 chunk id 去重
         ↓
    Cross-Encoder 重排 → 最终 top_k
         ↓
    现有 qa_chain / stream（返回契约不变）
```

### 文件边界

| 模块 | 职责 |
|------|------|
| `src/retrieval/bm25_index.py` | 从 Chroma 同步构建/刷新 BM25；中文分词检索 |
| `src/retrieval/hybrid.py` | 向量 + BM25 + RRF |
| `src/retrieval/rerank.py` | Cross-Encoder `predict` → 排序截断 |
| `src/retrieval/multi_query.py` | Multi-Query 生成与去重 |
| `src/retrieval/pipeline.py` | 编排；对外 `retrieve(question, top_k) -> List[Document]` |
| `src/services.py` | `_prepare_rag_context` 调 pipeline；`add_documents` 同名先删；目录意图短路；`dedupe_vectorstore` |
| `src/config.py` | 检索相关配置项 |

## 组件与数据流

### BM25 索引

- 数据源：`vectorstore.get(include=["documents","metadatas","ids"])`
- 分词：`jieba.lcut`
- 引擎：`rank_bm25.BM25Okapi`
- API：`search(query, k) -> List[Document]`
- 刷新：服务启动构建一次；`add_documents` 成功后刷新
- 去重键：优先 Chroma `id`；否则 `filename|page|content_hash`

### Hybrid

- 输入：单条 query、`HYBRID_FETCH_K`（默认 10）
- 向量：`similarity_search(query, k=fetch_k)`
- BM25：同 k
- RRF：`score = Σ 1/(RRF_K + rank)`，默认 `RRF_K=60`
- 输出：按 id 去重后的候选列表

### Multi-Query

- LLM 生成 `MULTI_QUERY_N`（默认 3）条改写（同义 / 关键词 / 接口号侧重）
- 与 `standalone_question` 合并去重
- 每条 query 各跑一轮 Hybrid，候选再按 id / RRF 合并

### Cross-Encoder 重排

- 默认模型：`cross-encoder/ms-marco-MiniLM-L-6-v2`
- `predict([(rerank_query, doc.page_content), ...])` 降序，截断为请求 `top_k`
- `rerank_query` 使用 **standalone_question**（或用户原问），不用改写句列表

### Pipeline

```text
retrieve(standalone_q, top_k) -> List[Document]
```

编排：Multi-Query → Hybrid×N → merge → Rerank → top_k

### services.py

1. **目录意图**：关键词/轻量规则匹配「几个文档」「有哪些文件」「文档列表」等 → `list_documents()` 拼答案；`query_multi_turn` / `query_multi_turn_stream` 均短路。
2. **`add_documents`**：按 `filename` 查旧 ids → `delete` → `add` → 刷新 BM25。
3. **`dedupe_vectorstore()`**：同 `content_hash` 只留一条；启动或首次 pipeline 前幂等调用一次。
4. **`_prepare_rag_context`**：非目录意图时用 `pipeline.retrieve` 替换纯向量 retriever。

### 对外契约

- `query_multi_turn` / `query_multi_turn_stream` 返回结构不变
- FastAPI / Streamlit 无需改接口签名

## 依赖与默认参数

### 新增依赖

- `rank-bm25`
- `jieba`
- `sentence-transformers`

### 配置默认值

| 项 | 默认 | 说明 |
|----|------|------|
| `MULTI_QUERY_N` | `3` | 改写条数 |
| `HYBRID_FETCH_K` | `10` | 每路召回条数 |
| `RRF_K` | `60` | RRF 常数 |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | CE 模型 |
| `ENABLE_MULTI_QUERY` | `true` | 可关 |
| `ENABLE_RERANK` | `true` | 可关 |

最终进 LLM 的条数 = 请求参数 `top_k`。

## 错误处理

| 场景 | 行为 |
|------|------|
| Multi-Query LLM 失败 | 降级为仅 `standalone_question`，不中断问答 |
| BM25 空库 | 只走向量 |
| 向量检索失败 | 向上抛错（与现网一致） |
| Cross-Encoder 加载/推理失败 | 降级为 RRF 结果直接截断 `top_k` |
| 同名删除失败 | 中止上传并报错，避免半写入 |
| 目录意图识别 | 宁可漏检走检索，也不误杀正常内容问答 |

## 性能预期

- 首启：下载/加载 CE 模型一次（进程内单例 / Streamlit `cache_resource` 复用 service）
- 每次问答：+1 次 Multi-Query LLM；候选约几十条本地 CE 打分

## 测试要点

- BM25：接口号 `2076134` 能命中对应 TXT
- Hybrid+RRF：按文件名提问不再被 PDF 重复块占满
- Rerank：相关 TXT 进入最终 `top_k`
- 同名重传：chunk 数不翻倍
- 「一共几个文档」→ 返回真实文件清单（当前 3 个），不编造
- `query_multi_turn` 返回字段回归：`question` / `standalone_question` / `answer` / `sources`

## 实现顺序建议

1. BM25 索引 + 单测（含接口号）
2. Hybrid RRF + 单测
3. Multi-Query + 降级行为单测
4. Cross-Encoder rerank + 失败降级单测
5. Pipeline 编排
6. services：去重上传、`dedupe_vectorstore`、目录短路、接上 pipeline
7. requirements / config / README
