# 个人知识库系统（多轮强化版）

基于 **LangChain + Chroma + FastAPI** 的个人 RAG 知识库后端：上传 PDF/TXT，向量化入库，支持多轮对话问答，并返回引用来源。

## 功能特性

- 文档上传与解析（PDF / TXT，单文件上限 20MB）
- 文本切片后写入 Chroma 向量库
- 智谱 `embedding-3` 做向量化，DeepSeek 做生成
- 多轮对话：根据历史将追问重写为独立问题再检索
- 回答附带来源文件名、页码与内容片段

## 技术栈

| 组件 | 说明 |
|------|------|
| FastAPI / Uvicorn | HTTP API |
| LangChain | RAG 编排、多轮重写与问答 |
| Chroma | 本地向量存储 |
| PyMuPDF | PDF 解析 |
| 智谱 AI | Embeddings（`embedding-3`） |
| DeepSeek | LLM（`deepseek-v4-flash`） |

## 项目结构

```
personal-kb/
├── src/
│   ├── main.py        # FastAPI 路由与请求/响应模型
│   ├── services.py    # 解析、向量化、多轮检索与生成
│   └── config.py      # 环境变量与路径配置
├── uploads/           # 上传文件（运行时生成，已 gitignore）
├── chroma_db/         # 向量库（运行时生成，已 gitignore）
├── requirements.txt
├── .env               # 本地密钥（勿提交）
└── README.md
```

## 快速开始

### 1. 环境要求

- Python 3.11+
- 智谱 AI API Key、DeepSeek API Key

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

在项目根目录创建 `.env`：

```env
ANONYMIZED_TELEMETRY=False

# DeepSeek（LLM）
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 智谱 AI（Embeddings）
ZHIPUAI_API_KEY=your_zhipu_api_key
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
```

### 4. 启动服务

```bash
python -m src.main
```

默认监听：`http://127.0.0.1:8000`  
交互文档：`http://127.0.0.1:8000/docs`

## API 说明

### `POST /upload`

上传文档并构建索引。

- 表单字段：`file`（multipart/form-data）
- 支持：`.pdf`、`.txt`
- 限制：单文件 ≤ 20MB

成功响应示例：

```json
{
  "status": "success",
  "filename": "demo.pdf",
  "chunks_created": 12,
  "message": "文档解析并向量化入库成功"
}
```

### `POST /query`

检索知识库并生成回答（支持多轮）。

请求体：

```json
{
  "question": "它支持哪些文件格式？",
  "history": [
    { "role": "user", "content": "这个知识库是做什么的？" },
    { "role": "assistant", "content": "这是一个个人 RAG 知识库后端服务。" }
  ],
  "top_k": 3
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | string | 当前问题（必填） |
| `history` | array | 对话历史，`role` 为 `user` / `assistant` |
| `top_k` | int | 检索条数，默认 3，范围 1–10 |

响应体：

```json
{
  "question": "它支持哪些文件格式？",
  "standalone_question": "个人知识库支持哪些文件格式？",
  "answer": "...",
  "sources": [
    {
      "filename": "demo.pdf",
      "page": 1,
      "content_snippet": "..."
    }
  ]
}
```

`standalone_question` 为结合历史重写后的独立检索问题；无历史时与 `question` 相同。

## 多轮问答流程

1. 若存在 `history`，先用 LLM 将当前追问重写为独立问题
2. 用 `standalone_question` 在 Chroma 中做相似度检索
3. 将检索上下文与对话历史一并交给 LLM 生成回答
4. 去重后返回引用来源

## 注意事项

- `.env`、`uploads/`、`chroma_db/` 已在 `.gitignore` 中，请勿提交密钥与本地数据
- 空文件、非 UTF-8 TXT、无法解析的 PDF 会返回 422
- 知识库中无相关内容时，模型应明确说明「知识库中未找到相关内容」
