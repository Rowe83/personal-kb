# Personal KB — 开源本地个人知识库

基于 **LangChain + Chroma + Streamlit / FastAPI** 的本地 RAG 知识库：文档上传、向量化、多轮对话问答，数据与索引均保存在本机。

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()

## 功能特性

- **本地部署**：向量库与上传文件存于本机，适合个人/团队内网使用
- **UI 配置 API Key**：Streamlit 设置页 + 供应商模板（DeepSeek / 智谱 / OpenAI 兼容）
- **共享配置**：`config.local.yaml` 同时供 Streamlit 与 FastAPI 使用
- **测试连接**：保存前可分别测试 Embedding / LLM
- **多轮对话**：追问重写 + 流式打字机输出 + 引用来源
- **Hybrid 检索**：向量相似度 + BM25（RRF 融合），提升接口号 / 文件名命中
- **同名覆盖提示**：重复上传同名文件前会明确提示并删除旧向量
- **目录类问题**：如「一共几个文档」直接列出清单，不走纯 RAG
- **文档格式**：PDF / TXT（P1 将支持 MD、Excel、PDF OCR、批量导入）

## 快速开始

### 1. 克隆与安装

```bash
git clone https://github.com/Rowe83/personal-kb.git
cd personal-kb
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置（二选一）

**方式 A — UI 配置（推荐）**

```bash
streamlit run streamlit_app.py
```

打开左侧 **⚙️ 设置** 页 → 选择供应商模板 → 填写 API Key → **测试连接** → **保存配置**。

**方式 B — 复制示例文件**

```bash
cp config.local.yaml.example config.local.yaml
# 编辑 config.local.yaml 填入 api_key
```

也支持传统 `.env`（见 `config.local.yaml.example` 内字段说明）；YAML 优先。

### 3. 使用

| 入口 | 命令 | 说明 |
|------|------|------|
| **Streamlit UI** | `streamlit run streamlit_app.py` | 推荐：对话 / 知识库 / 设置 |
| **FastAPI** | `python -m src.main` | HTTP API，文档见 `/docs` |

### 4. 页面导航

- **首页** — 使用说明与路线图
- **💬 对话** — 基于知识库的流式问答
- **📚 知识库** — 上传 PDF/TXT、查看已入库文档
- **⚙️ 设置** — Embedding / LLM API 配置

## 配置说明

配置文件：`config.local.yaml`（已 gitignore，不会提交）

```yaml
embedding:
  provider: zhipu          # zhipu | openai_compatible | custom
  api_key: "your-key"
  base_url: "https://open.bigmodel.cn/api/paas/v4/"
  model: "embedding-3"

llm:
  provider: deepseek
  api_key: "your-key"
  base_url: "https://api.deepseek.com/v1"
  model: "deepseek-chat"
  temperature: 0.3
```

**OpenAI 兼容**：Ollama、One API 等填写对应 Base URL 即可。

## 项目结构

```
personal-kb/
├── streamlit_app.py          # Streamlit 首页
├── pages/                    # 多页 UI（对话 / 知识库 / 设置）
├── ui/shared.py              # UI 公共组件
├── src/
│   ├── services.py           # RAG 核心服务
│   ├── config_loader.py      # 配置加载
│   ├── settings_store.py     # YAML 读写
│   ├── provider_templates.py # 供应商模板
│   ├── connection_test.py    # API 连通性测试
│   └── main.py               # FastAPI
├── config.local.yaml.example
├── uploads/                  # 上传文件（gitignore）
└── chroma_db/                # 向量库（gitignore）
```

## API 概览

- `GET /health` — 配置状态
- `POST /upload` — 上传文档
- `POST /query` — 多轮问答
- `GET /documents` — 文档列表

未配置 API Key 时，`/upload` 与 `/query` 返回 `503` 及明确提示。

## 路线图

| 阶段 | 内容 |
|------|------|
| **P0 ✅** | UI 配置 API Key、本地 YAML、测试连接 |
| **P1** | MD / Excel、PDF OCR、批量导入 |
| **P2** | 现代化 UI、导入进度 |
| **P3** | Hybrid 检索增强、Docker 一键部署 |

## 开发

```bash
python -m pytest tests/ -v
```

## 注意事项

- `config.local.yaml`、`.env`、`uploads/`、`chroma_db/` 请勿提交
- Streamlit 与 FastAPI 请勿同时对同一 `chroma_db/` 写入
- 首次使用 Cross-Encoder 重排（后续版本）需下载本地模型

## License

MIT
