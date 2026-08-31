import streamlit as st

from ui.shared import apply_css, apply_page_config, render_config_banner

apply_page_config()
apply_css()

st.title("🤖 Personal KB")
st.caption("开源本地个人知识库 — 数据与向量库均保存在本机")

render_config_banner()

st.markdown(
    """
### 快速导航

- **💬 对话** — 基于知识库的多轮问答（流式输出 + 引用来源）
- **📚 知识库** — 上传 PDF / TXT 并向量化入库
- **⚙️ 设置** — 配置 Embedding / LLM API Key（支持 DeepSeek、智谱、OpenAI 兼容）

### 首次使用

1. 打开左侧 **设置** 页，选择供应商模板并填写 API Key  
2. 点击 **测试连接** 确认配置正确，然后 **保存配置**  
3. 在 **知识库** 页上传文档  
4. 在 **对话** 页开始提问  

### 路线图

| 阶段 | 内容 |
|------|------|
| **P0（当前）** | UI 配置 API Key、本地 YAML、测试连接 |
| P1 | MD / Excel、PDF OCR、批量导入 |
| P2 | 现代化 UI、导入进度 |
| P3 | Hybrid 检索增强、Docker 部署 |
"""
)

st.info("请从左侧导航栏选择页面开始使用。")
