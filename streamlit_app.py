import streamlit as st

from src.config_loader import load_app_settings
from ui.shared import (
    apply_css,
    apply_page_config,
    cached_kb_service,
    render_config_banner,
    render_sidebar_brand,
)

apply_page_config()
apply_css()
render_sidebar_brand()
render_config_banner()

settings = load_app_settings()
st.title("Personal KB")
st.caption("开源本地个人知识库 — 数据与向量库均保存在本机")

emb_ok = settings.is_embedding_configured()
llm_ok = settings.is_llm_configured()
try:
    doc_count = len(cached_kb_service().list_documents())
except Exception:
    doc_count = 0

c1, c2 = st.columns(2)
with c1:
    st.markdown(
        f"<div class='kb-stat'><div class='kb-stat-label'>模型配置</div>"
        f"<div class='kb-stat-value'>{'已就绪' if emb_ok and llm_ok else '待配置'}</div>"
        f"<div class='kb-stat-label'>Embedding {'✓' if emb_ok else '✗'} · LLM {'✓' if llm_ok else '✗'}</div></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"<div class='kb-stat'><div class='kb-stat-label'>已入库文档</div>"
        f"<div class='kb-stat-value'>{doc_count}</div></div>",
        unsafe_allow_html=True,
    )

st.markdown("### 首次使用")
st.markdown(
    "1. 打开左侧 **设置**，选择供应商模板并填写 API Key，测试连接后保存\n"
    "2. 在 **知识库** 上传 PDF / TXT / MD / XLSX（支持批量）\n"
    "3. 在 **对话** 开始提问"
)

st.markdown("### 路线图")
st.markdown(
    "| 阶段 | 内容 |\n|------|------|\n"
    "| **P0 ✅** | UI 配置 API Key、本地 YAML、测试连接 |\n"
    "| **P1 ✅** | MD / Excel、批量导入 |\n"
    "| **P2 ✅** | 现代化 UI、导入进度、文档删除 |\n"
    "| P3 | Hybrid 检索增强、Docker |"
)
