import streamlit as st

from src.config_loader import load_app_settings
from ui.shared import (
    apply_css,
    apply_page_config,
    cached_kb_service,
    handle_config_error,
    render_config_banner,
    render_sidebar_brand,
    render_sources,
)

apply_page_config()
apply_css()
render_sidebar_brand()

settings = load_app_settings()
kb_service = cached_kb_service()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 你好！我是你的 **AI 个人知识库助手**。请先在「知识库」页上传文档，然后在此提问。",
            "sources": [],
        }
    ]

st.title("💬 智能问答")
st.caption("本地 RAG 问答 · 流式输出与引用来源")
render_config_banner()

with st.sidebar:
    st.subheader("检索参数")
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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if user_input := st.chat_input("请针对知识库文档提出你的问题..."):
    if not settings.is_llm_configured():
        st.warning("请先在「设置」页配置 LLM API Key。")
        st.stop()

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
        except Exception as exc:
            handle_config_error(exc)
