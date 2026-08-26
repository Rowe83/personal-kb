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
