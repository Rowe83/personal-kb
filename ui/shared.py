import streamlit as st

from src.config_loader import load_app_settings
from src.service_factory import get_kb_service, reset_kb_service
from src.settings_store import ConfigurationError

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

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
    .config-banner {
        background: #fff3cd;
        border: 1px solid #ffecb5;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        color: #664d03;
    }
</style>
"""


def apply_page_config():
    st.set_page_config(
        page_title="Personal KB",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def cached_kb_service():
    return get_kb_service(force_reload=True)


def invalidate_service_cache():
    reset_kb_service()
    cached_kb_service.clear()


def render_config_banner():
    settings = load_app_settings()
    missing = []
    if not settings.is_embedding_configured():
        missing.append("Embedding（向量模型）")
    if not settings.is_llm_configured():
        missing.append("LLM（大模型）")
    if not missing:
        return
    st.markdown(
        f"<div class='config-banner'>⚠️ 尚未配置 {'、'.join(missing)} API Key。"
        f"请前往左侧 <b>设置</b> 页完成配置后再使用相关功能。</div>",
        unsafe_allow_html=True,
    )


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


def handle_config_error(exc: Exception) -> None:
    if isinstance(exc, ConfigurationError):
        st.warning(str(exc))
    else:
        st.error(f"⚠️ 操作失败: {exc}")
