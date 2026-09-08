import streamlit as st

from src.config_loader import load_app_settings
from src.service_factory import get_kb_service, reset_kb_service
from src.settings_store import ConfigurationError

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB

KB_ACCENT = "#1D4E89"

CUSTOM_CSS = f"""
<style>
    :root {{
        --kb-bg: #f4f6f8;
        --kb-surface: #ffffff;
        --kb-text: #1a1d21;
        --kb-muted: #5c6570;
        --kb-accent: {KB_ACCENT};
        --kb-border: #e2e6ea;
    }}
    .stApp {{ background-color: var(--kb-bg); color: var(--kb-text); }}
    [data-testid="stSidebar"] {{
        background-color: var(--kb-surface);
        border-right: 1px solid var(--kb-border);
    }}
    .kb-brand {{
        padding: 0.25rem 0 1rem 0;
        border-bottom: 1px solid var(--kb-border);
        margin-bottom: 1rem;
    }}
    .kb-brand-title {{
        font-size: 1.15rem; font-weight: 700; color: var(--kb-text); margin: 0;
    }}
    .kb-brand-sub {{
        font-size: 0.8rem; color: var(--kb-muted); margin: 0.25rem 0 0 0;
    }}
    .stChatMessage {{ border-radius: 12px; padding: 12px; margin-bottom: 10px; }}
    /* 用户消息：醒目底色，便于与助手区分 */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessageContent"][aria-label="Chat message from user"] {{
        background-color: #d6e6f7 !important;
        border: 1px solid #9bb8d9 !important;
        border-radius: 12px !important;
        padding: 0.65rem 0.9rem !important;
        box-shadow: 0 1px 2px rgba(29, 78, 137, 0.08);
    }}
    .source-box {{
        background-color: #eef2f6;
        border-left: 4px solid var(--kb-accent);
        padding: 8px 12px; margin-top: 8px; border-radius: 4px;
        font-size: 0.85rem; color: var(--kb-muted);
    }}
    .stButton>button {{
        border-radius: 8px; font-weight: 500;
        border-color: var(--kb-accent);
    }}
    .stButton>button[kind="primary"],
    .stButton>button[data-testid="baseButton-primary"] {{
        background-color: var(--kb-accent);
        border-color: var(--kb-accent);
    }}
    .config-banner {{
        background: #fff8e6;
        border: 1px solid #f0e0a8;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 16px;
        color: #664d03;
    }}
    .kb-stat {{
        background: var(--kb-surface);
        border: 1px solid var(--kb-border);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }}
    .kb-stat-label {{ color: var(--kb-muted); font-size: 0.85rem; }}
    .kb-stat-value {{ color: var(--kb-text); font-size: 1.25rem; font-weight: 600; }}
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


def render_sidebar_brand():
    with st.sidebar:
        st.markdown(
            "<div class='kb-brand'>"
            "<p class='kb-brand-title'>Personal KB</p>"
            "<p class='kb-brand-sub'>本地开源个人知识库</p>"
            "</div>",
            unsafe_allow_html=True,
        )


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
