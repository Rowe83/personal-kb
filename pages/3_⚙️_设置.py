import streamlit as st

from src.app_settings import AppSettings
from src.config_loader import load_app_settings
from src.connection_test import test_embedding_connection, test_llm_connection
from src.provider_templates import EMBEDDING_TEMPLATES, LLM_TEMPLATES, get_template
from src.settings_store import load_local_settings, save_local_settings
from ui.shared import apply_css, apply_page_config, invalidate_service_cache, render_sidebar_brand

apply_page_config()
apply_css()
render_sidebar_brand()

st.title("⚙️ 模型设置")
st.caption("配置向量模型与大模型 API，保存至本地 `config.local.yaml`（不会提交到 Git）")

current = load_local_settings() or load_app_settings()

if "draft_settings" not in st.session_state:
    st.session_state.draft_settings = current

draft: AppSettings = st.session_state.draft_settings

st.subheader("Embedding（向量模型）")
emb_labels = {t["id"]: t["label"] for t in EMBEDDING_TEMPLATES}
emb_ids = list(emb_labels.keys())
emb_idx = emb_ids.index(draft.embedding.provider) if draft.embedding.provider in emb_ids else 0
emb_choice = st.selectbox(
    "供应商模板",
    options=emb_ids,
    format_func=lambda x: emb_labels[x],
    index=emb_idx,
    key="emb_provider",
)
if emb_choice != draft.embedding.provider:
    tpl = get_template("embedding", emb_choice)
    draft.embedding.provider = emb_choice
    draft.embedding.base_url = tpl["base_url"]
    draft.embedding.model = tpl["default_model"]

draft.embedding.api_key = st.text_input(
    "Embedding API Key",
    value=draft.embedding.api_key,
    type="password",
    placeholder="sk-...",
)
draft.embedding.base_url = st.text_input(
    "Embedding Base URL",
    value=draft.embedding.base_url,
)
draft.embedding.model = st.text_input(
    "Embedding 模型名",
    value=draft.embedding.model,
)

col_emb, _ = st.columns([1, 1])
with col_emb:
    if st.button("🔌 测试 Embedding 连接", use_container_width=True):
        result = test_embedding_connection(draft)
        if result.ok:
            st.success(result.message)
        else:
            st.error(result.message)
            if result.detail:
                st.caption(result.detail)

st.markdown("---")
st.subheader("LLM（大模型）")
llm_labels = {t["id"]: t["label"] for t in LLM_TEMPLATES}
llm_ids = list(llm_labels.keys())
llm_idx = llm_ids.index(draft.llm.provider) if draft.llm.provider in llm_ids else 0
llm_choice = st.selectbox(
    "供应商模板",
    options=llm_ids,
    format_func=lambda x: llm_labels[x],
    index=llm_idx,
    key="llm_provider",
)
if llm_choice != draft.llm.provider:
    tpl = get_template("llm", llm_choice)
    draft.llm.provider = llm_choice
    draft.llm.base_url = tpl["base_url"]
    draft.llm.model = tpl["default_model"]

draft.llm.api_key = st.text_input(
    "LLM API Key",
    value=draft.llm.api_key,
    type="password",
    placeholder="sk-...",
)
draft.llm.base_url = st.text_input("LLM Base URL", value=draft.llm.base_url)
draft.llm.model = st.text_input("LLM 模型名", value=draft.llm.model)
draft.llm.temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=float(draft.llm.temperature),
    step=0.1,
)

col_llm, _ = st.columns([1, 1])
with col_llm:
    if st.button("🔌 测试 LLM 连接", use_container_width=True):
        result = test_llm_connection(draft)
        if result.ok:
            st.success(result.message)
        else:
            st.error(result.message)
            if result.detail:
                st.caption(result.detail)

st.markdown("---")
save_col, reset_col = st.columns(2)
with save_col:
    if st.button("💾 保存配置", type="primary", use_container_width=True):
        try:
            save_local_settings(draft)
            invalidate_service_cache()
            st.session_state.draft_settings = draft
            st.success("配置已保存。请重新提问或刷新页面使新配置生效。")
        except Exception as exc:
            st.error(f"保存失败: {exc}")

with reset_col:
    if st.button("↩️ 重新加载", use_container_width=True):
        st.session_state.draft_settings = load_local_settings() or load_app_settings()
        st.rerun()

st.markdown("---")
st.markdown(
    """
**OpenAI 兼容示例**
- Ollama: `http://127.0.0.1:11434/v1`
- 其他网关：填写兼容 OpenAI API 的 Base URL 即可

配置文件路径：`config.local.yaml`（已加入 `.gitignore`）
"""
)
