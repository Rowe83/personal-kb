"""合并默认配置、本地 YAML 与环境变量。"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from src.app_settings import AppSettings, LLMConfig, ModelConfig
from src.settings_store import ensure_data_dirs, load_local_settings

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def _apply_env_fallback(settings: AppSettings) -> AppSettings:
    """兼容旧 .env：仅当 YAML 未填写时回填。"""
    if not settings.embedding.api_key:
        settings.embedding.api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv(
            "ZHIPU_API_KEY", ""
        )
    if not settings.embedding.base_url:
        settings.embedding.base_url = os.getenv("ZHIPU_BASE_URL", "")
    if not settings.embedding.model:
        settings.embedding.model = os.getenv("EMBEDDING_MODEL", "embedding-3")
    if settings.embedding.api_key and not settings.embedding.provider:
        settings.embedding.provider = "zhipu"

    if not settings.llm.api_key:
        settings.llm.api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not settings.llm.base_url:
        settings.llm.base_url = os.getenv("DEEPSEEK_BASE_URL", "")
    if not settings.llm.model:
        settings.llm.model = os.getenv("LLM_MODEL", "deepseek-chat")
    if settings.llm.api_key and settings.llm.provider == "custom":
        settings.llm.provider = "deepseek"

    settings.retrieval.multi_query_n = int(
        os.getenv("MULTI_QUERY_N", settings.retrieval.multi_query_n)
    )
    settings.retrieval.hybrid_fetch_k = int(
        os.getenv("HYBRID_FETCH_K", settings.retrieval.hybrid_fetch_k)
    )
    settings.retrieval.rrf_k = int(os.getenv("RRF_K", settings.retrieval.rrf_k))
    settings.retrieval.rerank_model = os.getenv(
        "RERANK_MODEL", settings.retrieval.rerank_model
    )
    settings.retrieval.enable_multi_query = _env_bool(
        "ENABLE_MULTI_QUERY", settings.retrieval.enable_multi_query
    )
    settings.retrieval.enable_rerank = _env_bool(
        "ENABLE_RERANK", settings.retrieval.enable_rerank
    )
    return settings


def load_app_settings() -> AppSettings:
    local = load_local_settings()
    settings = local if local is not None else AppSettings()
    settings = _apply_env_fallback(settings)
    ensure_data_dirs(settings)
    return settings


def default_settings_with_template(
    embedding_provider: str = "zhipu",
    llm_provider: str = "deepseek",
) -> AppSettings:
    from src.provider_templates import get_template

    emb_tpl = get_template("embedding", embedding_provider)
    llm_tpl = get_template("llm", llm_provider)
    settings = AppSettings(
        embedding=ModelConfig(
            provider=embedding_provider,
            base_url=emb_tpl["base_url"],
            model=emb_tpl["default_model"],
        ),
        llm=LLMConfig(
            provider=llm_provider,
            base_url=llm_tpl["base_url"],
            model=llm_tpl["default_model"],
            temperature=0.3,
        ),
    )
    return _apply_env_fallback(settings)
