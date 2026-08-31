"""config.local.yaml 读写与校验。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.app_settings import AppSettings, LLMConfig, ModelConfig, PathConfig, RetrievalConfig

CONFIG_FILENAME = "config.local.yaml"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / CONFIG_FILENAME


class ConfigurationError(Exception):
    """配置缺失或无效。"""


def mask_api_key(key: str) -> str:
    key = (key or "").strip()
    if len(key) <= 8:
        return "****" if key else ""
    return f"{key[:3]}****{key[-4:]}"


def _model_from_dict(data: Optional[Dict[str, Any]], default: ModelConfig) -> ModelConfig:
    if not data:
        return default
    return ModelConfig(
        provider=str(data.get("provider", default.provider)),
        api_key=str(data.get("api_key", default.api_key)),
        base_url=str(data.get("base_url", default.base_url)),
        model=str(data.get("model", default.model)),
    )


def _llm_from_dict(data: Optional[Dict[str, Any]], default: LLMConfig) -> LLMConfig:
    if not data:
        return default
    return LLMConfig(
        provider=str(data.get("provider", default.provider)),
        api_key=str(data.get("api_key", default.api_key)),
        base_url=str(data.get("base_url", default.base_url)),
        model=str(data.get("model", default.model)),
        temperature=float(data.get("temperature", default.temperature)),
    )


def settings_from_dict(data: Dict[str, Any]) -> AppSettings:
    defaults = AppSettings()
    paths_data = data.get("paths") or {}
    retrieval_data = data.get("retrieval") or {}
    return AppSettings(
        version=int(data.get("version", 1)),
        embedding=_model_from_dict(data.get("embedding"), defaults.embedding),
        llm=_llm_from_dict(data.get("llm"), defaults.llm),
        paths=PathConfig(
            upload_dir=str(paths_data.get("upload_dir", defaults.paths.upload_dir)),
            chroma_db_dir=str(
                paths_data.get("chroma_db_dir", defaults.paths.chroma_db_dir)
            ),
        ),
        retrieval=RetrievalConfig(
            multi_query_n=int(
                retrieval_data.get("multi_query_n", defaults.retrieval.multi_query_n)
            ),
            hybrid_fetch_k=int(
                retrieval_data.get("hybrid_fetch_k", defaults.retrieval.hybrid_fetch_k)
            ),
            rrf_k=int(retrieval_data.get("rrf_k", defaults.retrieval.rrf_k)),
            rerank_model=str(
                retrieval_data.get("rerank_model", defaults.retrieval.rerank_model)
            ),
            enable_multi_query=bool(
                retrieval_data.get(
                    "enable_multi_query", defaults.retrieval.enable_multi_query
                )
            ),
            enable_rerank=bool(
                retrieval_data.get("enable_rerank", defaults.retrieval.enable_rerank)
            ),
        ),
    )


def settings_to_dict(settings: AppSettings) -> Dict[str, Any]:
    return {
        "version": settings.version,
        "embedding": {
            "provider": settings.embedding.provider,
            "api_key": settings.embedding.api_key,
            "base_url": settings.embedding.base_url,
            "model": settings.embedding.model,
        },
        "llm": {
            "provider": settings.llm.provider,
            "api_key": settings.llm.api_key,
            "base_url": settings.llm.base_url,
            "model": settings.llm.model,
            "temperature": settings.llm.temperature,
        },
        "paths": {
            "upload_dir": settings.paths.upload_dir,
            "chroma_db_dir": settings.paths.chroma_db_dir,
        },
        "retrieval": {
            "multi_query_n": settings.retrieval.multi_query_n,
            "hybrid_fetch_k": settings.retrieval.hybrid_fetch_k,
            "rrf_k": settings.retrieval.rrf_k,
            "rerank_model": settings.retrieval.rerank_model,
            "enable_multi_query": settings.retrieval.enable_multi_query,
            "enable_rerank": settings.retrieval.enable_rerank,
        },
    }


def load_local_settings(path: Optional[Path] = None) -> Optional[AppSettings]:
    target = path or CONFIG_PATH
    if not target.exists():
        return None
    with open(target, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"配置文件格式错误: {target}")
    return settings_from_dict(raw)


def save_local_settings(settings: AppSettings, path: Optional[Path] = None) -> None:
    target = path or CONFIG_PATH
    payload = settings_to_dict(settings)
    with open(target, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


def ensure_data_dirs(settings: AppSettings) -> None:
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.chroma_db_dir, exist_ok=True)
