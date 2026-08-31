"""Lazy factory for KnowledgeBaseService."""

from __future__ import annotations

from typing import Optional

from src.config_loader import load_app_settings
from src.services import KnowledgeBaseService

_service: Optional[KnowledgeBaseService] = None


def get_kb_service(force_reload: bool = False) -> KnowledgeBaseService:
    global _service
    if _service is None or force_reload:
        _service = KnowledgeBaseService(load_app_settings())
    return _service


def reset_kb_service() -> None:
    global _service
    _service = None
