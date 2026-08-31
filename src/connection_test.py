"""Embedding / LLM 连通性测试。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.app_settings import AppSettings


@dataclass
class ConnectionTestResult:
    ok: bool
    message: str
    detail: Optional[str] = None


def test_embedding_connection(settings: AppSettings) -> ConnectionTestResult:
    if not settings.is_embedding_configured():
        return ConnectionTestResult(False, "Embedding 未配置完整（需要 API Key 与 Base URL）")
    try:
        client = OpenAIEmbeddings(
            model=settings.embedding.model,
            api_key=settings.embedding.api_key,
            base_url=settings.embedding.base_url,
        )
        vector = client.embed_query("连接测试")
        dim = len(vector) if vector else 0
        return ConnectionTestResult(True, f"Embedding 连接成功（向量维度 {dim}）")
    except Exception as exc:
        return ConnectionTestResult(False, "Embedding 连接失败", str(exc))


def test_llm_connection(settings: AppSettings) -> ConnectionTestResult:
    if not settings.is_llm_configured():
        return ConnectionTestResult(False, "LLM 未配置完整（需要 API Key 与 Base URL）")
    try:
        client = ChatOpenAI(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            temperature=0,
            max_tokens=16,
        )
        reply = client.invoke("回复 OK")
        text = reply.content if hasattr(reply, "content") else str(reply)
        preview = (text or "")[:80]
        return ConnectionTestResult(True, f"LLM 连接成功：{preview}")
    except Exception as exc:
        return ConnectionTestResult(False, "LLM 连接失败", str(exc))
