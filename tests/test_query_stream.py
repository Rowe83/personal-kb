from typing import Iterator, List, Dict, Tuple
from unittest.mock import MagicMock, patch
import pytest

from langchain_core.documents import Document


def test_query_multi_turn_stream_yields_tokens_and_sources():
    from src.services import KnowledgeBaseService

    service = KnowledgeBaseService.__new__(KnowledgeBaseService)
    service._convert_chat_history = MagicMock(return_value=[])
    service.rephrase_chain = MagicMock()

    doc = Document(
        page_content="LangChain 是一个 LLM 应用框架。",
        metadata={"filename": "demo.txt", "page": 1},
    )
    retriever = MagicMock()
    retriever.invoke.return_value = [doc]
    service.vectorstore = MagicMock()
    service.vectorstore.as_retriever.return_value = retriever

    def fake_stream(_inputs):
        yield "你好"
        yield "世界"

    service.qa_chain = MagicMock()
    service.qa_chain.stream.side_effect = fake_stream

    token_iter, sources = service.query_multi_turn_stream(
        question="什么是 LangChain？", history=[], top_k=3
    )

    assert isinstance(token_iter, Iterator) or hasattr(token_iter, "__iter__")
    assert "".join(list(token_iter)) == "你好世界"
    assert len(sources) == 1
    assert sources[0]["filename"] == "demo.txt"
    assert sources[0]["page"] == 1
    assert "LangChain" in sources[0]["content_snippet"]
    service.qa_chain.stream.assert_called_once()
    service.rephrase_chain.invoke.assert_not_called()


def test_query_multi_turn_stream_rephrases_when_history_exists():
    from src.services import KnowledgeBaseService
    from langchain_core.messages import HumanMessage, AIMessage

    service = KnowledgeBaseService.__new__(KnowledgeBaseService)
    history = [
        {"role": "user", "content": "介绍一下知识库"},
        {"role": "assistant", "content": "这是个人 RAG"},
    ]
    chat_history = [
        HumanMessage(content="介绍一下知识库"),
        AIMessage(content="这是个人 RAG"),
    ]
    service._convert_chat_history = MagicMock(return_value=chat_history)
    service.rephrase_chain = MagicMock()
    service.rephrase_chain.invoke.return_value = "个人知识库支持什么格式？"

    retriever = MagicMock()
    retriever.invoke.return_value = []
    service.vectorstore = MagicMock()
    service.vectorstore.as_retriever.return_value = retriever
    service.qa_chain = MagicMock()
    service.qa_chain.stream.return_value = iter(["无相关内容"])

    token_iter, sources = service.query_multi_turn_stream(
        question="它支持什么格式？", history=history, top_k=2
    )
    assert list(token_iter) == ["无相关内容"]
    assert sources == []
    service.rephrase_chain.invoke.assert_called_once()
    retriever.invoke.assert_called_once_with("个人知识库支持什么格式？")
