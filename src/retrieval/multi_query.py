"""Multi-query expansion for retrieval."""

from __future__ import annotations

from typing import Any, List


_PROMPT = (
    "请根据下面的问题，生成 {n} 条中文改写，用于检索同一知识库。"
    "改写侧重：同义表述、关键词、接口号/文件名。"
    "每行一条，不要编号，不要解释。\n\n问题：{question}"
)


def _content(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    return str(response)


def generate_multi_queries(
    standalone_question: str, llm: Any, n: int = 3
) -> List[str]:
    q = (standalone_question or "").strip()
    if not q:
        return []
    if llm is None or n <= 0:
        return [q]
    try:
        raw = _content(llm.invoke(_PROMPT.format(n=n, question=q)))
    except Exception:
        return [q]
    seen = {q}
    out = [q]
    for line in raw.splitlines():
        line = line.strip().lstrip("0123456789.-、)） ").strip()
        if not line or line in seen:
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= n + 1:
            break
    return out
