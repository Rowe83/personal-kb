"""Catalog / inventory question short-circuit helpers."""

from __future__ import annotations

import re
from typing import Dict, List

_CATALOG_PATTERNS = [
    r"几个文档",
    r"多少(?:个)?文档",
    r"有哪些文件",
    r"有哪些文档",
    r"文档列表",
    r"文件列表",
    r"上传了什么",
    r"知识库里有哪些",
    r"一共有几",
]


def is_catalog_question(question: str) -> bool:
    q = (question or "").strip()
    return any(re.search(p, q) for p in _CATALOG_PATTERNS)


def format_catalog_answer(docs: List[Dict]) -> str:
    if not docs:
        return "知识库中当前没有已入库文档。"
    lines = [f"知识库中共有 {len(docs)} 个文档："]
    for i, d in enumerate(docs, 1):
        lines.append(f"{i}. {d['filename']}（{d['chunk_count']} 个分块）")
    return "\n".join(lines)
