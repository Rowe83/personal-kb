"""Question intent helpers beyond catalog listing."""

from __future__ import annotations

import re
from typing import List

_INTERFACE_COUNT_PATTERNS = [
    r"多少(?:个)?接口",
    r"几个接口",
    r"有多少接口",
    r"共(?:有)?多少(?:个)?(?:接口|条目|功能)",
    r"接口(?:一共|总共|共)有几",
    r"整理(?:了|出)?多少",
]

_AMBIGUOUS_DOC_PATTERNS = [
    r"这个文档",
    r"该文档",
    r"这份文档",
    r"此文档",
    r"这篇文档",
]


def is_interface_count_question(question: str) -> bool:
    q = (question or "").strip()
    return any(re.search(p, q) for p in _INTERFACE_COUNT_PATTERNS)


def is_ambiguous_document_reference(question: str) -> bool:
    q = (question or "").strip()
    return any(re.search(p, q) for p in _AMBIGUOUS_DOC_PATTERNS)


def format_interface_count_answer(filename: str, api_lines: List[str]) -> str:
    count = len(api_lines)
    header = f"根据知识库中的文档《{filename}》，共整理了 {count} 个接口"
    if count == 0:
        return f"{header}。"
    lines = [header + "，具体如下：", ""]
    for i, line in enumerate(api_lines, 1):
        if "-" in line:
            code, desc = line.split("-", 1)
            lines.append(f"{i}. {code}：{desc.strip()}")
        else:
            lines.append(f"{i}. {line}")
    return "\n".join(lines)
