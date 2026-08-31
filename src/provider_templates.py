"""内置模型供应商模板。"""

from typing import Dict, List, TypedDict


class ProviderTemplate(TypedDict):
    id: str
    label: str
    base_url: str
    default_model: str


EMBEDDING_TEMPLATES: List[ProviderTemplate] = [
    {
        "id": "zhipu",
        "label": "智谱 AI",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "embedding-3",
    },
    {
        "id": "openai_compatible",
        "label": "OpenAI 兼容",
        "base_url": "https://api.openai.com/v1",
        "default_model": "text-embedding-3-small",
    },
    {
        "id": "custom",
        "label": "自定义",
        "base_url": "",
        "default_model": "",
    },
]

LLM_TEMPLATES: List[ProviderTemplate] = [
    {
        "id": "deepseek",
        "label": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
    },
    {
        "id": "zhipu",
        "label": "智谱 AI",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "glm-4-flash",
    },
    {
        "id": "openai_compatible",
        "label": "OpenAI 兼容",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
    },
    {
        "id": "custom",
        "label": "自定义",
        "base_url": "",
        "default_model": "",
    },
]

TEMPLATE_MAP: Dict[str, Dict[str, ProviderTemplate]] = {
    "embedding": {t["id"]: t for t in EMBEDDING_TEMPLATES},
    "llm": {t["id"]: t for t in LLM_TEMPLATES},
}


def get_template(kind: str, provider_id: str) -> ProviderTemplate:
    return TEMPLATE_MAP[kind].get(provider_id, TEMPLATE_MAP[kind]["custom"])
