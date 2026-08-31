import os
from dotenv import load_dotenv

from src.config_loader import load_app_settings
from src.settings_store import ensure_data_dirs

load_dotenv()

# 向后兼容：模块级 settings 代理，不再在 import 时强制校验 API Key
settings = load_app_settings()
ensure_data_dirs(settings)

# 旧代码可能引用的别名（逐步迁移到 app_settings）
UPLOAD_DIR = settings.upload_dir
CHROMA_DB_DIR = settings.chroma_db_dir
EMBEDDING_MODEL = settings.embedding.model
ZHIPU_API_KEY = settings.embedding.api_key
ZHIPU_BASE_URL = settings.embedding.base_url
DEEPSEEK_API_KEY = settings.llm.api_key
DEEPSEEK_BASE_URL = settings.llm.base_url
LLM_MODEL = settings.llm.model
MULTI_QUERY_N = settings.retrieval.multi_query_n
HYBRID_FETCH_K = settings.retrieval.hybrid_fetch_k
RRF_K = settings.retrieval.rrf_k
RERANK_MODEL = settings.retrieval.rerank_model
ENABLE_MULTI_QUERY = settings.retrieval.enable_multi_query
ENABLE_RERANK = settings.retrieval.enable_rerank
