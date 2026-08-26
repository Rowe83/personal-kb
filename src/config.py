import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # 路径配置
    UPLOAD_DIR: str = "./uploads"
    CHROMA_DB_DIR: str = "./chroma_db"

    # 智谱 Embeddings 配置（兼容 ZHIPUAI_API_KEY / ZHIPU_API_KEY）
    ZHIPU_API_KEY: str = os.getenv("ZHIPUAI_API_KEY")
    ZHIPU_BASE_URL: str = os.getenv("ZHIPU_BASE_URL")
    EMBEDDING_MODEL: str = "embedding-3"

    # DEEPSEEK LLM 配置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL")
    LLM_MODEL: str = "deepseek-v4-flash"

    # RAG retrieval
    MULTI_QUERY_N: int = int(os.getenv("MULTI_QUERY_N", "3"))
    HYBRID_FETCH_K: int = int(os.getenv("HYBRID_FETCH_K", "10"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    RERANK_MODEL: str = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    ENABLE_MULTI_QUERY: bool = os.getenv("ENABLE_MULTI_QUERY", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    ENABLE_RERANK: bool = os.getenv("ENABLE_RERANK", "true").lower() in (
        "1",
        "true",
        "yes",
    )

settings = Settings()

if not settings.ZHIPU_API_KEY:
    raise ValueError("缺少智谱 API Key：请在 .env 设置 ZHIPUAI_API_KEY 或 ZHIPU_API_KEY")
if not settings.DEEPSEEK_API_KEY:
    raise ValueError("缺少 DeepSeek API Key：请在 .env 设置 DEEPSEEK_API_KEY")

# 确保必要的本地存储目录存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
