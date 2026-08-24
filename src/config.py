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

settings = Settings()

if not settings.ZHIPU_API_KEY:
    raise ValueError("缺少智谱 API Key：请在 .env 设置 ZHIPUAI_API_KEY 或 ZHIPU_API_KEY")
if not settings.DEEPSEEK_API_KEY:
    raise ValueError("缺少 DeepSeek API Key：请在 .env 设置 DEEPSEEK_API_KEY")

# 确保必要的本地存储目录存在
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_DB_DIR, exist_ok=True)
