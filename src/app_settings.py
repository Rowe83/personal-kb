"""应用配置数据模型。"""

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    provider: str = "custom"
    api_key: str = ""
    base_url: str = ""
    model: str = ""


@dataclass
class LLMConfig(ModelConfig):
    temperature: float = 0.3


@dataclass
class PathConfig:
    upload_dir: str = "./uploads"
    chroma_db_dir: str = "./chroma_db"


@dataclass
class RetrievalConfig:
    multi_query_n: int = 3
    hybrid_fetch_k: int = 10
    rrf_k: int = 60
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_multi_query: bool = True
    enable_rerank: bool = True


@dataclass
class AppSettings:
    version: int = 1
    embedding: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @property
    def upload_dir(self) -> str:
        return self.paths.upload_dir

    @property
    def chroma_db_dir(self) -> str:
        return self.paths.chroma_db_dir

    def is_embedding_configured(self) -> bool:
        return bool(self.embedding.api_key.strip() and self.embedding.base_url.strip())

    def is_llm_configured(self) -> bool:
        return bool(self.llm.api_key.strip() and self.llm.base_url.strip())

    def is_fully_configured(self) -> bool:
        return self.is_embedding_configured() and self.is_llm_configured()
