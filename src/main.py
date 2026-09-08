import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
from typing import List

from src.config_loader import load_app_settings
from src.ingest.parsers import SUPPORTED_EXTENSIONS, normalize_ext
from src.service_factory import get_kb_service
from src.settings_store import ConfigurationError

app = FastAPI(
    title="个人知识库系统 API",
    description="基于 LangChain + Chroma 的开源本地 RAG 知识库",
    version="2.1.0",
)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: user 或 assistant")
    content: str = Field(..., description="消息内容")


class MultiTurnQueryRequest(BaseModel):
    question: str = Field(..., example="什么是 LangChain？")
    history: List[ChatMessage] = Field(default=[], description="对话历史")
    top_k: int = Field(default=3, ge=1, le=10, example="3")


class SourceItem(BaseModel):
    filename: str
    page: int
    content_snippet: str


class QueryResponse(BaseModel):
    question: str
    standalone_question: str
    answer: str
    sources: List[SourceItem]


def _config_error_to_http(exc: ConfigurationError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
    )


@app.get("/health")
async def health():
    settings = load_app_settings()
    return {
        "status": "ok",
        "embedding_configured": settings.is_embedding_configured(),
        "llm_configured": settings.is_llm_configured(),
    }


@app.post("/upload", summary="上传文档构建索引")
async def upload_document(file: UploadFile = File(...)):
    settings = load_app_settings()
    if not settings.is_embedding_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="请先配置 Embedding API Key（config.local.yaml 或环境变量）",
        )

    ext = normalize_ext(file.filename or "")
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="目前仅支持上传 .pdf / .txt / .md / .xlsx 文件",
        )

    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件大小超过限制 (最大 {MAX_FILE_SIZE // (1024 * 1024):.2f}MB)。单文件上限为 20MB",
        )

    file_path = os.path.join(settings.upload_dir, file.filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"写入文件失败: {str(e)}",
        )

    try:
        kb_service = get_kb_service(force_reload=True)
        result = kb_service.add_documents(file_path, file.filename, overwrite=True)
        message = "文档解析并向量化入库成功"
        if result.get("overwritten"):
            message = (
                f"已覆盖同名文档：删除旧分块 {result['deleted_chunks']} 个，"
                f"新建 {result['chunks_created']} 个"
            )
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": result["chunks_created"],
            "overwritten": result.get("overwritten", False),
            "deleted_chunks": result.get("deleted_chunks", 0),
            "message": message,
        }
    except ConfigurationError as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise _config_error_to_http(exc) from exc
    except ValueError as ve:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"文件解析失败: {str(ve)}",
        )
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"向量入库异常: {str(e)}",
        )


@app.post("/query", response_model=QueryResponse, summary="检索知识库并生成回答")
async def query_knowledge_base(request: MultiTurnQueryRequest):
    settings = load_app_settings()
    if not settings.is_llm_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="请先配置 LLM API Key（config.local.yaml 或环境变量）",
        )

    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="提问不能为空"
        )

    try:
        kb_service = get_kb_service(force_reload=True)
        history_dicts = [msg.model_dump() for msg in request.history]
        result = kb_service.query_multi_turn(
            question=request.question, history=history_dicts, top_k=request.top_k
        )
        return result
    except ConfigurationError as exc:
        raise _config_error_to_http(exc) from exc
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答服务处理失败: {str(e)}",
        )


@app.get("/documents", summary="获取已上传文档列表")
async def get_documents():
    kb_service = get_kb_service()
    docs = kb_service.list_documents()
    return {"total_documents": len(docs), "documents": docs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
