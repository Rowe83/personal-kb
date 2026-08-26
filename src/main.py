import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from pydantic import BaseModel, Field
from typing import List

from src.config import settings
from src.services import kb_service

app = FastAPI(
    title="个人知识库系统（多轮强化版） API",
    description="基于 LangChain + Chroma + FastAPI 的个人 RAG 知识库后端服务",
    version="2.0.0",
)

# 限制最大文件上传为 20MB
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB


# Pydantic 模型定义
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


# 接口路由定义
@app.post("/upload", summary="上传文档构建索引")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="目前仅支持上传 .pdf 和 .txt 文件",
        )

    # 校验文件大小
    file.file.seek(0, os.SEEK_END)
    file_size = file.file.tell()
    file.file.seek(0)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"文件大小超过限制 (最大 {MAX_FILE_SIZE // (1024 * 1024):.2f}MB)。单文件上限为 20MB",
        )

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

    # 写入到本地
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"写入文件失败: {str(e)}",
        )

    # 解析与向量化
    try:
        chunks_count = kb_service.add_documents(file_path, file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": chunks_count,
            "message": "文档解析并向量化入库成功",
        }
    except ValueError as ve:
        # 清理非法或损坏的本地残余文件
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
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="提问不能为空"
        )

    try:
        history_dicts = [msg.model_dump() for msg in request.history]
        result = kb_service.query_multi_turn(
            question=request.question, history=history_dicts, top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"问答服务处理失败: {str(e)}",
        )


@app.get("/documents", summary="获取已上传文档列表")
async def get_documents():
    docs = kb_service.list_documents()
    return {"total_documents": len(docs), "documents": docs}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
