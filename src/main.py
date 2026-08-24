import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import List

from src.config import settings
from src.services import kb_service

app = FastAPI(
    title="个人知识库 API",
    description="基于 LangChain + Chroma + FastAPI 的个人 RAG 知识库后端服务",
    version="1.0.0"
)

# Pydantic 模型定义
class QueryRequest(BaseModel):
    question: str = Field(..., example="什么是 LangChain？")
    top_k: int = Field(default=3, ge=1, le=10, example="3")

class SourceItem(BaseModel):
    filename: str
    page: int
    content_snippet: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]

# 接口路由定义
@app.post("/upload", summary="上传文档构建索引")
async def upload_document(file: UploadFile = File(...)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="目前仅支持上传 .pdf 和 .txt 文件")

    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)

    # 写入到本地
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_count = kb_service.add_documents(file_path, file.filename)
        return {
            "status": "success",
            "filename": file.filename,
            "chunks_created": chunks_count,
            "message": "文档解析并向量化入库成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文件失败: {str(e)}")

@app.post("/query", response_model=QueryResponse, summary="检索知识库并生成回答")
async def query_knowledge_base(request: QueryRequest):
    try:
        result = kb_service.query(question=request.question, top_k=request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {str(e)}")

@app.get("/documents", summary="获取已上传文档列表")
async def get_documents():
    docs = kb_service.list_documents()
    return {
        "total_documents": len(docs),
        "documents": docs
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)