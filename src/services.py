import os
import fitz
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from src.config import settings

class KnowledgeBaseService:
    def __init__(self):
        # 1. 初始化 Embeddings 模型
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.ZHIPU_API_KEY,
            base_url=settings.ZHIPU_BASE_URL,
        )

        # 2. 初始化 LLM 模型
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )
        
        # 3. 初始化 Chroma 数据库
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DB_DIR,
        )
        
        # 4. 中英文优化切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""]
        )

    def parse_pdf(self, file_path: str, filename: str) -> List[Document]:
        """使用 PyMuPDF 解析 PDF 文件按页提取文本"""
        doc = fitz.open(file_path)
        documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"filename": filename, "page": page_num + 1},
                    )
                )
        doc.close()
        return documents

    def add_documents(self, file_path: str, filename: str) -> int:
        """解析文件、切片并保存至 Chroma 向量库"""
        if filename.endswith(".pdf"):
            raw_docs = self.parse_pdf(file_path, filename)
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            raw_docs = [Document(page_content=content, metadata={"filename": filename, "page": 1})]

        # 切片处理
        chunks = self.text_splitter.split_documents(raw_docs)

        # 保存至 Chroma 向量库
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def query(self, question: str, top_k: int = 3) -> Dict:
        """检索并生成回答与出处引用"""
        retrieval = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retrieval.invoke(question)

        # 拼接参考上下文
        context_str = "\n\n".join([
            f"出处 {doc.metadata['filename']} 第 {doc.metadata['page']} 页: {doc.page_content}"
            for doc in retrieved_docs
        ])

        # 构建提示词模板
        prompt_str = (
            "你是一个严谨的个人知识库助手。请严格根据给出的 [参考文档] 回答用户的 [提问]。\n"
            "如果文档中没有提及，请明确回答'知识库中未找到相关内容'，切勿编造。\n\n"
            "[参考文档]:\n{context}\n\n"
            "[提问]: {question}"
        )
        prompt = ChatPromptTemplate.from_template(prompt_str)
        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({"context": context_str, "question": question})

        # 构建来源引用列表
        sources = [
            {
                "filename": doc.metadata.get("filename", "未知"),
                "page": doc.metadata.get("page", 1),
                "content_snippet": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            }
            for doc in retrieved_docs
        ]

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    def list_documents(self) -> List[Dict]:
        """获取已有向量数据库的统计信息"""
        results = self.vectorstore.get(include=["metadatas"])

        file_stats = {}
        if results and results.get("metadatas"):
            for meta in results["metadatas"]:
                if not meta:
                    continue
                fname = meta.get("filename", "未命名文件")
                file_stats[fname] = file_stats.get(fname, 0) + 1

        return [{"filename": fname, "chunk_count": count} for fname, count in file_stats.items()]

# 实例化全局服务单例
kb_service = KnowledgeBaseService()