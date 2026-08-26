import os
import fitz  # PyMuPDF 库
from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from src.config import settings


class KnowledgeBaseService:
    def __init__(self):
        # 初始化 Embeddings 模型
        self.embeddings = OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.ZHIPU_API_KEY,
            base_url=settings.ZHIPU_BASE_URL,
        )

        # 初始化 LLM 模型
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
            temperature=0.0,
        )

        # 初始化 Chroma 数据库
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_DB_DIR,
        )

        # 中英文优化切分器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        )

        # 追问重写 Prompt (将带有代词的追问转化为独立查询)
        rephrase_prompt_str = (
            "给定一段对话历史和用户的最新提问，如果最新提问依赖于对话历史（例如包含'它'、'上述'、'后者'等指代），"
            "请将其重写为一个无须结合上下文就能独立理解的全新问题。"
            "如果最新提问本身就是独立的，请原样返回，不要做多余回答。"
        )
        self.rephrase_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rephrase_prompt_str),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.rephrase_chain = self.rephrase_prompt | self.llm | StrOutputParser()

        # 最终回答 Prompt
        qa_prompt_str = (
            "你是一个严谨的个人知识库助手。请严格根据给出的 [参考文档] 回答用户的 [提问]。\n"
            "如果文档中没有提及，请明确回答'知识库中未找到相关内容'，切勿编造。\n\n"
            "[参考文档]:\n{context}\n"
        )
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_prompt_str),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.qa_chain = self.qa_prompt | self.llm | StrOutputParser()

    def parse_file(self, file_path: str, filename: str) -> List[Document]:
        """校验并解析文件，包含空文档与非法格式拦截"""
        documents = []
        file_size = os.path.getsize(file_path)

        if file_size == 0:
            raise ValueError("上传的文件为空文件 (0字节)")

        if filename.endswith(".pdf"):
            try:
                doc = fitz.open(file_path)
                if doc.page_count == 0:
                    raise ValueError("PDF 文件未包含任何页面")

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
            except Exception as e:
                raise ValueError(f"解析 PDF 文件时发生错误: {e}")

        elif filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if not content:
                    raise ValueError("TXT 文件内容为空")
                documents.append(
                    Document(
                        page_content=content, metadata={"filename": filename, "page": 1}
                    )
                )
            except UnicodeDecodeError:
                raise ValueError("TXT 文件不是 UTF-8 编码格式")
        else:
            raise ValueError("不支持的文件类型，仅支持 PDF 和 TXT 文件")

        if not documents:
            raise ValueError("未能从文件中提取出任何有效文件")

        return documents

    def add_documents(self, file_path: str, filename: str) -> int:
        """解析文件、切片并保存至 Chroma 向量库"""
        raw_docs = self.parse_file(file_path, filename)
        # 切片处理
        chunks = self.text_splitter.split_documents(raw_docs)
        # 保存至 Chroma 向量库
        self.vectorstore.add_documents(chunks)
        return len(chunks)

    def _convert_chat_history(
        self, history_list: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        """将对话历史转换为 LangChain Message 对象列表"""
        messages = []
        for item in history_list:
            role = item.get("role")
            content = item.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def query_multi_turn(
        self, question: str, history: List[Dict[str, str]], top_k: int = 3
    ) -> Dict:
        """检索并生成回答与出处引用"""
        chat_history = self._convert_chat_history(history)

        # 如果有历史对话，先将提问重构成独立的 Query
        if chat_history:
            standalone_question = self.rephrase_chain.invoke(
                {"chat_history": chat_history, "question": question}
            )
        else:
            standalone_question = question

        # 向量检索
        retrieval = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retrieval.invoke(standalone_question)

        # 拼接参考上下文
        context_str = "\n\n".join(
            [
                f"出处 {doc.metadata['filename']} 第 {doc.metadata['page']} 页: {doc.page_content}"
                for doc in retrieved_docs
            ]
        )

        # 生成多轮回答
        answer = self.qa_chain.invoke(
            {"context": context_str, "chat_history": chat_history, "question": question}
        )

        # 构建来源引用列表
        sources = []
        seen = set()
        for doc in retrieved_docs:
            fname = doc.metadata.get("filename", "未知")
            page = doc.metadata.get("page", 1)
            key = f"{fname}-{page}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "filename": fname,
                        "page": page,
                        "content_snippet": doc.page_content[:120] + "...",
                    }
                )

        return {
            "question": question,
            "standalone_question": standalone_question,
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

        return [
            {"filename": fname, "chunk_count": count}
            for fname, count in file_stats.items()
        ]


# 实例化全局服务单例
kb_service = KnowledgeBaseService()
