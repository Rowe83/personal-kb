from typing import List, Dict, Iterator, Tuple, Optional, Any

import chromadb
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from src.app_settings import AppSettings
from src.retrieval.bm25_index import BM25Index, content_hash
from src.retrieval.catalog import format_catalog_answer, is_catalog_question
from src.retrieval.filters import (
    dedupe_documents,
    extract_api_lines_from_docs,
    filenames_with_api_lines,
    resolve_target_filename,
)
from src.retrieval.intent import (
    format_interface_count_answer,
    is_ambiguous_document_reference,
    is_interface_count_question,
)
from src.retrieval.pipeline import RetrievalPipeline
from src.settings_store import ConfigurationError, ensure_data_dirs

SHORT_TXT_CHARS = 800


class KnowledgeBaseService:
    def __init__(self, app_settings: AppSettings):
        self.settings = app_settings
        ensure_data_dirs(app_settings)

        self._embeddings: Optional[OpenAIEmbeddings] = None
        self._llm: Optional[ChatOpenAI] = None
        self._vectorstore: Optional[Chroma] = None
        self._rephrase_chain = None
        self._qa_chain = None
        self._bm25_index: Optional[BM25Index] = None
        self._pipeline: Optional[RetrievalPipeline] = None
        self._deduped = False

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40,
            separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        )

        self.rephrase_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "给定一段对话历史和用户的最新提问，如果最新提问依赖于对话历史（例如包含'它'、'上述'、'后者'等指代），"
                    "请将其重写为一个无须结合上下文就能独立理解的全新问题。"
                    "如果最新提问本身就是独立的，请原样返回，不要做多余回答。",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )

        qa_prompt_str = (
            "你是一个严谨的个人知识库助手。请严格根据给出的 [参考文档] 回答用户的 [提问]。\n"
            "只能引用 [参考文档] 中明确出现的条目、数字和名称，禁止根据常识或领域知识补充未在文档中出现的内容。\n"
            "统计接口/条目数量时，必须逐条列出文档中的原文编号行，不得推断或编造。\n"
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

    def _require_embedding(self) -> OpenAIEmbeddings:
        if not self.settings.is_embedding_configured():
            raise ConfigurationError(
                "未配置 Embedding API Key。请在「设置」页配置向量模型后再上传文档。"
            )
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.settings.embedding.model,
                api_key=self.settings.embedding.api_key,
                base_url=self.settings.embedding.base_url,
            )
        return self._embeddings

    def _require_llm(self) -> ChatOpenAI:
        if not self.settings.is_llm_configured():
            raise ConfigurationError(
                "未配置 LLM API Key。请在「设置」页配置大模型后再进行问答。"
            )
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.settings.llm.model,
                api_key=self.settings.llm.api_key,
                base_url=self.settings.llm.base_url,
                temperature=self.settings.llm.temperature,
            )
            self._rephrase_chain = self.rephrase_prompt | self._llm | StrOutputParser()
            self._qa_chain = self.qa_prompt | self._llm | StrOutputParser()
        return self._llm

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                embedding_function=self._require_embedding(),
                persist_directory=self.settings.chroma_db_dir,
            )
        return self._vectorstore

    @property
    def bm25_index(self) -> BM25Index:
        if self._bm25_index is None:
            self._bm25_index = BM25Index.from_vectorstore(self.vectorstore)
        return self._bm25_index

    @property
    def retrieval_pipeline(self) -> RetrievalPipeline:
        if self._pipeline is None:
            self._pipeline = RetrievalPipeline(
                vectorstore=self.vectorstore,
                bm25_index=self.bm25_index,
                hybrid_fetch_k=self.settings.retrieval.hybrid_fetch_k,
                rrf_k=self.settings.retrieval.rrf_k,
            )
            if not self._deduped:
                self.dedupe_vectorstore()
                self._deduped = True
        return self._pipeline

    @property
    def rephrase_chain(self):
        self._require_llm()
        return self._rephrase_chain

    @property
    def qa_chain(self):
        self._require_llm()
        return self._qa_chain

    def parse_file(self, file_path: str, filename: str) -> List[Document]:
        from src.ingest.parsers import parse_document
        return parse_document(file_path, filename)

    def _chunk_documents(self, raw_docs: List[Document], filename: str) -> List[Document]:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if (
            len(raw_docs) == 1
            and ext in {"txt", "md"}
            and len(raw_docs[0].page_content) <= SHORT_TXT_CHARS
        ):
            chunks = raw_docs
        else:
            chunks = self.text_splitter.split_documents(raw_docs)

        prefixed: List[Document] = []
        for chunk in chunks:
            meta = dict(chunk.metadata or {})
            meta["filename"] = meta.get("filename", filename)
            content = chunk.page_content
            prefix = f"[{filename}] "
            if not content.startswith(prefix):
                content = prefix + content
            prefixed.append(Document(page_content=content, metadata=meta))
        return prefixed

    def find_chunk_ids_by_filename(self, filename: str) -> List[str]:
        raw = self.vectorstore.get(include=["metadatas"])
        ids = raw.get("ids") or []
        metadatas = raw.get("metadatas") or []
        return [
            ids[i]
            for i, meta in enumerate(metadatas)
            if meta and meta.get("filename") == filename and i < len(ids)
        ]

    def get_documents_by_filename(self, filename: str) -> List[Document]:
        raw = self.vectorstore.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        docs: List[Document] = []
        for i, content in enumerate(documents):
            if content is None or i >= len(metadatas):
                continue
            meta = metadatas[i] or {}
            if meta.get("filename") != filename:
                continue
            meta = dict(meta)
            if i < len(ids):
                meta["_id"] = ids[i]
            docs.append(Document(page_content=content, metadata=meta))
        return docs

    def delete_by_filename(self, filename: str) -> int:
        old_ids = self.find_chunk_ids_by_filename(filename)
        if not old_ids:
            return 0
        self.vectorstore.delete(ids=old_ids)
        return len(old_ids)

    def dedupe_vectorstore(self) -> int:
        try:
            raw = self.vectorstore.get(include=["documents", "metadatas"])
        except Exception:
            return 0
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        seen: Dict[str, str] = {}
        to_delete: List[str] = []
        for i, content in enumerate(documents):
            if content is None or i >= len(ids):
                continue
            h = content_hash(content)
            if h in seen:
                to_delete.append(ids[i])
            else:
                seen[h] = ids[i]
        if to_delete:
            self.vectorstore.delete(ids=to_delete)
            if self._bm25_index is not None:
                self._bm25_index.rebuild_from_vectorstore(self.vectorstore)
        return len(to_delete)

    def add_documents(
        self, file_path: str, filename: str, overwrite: bool = True
    ) -> Dict[str, Any]:
        self._require_embedding()
        deleted_chunks = 0
        overwritten = False
        old_ids = self.find_chunk_ids_by_filename(filename)
        if old_ids:
            if not overwrite:
                raise ValueError(
                    f"文档「{filename}」已存在。请确认覆盖后重新上传。"
                )
            deleted_chunks = self.delete_by_filename(filename)
            overwritten = True

        raw_docs = self.parse_file(file_path, filename)
        chunks = self._chunk_documents(raw_docs, filename)
        self.vectorstore.add_documents(chunks)
        self.bm25_index.rebuild_from_vectorstore(self.vectorstore)
        if self._pipeline is not None:
            self._pipeline.bm25_index = self._bm25_index

        return {
            "chunks_created": len(chunks),
            "overwritten": overwritten,
            "deleted_chunks": deleted_chunks,
        }

    def _convert_chat_history(
        self, history_list: List[Dict[str, str]]
    ) -> List[BaseMessage]:
        messages = []
        for item in history_list:
            role = item.get("role")
            content = item.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        return messages

    def _build_sources(self, retrieved_docs: List[Document]) -> List[Dict]:
        sources = []
        seen = set()
        for doc in retrieved_docs:
            fname = doc.metadata.get("filename", "未知")
            page = doc.metadata.get("page", 1)
            key = f"{fname}-{page}"
            if key not in seen:
                seen.add(key)
                snippet = doc.page_content
                if snippet.startswith(f"[{fname}] "):
                    snippet = snippet[len(fname) + 3 :]
                sources.append(
                    {
                        "filename": fname,
                        "page": page,
                        "content_snippet": snippet[:120] + "...",
                    }
                )
        return sources

    def _try_interface_count_answer(
        self,
        question: str,
        standalone_question: str,
        retrieved_docs: List[Document],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Optional[str]:
        if not is_interface_count_question(question):
            return None

        known_filenames = [d["filename"] for d in self.list_documents()]
        history_text = "\n".join(
            item.get("content", "")
            for item in (history or [])[-6:]
            if item.get("content")
        )
        combined_question = f"{history_text}\n{question}".strip()
        target = resolve_target_filename(
            combined_question,
            standalone_question,
            known_filenames,
            retrieved_docs,
            vectorstore=self.vectorstore,
        )
        if not target and is_ambiguous_document_reference(combined_question):
            api_docs = filenames_with_api_lines(self.vectorstore, known_filenames)
            if len(api_docs) > 1:
                examples = "、".join(f"「{name}」" for name in api_docs[:3])
                return (
                    f"知识库中有 {len(api_docs)} 个接口类文档（{examples}），"
                    "「这个文档」指代不明确。请指定文件名，"
                    "例如：「原生场外开放式基金接口整理.txt 有多少个接口」。"
                )
        if not target:
            return None

        target_docs = self.get_documents_by_filename(target)
        if not target_docs:
            target_docs = [
                d for d in retrieved_docs if d.metadata.get("filename") == target
            ]
        api_lines = extract_api_lines_from_docs(target_docs)
        if not api_lines:
            return None
        return format_interface_count_answer(target, api_lines)

    def _prepare_rag_context(
        self, question: str, history: List[Dict[str, str]], top_k: int = 3
    ) -> Dict:
        self._require_llm()
        chat_history = self._convert_chat_history(history)

        if chat_history:
            standalone_question = self.rephrase_chain.invoke(
                {"chat_history": chat_history, "question": question}
            )
        else:
            standalone_question = question

        known_filenames = [d["filename"] for d in self.list_documents()]
        history_text = "\n".join(
            item.get("content", "")
            for item in history[-6:]
            if item.get("content")
        )
        combined_question = f"{history_text}\n{question}".strip()
        filename_hint = resolve_target_filename(
            combined_question,
            standalone_question,
            known_filenames,
            [],
            vectorstore=self.vectorstore,
        )

        if (
            filename_hint
            and is_interface_count_question(question)
            and self.settings.is_embedding_configured()
        ):
            retrieved_docs = dedupe_documents(
                self.get_documents_by_filename(filename_hint)
            )
        else:
            retrieved_docs = self.retrieval_pipeline.retrieve(
                standalone_question,
                top_k=top_k,
                filename_hint=filename_hint,
            )

        context_str = "\n\n".join(
            [
                f"出处 {doc.metadata.get('filename', '未知')} 第 {doc.metadata.get('page', 1)} 页: {doc.page_content}"
                for doc in retrieved_docs
            ]
        )

        return {
            "chat_history": chat_history,
            "standalone_question": standalone_question,
            "context_str": context_str,
            "sources": self._build_sources(retrieved_docs),
            "retrieved_docs": retrieved_docs,
        }

    def query_multi_turn_stream(
        self, question: str, history: List[Dict[str, str]], top_k: int = 3
    ) -> Tuple[Iterator[str], List[Dict]]:
        if is_catalog_question(question):
            answer = format_catalog_answer(self.list_documents())
            return iter([answer]), []

        prepared = self._prepare_rag_context(question, history, top_k)
        count_answer = self._try_interface_count_answer(
            question,
            prepared["standalone_question"],
            prepared["retrieved_docs"],
            history,
        )
        if count_answer:
            return iter([count_answer]), prepared["sources"]

        token_iter = self.qa_chain.stream(
            {
                "context": prepared["context_str"],
                "chat_history": prepared["chat_history"],
                "question": question,
            }
        )
        return token_iter, prepared["sources"]

    def query_multi_turn(
        self, question: str, history: List[Dict[str, str]], top_k: int = 3
    ) -> Dict:
        if is_catalog_question(question):
            answer = format_catalog_answer(self.list_documents())
            return {
                "question": question,
                "standalone_question": question,
                "answer": answer,
                "sources": [],
            }

        prepared = self._prepare_rag_context(question, history, top_k)
        count_answer = self._try_interface_count_answer(
            question,
            prepared["standalone_question"],
            prepared["retrieved_docs"],
            history,
        )
        if count_answer:
            return {
                "question": question,
                "standalone_question": prepared["standalone_question"],
                "answer": count_answer,
                "sources": prepared["sources"],
            }

        answer = self.qa_chain.invoke(
            {
                "context": prepared["context_str"],
                "chat_history": prepared["chat_history"],
                "question": question,
            }
        )
        return {
            "question": question,
            "standalone_question": prepared["standalone_question"],
            "answer": answer,
            "sources": prepared["sources"],
        }

    def list_documents(self) -> List[Dict]:
        file_stats: Dict[str, int] = {}
        try:
            client = chromadb.PersistentClient(path=self.settings.chroma_db_dir)
            for col in client.list_collections():
                results = client.get_collection(col.name).get(include=["metadatas"])
                metadatas = results.get("metadatas") or []
                for meta in metadatas:
                    if not meta:
                        continue
                    fname = meta.get("filename", "未命名文件")
                    file_stats[fname] = file_stats.get(fname, 0) + 1
        except Exception:
            if self.settings.is_embedding_configured():
                results = self.vectorstore.get(include=["metadatas"])
                for meta in results.get("metadatas") or []:
                    if not meta:
                        continue
                    fname = meta.get("filename", "未命名文件")
                    file_stats[fname] = file_stats.get(fname, 0) + 1

        return [
            {"filename": fname, "chunk_count": count}
            for fname, count in file_stats.items()
        ]
