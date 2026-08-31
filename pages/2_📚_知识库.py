import os
import streamlit as st

from src.config_loader import load_app_settings
from ui.shared import (
    MAX_FILE_SIZE,
    apply_css,
    apply_page_config,
    cached_kb_service,
    handle_config_error,
    render_config_banner,
)

apply_page_config()
apply_css()

settings = load_app_settings()
kb_service = cached_kb_service()

st.title("📚 知识库管理")
st.caption("上传 PDF / TXT 文档并向量化入库（同名文件将覆盖旧索引）")
render_config_banner()

uploaded_file = st.file_uploader(
    "选择 PDF 或 TXT 文件", type=["pdf", "txt"], label_visibility="collapsed"
)

if uploaded_file is not None:
    existing = {
        d["filename"]: d["chunk_count"] for d in kb_service.list_documents()
    }
    is_overwrite = uploaded_file.name in existing

    if is_overwrite:
        st.warning(
            f"⚠️ 知识库中已存在同名文档「{uploaded_file.name}」"
            f"（{existing[uploaded_file.name]} 个分块）。"
            f"继续上传将**删除旧向量并重新入库**。"
        )

    if not settings.is_embedding_configured():
        st.warning("请先在「设置」页配置 Embedding API Key 后再上传。")
    else:
        confirm_label = (
            "⚠️ 确认覆盖并重新入库"
            if is_overwrite
            else "🚀 开始解析并向量化"
        )
        if st.button(confirm_label, use_container_width=True, type="primary"):
            file_bytes = uploaded_file.getvalue()
            if len(file_bytes) > MAX_FILE_SIZE:
                st.error("文件大小超过限制（最大 20MB）")
            else:
                file_path = os.path.join(settings.upload_dir, uploaded_file.name)
                with st.spinner("文档上传与切片处理中..."):
                    try:
                        with open(file_path, "wb") as buffer:
                            buffer.write(file_bytes)
                        result = kb_service.add_documents(
                            file_path, uploaded_file.name, overwrite=True
                        )
                        if result.get("overwritten"):
                            st.success(
                                f"✅ 已覆盖「{uploaded_file.name}」："
                                f"删除旧分块 {result['deleted_chunks']} 个，"
                                f"新建 {result['chunks_created']} 个。"
                            )
                        else:
                            st.success(
                                f"✅ 上传成功！创建了 {result['chunks_created']} 个文档分块。"
                            )
                        st.rerun()
                    except Exception as exc:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        handle_config_error(exc)

st.markdown("---")
st.subheader("📑 已入库文档")

try:
    docs_data = kb_service.list_documents()
    if not docs_data:
        st.info("暂无已入库文档，请先上传。")
    else:
        for doc in docs_data:
            st.text(f"📄 {doc['filename']} ({doc['chunk_count']} 块)")
except Exception as exc:
    handle_config_error(exc)

st.markdown("---")
st.caption("检索已启用 Hybrid（向量 + BM25）。P1 将支持 MD / Excel、批量导入与 PDF OCR。")
