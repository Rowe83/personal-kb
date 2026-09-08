import os
from typing import Dict, List, Set

import streamlit as st

from src.config_loader import load_app_settings
from src.ingest.batch import run_batch_ingest
from src.ingest.parsers import SUPPORTED_EXTENSIONS, normalize_ext
from ui.shared import (
    MAX_FILE_SIZE,
    apply_css,
    apply_page_config,
    cached_kb_service,
    handle_config_error,
    render_config_banner,
    render_sidebar_brand,
)

apply_page_config()
apply_css()
render_sidebar_brand()

settings = load_app_settings()
kb_service = cached_kb_service()

UPLOADER_TYPES = ["pdf", "txt", "md", "xlsx"]


def _save_upload_item(item: Dict, upload_dir: str) -> str:
    file_path = os.path.join(upload_dir, item["filename"])
    with open(file_path, "wb") as buffer:
        buffer.write(item["bytes"])
    return file_path


def _overwrite_key(filename: str) -> str:
    return f"overwrite_{filename}"


def _init_overwrite_widgets(existing_names: Set[str]) -> None:
    for name in existing_names:
        key = _overwrite_key(name)
        if key not in st.session_state:
            st.session_state[key] = True


def _set_all_overwrite(existing_names: Set[str], value: bool) -> None:
    for name in existing_names:
        st.session_state[_overwrite_key(name)] = value


def _render_batch_summary(results) -> None:
    created = [r for r in results if r.status == "created"]
    overwritten = [r for r in results if r.status == "overwritten"]
    skipped = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]

    if created:
        lines = [
            f"• {r.filename}（{r.chunks_created} 块）" for r in created
        ]
        st.success(f"✅ 新建 {len(created)} 个：\n" + "\n".join(lines))
    if overwritten:
        lines = [
            f"• {r.filename}（删除 {r.deleted_chunks} 块，新建 {r.chunks_created} 块）"
            for r in overwritten
        ]
        st.success(f"✅ 覆盖 {len(overwritten)} 个：\n" + "\n".join(lines))
    if skipped:
        lines = [f"• {r.filename}" for r in skipped]
        st.info(f"⏭️ 跳过 {len(skipped)} 个（未勾选覆盖）：\n" + "\n".join(lines))
    if failed:
        lines = [f"• {r.filename}：{r.error}" for r in failed]
        st.error(f"❌ 失败 {len(failed)} 个：\n" + "\n".join(lines))


st.title("📚 知识库管理")
st.caption("上传 PDF / TXT / MD / XLSX 文档并向量化入库（支持批量；同名文件可勾选覆盖）")
render_config_banner()

if "batch_results" in st.session_state:
    _render_batch_summary(st.session_state.pop("batch_results"))

if "delete_flash" in st.session_state:
    st.success(st.session_state.pop("delete_flash"))

uploaded_files = st.file_uploader(
    "选择 PDF / TXT / MD / XLSX 文件（可多选）",
    type=UPLOADER_TYPES,
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    try:
        existing_docs = {
            d["filename"]: d["chunk_count"] for d in kb_service.list_documents()
        }
    except Exception as exc:
        handle_config_error(exc)
        existing_docs = {}

    seen_names: Set[str] = set()
    valid_items: List[Dict] = []
    invalid_messages: List[str] = []

    for uploaded in uploaded_files:
        name = uploaded.name
        ext = normalize_ext(name)
        size = len(uploaded.getvalue())

        if name in seen_names:
            invalid_messages.append(f"「{name}」在本次选择中重复，已忽略后续重复项。")
            continue
        seen_names.add(name)

        if ext not in SUPPORTED_EXTENSIONS:
            invalid_messages.append(
                f"「{name}」类型不支持（仅支持 PDF / TXT / MD / XLSX）。"
            )
            continue
        if size > MAX_FILE_SIZE:
            invalid_messages.append(f"「{name}」超过 20MB 大小限制。")
            continue
        if size == 0:
            invalid_messages.append(f"「{name}」为空文件。")
            continue

        valid_items.append({"filename": name, "bytes": uploaded.getvalue()})

    for msg in invalid_messages:
        st.warning(msg)

    if valid_items:
        existing_names = {item["filename"] for item in valid_items if item["filename"] in existing_docs}
        _init_overwrite_widgets(existing_names)

        st.markdown("**待入库文件**")
        for item in valid_items:
            name = item["filename"]
            if name in existing_docs:
                chunk_count = existing_docs[name]
                st.checkbox(
                    f"📄 {name} — 将覆盖（现有 {chunk_count} 块）",
                    key=_overwrite_key(name),
                )
            else:
                st.text(f"📄 {name} — 新建")

        if existing_names:
            col_all, col_none = st.columns(2)
            with col_all:
                if st.button("全部覆盖", use_container_width=True):
                    _set_all_overwrite(existing_names, True)
                    st.rerun()
            with col_none:
                if st.button("全部取消", use_container_width=True):
                    _set_all_overwrite(existing_names, False)
                    st.rerun()

        if not settings.is_embedding_configured():
            st.warning("请先在「设置」页配置 Embedding API Key 后再上传。")
        elif st.button("🚀 批量解析并向量化", use_container_width=True, type="primary"):
            overwrite_names = {
                name for name in existing_names if st.session_state.get(_overwrite_key(name), True)
            }
            progress_bar = st.progress(0.0)
            status_box = st.empty()

            def on_progress(index, total, filename, phase):
                progress_bar.progress(index / total)
                label = {"start": "处理中", "skipped": "已跳过", "done": "完成"}.get(phase, phase)
                status_box.caption(f"{label} {index}/{total}：{filename}")

            try:
                results = run_batch_ingest(
                    items=valid_items,
                    existing_filenames=set(existing_docs),
                    overwrite_names=overwrite_names,
                    save_file=lambda item: _save_upload_item(item, settings.upload_dir),
                    ingest_fn=kb_service.add_documents,
                    on_progress=on_progress,
                )
                if any(r.status in ("created", "overwritten") for r in results):
                    st.session_state["batch_results"] = results
                    st.rerun()
                else:
                    _render_batch_summary(results)
            except Exception as exc:
                handle_config_error(exc)

st.markdown("---")
st.subheader("📑 已入库文档")

try:
    docs_data = kb_service.list_documents()
    if not docs_data:
        st.info("暂无已入库文档，请先上传。")
    else:
        for doc in docs_data:
            name = doc["filename"]
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.text(f"📄 {name} ({doc['chunk_count']} 块)")
            with col_b:
                confirm_key = f"confirm_delete_{name}"
                if st.session_state.get(confirm_key):
                    if st.button("确认删除", key=f"do_del_{name}", type="primary"):
                        try:
                            n = kb_service.delete_by_filename(name)
                            st.session_state.pop(confirm_key, None)
                            st.session_state["delete_flash"] = f"已删除「{name}」（{n} 个分块）"
                            st.rerun()
                        except Exception as exc:
                            handle_config_error(exc)
                    if st.button("取消", key=f"cancel_del_{name}"):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
                else:
                    if st.button("删除", key=f"del_{name}"):
                        st.session_state[confirm_key] = True
                        st.rerun()
except Exception as exc:
    handle_config_error(exc)

st.markdown("---")
st.caption(
    "批量入库时显示逐文件进度；文档列表支持二次确认删除。"
    "检索已启用 Hybrid（向量 + BM25）。"
)
