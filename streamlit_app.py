import streamlit as st
import requests

# =====================================================================
# 🎨 1. 页面配置与自定义 CSS 美化
# =====================================================================
st.set_page_config(
    page_title="AI 个人知识库",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 注入 CSS 提升界面美观度（渐变背景、卡片阴影与圆角等）
CUSTOM_CSS = """
<style>
    /* 主背景色轻微微调 */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* 侧边栏卡片样式 */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    
    /* 聊天消息气泡美化 */
    .stChatMessage {
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    /* 来源引用卡片样式 */
    .source-box {
        background-color: #f1f3f5;
        border-left: 4px solid #4c6ef5;
        padding: 8px 12px;
        margin-top: 8px;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #495057;
    }
    
    /* 按钮样式强化 */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 后端 API 基础URL
API_BASE_URL = "http://127.0.0.1:8000"

# =====================================================================
# 🧠 2. Session State 初始化 (用于保存多轮对话历史)
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 你好！我是你的 **AI 个人知识库助手**。请在左侧上传文档，然后向我提问吧！",
            "sources": [],
        }
    ]

# =====================================================================
# 📂 3. 侧边栏构建：文档上传、列表展示与设置
# =====================================================================
with st.sidebar:
    st.title("📚 知识库管理")
    st.markdown("---")

    # --- 3.1 文件上传组件 ---
    st.subheader("📤 上传新文档")
    uploaded_file = st.file_uploader("选择 PDF 或 TXT 文件", type=["pdf", "txt"])

    if uploaded_file is not None:
        if st.button("🚀 开始解析并向量化", use_container_width=True):
            with st.spinner("文档上传与切片处理中..."):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    }
                    response = requests.post(f"{API_BASE_URL}/upload", files=files)

                    if response.status_code == 200:
                        res_data = response.json()
                        st.success(
                            f"✅ 上传成功！创建了 {res_data.get('chunks_created')} 个文档分块。"
                        )
                        st.rerun()  # 刷新页面更新文档列表
                    else:
                        st.error(f"❌ 上传失败: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"⚠️ 无法连接后端 API 服务器: {str(e)}")

    st.markdown("---")

    # --- 3.2 已入库文档列表 ---
    st.subheader("📑 已入库文档")
    try:
        doc_res = requests.get(f"{API_BASE_URL}/documents")
        if doc_res.status_code == 200:
            docs_data = doc_res.json().get("documents", [])
            if not docs_data:
                st.info("暂无已入库文档，请先上传。")
            else:
                for doc in docs_data:
                    st.text(f"📄 {doc['filename']} ({doc['chunk_count']} 块)")
        else:
            st.warning("未能获取文档列表")
    except Exception:
        st.warning("⚠️ 后端 API 连接离线")

    st.markdown("---")

    # --- 3.3 系统设置区 ---
    st.subheader("⚙️ 检索参数与设置")
    top_k = st.slider("检索匹配块数量 (Top-K)", min_value=1, max_value=5, value=3)

    if st.button("🧹 清空对话历史", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "对话已重置。请随时向我提问！",
                "sources": [],
            }
        ]
        st.rerun()

# =====================================================================
# 💬 4. 主界面：聊天历史渲染与用户交互
# =====================================================================
st.title("🤖 个人知识库智能问答")
st.caption("基于 LangChain + DeepSeek + Chroma + Streamlit 驱动")

# 渲染已有对话历史
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # 如果存在引用来源，使用可折叠组件或扩展卡片展示
        if msg.get("sources"):
            with st.expander("📌 查看参考引用来源"):
                for idx, src in enumerate(msg["sources"], start=1):
                    st.markdown(
                        f"<div class='source-box'>"
                        f"<b>来源 {idx}:</b> {src['filename']} (第 {src['page']} 页)<br>"
                        f"<i>\"{src['content_snippet']}\"</i>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

# 监听用户输入
if user_input := st.chat_input("请针对知识库文档提出你的问题..."):
    # 1. 立即将用户提问渲染到界面
    st.session_state.messages.append(
        {"role": "user", "content": user_input, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 准备历史对话（提取 role 与 content 构建多轮上下文）
    history_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ["user", "assistant"]
    ]

    # 3. 调用 FastAPI `/query` 接口
    with st.chat_message("assistant"):
        with st.spinner("正在检索知识库并生成回答..."):
            try:
                payload = {
                    "question": user_input,
                    "history": history_payload,
                    "top_k": top_k,
                }
                response = requests.post(f"{API_BASE_URL}/query", json=payload)

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "")
                    sources = data.get("sources", [])

                    # 显示回答
                    st.markdown(answer)

                    # 显示来源
                    if sources:
                        with st.expander("📌 查看参考引用来源"):
                            for idx, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"<div class='source-box'>"
                                    f"<b>来源 {idx}:</b> {src['filename']} (第 {src['page']} 页)<br>"
                                    f"<i>\"{src['content_snippet']}\"</i>"
                                    f"</div>",
                                    unsafe_allow_html=True,
                                )

                    # 将 AI 响应与引用追加到 Session State
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    err_msg = f"❌ 后端服务响应异常: {response.json().get('detail')}"
                    st.error(err_msg)
            except Exception as e:
                st.error(f"⚠️ 无法连接到知识库后端: {str(e)}")
