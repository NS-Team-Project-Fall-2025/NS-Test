import streamlit as st
import os
from typing import List
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.rag_chain import RAGChain
from utils.session_store import (
    list_sessions,
    load_session,
    save_session,
    delete_session,
    create_new_session,
    most_recent_session,
)
from config import Config

# ------------------ Main Page UI Config ------------------
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide"
)

# ------------------ Custom Styling using CSS with "unsafe_allow_html=True" ------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    overflow: hidden; /* prevent whole-page scroll */
}

/* ===== Fixed Header ===== */
.header-container {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    text-align: center;
    background: var(--background-color, #0e1117);
    z-index: 1001;
    padding: 20px 0 15px 0;
    border-bottom: 1px solid rgba(128,128,128,0.2);
}

.header-container .main-title {
    color: var(--text-color, #fff);
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 4px;
}

.header-container .subheader {
    font-size: 18px;
    color: var(--text-color, #ccc);
    opacity: 0.8;
    margin: 0;
}

/* ===== Layout adjustments ===== */
[data-testid="stAppViewContainer"] .block-container {
    padding-top: 120px !important; /* Space for header */
    padding-bottom: 100px !important;
    overflow: hidden;
}

/* ===== Scrollable chat area only ===== */
div[data-testid="stVerticalBlock"]:has(.stChatMessage) {
    max-height: calc(100vh - 220px);
    overflow-y: auto;
    padding-right: 12px;
    scroll-behavior: smooth;
}

/* ===== Fixed chat input ===== */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: var(--chat-left-offset, 300px) !important;
    right: 0 !important;
    z-index: 1000 !important;
    background: var(--background-color, #0e1117);
    padding: 8px 16px 16px 16px;
    box-shadow: 0 -2px 8px rgba(0,0,0,0.1);
}

/* ===== Mobile sidebar collapse ===== */
@media (max-width: 900px) {
    [data-testid="stChatInput"] {
        left: 0 !important;
    }
}

/* ===== Smooth auto-scroll ===== */
</style>

<script>
const chatContainerObserver = new MutationObserver(() => {
    const chatBlocks = document.querySelectorAll('div[data-testid="stVerticalBlock"]');
    chatBlocks.forEach(block => {
        if (block.querySelector('.stChatMessage')) {
            block.scrollTop = block.scrollHeight;
        }
    });
});
chatContainerObserver.observe(document.body, { childList: true, subtree: true });
</script>
<style>
/* Hide Streamlit's built-in 'Limit 200MB per file' text */
[data-testid="stFileUploaderDropzone"] small {
    visibility: hidden;
}

/* Inject our own limit text */
[data-testid="stFileUploaderDropzone"]::after {
    content: "Limit 25 MB per file • PDF, DOCX, TXT";
    display: block;
    font-size: 0.8rem;
    color: rgba(255,255,255,0.7);
    text-align: center;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ------------------ Initialize Session State ------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False
if "active_view_file" not in st.session_state:
    st.session_state.active_view_file = None
if "viewer_mode" not in st.session_state:
    st.session_state.viewer_mode = False
if "active_view_page" not in st.session_state:
    st.session_state.active_view_page = None
# Chat session persistence
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "session_title" not in st.session_state:
    st.session_state.session_title = None
if "session_loaded" not in st.session_state:
    st.session_state.session_loaded = False
if "ignore_next_session_select" not in st.session_state:
    st.session_state.ignore_next_session_select = False

# ------------------ Helpers ------------------
def _get_kb_files() -> List[str]:
    files = []
    kb_dir = Config.DATA_DIR
    try:
        if os.path.isdir(kb_dir):
            for name in sorted(os.listdir(kb_dir)):
                path = os.path.join(kb_dir, name)
                if os.path.isfile(path) and os.path.splitext(name)[1].lower() in {'.pdf', '.docx', '.txt'}:
                    files.append(path)
    except Exception:
        pass
    return files

def _render_kb_listing():
    files = _get_kb_files()
    st.subheader("📚 Knowledge Base Files")
    if not files:
        st.info("No documents in the knowledge base yet. Upload to get started.")
        return
    for path in files:
        name = os.path.basename(path)
        size = os.path.getsize(path)
        label = f"📄 {name} ({size/1024:.1f} KB)"
        with st.expander(label):
            try:
                with open(path, 'rb') as fh:
                    st.download_button("⬇️ Download", data=fh.read(), file_name=name, use_container_width=True)
            except Exception:
                st.warning("Download unavailable for this file.")

            if st.button("👁️ View", key=f"view_{name}", use_container_width=True):
                st.session_state.active_view_file = path
                st.session_state.viewer_mode = True
                st.rerun()


def initialize_components():
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = VectorStore(
            embedding_model=Config.EMBEDDING_MODEL,
            persist_directory=Config.VECTOR_STORE_DIR
        )
        try:
            loaded = st.session_state.vectorstore.load_vectorstore()
            if loaded:
                info = st.session_state.vectorstore.get_collection_info()
                if info.get("count", 0) > 0:
                    st.session_state.documents_processed = True
        except Exception:
            pass
    if st.session_state.rag_chain is None:
        st.session_state.rag_chain = RAGChain(st.session_state.vectorstore)

# --- PDF/DOCX/TXT viewer utilities ---
def display_pdf(file_path: str, height: int = 820) -> bool:
    """
    Try 3 ways to display the PDF.
    Returns True if something was rendered; False otherwise.
    """
    # streamlit-pdf-viewer
    try:
        from streamlit_pdf_viewer import pdf_viewer
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        pdf_viewer(input=pdf_bytes, width=0, height=height)  # width=0 makes it responsive
        return True
    except Exception:
        pass

    #Fallback: base64 iframe (works widely)
    try:
        import base64
        from streamlit.components.v1 import html as st_html
        with open(file_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st_html(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="{height}px" style="border:none;border-radius:10px;"></iframe>',
            height=height + 20,
        )
        return True
    except Exception:
        pass

    # render page images with PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(file_path)
        total = doc.page_count
        col_left, col_right = st.columns([9, 3])
        with col_left:
            page = st.number_input("Page", min_value=1, max_value=total, value=1, step=1)
        with col_right:
            zoom = st.slider("Zoom", 100, 300, 200, step=25, help="Image scale (fallback viewer)")
        mat = fitz.Matrix(zoom/100.0, zoom/100.0)
        pix = doc.load_page(page-1).get_pixmap(matrix=mat, alpha=False)
        st.image(pix.tobytes("png"), use_container_width=True)
        st.caption(f"Page {page} of {total} — image preview (PDF fallback).")
        return True
    except Exception as e:
        st.error(f"Unable to render PDF: {e}")
        return False

def display_text_file(path: str):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            st.text_area("Preview", f.read(), height=800)
    except Exception as e:
        st.warning(f"Unable to preview file: {e}")

def display_docx_file(path: str):
    try:
        import docx
        doc = docx.Document(path)
        text = "\n".join(p.text for p in doc.paragraphs)
        st.text_area("Preview", text, height=800)
    except Exception as e:
        st.warning(f"Unable to preview DOCX: {e}")

# ------------------ Main UI ------------------
def main():
    st.markdown('<h1 class="main-title">NetSec Tutor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">An easy way to study and test your network security skills...</p>', unsafe_allow_html=True)

    # Checking Ollama availability
    try:
        import requests
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
        if response.status_code != 200:
            st.error("Could Not find any model from Ollama is running, make sure a model from Ollama is running.")
            st.info("Run ollama by 'ollama run mistral' ")
            return
    except Exception:
        st.error("Could Not find any model from Ollama is running, make sure a model from Ollama is running.")
        st.info("Run ollama by 'ollama run mistral' ")
        return

    # Initialize components
    initialize_components()

    # Load most recent session if any
    if not st.session_state.session_loaded:
        recent = most_recent_session()
        if recent:
            st.session_state.session_id = recent.get("session_id")
            st.session_state.session_title = recent.get("title")
            st.session_state.chat_history = recent.get("messages", [])
        else:
            st.session_state.session_id = None
            st.session_state.session_title = None
            st.session_state.chat_history = []
        st.session_state.session_loaded = True

    # Sidebar
    with st.sidebar:
        st.header("📂 Document Management")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Each file must be under 25 MB. Supported formats: PDF, DOCX, TXT."
        )
        max_size = 25 * 1024 * 1024 #25MB max file size for each file
        if uploaded_files:
            small_enough = []
            for f in uploaded_files:
                if f.size > max_size:
                    st.error(f"{f.name} exceeds 25MB size limit and will be skipped.")
                else:
                    small_enough.append(f)
            uploaded_files = small_enough
            # Deduplicate by filename
            seen = set(); unique_files = []
            for f in uploaded_files:
                if f.name not in seen:
                    unique_files.append(f); seen.add(f.name)
            uploaded_files = unique_files
        if uploaded_files and st.button("🚀 Process Documents", use_container_width=True):
            process_documents(uploaded_files)

        st.divider()
        _render_kb_listing()

        with st.expander("🔧 Advanced Settings"):
            # keep these in session via return values below (read inside chat section)
            st.session_state.setdefault("show_sources", True)
            st.session_state.setdefault("show_context", False)
            st.session_state.show_sources = st.checkbox("Show sources", value=st.session_state.show_sources)
            st.session_state.show_context = st.checkbox("Show context", value=st.session_state.show_context)

        if st.button("🗑 Clear Knowledge Base", use_container_width=True):
            clear_knowledge_base()

    # ---------- Viewer Mode (full width) ----------
    if st.session_state.viewer_mode and st.session_state.active_view_file:
        path = st.session_state.active_view_file
        if os.path.exists(path):
            filename = os.path.basename(path)
            # Top row: title (left) and close button (right)
            left, right = st.columns([8, 1])
            with left:
                st.subheader(f"👁️ Viewer: {filename}")
            with right:
                if st.button("Close ✖", key="close_viewer", use_container_width=True):
                    st.session_state.viewer_mode = False
                    st.session_state.active_view_file = None
                    st.rerun()

            st.markdown('<div class="viewer-container">', unsafe_allow_html=True)
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".pdf":
                displayed = display_pdf(path, height=820)
                if not displayed:
                    st.info("If you see nothing, install `streamlit-pdf-viewer` or `pymupdf` for robust rendering.")
            elif ext == ".txt":
                display_text_file(path)
            elif ext == ".docx":
                display_docx_file(path)
            else:
                st.info("Viewer supports PDF, DOCX, and TXT.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Selected file no longer exists.")
        return  # don't render chat while in viewer

    # ---------- Normal Chat Layout (ChatGPT-like) ----------
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("💬 Chat")

        # Render existing conversation sequentially
        for entry in st.session_state.chat_history:
            # Support legacy Q/A entries as well as new role-based messages
            if isinstance(entry, dict) and "role" in entry:
                role = entry.get("role", "assistant")
                content = entry.get("content", "")
                with st.chat_message(role):
                    st.markdown(content)
                    # Render optional sources/context if present on assistant messages
                    if role == "assistant":
                        if st.session_state.get("show_sources", True) and entry.get("sources"):
                            st.markdown("**Sources**")
                            for i, source in enumerate(entry.get("sources", []), 1):
                                with st.expander(f"📄 Source {i}: {source.get('filename', 'unknown')}"):
                                    st.text(source.get('content_preview', ''))
                        if st.session_state.get("show_context", False) and entry.get("context"):
                            with st.expander("🔍 Retrieved Context"):
                                st.text_area("Context used for answering:", entry.get("context", ""), height=200)
            elif isinstance(entry, dict) and "question" in entry and "answer" in entry:
                # Legacy record: render as two chat bubbles
                with st.chat_message("user"):
                    st.markdown(entry.get("question", ""))
                with st.chat_message("assistant"):
                    st.markdown(entry.get("answer", ""))
                    if st.session_state.get("show_sources", True) and entry.get("sources"):
                        st.markdown("**Sources**")
                        for i, source in enumerate(entry.get("sources", []), 1):
                            with st.expander(f"📄 Source {i}: {source.get('filename', 'unknown')}"):
                                st.text(source.get('content_preview', ''))
            else:
                # Unknown format fallback
                with st.chat_message("assistant"):
                    st.markdown(str(entry))

        # Chat input at bottom
        prompt = st.chat_input("Ask Anything related to Network Security Course...")
        if prompt is not None:
            if not st.session_state.documents_processed:
                st.warning("Please upload and process documents before asking questions.", icon="⚠️")
            elif not prompt.strip():
                st.warning("Please enter a message.", icon="⚠️")
            elif not st.session_state.rag_chain:
                st.error("Unexpected error: RAG chain not initialized.")
            else:
                # Appending user message
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                # Persist after user message; if first message of a new chat, create a new chat session
                if st.session_state.get("session_id"):
                    try:
                        save_session(st.session_state.session_id, st.session_state.chat_history, st.session_state.get("session_title"))
                    except Exception:
                        pass
                else:
                    try:
                        # Create new session id and title: Time followed by date and a session ID
                        import uuid as _uuid
                        from datetime import datetime as _dt
                        now = _dt.now()
                        time_str = now.strftime("%H:%M")
                        date_str = now.strftime("%Y-%m-%d")
                        new_sid = str(_uuid.uuid4())
                        short_sid = new_sid[:8]
                        title = f"{time_str} {date_str} — {short_sid}"
                        save_session(new_sid, st.session_state.chat_history, title)
                        st.session_state.session_id = new_sid
                        st.session_state.session_title = title
                    except Exception:
                        pass
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate assistant reply using RAG context history
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Thinking..."):
                        result = st.session_state.rag_chain.chat_with_context(prompt, chat_history=st.session_state.chat_history)
                    answer = result.get("answer", "")
                    st.markdown(answer)

                    # Append assistant message with required metadata
                    assistant_entry = {
                        "role": "assistant",
                        "content": answer,
                        "sources": result.get("sources", []),
                        "context": result.get("context", "")
                    }
                    st.session_state.chat_history.append(assistant_entry)

                    # Persist session after each of the assistant reply
                    if st.session_state.get("session_id"):
                        try:
                            save_session(st.session_state.session_id, st.session_state.chat_history, st.session_state.get("session_title"))
                        except Exception:
                            pass

                    # Optionally render sources/context inline
                    if st.session_state.get("show_sources", True) and assistant_entry["sources"]:
                        st.markdown("**Sources**")
                        for i, source in enumerate(assistant_entry["sources"], 1):
                            with st.expander(f"📄 Source {i}: {source.get('filename', 'unknown')}"):
                                st.text(source.get('content_preview', ''))
                    if st.session_state.get("show_context", False) and assistant_entry.get("context"):
                        with st.expander("🔍 Retrieved Context"):
                            st.text_area("Context used for answering:", assistant_entry.get("context", ""), height=200)

    with col2:
        st.subheader("⏱ Chat History")

        sessions_meta = list_sessions()
        session_options = [
            f"{s.get('title', 'Untitled')} ({s.get('message_count', 0)} msgs) — {s.get('session_id')[:8]}"
            for s in sessions_meta
        ]
        current_sid = st.session_state.get("session_id")
        current_index = 0
        for idx, meta in enumerate(sessions_meta):
            if meta.get("session_id") == current_sid:
                current_index = idx
                break

        selected = st.selectbox(
            "Select Chat Session, for new chat session click 'New Chat' below.",
            options=range(len(sessions_meta)) if sessions_meta else [0],
            format_func=lambda i: session_options[i] if sessions_meta else "(No sessions)",
            index=current_index if sessions_meta else 0,
            disabled=not sessions_meta
        )

        cols = st.columns(2)
        with cols[0]:
            if st.button("🆕 New Chat", use_container_width=True):
                # Do not persist yet; wait until first message to create the session
                st.session_state.session_id = None
                st.session_state.session_title = None
                st.session_state.chat_history = []
                # Prevent immediate sidebar auto-switch back to previously selected session
                st.session_state.ignore_next_session_select = True
                st.rerun()

        with cols[1]:
            if st.button("🗑️ Delete Chat", use_container_width=True, disabled=not sessions_meta):
                sid_to_delete = sessions_meta[selected].get("session_id") if sessions_meta else None
                if sid_to_delete:
                    delete_session(sid_to_delete)
                    if sid_to_delete == current_sid:
                        recent = most_recent_session()
                        if recent:
                            st.session_state.session_id = recent.get("session_id")
                            st.session_state.session_title = recent.get("title")
                            st.session_state.chat_history = recent.get("messages", [])
                        else:
                            # No sessions left; clear current and wait for first message to create a new session
                            st.session_state.session_id = None
                            st.session_state.session_title = None
                            st.session_state.chat_history = []
                    st.rerun()

        if sessions_meta:
            selected_sid = sessions_meta[selected].get("session_id")
            if selected_sid and selected_sid != current_sid and not st.session_state.get("ignore_next_session_select"):
                loaded = load_session(selected_sid)
                if loaded:
                    st.session_state.session_id = loaded.get("session_id")
                    st.session_state.session_title = loaded.get("title")
                    st.session_state.chat_history = loaded.get("messages", [])
                    st.rerun()
        # Reset the ignore flag after rendering the sidebar to allow future switches
        if st.session_state.get("ignore_next_session_select"):
            st.session_state.ignore_next_session_select = False

        # --- Optional Viewer Section (if file selected) ---
        active_file = st.session_state.get("active_view_file")
        if active_file and os.path.exists(active_file):
            file_name = os.path.basename(active_file)
            ext = os.path.splitext(file_name)[1].lower()
            st.subheader(f"👁️ Viewer: {file_name}")
            if ext == ".pdf":
                display_pdf(active_file, height=600)
            elif ext == ".txt":
                display_text_file(active_file)
            elif ext == ".docx":
                display_docx_file(active_file)


# ------------------ Core Functions ------------------
def process_documents(uploaded_files):
    try:
        with st.spinner("⏳ Processing documents..."):
            os.makedirs(Config.DATA_DIR, exist_ok=True)
            file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(Config.DATA_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_paths.append(file_path)
            processor = DocumentProcessor(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            documents = processor.process_multiple_documents(file_paths)
            if documents:
                st.session_state.vectorstore.add_documents(documents)
                st.session_state.documents_processed = True
                st.success("✅ Successfully processed document(s). You can now ask questions!", icon="🎉")
            else:
                st.error("❌ No documents were successfully processed.", icon="🚫")
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}", icon="🚫")

def ask_question(question: str, show_sources: bool, show_context: bool):
    try:
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.rag_chain.generate_answer(question)
        st.markdown("### 🧠 Answer")
        st.info(result["answer"])
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", [])
        })
        if show_sources and result.get("sources"):
            st.markdown("### 📚 Sources")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"📄 Source {i}: {source.get('filename', 'unknown')}"):
                    st.text(source.get('content_preview', ''))
        if show_context and result.get("context"):
            with st.expander("🔍 Retrieved Context"):
                st.text_area("Context used for answering:", result["context"], height=200)
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")

def display_chat_history():
    history = st.session_state.get("chat_history", [])
    if not history:
        st.info("No messages yet.")
        return

    # Build user/assistant pairs supporting both legacy and new formats
    pairs = []
    i = 0
    n = len(history)
    while i < n:
        entry = history[i]
        if isinstance(entry, dict) and "question" in entry and "answer" in entry:
            pairs.append((entry.get("question", ""), entry.get("answer", "")))
            i += 1
            continue
        if isinstance(entry, dict) and entry.get("role") == "user":
            user_msg = entry.get("content", "")
            # Pair with next assistant message if present
            if i + 1 < n and isinstance(history[i+1], dict) and history[i+1].get("role") == "assistant":
                assistant_msg = history[i+1].get("content", "")
                pairs.append((user_msg, assistant_msg))
                i += 2
                continue
            else:
                pairs.append((user_msg, ""))
                i += 1
                continue
        i += 1

    # Show last 5 pairs in reverse order (most recent first)
    for q, a in reversed(pairs[-5:]):
        title = (q or a or "(empty)")[:50]
        with st.expander(f"💭 {title}..."):
            if q:
                st.write("**Q:**", q)
            if a:
                st.write("**A:**", a)

def clear_knowledge_base():
    if st.session_state.vectorstore:
        st.session_state.vectorstore.delete_collection()
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.documents_processed = False
        st.success("🧹 Knowledge base cleared!")
        st.rerun()

# ------------------ Run App ------------------
if __name__ == "__main__":
    main()
