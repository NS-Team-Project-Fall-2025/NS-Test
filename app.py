import streamlit as st
import os
from typing import List
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.rag_chain import RAGChain
from config import Config

# Configure Streamlit page
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide"
)

# ---------- Theme-aware Custom CSS ----------
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        color: var(--text-color);
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .subheader {
        text-align: center;
        font-size: 18px;
        color: var(--text-color);
        opacity: 0.75;
        margin-bottom: 30px;
    }
    .card {
        background: var(--background-color);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 20px;
        border: 1px solid var(--secondary-background-color);
    }
    section[data-testid="stSidebar"] {
        background-color: var(--secondary-background-color);
        border-right: 1px solid rgba(128,128,128,0.2);
    }
    button[kind="primary"], button[kind="secondary"] {
        width: 100% !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

def initialize_components():
    """Initialize the RAG components."""
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = VectorStore(
            embedding_model=Config.EMBEDDING_MODEL,
            persist_directory=Config.VECTOR_STORE_DIR
        )
    if st.session_state.rag_chain is None:
        st.session_state.rag_chain = RAGChain(st.session_state.vectorstore)

def main():
    st.markdown('<h1 class="main-title">🤖 RAG Q&A System with Ollama</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload documents and ask AI-powered questions based on their content.</p>', unsafe_allow_html=True)
    
    # Check for Ollama availability
    try:
        import requests
        response = requests.get(f"{Config.OLLAMA_BASE_URL}/api/tags")
        if response.status_code != 200:
            st.error("⚠ Could not connect to Ollama service! Please ensure Ollama is running.")
            st.info("Install Ollama from: https://ollama.ai")
            return
    except Exception as e:
        st.error("⚠ Could not connect to Ollama service! Please ensure Ollama is running.")
        st.info("Install Ollama from: https://ollama.ai")
        return
    
    # Initialize components
    initialize_components()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        with st.container():
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Upload PDF, DOCX, or TXT files"
            )
            # File size limit (10MB per file)
            max_size = 10 * 1024 * 1024
            if uploaded_files:
                for f in uploaded_files:
                    if f.size > max_size:
                        st.error(f"{f.name} exceeds 10MB size limit and will be skipped.")
                uploaded_files = [f for f in uploaded_files if f.size <= max_size]
            # Deduplicate by filename
            if uploaded_files:
                seen = set()
                unique_files = []
                for f in uploaded_files:
                    if f.name not in seen:
                        unique_files.append(f)
                        seen.add(f.name)
                uploaded_files = unique_files
            if uploaded_files and st.button("🚀 Process Documents", use_container_width=True):
                process_documents(uploaded_files)

        if st.session_state.vectorstore:
            info = st.session_state.vectorstore.get_collection_info()
    

        with st.expander("🔧 Advanced Settings"):
            show_sources = st.checkbox("Show sources", value=True)
            show_context = st.checkbox("Show context", value=False)
            
        if st.button("🗑 Clear Knowledge Base", use_container_width=True):
            clear_knowledge_base()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container():
            st.subheader("💬 Ask Questions")
            user_question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about your documents?",
                key="question_input"
            )
            if st.button("❓ Ask Question", type="primary", use_container_width=True):
                if not st.session_state.documents_processed:
                    st.warning("Please upload and process documents before asking questions.", icon="⚠️")
                elif not user_question.strip():
                    st.warning("Please enter a question to ask.", icon="⚠️")
                elif st.session_state.rag_chain:
                    ask_question(user_question, show_sources, show_context)
                else:
                    st.error("Unexpected error: RAG chain not initialized.")
    with col2:
        with st.container():
            st.subheader("📋 Chat History")
            display_chat_history()

def process_documents(uploaded_files):
    """Process uploaded documents and add to vector store."""
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
                st.success(f"✅ Successfully processed document(s). You can now ask questions!", icon="🎉")
            else:
                st.session_state.documents_processed = False
                st.error("❌ No documents were successfully processed.", icon="🚫")
    except Exception as e:
        st.session_state.documents_processed = False
        st.error(f"Error processing documents: {str(e)}", icon="🚫")

def ask_question(question: str, show_sources: bool, show_context: bool):
    """Process user question and display answer."""
    try:
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.rag_chain.generate_answer(question)
        st.markdown("### 🧠 Answer")
        st.info(result["answer"])
        st.session_state.chat_history.append({
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"]
        })
        if show_sources and result["sources"]:
            st.markdown("### 📚 Sources")
            for i, source in enumerate(result["sources"], 1):
                with st.expander(f"📄 Source {i}: {source['filename']}"):
                    st.text(source['content_preview'])
        if show_context and result["context"]:
            with st.expander("🔍 Retrieved Context"):
                st.text_area("Context used for answering:", result["context"], height=200)
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")

def display_chat_history():
    """Display chat history."""
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"💭 Q: {chat['question'][:50]}..."):
                st.write("**A:**", chat['answer'])
    else:
        st.info("No questions asked yet.")

def clear_knowledge_base():
    """Clear the vector database."""
    if st.session_state.vectorstore:
        st.session_state.vectorstore.delete_collection()
        st.session_state.vectorstore = None
        st.session_state.rag_chain = None
        st.session_state.documents_processed = False
        st.success("🧹 Knowledge base cleared!")
        st.rerun()

if __name__ == "__main__":
    main()