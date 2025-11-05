import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables early so Config picks up overrides.
load_dotenv()


class Config:
    """Shared configuration for both backend services and worker scripts."""

    PROJECT_ROOT = Path(
        os.getenv("NETSEC_PROJECT_ROOT", Path(__file__).resolve().parent)
    )

    # Ollama Configuration
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.7"))

    # Vector Store Configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chromadb")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

    # Directories
    DATA_DIR = os.getenv(
        "NETSEC_DATA_DIR", str(PROJECT_ROOT / "data" / "documents")
    )
    VECTOR_STORE_DIR = os.getenv(
        "NETSEC_VECTOR_DIR", str(PROJECT_ROOT / "vectorstore")
    )

    DATA_DIR_TEXTBOOKS = os.getenv(
        "NETSEC_DATA_TEXTBOOKS", str(PROJECT_ROOT / "data" / "textbooks")
    )
    DATA_DIR_SLIDES = os.getenv(
        "NETSEC_DATA_SLIDES", str(PROJECT_ROOT / "data" / "slides")
    )
    VECTOR_STORE_DIR_TEXTBOOKS = os.getenv(
        "NETSEC_VECTOR_TEXTBOOKS", str(PROJECT_ROOT / "vectorstore" / "textbooks")
    )
    VECTOR_STORE_DIR_SLIDES = os.getenv(
        "NETSEC_VECTOR_SLIDES", str(PROJECT_ROOT / "vectorstore" / "slides")
    )

    # Frontend defaults (may be used for metadata responses)
    PAGE_TITLE = os.getenv("NETSEC_PAGE_TITLE", "NetSec Tutor")
    PAGE_ICON = os.getenv("NETSEC_PAGE_ICON", "")



