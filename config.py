import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

class Config:
    # Ollama Configuration
    OLLAMA_BASE_URL = "http://localhost:11434"  # Default Ollama URL
    OLLAMA_MODEL = "mistral"  # Default model, can be changed to other models like mistral
    OLLAMA_TEMPERATURE = 0.7
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE = "chromadb"  # Options: chromadb, faiss
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Directories
    DATA_DIR = "data/documents"
    VECTOR_STORE_DIR = "vectorstore"
    
    # Streamlit Configuration
    PAGE_TITLE = "NetSec-Tutor"

    PAGE_ICON = "🤖"



