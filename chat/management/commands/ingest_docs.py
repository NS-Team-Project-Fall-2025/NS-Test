import os
from django.core.management.base import BaseCommand, CommandError
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
# --- This is the new, non-deprecated import ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- We no longer need a Google API key check for ingestion ---

DOCS_DIRECTORY = "docs"
VECTORSTORE_FILE = "faiss_index"

class Command(BaseCommand):
    help = "Ingests documents from the 'docs' directory into a FAISS vector store using free, local HuggingFace embeddings."

    def handle(self, *args, **options):
        if os.path.exists(VECTORSTORE_FILE):
            self.stdout.write(self.style.SUCCESS(f"Vector store '{VECTORSTORE_FILE}' already exists. Skipping ingestion."))
            return

        self.stdout.write("Starting document ingestion process...")
        
        # 1. Load Documents
        self.stdout.write(f"Loading documents from '{DOCS_DIRECTORY}'...")
        loader = DirectoryLoader(
            DOCS_DIRECTORY,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()

        if not documents:
            raise CommandError(f"No PDF documents found in '{DOCS_DIRECTORY}'.")

        self.stdout.write(f"Loaded {len(documents)} document pages.")

        # 2. Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        self.stdout.write(f"Split documents into {len(chunks)} chunks.")

        # 3. Create Embeddings and Store in FAISS
        self.stdout.write("Creating local embeddings with HuggingFace... (This may take a while the first time as it downloads the model)")
        try:
            # --- This uses the free, local model ---
            # Using a popular, lightweight model
            model_name = "all-MiniLM-L6-v2"
            model_kwargs = {'device': 'cpu'} # Use CPU
            encode_kwargs = {'normalize_embeddings': False}
            
            # --- This uses the new, non-deprecated class ---
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            vectorstore = FAISS.from_documents(chunks, embeddings)
        except Exception as e:
            # --- This error message is for a local model ---
            raise CommandError(f"Failed to create local embeddings. Error: {e}")
        
        # 4. Save Locally
        vectorstore.save_local(VECTORSTORE_FILE)
        self.stdout.write(self.style.SUCCESS(f"Vector store saved to '{VECTORSTORE_FILE}'."))

