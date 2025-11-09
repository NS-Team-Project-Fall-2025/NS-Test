import os
try:
    import pysqlite3
    import sys
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    import sqlite3

import chromadb
from typing import List, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import numpy as np
from langchain_core.documents import Document as LCDocument
from ..logging_utils import get_app_logger, summarize_text

logger = get_app_logger()

class VectorStore:
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "vectorstore"):
        self.embedding_model_name = embedding_model
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model
        )
        self.vectorstore = None
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        logger.info(
            "VectorStore.init model=%s persist_directory=%s",
            self.embedding_model_name,
            self.persist_directory,
        )
    
    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create a new vector store from documents."""
        logger.info("VectorStore.create_vectorstore documents=%d", len(documents))
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name="rag_collection"
        )
    
    def load_vectorstore(self) -> bool:
        """Load existing vector store."""
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="rag_collection"
            )
            logger.info("VectorStore.load_vectorstore success path=%s", self.persist_directory)
            return True
        except Exception as e:
            logger.exception("VectorStore.load_vectorstore failed: %s", e)
            return False
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to existing vector store."""
        if self.vectorstore is None:
            logger.info("VectorStore.add_documents auto-create store")
            self.create_vectorstore(documents)
        else:
            self.vectorstore.add_documents(documents)
            logger.info("VectorStore.add_documents appended=%d", len(documents))

    def _ensure_store(self, method: str) -> bool:
        if self.vectorstore is None:
            logger.warning("VectorStore.%s attempted with empty store", method)
            return False
        return True

    def _log_similarity_results(
        self,
        method: str,
        query: str,
        k: int,
        results: List[Tuple[Document, Optional[float]]],
        extra: Optional[str] = None,
    ) -> None:
        preview = summarize_text(query, 120)
        snippet_parts = []
        for doc, score in results[:5]:
            meta = doc.metadata or {}
            filename = meta.get("filename") or "Unknown"
            page = meta.get("page_number")
            label = f"{filename}{f'#{page}' if page else ''}"
            formatted_score = "n/a" if score is None else f"{score:.4f}"
            snippet_parts.append(f"{label}={formatted_score}")
        snippet = "; ".join(snippet_parts) if snippet_parts else "no-results"
        logger.info(
            "VectorStore.%s query='%s' k=%d hits=%d %s samples=[%s]",
            method,
            preview,
            k,
            len(results),
            extra or "",
            snippet,
        )

    def _similarity_with_scores(
        self,
        method: str,
        query: str,
        k: int,
    ) -> List[Tuple[Document, Optional[float]]]:
        if not self._ensure_store(method):
            return []
        try:
            scored = self.vectorstore.similarity_search_with_score(query, k=k)
        except Exception as exc:
            logger.exception("VectorStore.%s score retrieval failed, falling back: %s", method, exc)
            docs = self.vectorstore.similarity_search(query, k=k)
            scored = [(doc, None) for doc in docs]
        self._log_similarity_results(method, query, k, scored)
        return scored
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 4) -> List[Document]:
        """Perform similarity search and return relevant documents."""
        scored = self._similarity_with_scores("similarity_search", query, k)
        return [doc for doc, _ in scored]
    
    def similarity_search_with_score(self, 
                                   query: str, 
                                   k: int = 4) -> List[tuple]:
        """Perform similarity search with scores."""
        return self._similarity_with_scores("similarity_search_with_score", query, k)

    def similarity_search_filename_contains(self, query: str, k: int = 4, filename_contains: Optional[str] = None) -> List[Document]:
        """Similarity search but only keep results whose metadata.filename contains the given substring."""
        if not self._ensure_store("similarity_search_filename_contains"):
            return []
        if not filename_contains:
            return self.similarity_search(query, k=k)
        # pull more candidates, then filter
        candidates = self._similarity_with_scores("similarity_search_filename_contains", query, max(k * 5, 10))
        needle = filename_contains.lower()
        filtered: List[Document] = []
        for doc, score in candidates:
            try:
                fn = (doc.metadata or {}).get("filename")
                if fn and needle in fn.lower():
                    filtered.append(doc)
            except Exception:
                continue
            if len(filtered) >= k:
                break
        self._log_similarity_results(
            "similarity_search_filename_contains.filtered",
            query,
            k,
            [(doc, None) for doc in filtered],
            extra=f"filter='{filename_contains}'",
        )
        return filtered

    def similarity_search_with_score_filename_contains(self, query: str, k: int = 4, filename_contains: Optional[str] = None) -> List[tuple]:
        """Similarity search with scores filtered by filename substring."""
        if not self._ensure_store("similarity_search_with_score_filename_contains"):
            return []
        if not filename_contains:
            return self.similarity_search_with_score(query, k=k)
        candidates = self._similarity_with_scores(
            "similarity_search_with_score_filename_contains", query, max(k * 5, 10)
        )
        needle = filename_contains.lower()
        filtered: List[tuple] = []
        for doc, score in candidates:
            try:
                fn = (doc.metadata or {}).get("filename")
                if fn and needle in fn.lower():
                    filtered.append((doc, score))
            except Exception:
                continue
            if len(filtered) >= k:
                break
        self._log_similarity_results(
            "similarity_search_with_score_filename_contains.filtered",
            query,
            k,
            filtered,
            extra=f"filter='{filename_contains}'",
        )
        return filtered
    
    def delete_collection(self) -> None:
        """Delete the vector store collection."""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            self.vectorstore = None
            logger.info("VectorStore.delete_collection completed")
    
    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        if self.vectorstore is None:
            return {"status": "No collection loaded", "count": 0}
        
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            info = {"status": "Collection loaded", "count": count}
            logger.info("VectorStore.get_collection_info count=%d", count)
            return info
        except Exception as e:
            logger.exception("VectorStore.get_collection_info error: %s", e)
            return {"status": f"Error: {str(e)}", "count": 0}

    def get_page_document(self, page_number: int, filename_contains: Optional[str] = None) -> List[LCDocument]:
        """Fetch a specific PDF page by metadata.
        Returns a list with one Document if found, else empty list.
        Note: This bypasses vector similarity and reads the underlying collection.
        """
        if not self._ensure_store("get_page_document"):
            return []
        try:
            collection = self.vectorstore._collection
            data = collection.get(include=["metadatas", "documents", "ids"]) or {}
            metadatas = data.get("metadatas", []) or []
            documents = data.get("documents", []) or []
            results: List[LCDocument] = []
            for meta, text in zip(metadatas, documents):
                try:
                    pn_raw = meta.get("page_number") if isinstance(meta, dict) else None
                    # Normalize page number to int when possible (Chroma may store metadata as strings)
                    pn = None
                    if pn_raw is not None:
                        try:
                            pn = int(str(pn_raw).strip())
                        except Exception:
                            pn = pn_raw  # leave as-is if not convertible
                    fn = meta.get("filename") if isinstance(meta, dict) else None
                    src = meta.get("source") if isinstance(meta, dict) else None
                    if pn == page_number and (not filename_contains or (fn and filename_contains.lower() in fn.lower())):
                        results.append(LCDocument(page_content=text or "", metadata={"page_number": pn, "filename": fn, "source": src}))
                        # Only return the first match (one page)
                        break
                except Exception:
                    continue
            logger.info(
                "VectorStore.get_page_document page=%s filter='%s' hits=%d",
                page_number,
                filename_contains or "",
                len(results),
            )
            return results
        except Exception as exc:
            logger.exception("VectorStore.get_page_document error page=%s: %s", page_number, exc)
            return []
    
    def retrieve_by_topics(self, topics: list, num_contexts: int = 10) -> list:
        """Retrieve relevant documents for each topic using similarity search."""
        results = []
        k = max(1, num_contexts // max(1, len(topics)))
        for topic in topics:
            docs = self.similarity_search(topic, k=k)
            for doc in docs:
                results.append({
                    "content": doc.page_content if hasattr(doc, 'page_content') else getattr(doc, 'content', ''),
                    "metadata": getattr(doc, 'metadata', {})
                })
        # Deduplicate by content
        seen = set()
        unique_results = []
        for r in results:
            c = r["content"]
            if c not in seen:
                unique_results.append(r)
                seen.add(c)
            if len(unique_results) >= num_contexts:
                break
        logger.info(
            "VectorStore.retrieve_by_topics topics=%d unique_contexts=%d limit=%d",
            len(topics),
            len(unique_results),
            num_contexts,
        )
        return unique_results

