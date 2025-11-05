"""Knowledge base management utilities."""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional

from config import Config

from .document_processor import DocumentProcessor
from .vector_store import VectorStore


KBCategory = str


class KnowledgeBaseManager:
    """Load, index, and expose knowledge base assets."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        self._stores: Dict[KBCategory, VectorStore] = {}
        self._init_categories()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_categories(self) -> List[KBCategory]:
        return list(self._category_config().keys())

    def list_files(self, category: Optional[KBCategory] = None) -> Dict[str, List[Dict[str, str]]]:
        """Return file metadata for one or all categories."""
        cats = (
            [category]
            if category
            else self.list_categories()
        )
        result: Dict[str, List[Dict[str, str]]] = {}
        for cat in cats:
            cfg = self._category_config().get(cat)
            if not cfg:
                continue
            folder = Path(cfg["data_dir"])
            items: List[Dict[str, str]] = []
            if folder.is_dir():
                for name in sorted(folder.iterdir()):
                    if not name.is_file():
                        continue
                    if name.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                        continue
                    stat = name.stat()
                    items.append(
                        {
                            "filename": name.name,
                            "path": str(name.resolve()),
                            "size_bytes": str(stat.st_size),
                            "modified": stat.st_mtime_ns and str(stat.st_mtime_ns),
                        }
                    )
            result[cat] = items
        return result

    def ensure_vector_store(self, category: KBCategory) -> Optional[VectorStore]:
        """Load or build the vector store for the requested category."""
        cfg = self._category_config().get(category)
        if not cfg:
            return None
        with self._lock:
            store = self._stores.get(category)
            if store is None:
                store = VectorStore(
                    embedding_model=Config.EMBEDDING_MODEL,
                    persist_directory=cfg["vector_dir"],
                )
                self._stores[category] = store
            try:
                loaded = store.load_vectorstore()
            except Exception:
                loaded = False
            if loaded:
                return store
            # Build from scratch if data exists
            documents, _ = self._collect_documents(cfg["data_dir"])
            if not documents:
                return store
            store.create_vectorstore(documents)
            return store

    def rebuild_vector_store(self, category: KBCategory) -> Optional[tuple[VectorStore, List[str]]]:
        """Force a rebuild of the vector store from the raw documents.

        Returns a tuple of (VectorStore, processed file paths) when successful.
        """
        cfg = self._category_config().get(category)
        if not cfg:
            return None
        documents, file_paths = self._collect_documents(cfg["data_dir"])
        with self._lock:
            store = self._stores.get(category)
            if store is None:
                store = VectorStore(
                    embedding_model=Config.EMBEDDING_MODEL,
                    persist_directory=cfg["vector_dir"],
                )
                self._stores[category] = store
            if documents:
                store.delete_collection()
                store.create_vectorstore(documents)
            return store, file_paths

    def get_vector_store(self, category: KBCategory) -> Optional[VectorStore]:
        """Return the store if it exists without loading/building."""
        return self._stores.get(category)

    def clear_category(self, category: KBCategory) -> Dict[str, int]:
        """Remove all raw documents and vector data for a category."""
        cfg = self._category_config().get(category)
        if not cfg:
            raise ValueError(f"Unsupported category '{category}'")
        data_dir = Path(cfg["data_dir"])
        vector_dir = Path(cfg["vector_dir"])
        removed_files = 0
        removed_vectors = 0
        # Count existing files before deletion for reporting
        if data_dir.exists():
            removed_files = sum(1 for path in data_dir.rglob("*") if path.is_file())
        if vector_dir.exists():
            removed_vectors = sum(1 for path in vector_dir.rglob("*") if path.is_file() or path.is_dir())
        with self._lock:
            store = self._stores.pop(category, None)
            if store is not None:
                try:
                    store.delete_collection()
                except Exception:
                    pass
        if data_dir.exists():
            shutil.rmtree(data_dir, ignore_errors=True)
        if vector_dir.exists():
            shutil.rmtree(vector_dir, ignore_errors=True)
        # Recreate empty directories so later uploads succeed
        data_dir.mkdir(parents=True, exist_ok=True)
        vector_dir.mkdir(parents=True, exist_ok=True)
        return {
            "files_removed": removed_files,
            "vector_items_removed": removed_vectors,
        }

    def clear_all(self, categories: Optional[List[KBCategory]] = None) -> Dict[KBCategory, Dict[str, int]]:
        """Clear every requested category."""
        cats = categories if categories is not None else self.list_categories()
        result: Dict[KBCategory, Dict[str, int]] = {}
        for cat in cats:
            cfg = self._category_config().get(cat)
            if not cfg:
                continue
            result[cat] = self.clear_category(cat)
        return result

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _init_categories(self) -> None:
        for cfg in self._category_config().values():
            Path(cfg["data_dir"]).mkdir(parents=True, exist_ok=True)
            Path(cfg["vector_dir"]).mkdir(parents=True, exist_ok=True)

    def _collect_documents(self, data_dir: str):
        folder = Path(data_dir)
        if not folder.is_dir():
            return [], []
        files = [
            str(path)
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        if not files:
            return [], []
        documents = self._processor.process_multiple_documents(files)
        return documents, files

    @staticmethod
    def _category_config() -> Dict[KBCategory, Dict[str, str]]:
        """Configuration for each knowledge base segment."""
        return {
            "textbooks": {
                "data_dir": Config.DATA_DIR_TEXTBOOKS,
                "vector_dir": Config.VECTOR_STORE_DIR_TEXTBOOKS,
            },
            "slides": {
                "data_dir": Config.DATA_DIR_SLIDES,
                "vector_dir": Config.VECTOR_STORE_DIR_SLIDES,
            },
        }


# Singleton-style accessor used by views.
_KB_MANAGER: Optional[KnowledgeBaseManager] = None
_KB_LOCK = threading.Lock()


def get_kb_manager() -> KnowledgeBaseManager:
    global _KB_MANAGER
    with _KB_LOCK:
        if _KB_MANAGER is None:
            _KB_MANAGER = KnowledgeBaseManager()
        return _KB_MANAGER
