"""Factory helpers for building RAG chains."""
from __future__ import annotations

from typing import List, Tuple

from .knowledge_base import KnowledgeBaseManager, get_kb_manager
from .rag_chain import CombinedRAGChain, RAGChain
from .vector_store import VectorStore


class RAGService:
    """Builds RAG chains using the knowledge base manager."""

    DEFAULT_MODE = "combined"

    def __init__(self, kb_manager: KnowledgeBaseManager | None = None) -> None:
        self.kb_manager = kb_manager or get_kb_manager()

    def get_chain(
        self,
        mode: str | None = None,
    ) -> Tuple[object, List[VectorStore]]:
        mode = (mode or self.DEFAULT_MODE).lower()
        if mode == "textbooks":
            store = self.kb_manager.ensure_vector_store("textbooks")
            if store is None:
                raise ValueError("Textbook vector store is not available.")
            return RAGChain(store), [store]
        if mode == "slides":
            store = self.kb_manager.ensure_vector_store("slides")
            if store is None:
                raise ValueError("Lecture slides vector store is not available.")
            return RAGChain(store), [store]

        stores = self._collect_available_stores()
        if not stores:
            raise ValueError("No knowledge base is initialized yet.")
        if len(stores) == 1:
            return RAGChain(stores[0]), stores
        return CombinedRAGChain(stores), stores

    def _collect_available_stores(self) -> List[VectorStore]:
        stores: List[VectorStore] = []
        for category in ["textbooks", "slides"]:
            store = self.kb_manager.ensure_vector_store(category)
            if store is not None and getattr(store, "vectorstore", None) is not None:
                stores.append(store)
        return stores


_RAG_SERVICE: RAGService | None = None


def get_rag_service() -> RAGService:
    global _RAG_SERVICE
    if _RAG_SERVICE is None:
        _RAG_SERVICE = RAGService()
    return _RAG_SERVICE
