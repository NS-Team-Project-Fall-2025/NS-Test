"""Quiz orchestration helpers."""
from __future__ import annotations

import threading
from typing import Iterable, List, Sequence, Tuple

from .knowledge_base import KnowledgeBaseManager, get_kb_manager
from .quiz_agent import QuizAgent
from .vector_store import VectorStore
from ..logging_utils import get_app_logger


ALLOWED_QUIZ_CATEGORIES: Tuple[str, str] = ("textbooks", "slides")
logger = get_app_logger()


class QuizService:
    """Manages QuizAgent instances keyed by requested knowledge base categories."""

    def __init__(self, kb_manager: KnowledgeBaseManager | None = None) -> None:
        self.kb_manager = kb_manager or get_kb_manager()
        self._agents: dict[Tuple[str, ...], QuizAgent] = {}
        self._lock = threading.Lock()

    def get_agent(self, categories: Iterable[str] | None = None) -> QuizAgent:
        normalized = self._normalize_categories(categories)
        key = tuple(normalized)
        with self._lock:
            agent = self._agents.get(key)
            if agent is None:
                stores = self._collect_vector_stores(normalized)
                if not stores:
                    raise ValueError("Selected knowledge base is not ready for quiz generation.")
                agent = QuizAgent(stores)
                self._agents[key] = agent
                logger.info("QuizService.get_agent created key=%s stores=%d", key, len(stores))
            else:
                logger.info("QuizService.get_agent cache-hit key=%s", key)
            return agent

    def normalize_categories(self, categories: Iterable[str] | None = None) -> List[str]:
        """Public helper to validate/normalize category selections."""
        return self._normalize_categories(categories)

    def invalidate(self) -> None:
        with self._lock:
            self._agents.clear()
        logger.info("QuizService.invalidate cleared cached quiz agents")

    def _normalize_categories(self, categories: Iterable[str] | None) -> List[str]:
        if not categories:
            return list(ALLOWED_QUIZ_CATEGORIES)
        normalized: List[str] = []
        for cat in categories:
            value = (cat or "").strip().lower()
            if not value:
                continue
            if value not in ALLOWED_QUIZ_CATEGORIES:
                raise ValueError(f"Unsupported knowledge base category '{cat}'.")
            if value not in normalized:
                normalized.append(value)
        if not normalized:
            raise ValueError("No valid knowledge base categories selected.")
        return normalized

    def _collect_vector_stores(self, categories: Sequence[str]) -> List[VectorStore]:
        stores: List[VectorStore] = []
        for category in categories:
            store = self.kb_manager.ensure_vector_store(category)
            if store is not None and getattr(store, "vectorstore", None) is not None:
                stores.append(store)
        logger.info(
            "QuizService._collect_vector_stores categories=%s stores=%d",
            list(categories),
            len(stores),
        )
        return stores


_QUIZ_SERVICE: QuizService | None = None


def get_quiz_service() -> QuizService:
    global _QUIZ_SERVICE
    if _QUIZ_SERVICE is None:
        _QUIZ_SERVICE = QuizService()
    return _QUIZ_SERVICE
