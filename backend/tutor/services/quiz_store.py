import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import Config
from ..logging_utils import get_app_logger

QUIZ_DIR_NAME = "quiz_history"
logger = get_app_logger()


def _quiz_dir() -> str:
    base = getattr(Config, "DATA_DIR", "data/documents")
    path = os.path.join(base, QUIZ_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _quiz_path(quiz_id: str) -> str:
    return os.path.join(_quiz_dir(), f"{quiz_id}.json")


def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def list_quiz_attempts() -> List[Dict[str, Any]]:
    """Return lightweight metadata for all saved quiz attempts."""
    directory = _quiz_dir()
    items: List[Dict[str, Any]] = []
    try:
        for name in os.listdir(directory):
            if not name.endswith(".json"):
                continue
            full = os.path.join(directory, name)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results = data.get("results", {}) or {}
                items.append(
                    {
                        "quiz_id": data.get("quiz_id", name[:-5]),
                        "title": data.get("title", "Quiz Attempt"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "graded_at": data.get("graded_at"),
                        "percentage": results.get("percentage"),
                        "topics": data.get("topics") or [],
                        "difficulty": data.get("difficulty"),
                        "source_categories": data.get("source_categories") or [],
                    }
                )
            except Exception:
                continue
    except Exception:
        pass

    def _sort_key(entry: Dict[str, Any]):
        return (
            entry.get("updated_at")
            or entry.get("graded_at")
            or entry.get("created_at")
            or ""
        )

    items.sort(key=_sort_key, reverse=True)
    logger.info("QuizStore.list_quiz_attempts count=%d", len(items))
    return items


def load_quiz_attempt(quiz_id: str) -> Optional[Dict[str, Any]]:
    path = _quiz_path(quiz_id)
    if not os.path.exists(path):
        logger.info("QuizStore.load_quiz_attempt missing quiz_id=%s", quiz_id)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "QuizStore.load_quiz_attempt quiz_id=%s topics=%d difficulty=%s",
            quiz_id,
            len(data.get("topics") or []),
            data.get("difficulty"),
        )
        return data
    except Exception as exc:
        logger.exception("QuizStore.load_quiz_attempt error quiz_id=%s: %s", quiz_id, exc)
        return None


def save_quiz_attempt(attempt: Dict[str, Any]) -> Dict[str, Any]:
    """Persist a quiz attempt to disk."""
    quiz_id = attempt.get("quiz_id") or str(uuid.uuid4())
    existing = load_quiz_attempt(quiz_id)
    now = _now_iso()

    data = existing or {}
    if not existing:
        data["quiz_id"] = quiz_id
        data["created_at"] = now

    data["title"] = attempt.get("title") or data.get("title") or "Quiz Attempt"
    data["mode"] = attempt.get("mode") or data.get("mode")
    data["topics"] = attempt.get("topics") or data.get("topics") or []
    data["difficulty"] = attempt.get("difficulty") or data.get("difficulty")
    data["source_categories"] = attempt.get("source_categories") or data.get("source_categories") or []
    data["results"] = attempt.get("results") or data.get("results") or {}
    data["quiz_data"] = attempt.get("quiz_data") or data.get("quiz_data")
    data["user_answers"] = attempt.get("user_answers") or data.get("user_answers") or {}
    data["graded_at"] = attempt.get("graded_at") or data.get("graded_at") or now
    data["updated_at"] = now

    path = _quiz_path(quiz_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(
        "QuizStore.save_quiz_attempt quiz_id=%s title='%s' topics=%d percentage=%s",
        quiz_id,
        data.get("title"),
        len(data.get("topics") or []),
        (data.get("results") or {}).get("percentage"),
    )
    return data


def delete_quiz_attempt(quiz_id: str) -> bool:
    path = _quiz_path(quiz_id)
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("QuizStore.delete_quiz_attempt quiz_id=%s deleted=True", quiz_id)
            return True
    except Exception as exc:
        logger.exception("QuizStore.delete_quiz_attempt error quiz_id=%s: %s", quiz_id, exc)
    return False
