import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import Config

QUIZ_DIR_NAME = "quiz_history"


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
    return items


def load_quiz_attempt(quiz_id: str) -> Optional[Dict[str, Any]]:
    path = _quiz_path(quiz_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
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
    data["results"] = attempt.get("results") or data.get("results") or {}
    data["quiz_data"] = attempt.get("quiz_data") or data.get("quiz_data")
    data["user_answers"] = attempt.get("user_answers") or data.get("user_answers") or {}
    data["graded_at"] = attempt.get("graded_at") or data.get("graded_at") or now
    data["updated_at"] = now

    path = _quiz_path(quiz_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def delete_quiz_attempt(quiz_id: str) -> bool:
    path = _quiz_path(quiz_id)
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False
