import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from config import Config
from ..logging_utils import get_app_logger, summarize_text


SESSIONS_DIR_NAME = "chat_sessions"
logger = get_app_logger()


def _sessions_dir() -> str:
    base = Config.DATA_DIR
    path = os.path.join(base, SESSIONS_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def _session_path(session_id: str) -> str:
    return os.path.join(_sessions_dir(), f"{session_id}.json")


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def list_sessions() -> List[Dict[str, Any]]:
    """Return lightweight metadata for all saved chat sessions."""
    dir_path = _sessions_dir()
    sessions: List[Dict[str, Any]] = []
    try:
        for name in os.listdir(dir_path):
            if not name.endswith(".json"):
                continue
            full = os.path.join(dir_path, name)
            try:
                with open(full, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": data.get("session_id", name[:-5]),
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "message_count": len(data.get("messages", [])),
                })
            except Exception:
                # Skip corrupted files
                continue
    except Exception:
        pass

    # Sort by updated_at desc, then created_at desc
    def _sort_key(s: Dict[str, Any]):
        return (
            s.get("updated_at") or s.get("created_at") or "",
            s.get("created_at") or "",
        )
    sessions.sort(key=_sort_key, reverse=True)
    logger.info("SessionStore.list_sessions count=%d", len(sessions))
    return sessions


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    path = _session_path(session_id)
    if not os.path.exists(path):
        logger.info("SessionStore.load_session missing session_id=%s", session_id)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(
            "SessionStore.load_session session_id=%s message_count=%d",
            session_id,
            len(data.get("messages", [])),
        )
        return data
    except Exception as exc:
        logger.exception("SessionStore.load_session error session_id=%s: %s", session_id, exc)
        return None


def save_session(session_id: str, messages: List[Dict[str, Any]], title: Optional[str] = None) -> Dict[str, Any]:
    """Persist the given messages for the session id. Creates file if missing."""
    path = _session_path(session_id)
    existing = load_session(session_id)
    now = _now_iso()
    if not existing:
        data = {
            "session_id": session_id,
            "title": title or f"Chat {now[:10]}",
            "created_at": now,
            "updated_at": now,
            "messages": messages or [],
        }
    else:
        data = existing
        if title is not None:
            data["title"] = title
        data["messages"] = messages or []
        data["updated_at"] = now

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    preview = summarize_text(messages[-1]["content"] if messages else "", 80)  # type: ignore[index]
    logger.info(
        "SessionStore.save_session session_id=%s title='%s' messages=%d preview='%s'",
        session_id,
        data.get("title"),
        len(messages or []),
        preview,
    )
    return data


def delete_session(session_id: str) -> bool:
    path = _session_path(session_id)
    try:
        if os.path.exists(path):
            os.remove(path)
            logger.info("SessionStore.delete_session session_id=%s deleted=True", session_id)
            return True
    except Exception as exc:
        logger.exception("SessionStore.delete_session error session_id=%s: %s", session_id, exc)
    return False


def create_new_session(title: Optional[str] = None) -> Dict[str, Any]:
    sid = str(uuid.uuid4())
    data = save_session(sid, messages=[], title=title)
    logger.info("SessionStore.create_new_session session_id=%s title='%s'", sid, data.get("title"))
    return data


def most_recent_session() -> Optional[Dict[str, Any]]:
    sessions = list_sessions()
    if not sessions:
        return None
    # Load full content for the first (most recent) entry
    return load_session(sessions[0]["session_id"]) or None
