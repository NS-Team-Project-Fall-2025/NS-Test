import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from config import Config


SESSIONS_DIR_NAME = "chat_sessions"


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
    return sessions


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    path = _session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
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
    return data


def delete_session(session_id: str) -> bool:
    path = _session_path(session_id)
    try:
        if os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False


def create_new_session(title: Optional[str] = None) -> Dict[str, Any]:
    sid = str(uuid.uuid4())
    data = save_session(sid, messages=[], title=title)
    return data


def most_recent_session() -> Optional[Dict[str, Any]]:
    sessions = list_sessions()
    if not sessions:
        return None
    # Load full content for the first (most recent) entry
    return load_session(sessions[0]["session_id"]) or None
