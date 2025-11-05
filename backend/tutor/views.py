"""Tutor API views."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from django.http import (
    HttpRequest,
    HttpResponse,
    HttpResponseBadRequest,
    HttpResponseNotAllowed,
    JsonResponse,
    StreamingHttpResponse,
)
from django.views.decorators.csrf import csrf_exempt

from config import Config

from .services.knowledge_base import KnowledgeBaseManager, get_kb_manager
from .services.quiz_service import get_quiz_service
from .services.rag_service import get_rag_service
from .services.session_store import (
    create_new_session,
    delete_session,
    list_sessions,
    load_session,
    save_session,
)
from .services.quiz_store import (
    delete_quiz_attempt,
    list_quiz_attempts,
    load_quiz_attempt,
    save_quiz_attempt,
)
from .services.summarizer import (
    ask_llm_summary_stream,
    handle_summarization_request,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _json_body(request: HttpRequest) -> Dict[str, Any]:
    if not request.body:
        return {}
    try:
        return json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc


def _json(method: str, payload: Dict[str, Any], status: int = 200) -> JsonResponse:
    return JsonResponse(payload, status=status)


def _options_ok() -> JsonResponse:
    return JsonResponse({"status": "ok"})


def _method_not_allowed(allowed: Iterable[str]) -> HttpResponseNotAllowed:
    return HttpResponseNotAllowed(allowed)


def _ensure_category(manager: KnowledgeBaseManager, category: str) -> str:
    category = (category or "").lower()
    if category not in manager.list_categories():
        raise ValueError(f"Unsupported category '{category}'. Valid options: {manager.list_categories()}")
    return category


# --------------------------------------------------------------------------- #
# Basic metadata
# --------------------------------------------------------------------------- #
def health_check(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "GET":
        return _method_not_allowed(["GET", "OPTIONS"])
    manager = get_kb_manager()
    status = {
        "status": "ok",
        "categories": manager.list_categories(),
    }
    return _json("GET", status)


def api_config(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "GET":
        return _method_not_allowed(["GET", "OPTIONS"])
    payload = {
        "pageTitle": Config.PAGE_TITLE,
        "pageIcon": Config.PAGE_ICON,
        "ollama": {
            "baseUrl": Config.OLLAMA_BASE_URL,
            "model": Config.OLLAMA_MODEL,
            "temperature": Config.OLLAMA_TEMPERATURE,
        },
        "knowledgeBase": {
            "categories": get_kb_manager().list_categories(),
        },
    }
    return _json("GET", payload)


# --------------------------------------------------------------------------- #
# Knowledge base management
# --------------------------------------------------------------------------- #
def list_knowledge_base_files(request: HttpRequest) -> JsonResponse:
    manager = get_kb_manager()
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "GET":
        return _method_not_allowed(["GET", "OPTIONS"])
    category = request.GET.get("category")
    if category:
        try:
            category = _ensure_category(manager, category)
        except ValueError as exc:
            return _json("GET", {"error": str(exc)}, status=400)
    files = manager.list_files(category=category) if category else manager.list_files()
    return _json("GET", {"files": files})


@csrf_exempt
def upload_knowledge_base_file(request: HttpRequest) -> JsonResponse:
    manager = get_kb_manager()
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    uploads = request.FILES.getlist("files")
    if not uploads:
        single = request.FILES.get("file")
        if single:
            uploads = [single]
    if not uploads:
        return _json("POST", {"error": "No file uploaded."}, status=400)
    category = request.POST.get("category", "textbooks")
    try:
        category = _ensure_category(manager, category)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    cfg = KnowledgeBaseManager._category_config()[category]  # type: ignore[attr-defined]
    dest_dir = cfg["data_dir"]
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    stored: List[str] = []
    for upload in uploads:
        dest_path = Path(dest_dir) / upload.name
        with open(dest_path, "wb") as handle:
            for chunk in upload.chunks():
                handle.write(chunk)
        stored.append(str(dest_path))
    # Invalidate quiz agent to ensure refreshed context on next access
    get_quiz_service().invalidate()
    return _json(
        "POST",
        {
            "message": "Upload successful.",
            "paths": stored,
            "filenames": [Path(path).name for path in stored],
            "category": category,
        },
    )


@csrf_exempt
def rebuild_knowledge_base(request: HttpRequest) -> JsonResponse:
    manager = get_kb_manager()
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    categories = payload.get("categories") or []
    if categories and not isinstance(categories, list):
        return _json("POST", {"error": "categories must be a list of category names."}, status=400)
    if not categories:
        categories = manager.list_categories()
    rebuilt: List[str] = []
    errors: Dict[str, str] = {}
    rebuild_details: Dict[str, List[str]] = {}
    for cat in categories:
        try:
            cat_name = _ensure_category(manager, cat)
        except ValueError as exc:
            errors[str(cat)] = str(exc)
            continue
        try:
            result = manager.rebuild_vector_store(cat_name)
            if isinstance(result, tuple):
                _store, processed_files = result
            else:
                _store, processed_files = result, []
            rebuilt.append(cat_name)
            if processed_files:
                rebuild_details[cat_name] = [Path(path).name for path in processed_files]
        except Exception as exc:  # pragma: no cover - IO/DB
            errors[cat_name] = str(exc)
    if rebuilt:
        get_quiz_service().invalidate()
    return _json("POST", {"rebuilt": rebuilt, "errors": errors, "details": rebuild_details})


@csrf_exempt
def clear_knowledge_base(request: HttpRequest) -> JsonResponse:
    manager = get_kb_manager()
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    categories = payload.get("categories")
    if categories is not None and not isinstance(categories, list):
        return _json("POST", {"error": "categories must be a list of category names."}, status=400)
    targets = categories or manager.list_categories()
    cleared: Dict[str, Dict[str, int]] = {}
    errors: Dict[str, str] = {}
    for cat in targets:
        try:
            cat_name = _ensure_category(manager, cat)
        except ValueError as exc:
            errors[str(cat)] = str(exc)
            continue
        try:
            cleared[cat_name] = manager.clear_category(cat_name)
        except Exception as exc:  # pragma: no cover - IO/FS
            errors[cat_name] = str(exc)
    if cleared:
        get_quiz_service().invalidate()
    return _json("POST", {"cleared": cleared, "errors": errors})


# --------------------------------------------------------------------------- #
# Tutor chat endpoints
# --------------------------------------------------------------------------- #
@csrf_exempt
def tutor_chat_stream(request: HttpRequest) -> HttpResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    question = (payload.get("question") or "").strip()
    if not question:
        return _json("POST", {"error": "question is required."}, status=400)
    history = payload.get("history") or []
    if not isinstance(history, list):
        return _json("POST", {"error": "history must be a list."}, status=400)
    mode = payload.get("mode")
    rag_service = get_rag_service()
    try:
        chain, stores = rag_service.get_chain(mode)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)

    def _stream():
        try:
            generator = chain.chat_with_context_stream(question, chat_history=history)
        except AttributeError:
            # Fallback to non-streaming response.
            try:
                result = chain.chat_with_context(question, chat_history=history)
            except Exception as exc:  # pragma: no cover - network
                yield json.dumps({"type": "error", "error": str(exc)}) + "\n"
                return
            result["type"] = "final"
            yield json.dumps(result) + "\n"
            return
        try:
            for event in generator:
                yield json.dumps(event) + "\n"
        except Exception as exc:  # pragma: no cover - network
            yield json.dumps({"type": "error", "error": str(exc)}) + "\n"

    response = StreamingHttpResponse(_stream(), content_type="application/x-ndjson")
    response["Cache-Control"] = "no-cache"
    return response


# --------------------------------------------------------------------------- #
# Summaries
# --------------------------------------------------------------------------- #
@csrf_exempt
def summarize_document(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return _json("POST", {"error": "prompt is required."}, status=400)
    result = handle_summarization_request(prompt)
    return _json("POST", result)


@csrf_exempt
def summarize_document_stream(request: HttpRequest) -> HttpResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    text = payload.get("text")
    filename = payload.get("filename")
    number = payload.get("number")
    if not all([text, filename, number]):
        return _json(
            "POST",
            {"error": "text, filename, and number are required for streaming summary."},
            status=400,
        )
    is_slide = bool(payload.get("isSlide"))

    def _stream():
        for event in ask_llm_summary_stream(text, int(number), filename, is_slide=is_slide):
            yield json.dumps(event) + "\n"

    response = StreamingHttpResponse(_stream(), content_type="application/x-ndjson")
    response["Cache-Control"] = "no-cache"
    return response


# --------------------------------------------------------------------------- #
# Quiz endpoints
# --------------------------------------------------------------------------- #
@csrf_exempt
def generate_quiz(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)

    difficulty = (payload.get("difficulty") or "medium").strip().lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    raw_categories = payload.get("source_categories") or payload.get("sources")
    if isinstance(raw_categories, str):
        categories = [raw_categories]
    elif isinstance(raw_categories, list):
        categories = raw_categories
    else:
        categories = None

    service = get_quiz_service()
    try:
        normalized_categories = service.normalize_categories(categories)
        agent = service.get_agent(normalized_categories)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)

    try:
        quiz = agent.generate_quiz(
            num_mcq=int(payload.get("num_mcq", 2)),
            num_true_false=int(payload.get("num_true_false", 2)),
            num_open_ended=int(payload.get("num_open_ended", 1)),
            topics=payload.get("topics"),
            mode=payload.get("mode", "random"),
            difficulty=difficulty,
            source_categories=normalized_categories,
        )
    except Exception as exc:
        return _json("POST", {"error": str(exc)}, status=500)

    return _json("POST", quiz)


@csrf_exempt
def grade_quiz(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    quiz_data = payload.get("quiz_data")
    user_answers = payload.get("user_answers")
    if not quiz_data or not user_answers:
        return _json("POST", {"error": "quiz_data and user_answers are required."}, status=400)
    source_categories = quiz_data.get("source_categories")
    if isinstance(source_categories, str):
        source_categories = [source_categories]

    service = get_quiz_service()
    try:
        normalized_categories = service.normalize_categories(source_categories)
        agent = service.get_agent(normalized_categories)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    try:
        results = agent.grade_quiz(quiz_data, user_answers)
    except Exception as exc:
        return _json("POST", {"error": str(exc)}, status=500)

    attempt_payload = {
        "title": quiz_data.get("title"),
        "mode": quiz_data.get("mode"),
        "topics": quiz_data.get("topics"),
        "difficulty": quiz_data.get("difficulty"),
        "source_categories": quiz_data.get("source_categories"),
        "results": results,
        "quiz_data": quiz_data,
        "user_answers": user_answers,
    }
    try:
        saved = save_quiz_attempt(attempt_payload)
    except Exception as exc:  # pragma: no cover - IO
        saved = {"error": str(exc)}
    return _json("POST", {"results": results, "attempt": saved})


# --------------------------------------------------------------------------- #
# Quiz history endpoints
# --------------------------------------------------------------------------- #
def list_quiz_history(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "GET":
        return _method_not_allowed(["GET", "OPTIONS"])
    attempts = list_quiz_attempts()
    return _json("GET", {"attempts": attempts})


def quiz_history_detail(request: HttpRequest, quiz_id: str) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method == "GET":
        attempt = load_quiz_attempt(quiz_id)
        if attempt is None:
            return _json("GET", {"error": "Quiz attempt not found."}, status=404)
        return _json("GET", attempt)
    if request.method == "DELETE":
        deleted = delete_quiz_attempt(quiz_id)
        return _json("DELETE", {"deleted": deleted})
    return _method_not_allowed(["GET", "DELETE", "OPTIONS"])


# --------------------------------------------------------------------------- #
# Chat session endpoints
# --------------------------------------------------------------------------- #
def list_chat_sessions(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "GET":
        return _method_not_allowed(["GET", "OPTIONS"])
    sessions = list_sessions()
    return _json("GET", {"sessions": sessions})


@csrf_exempt
def create_chat_session(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method != "POST":
        return _method_not_allowed(["POST", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json("POST", {"error": str(exc)}, status=400)
    title = payload.get("title")
    session = create_new_session(title=title)
    return _json("POST", session)


@csrf_exempt
def chat_session_detail(request: HttpRequest, session_id: str) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method == "GET":
        session = load_session(session_id)
        if session is None:
            return _json("GET", {"error": "Session not found."}, status=404)
        return _json("GET", session)
    if request.method == "DELETE":
        deleted = delete_session(session_id)
        return _json("DELETE", {"deleted": deleted})
    return _method_not_allowed(["GET", "DELETE", "OPTIONS"])


@csrf_exempt
def update_chat_session(request: HttpRequest, session_id: str) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_ok()
    if request.method not in {"PUT", "PATCH"}:
        return _method_not_allowed(["PUT", "PATCH", "OPTIONS"])
    try:
        payload = _json_body(request)
    except ValueError as exc:
        return _json(request.method, {"error": str(exc)}, status=400)
    messages = payload.get("messages")
    title = payload.get("title")
    if messages is None:
        return _json(request.method, {"error": "messages array is required."}, status=400)
    updated = save_session(session_id, messages=messages, title=title)
    return _json(request.method, updated)
