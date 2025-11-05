"""Document summarization helpers."""
from __future__ import annotations

import os
import re
import json
from typing import Dict, Generator, List, Optional

import docx
import fitz  # PyMuPDF
import requests

from config import Config

from .knowledge_base import get_kb_manager


ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

SUMMARY_COURTESY_TERMS = re.compile(r"\b(?:please|thanks|thank you)\b.*$", re.IGNORECASE)


def _clean_doc_phrase(phrase: str) -> str:
    candidate = phrase.strip()
    if not candidate:
        return candidate
    candidate = SUMMARY_COURTESY_TERMS.sub("", candidate).strip()
    candidate = re.split(r"[.,!?\n]", candidate, maxsplit=1)[0].strip()
    return candidate


def _normalize_doc_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _extract_index(tokens: List[str]) -> Optional[int]:
    for token in tokens:
        if token.isdigit():
            try:
                return int(token)
            except Exception:
                continue
    for token in tokens:
        if token in ORDINAL_WORDS:
            return ORDINAL_WORDS[token]
    return None


def _match_kb_file(doc_phrase: str, files: List[str], category_hint: Optional[str] = None) -> Optional[str]:
    if not files:
        return None
    if (not doc_phrase or not doc_phrase.strip()) and len(files) == 1:
        return files[0]

    normalized_target = _normalize_doc_name(doc_phrase or "")
    if not normalized_target:
        return None

    tokens = normalized_target.split()
    index = _extract_index(tokens)

    category_keywords = {
        "TEXTBOOKs": {"textbook", "book", "tb"},
        "LECTURE SLIDEs": {"slide", "slides", "deck", "module", "presentation", "lecture"},
    }
    keywords = category_keywords.get(category_hint, set())

    if index is not None and 0 < index <= len(files):
        if not keywords or any(token in keywords for token in tokens):
            return files[index - 1]

    normalized_files = [
        (path, _normalize_doc_name(os.path.splitext(os.path.basename(path))[0]))
        for path in files
    ]

    for path, norm in normalized_files:
        if normalized_target == norm:
            return path

    for path, norm in normalized_files:
        if normalized_target in norm or norm in normalized_target:
            return path

    essential_tokens = [token for token in tokens if token not in keywords and len(token) > 1]
    if essential_tokens:
        for path, norm in normalized_files:
            if all(token in norm for token in essential_tokens):
                return path

    choices = [norm for _, norm in normalized_files]
    if choices:
        import difflib

        fuzzy = difflib.get_close_matches(normalized_target, choices, n=1, cutoff=0.6)
        if fuzzy:
            match_value = fuzzy[0]
            for path, norm in normalized_files:
                if norm == match_value:
                    return path

    return None


def _list_files_by_category() -> Dict[str, List[str]]:
    kb = get_kb_manager()
    results: Dict[str, List[str]] = {
        "TEXTBOOKs": [],
        "LECTURE SLIDEs": [],
    }
    files = kb.list_files()
    for key, items in files.items():
        display_key = "TEXTBOOKs" if key == "textbooks" else "LECTURE SLIDEs" if key == "slides" else "DOCUMENTs"
        target = results.setdefault(display_key, [])
        for item in items:
            path = item.get("path")
            if path:
                target.append(path)
    return results


def _read_document_slice(file_path: str, index: int, is_slide: bool) -> Optional[str]:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        try:
            doc = fitz.open(file_path)
            if index < 1 or index > doc.page_count:
                return None
            return doc.load_page(index - 1).get_text()
        except Exception:
            return None
    if ext == ".docx":
        try:
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception:
            return None
    if ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as handle:
                return handle.read()
        except Exception:
            return None
    return None


def ask_llm_summary_stream(
    text: str,
    num: int,
    filename: str,
    is_slide: bool = False,
) -> Generator[Dict[str, str], None, None]:
    """Stream summary tokens for a page or slide."""
    prompt = (
        f"Summarize the following {'slide' if is_slide else 'page'} "
        f"(number {num}) from {filename}:\n\n{text}\n\nSummary:"
    )
    url = f"{Config.OLLAMA_BASE_URL}/api/generate"
    try:
        with requests.post(
            url,
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": Config.OLLAMA_TEMPERATURE,
                "stream": True,
            },
            timeout=120,
            stream=True,
        ) as resp:
            if resp.status_code != 200:
                yield {"type": "error", "text": f"LLM error: {resp.text}"}
                return
            buffer = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    data = line.decode("utf-8")
                    payload = json.loads(data)
                except Exception:
                    continue
                token = payload.get("response", "")
                if token:
                    buffer += token
                    yield {"type": "token", "text": token, "buffer": buffer}
                if payload.get("done"):
                    break
            yield {"type": "final", "text": buffer}
    except Exception as exc:  # pragma: no cover - network/IO
        yield {"type": "error", "text": f"Error calling LLM: {exc}"}


def ask_llm_summary(
    text: str,
    num: int,
    filename: str,
    is_slide: bool = False,
) -> str:
    prompt = (
        f"Summarize the following {'slide' if is_slide else 'page'} "
        f"(number {num}) from {filename}:\n\n{text}\n\nSummary:"
    )
    try:
        response = requests.post(
            f"{Config.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "temperature": Config.OLLAMA_TEMPERATURE,
                "stream": False,
            },
            timeout=60,
        )
    except Exception as exc:  # pragma: no cover - network/IO
        return f"Error calling LLM: {exc}"
    if response.status_code == 200:
        data = response.json()
        return data.get("response", "[No summary returned]")
    return f"LLM error: {response.text}"


def handle_summarization_request(prompt: str) -> Dict[str, object]:
    """Interpret a user prompt and return summary results or error info."""
    page_pat = re.compile(r"summari[sz]e\s+page\s+(\d+)(?:\s+(?:of|from))?\s+(.*)", re.I)
    slide_pat = re.compile(
        r"summari[sz]e\s+slide(?:\s*(?:number|no\.?)?)?\s*(\d+)(?:\s+(?:of|from))?\s+(?:the\s+)?(.*)",
        re.I,
    )
    m_page = page_pat.search(prompt)
    m_slide = slide_pat.search(prompt)
    kb_files = _list_files_by_category()
    if m_page:
        page_num = int(m_page.group(1))
        doc_phrase = _clean_doc_phrase(m_page.group(2))
        files = kb_files.get("TEXTBOOKs", [])
        file_path = _match_kb_file(doc_phrase, files, category_hint="TEXTBOOKs")
        if not file_path:
            return {"error": f"No textbook found matching '{doc_phrase or 'your description'}'."}
        text = _read_document_slice(file_path, page_num, is_slide=False)
        if text is None:
            return {"error": f"Page {page_num} out of range for {os.path.basename(file_path)}."}
        summary = ask_llm_summary(text, page_num, os.path.basename(file_path), is_slide=False)
        citation = {
            "filename": os.path.basename(file_path),
            "page_number": page_num,
            "content_preview": text[:300] + ("..." if len(text) > 300 else ""),
            "full_text": text,
        }
        return {"summary": summary, "citation": citation}
    if m_slide:
        slide_num = int(m_slide.group(1))
        doc_phrase = _clean_doc_phrase(m_slide.group(2) or "")
        files = kb_files.get("LECTURE SLIDEs", [])
        file_path = _match_kb_file(doc_phrase, files, category_hint="LECTURE SLIDEs")
        if not file_path:
            return {"error": f"No lecture slide deck found matching '{doc_phrase or 'your description'}'."}
        text = _read_document_slice(file_path, slide_num, is_slide=True)
        if text is None:
            return {"error": f"Slide {slide_num} out of range for {os.path.basename(file_path)}."}
        summary = ask_llm_summary(text, slide_num, os.path.basename(file_path), is_slide=True)
        citation = {
            "filename": os.path.basename(file_path),
            "page_number": slide_num,
            "content_preview": text[:300] + ("..." if len(text) > 300 else ""),
            "full_text": text,
        }
        return {"summary": summary, "citation": citation}
    return {"error": "Unable to interpret the summarization request."}


__all__ = [
    "ask_llm_summary_stream",
    "ask_llm_summary",
    "handle_summarization_request",
]
