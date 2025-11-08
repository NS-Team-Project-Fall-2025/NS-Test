import json
import re
from typing import List, Dict, Any, Optional, Sequence, Tuple
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from .vector_store import VectorStore
from config import Config

OUTPUT_FORMAT_INSTRUCTIONS = (
    "\nOutput formatting requirements:\n"
    "1. The very first line must be a single-line JSON object containing a boolean field named \"show_sources\" "
    '(example: {"show_sources": true}).\n'
    "   - Use true when your answer is grounded in the provided context snippets.\n"
    "   - Use false only when you must refuse because the answer is not present in the provided context.\n"
    "2. After a blank line, write the natural-language answer exactly as it should appear to the user.\n"
    "3. Do not emit any other JSON or text before the JSON control line.\n"
)

REFUSAL_TEXT = "I can only assist with content from the provided Network Security course materials."
_STOP_WORDS = {
    "the",
    "this",
    "that",
    "with",
    "from",
    "what",
    "when",
    "where",
    "your",
    "about",
    "explain",
    "describe",
    "give",
    "tell",
    "does",
    "have",
    "into",
    "which",
    "will",
    "them",
    "they",
    "their",
    "page",
    "summarize",
    "summary",
    "detail",
    "details",
    "need",
    "help",
    "please",
}


def _with_output_format(prompt: str) -> str:
    """Append the standardized output-format instructions to a prompt."""
    return f"{prompt}{OUTPUT_FORMAT_INSTRUCTIONS}"


def _split_structured_answer(raw: Any) -> Tuple[bool, str]:
    """Split a model response into (show_sources, answer_text) according to the control JSON header."""
    if not isinstance(raw, str):
        text = str(raw)
    else:
        text = raw
    stripped = text.lstrip()
    newline_idx = stripped.find("\n")
    if not stripped.startswith("{") or newline_idx == -1:
        return True, text.strip()
    header_line = stripped[:newline_idx].strip()
    try:
        header = json.loads(header_line)
    except Exception:
        return True, text.strip()
    show_sources = bool(header.get("show_sources", True))
    body = stripped[newline_idx + 1 :].lstrip("\r\n")
    return show_sources, body.strip()


def _consume_control_prefix(buffer: str) -> Optional[Tuple[bool, int]]:
    """Attempt to parse the control JSON from the beginning of a streaming buffer.

    Returns (show_sources, end_index) if successful, where end_index is the position in the original buffer
    immediately after the JSON line and any trailing blank lines.
    """
    stripped = buffer.lstrip()
    if not stripped.startswith("{"):
        return None
    newline_idx = stripped.find("\n")
    if newline_idx == -1:
        return None
    header_line = stripped[:newline_idx].strip()
    try:
        header = json.loads(header_line)
    except Exception:
        return None
    show_sources = bool(header.get("show_sources", True))
    offset = len(buffer) - len(stripped)
    end_idx = offset + newline_idx + 1
    while end_idx < len(buffer) and buffer[end_idx] in "\r\n":
        end_idx += 1
    return show_sources, end_idx


def _extract_query_keywords(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [tok for tok in tokens if len(tok) >= 4 and tok not in _STOP_WORDS]


def _context_matches_keywords(documents: List[Document], keywords: List[str]) -> bool:
    if not keywords:
        return True
    combined = " ".join((doc.page_content or "").lower() for doc in documents)
    if not combined.strip():
        return False
    return any(keyword in combined for keyword in keywords)


def _question_supported_by_context(question: str, documents: List[Document]) -> bool:
    keywords = _extract_query_keywords(question)
    return _context_matches_keywords(documents, keywords)


class RAGChain:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.setup_ollama()
    
    def setup_ollama(self):
        """Setup Ollama LLM."""
        self.model = OllamaLLM(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL,
            temperature=Config.OLLAMA_TEMPERATURE
        )
    
    def retrieve_documents(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for the query.
        Supports explicit page-focused queries like: "Summarize/Explain/Describe Page 135 from Network Security Essentials".
        """
        # Detect page-focused intent
        m = re.search(
            r"(?:summari[sz]e|explain|describe|discuss|what(?:'s| is)?\s+on)\s+(?:the\s+)?page\s*(\d+)"
            r"|page\s*(\d+)\s+(?:summary|summari[sz]e|explanation|explain|details)",
            query,
            re.IGNORECASE,
        )
        page_number = None
        if m:
            page_number = int(m.group(1) or m.group(2)) if (m.group(1) or m.group(2)) else None
        if page_number is None:
            # Generic fallback: any standalone "page <num>" mention
            m2 = re.search(r"\bpage\s*(\d{1,4})\b", query, re.IGNORECASE)
            if m2:
                try:
                    page_number = int(m2.group(1))
                except Exception:
                    page_number = None
        if page_number is not None:
            # Optionally detect textbook name hint
            name_match = re.search(r"from\s+([^\n]+?)(?:\s+textbook|$)", query, re.IGNORECASE)
            name_hint = name_match.group(1).strip() if name_match else None
            # Fetch exact or nearest page document from vector store
            candidates = [0, -1, 1, -2, 2]
            for off in candidates:
                pn = page_number + off
                if pn <= 0:
                    continue
                try:
                    page_docs = self.vector_store.get_page_document(pn, filename_contains=name_hint)  # type: ignore[attr-defined]
                except Exception:
                    page_docs = []
                if page_docs:
                    return page_docs
            # If page intent but not found, constrain similarity to top-1 to avoid mixing many pages
            if name_hint:
                # restrict fallback to the hinted book only
                filtered = []
                try:
                    filtered = self.vector_store.similarity_search_filename_contains(query, k=1, filename_contains=name_hint)  # type: ignore[attr-defined]
                except Exception:
                    filtered = []
                return filtered
            return self.vector_store.similarity_search(query, k=1)
        # Default semantic retrieval
        return self.vector_store.similarity_search(query, k=k)
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string with page numbers when available."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('filename', 'Unknown source')
            page = doc.metadata.get('page_number')
            page_info = f", page {page}" if page else ""
            content = (doc.page_content or "").strip()
            context_parts.append(f"Document {i} (Source: {source}{page_info}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def generate_prompt(self, query: str, context: str, documents: Optional[List[Document]] = None) -> str:
        """Generate a strict, context-grounded prompt (single-turn).
        If the user asked to summarize a page, switch to a page-summary prompt.
        """
        # Detect requested page from the user's query
        req_page = None
        m = re.search(
            r"(?:summari[sz]e|explain|describe|discuss|what(?:'s| is)?\s+on)\s+(?:the\s+)?page\s*(\d+)"
            r"|page\s*(\d+)\s+(?:summary|summari[sz]e|explanation|explain|details)",
            query,
            re.IGNORECASE,
        )
        if m:
            try:
                req_page = int(m.group(1) or m.group(2))
            except Exception:
                req_page = None
        if req_page is None:
            m2 = re.search(r"\bpage\s*(\d{1,4})\b", query, re.IGNORECASE)
            if m2:
                try:
                    req_page = int(m2.group(1))
                except Exception:
                    req_page = None

        used_page = None
        used_filename = None
        if documents:
            try:
                first = documents[0]
                used_page = first.metadata.get("page_number")
                used_filename = first.metadata.get("filename")
            except Exception:
                pass

        # If summarization intent is detected and we have a page, use summary template
        if req_page is not None and used_page is not None:
            note = ""
            if req_page != used_page:
                note = f"(Requested page {req_page}, summarizing nearest available page {used_page}.)\n\n"
            prompt = (
                "You are NetSec Tutor, restricted to the provided Network Security materials.\n"
                "Summarize ONLY the specified page from the context. Do NOT use outside knowledge. If the page content is not present, reply exactly: "
                "\"I can only assist with content from the provided Network Security course materials.\"\n\n"
                f"Target: {used_filename or 'Unknown source'} — page {used_page}.\n"
                f"{note}"
                "Provide a concise, structured summary with the following sections (omit a section if not applicable):\n"
                "- Key points (bullets)\n"
                "- Important terms/definitions (term: definition)\n"
                "- Examples or figures mentioned\n"
                "- Equations or formulas (if any)\n"
                "- Practical takeaways\n\n"
                f"Context (page text):\n{context}\n\n"
                f"User request:\n{query}\n\n"
                "Final page summary:"
            )
            return _with_output_format(prompt)

        # Default QA template
        prompt = f"""You are NetSec Tutor, an expert assistant restricted to the provided Network Security materials.

Strict rules:
- Answer ONLY using the Context. If the Context does not contain the answer, reply exactly: "I can only assist with content from the provided Network Security course materials."
- Do NOT use outside knowledge. Do NOT hallucinate. If unsure, refuse.
- If asked to summarize a page, provide a concise summary of ONLY that page.
- Keep the answer precise and well-structured.

Context:
{context}

User question:
{query}

Final answer (follow the rules):
"""
        return _with_output_format(prompt)
    
    def _extract_last_user_questions(self, chat_history: Optional[List[Dict]]) -> List[str]:
        """Extract user messages from history, return last two before the most recent user message.
        Assumes chat_history includes the current user message at the end.
        """
        if not chat_history:
            return []
        user_msgs = [m.get("content", "") for m in chat_history if isinstance(m, dict) and m.get("role") == "user"]
        if not user_msgs:
            return []
        # The last entry in user_msgs is the current question; take the two before it
        if len(user_msgs) <= 1:
            return []
        prior = user_msgs[:-1]
        return prior[-2:]
    
    def _rewrite_retrieval_query(self, last_two: List[str], current: str) -> str:
        """Try to rewrite the three questions into a single standalone retrieval query.
        Falls back to simple concatenation if the model call fails.
        """
        # Fallback first (always available)
        concatenated = " \n".join([q for q in last_two if q] + [current])
        try:
            instruction = (
                "Rewrite the following conversation of user questions into ONE standalone search query. "
                "Resolve pronouns and references so that it is fully self-contained. "
                "Focus on the current question while keeping useful specifics from the prior two.\n\n"
                f"Previous Q1: {last_two[-2] if len(last_two) == 2 else ''}\n"
                f"Previous Q2: {last_two[-1] if last_two else ''}\n"
                f"Current Q: {current}\n\n"
                "Standalone search query:"
            )
            resp = self.model.invoke(instruction)
            if isinstance(resp, str) and len(resp.strip()) >= 5:
                return resp.strip()
        except Exception:
            pass
        return concatenated.strip()
    
    def _format_recent_turns(self, chat_history: Optional[List[Dict]], max_messages: int = 6) -> str:
        """Format the last few conversation messages for continuity."""
        if not chat_history:
            return "(no prior conversation)"
        recent = chat_history[-max_messages:]
        lines = []
        for m in recent:
            role = m.get("role", "assistant") if isinstance(m, dict) else "assistant"
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            role_name = "User" if role == "user" else "Assistant"
            lines.append(f"{role_name}: {content}")
        return "\n".join(lines)
    
    def _build_final_prompt(self, latest_question: str, context: str, recent_turns: str) -> str:
        """Build the final multi-part prompt for the LLM."""
        return _with_output_format(
            "You are NetSec Tutor, restricted to the provided Network Security context. "
            "Answer ONLY using the retrieved snippets below. If not answerable from them, reply exactly: "
            "\"I can only assist with content from the provided Network Security course materials.\"\n\n"
            "Recent conversation:\n"
            f"{recent_turns}\n\n"
            "Retrieved knowledge base snippets:\n"
            f"{context}\n\n"
            "User's latest question (answer this):\n"
            f"{latest_question}\n\n"
            "Final answer:"
        )
    
    def generate_answer(self, query: str, k: int = 4) -> Dict[str, Any]:
        """Generate answer using RAG pipeline (single-turn)."""
        try:
            # Step 1: Retrieve relevant documents
            documents = self.retrieve_documents(query, k=k)
            
            if not documents:
                return {
                    "answer": REFUSAL_TEXT,
                    "sources": [],
                    "context": "",
                    "query": query,
                    "show_sources": False,
                }

            if not _question_supported_by_context(query, documents):
                return {
                    "answer": REFUSAL_TEXT,
                    "sources": [],
                    "context": "",
                    "query": query,
                    "show_sources": False,
                }
            
            # Step 2: Format context
            context = self.format_context(documents)
            
            # Step 3: Generate prompt (page-summary aware)
            prompt = self.generate_prompt(query, context, documents)
            
            # Step 4: Get response from Ollama
            response = self.model.invoke(prompt)
            show_sources, answer = _split_structured_answer(response)
            
            # Step 5: Extract sources
            sources = []
            for doc in documents:
                src_path = doc.metadata.get('source') or ''
                page_num = doc.metadata.get('page_number')
                url = None
                try:
                    if src_path and page_num:
                        url = f"file://{src_path}#page={page_num}"
                    elif src_path:
                        url = f"file://{src_path}"
                except Exception:
                    url = None
                source_info = {
                    "filename": doc.metadata.get('filename', 'Unknown'),
                    "page_number": page_num,
                    "source": src_path,
                    "url": url,
                    "content_preview": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "query": query,
                "show_sources": show_sources,
            }
            
        except Exception as e:
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": [],
                "context": "",
                "query": query,
                "show_sources": False,
            }
    
    def chat_with_context(self, query: str, chat_history: List[Dict] = None, k: int = 4) -> Dict[str, Any]:
        """Conversational RAG with history-aware retrieval and grounded answering."""
        try:
            chat_history = chat_history or []
            last_two = self._extract_last_user_questions(chat_history)
            combined_retrieval_query = self._rewrite_retrieval_query(last_two, query)
            documents = self.retrieve_documents(combined_retrieval_query, k=k)
            if not documents or not _question_supported_by_context(query, documents):
                return {
                    "answer": REFUSAL_TEXT,
                    "sources": [],
                    "context": "",
                    "query": query,
                    "retrieval_query": combined_retrieval_query,
                    "show_sources": False,
                }
            context = self.format_context(documents)
            recent_turns = self._format_recent_turns(chat_history, max_messages=6)
            # If page-intent is detected, prefer the single-turn summary prompt; otherwise use conversation prompt
            page_aware_prompt = self.generate_prompt(query, context, documents)
            # Heuristic: if page-aware prompt is a summary prompt (it contains 'Final page summary:'), use it
            if "Final page summary:" in page_aware_prompt:
                raw_answer = self.model.invoke(page_aware_prompt)
            else:
                final_prompt = self._build_final_prompt(query, context, recent_turns)
                raw_answer = self.model.invoke(final_prompt)
            show_sources, answer = _split_structured_answer(raw_answer)
            sources = []
            for doc in documents:
                src_path = doc.metadata.get('source') or ''
                page_num = doc.metadata.get('page_number')
                url = None
                try:
                    if src_path and page_num:
                        url = f"file://{src_path}#page={page_num}"
                    elif src_path:
                        url = f"file://{src_path}"
                except Exception:
                    url = None
                source_info = {
                    "filename": doc.metadata.get('filename', 'Unknown'),
                    "page_number": page_num,
                    "source": src_path,
                    "url": url,
                    "content_preview": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "query": query,
                "retrieval_query": combined_retrieval_query,
                "show_sources": show_sources,
            }
        except Exception as e:
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": [],
                "context": "",
                "query": query,
                "show_sources": False,
            }

    def chat_with_context_stream(self, query: str, chat_history: Optional[List[Dict]] = None, k: int = 4):
        """Stream tokens from Ollama while keeping RAG grounding.
        Yields dict events: {"type": "token", "text": str} and final: {"type": "final", ...}.
        """
        import requests
        chat_history = chat_history or []
        last_two = self._extract_last_user_questions(chat_history)
        combined_retrieval_query = self._rewrite_retrieval_query(last_two, query)
        documents = self.retrieve_documents(combined_retrieval_query, k=k)
        if not documents or not _question_supported_by_context(query, documents):
            refusal = REFUSAL_TEXT
            yield {"type": "token", "text": refusal}
            yield {
                "type": "final",
                "answer": refusal,
                "sources": [],
                "context": "",
                "query": query,
                "retrieval_query": combined_retrieval_query,
                "show_sources": False,
            }
            return
        context = self.format_context(documents)
        recent_turns = self._format_recent_turns(chat_history, max_messages=6)
        # Build a page-aware prompt for streaming if applicable
        page_aware_prompt = self.generate_prompt(query, context, documents)
        use_page_summary = "Final page summary:" in page_aware_prompt
        final_prompt = page_aware_prompt if use_page_summary else self._build_final_prompt(query, context, recent_turns)

        # Prepare sources early
        sources = []
        for doc in documents:
            src_path = doc.metadata.get('source') or ''
            page_num = doc.metadata.get('page_number')
            url = None
            try:
                if src_path and page_num:
                    url = f"file://{src_path}#page={page_num}"
                elif src_path:
                    url = f"file://{src_path}"
            except Exception:
                url = None
            sources.append({
                "filename": doc.metadata.get('filename', 'Unknown'),
                "page_number": page_num,
                "source": src_path,
                "url": url,
                "content_preview": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content
            })

        # Stream from Ollama generate endpoint
        url = f"{Config.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": Config.OLLAMA_MODEL,
            "prompt": final_prompt,
            "stream": True,
            "options": {
                "temperature": Config.OLLAMA_TEMPERATURE
            }
        }
        control_buffer = ""
        control_parsed = False
        show_sources_flag = True
        answer_chunks: List[str] = []
        try:
            with requests.post(url, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("done"):
                        break
                    chunk = obj.get("response") or obj.get("data") or ""
                    if chunk:
                        if not control_parsed:
                            control_buffer += chunk
                            parsed = _consume_control_prefix(control_buffer)
                            if parsed:
                                show_sources_flag, consumed_idx = parsed
                                remainder = control_buffer[consumed_idx:]
                                if remainder:
                                    answer_chunks.append(remainder)
                                    yield {"type": "token", "text": remainder}
                                control_parsed = True
                                control_buffer = ""
                            continue
                        answer_chunks.append(chunk)
                        yield {"type": "token", "text": chunk}
                if not control_parsed and control_buffer:
                    control_parsed = True
                    remainder = control_buffer
                    if remainder:
                        answer_chunks.append(remainder)
                        yield {"type": "token", "text": remainder}
                answer = "".join(answer_chunks).strip()
        except Exception as e:
            answer = f"[Streaming error: {e}]"
            show_sources_flag = False
        yield {
            "type": "final",
            "answer": answer,
            "sources": sources,
            "context": context,
            "query": query,
            "retrieval_query": combined_retrieval_query,
            "show_sources": show_sources_flag,
        }


class CombinedRAGChain:
    """RAG over multiple vector stores (e.g., TEXTBOOKs + LECTURE SLIDEs).
    It merges retrieval results from all provided stores and keeps the top-k by score.
    """
    def __init__(self, vector_stores: Sequence[VectorStore]):
        self.vector_stores = [vs for vs in vector_stores if vs is not None]
        # Reuse Ollama config as in RAGChain
        self.model = OllamaLLM(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_MODEL,
            temperature=Config.OLLAMA_TEMPERATURE,
        )

    def chat_with_context_stream(self, query: str, chat_history: Optional[List[Dict]] = None, k: int = 4):
        """Stream tokens from Ollama while keeping RAG grounding across multiple stores.
        Yields dict events: {"type": "token", "text": str} and final: {"type": "final", ...}.
        """
        import requests
        chat_history = chat_history or []
        # Build a simple retrieval query from history similar to chat_with_context
        user_msgs = [m.get("content", "") for m in chat_history if isinstance(m, dict) and m.get("role") == "user"]
        last_two = user_msgs[-3:-1] if len(user_msgs) > 1 else []
        concatenated = " \n".join([q for q in last_two if q] + [query])
        retrieval_query = concatenated.strip() or query

        # Try page-specific retrieval across all stores
        page_number = None
        m = re.search(
            r"(?:summari[sz]e|explain|describe|discuss|what(?:'s| is)?\s+on)\s+(?:the\s+)?page\s*(\d+)"
            r"|page\s*(\d+)\s+(?:summary|summari[sz]e|explanation|explain|details)",
            retrieval_query,
            re.IGNORECASE,
        )
        if m:
            page_number = int(m.group(1) or m.group(2)) if (m.group(1) or m.group(2)) else None
        if page_number is None:
            m2 = re.search(r"\bpage\s*(\d{1,4})\b", retrieval_query, re.IGNORECASE)
            if m2:
                try:
                    page_number = int(m2.group(1))
                except Exception:
                    page_number = None
        name_hint = None
        if page_number is not None:
            name_match = re.search(r"from\s+([^\n]+?)(?:\s+textbook|$)", retrieval_query, re.IGNORECASE)
            name_hint = name_match.group(1).strip() if name_match else None

        documents = []
        if page_number is not None:
            # Try exact or nearest pages across all stores
            candidates = [0, -1, 1, -2, 2]
            for off in candidates:
                pn = page_number + off
                if pn <= 0:
                    continue
                for vs in self.vector_stores:
                    try:
                        docs = vs.get_page_document(pn, filename_contains=name_hint)  # type: ignore[attr-defined]
                    except Exception:
                        docs = []
                    if docs:
                        documents = docs
                        break
                if documents:
                    break

        # Fallback to similarity across all stores
        if not documents:
            # If this was a page-intent query, constrain to top-1 to avoid mixing content from multiple pages
            if page_number is not None and name_hint:
                # restrict fallback to the hinted book across stores
                all_scored = []
                for vs in self.vector_stores:
                    try:
                        scored = vs.similarity_search_with_score_filename_contains(retrieval_query, k=1, filename_contains=name_hint)  # type: ignore[attr-defined]
                        all_scored.extend(scored or [])
                    except Exception:
                        continue
                if all_scored:
                    # Chroma scores: lower is better
                    all_scored.sort(key=lambda t: (t[1] if len(t) > 1 else 1e9))
                    documents = [all_scored[0][0]]
                else:
                    documents = []
            else:
                documents = self._similarity_search_all(retrieval_query, k=1 if page_number is not None else k)

        if not documents:
            refusal = REFUSAL_TEXT
            yield {"type": "token", "text": refusal}
            yield {
                "type": "final",
                "answer": refusal,
                "sources": [],
                "context": "",
                "query": query,
                "retrieval_query": retrieval_query,
                "show_sources": False,
            }
            return

        if not _question_supported_by_context(query, documents):
            refusal = REFUSAL_TEXT
            yield {"type": "token", "text": refusal}
            yield {
                "type": "final",
                "answer": refusal,
                "sources": [],
                "context": "",
                "query": query,
                "retrieval_query": retrieval_query,
                "show_sources": False,
            }
            return

        context = self._format_context(documents)
        recent_turns = self._format_recent_turns(chat_history, max_messages=6)
        # Build a page-aware prompt for streaming if applicable
        page_aware_prompt = self._generate_prompt_combined(query, context, documents)
        use_page_summary = "Final page summary:" in page_aware_prompt
        if use_page_summary:
            final_prompt = page_aware_prompt
        else:
            base_prompt = (
                "You are NetSec Tutor, restricted to the provided Network Security context. "
                "Answer ONLY using the retrieved snippets below. If not answerable from them, reply exactly: "
                "\"I can only assist with content from the provided Network Security course materials.\"\n\n"
                "Recent conversation:\n"
                f"{recent_turns}\n\n"
                "Retrieved knowledge base snippets:\n"
                f"{context}\n\n"
                "User's latest question (answer this):\n"
                f"{query}\n\n"
                "Final answer:"
            )
            final_prompt = _with_output_format(base_prompt)

        # Prepare sources early
        sources: List[Dict[str, Any]] = []
        for doc in documents:
            src_path = doc.metadata.get('source') or ''
            page_num = doc.metadata.get('page_number')
            url = None
            try:
                if src_path and page_num:
                    url = f"file://{src_path}#page={page_num}"
                elif src_path:
                    url = f"file://{src_path}"
            except Exception:
                url = None
            sources.append({
                "filename": doc.metadata.get("filename", "Unknown"),
                "page_number": page_num,
                "source": src_path,
                "url": url,
                "content_preview": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content,
            })

        # Stream from Ollama generate endpoint
        url = f"{Config.OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": Config.OLLAMA_MODEL,
            "prompt": final_prompt,
            "stream": True,
            "options": {
                "temperature": Config.OLLAMA_TEMPERATURE
            }
        }
        control_buffer = ""
        control_parsed = False
        show_sources_flag = True
        answer_chunks: List[str] = []
        try:
            with requests.post(url, json=payload, stream=True) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("done"):
                        break
                    chunk = obj.get("response") or obj.get("data") or ""
                    if chunk:
                        if not control_parsed:
                            control_buffer += chunk
                            parsed = _consume_control_prefix(control_buffer)
                            if parsed:
                                show_sources_flag, consumed_idx = parsed
                                remainder = control_buffer[consumed_idx:]
                                if remainder:
                                    answer_chunks.append(remainder)
                                    yield {"type": "token", "text": remainder}
                                control_parsed = True
                                control_buffer = ""
                            continue
                        answer_chunks.append(chunk)
                        yield {"type": "token", "text": chunk}
                if not control_parsed and control_buffer:
                    control_parsed = True
                    remainder = control_buffer
                    if remainder:
                        answer_chunks.append(remainder)
                        yield {"type": "token", "text": remainder}
                answer = "".join(answer_chunks).strip()
        except Exception as e:
            answer = f"[Streaming error: {e}]"
            show_sources_flag = False
        yield {
            "type": "final",
            "answer": answer,
            "sources": sources,
            "context": context,
            "query": query,
            "retrieval_query": retrieval_query,
            "show_sources": show_sources_flag,
        }

    def _generate_prompt_combined(self, query: str, context: str, documents: List[Document]) -> str:
        """Generate a page-aware prompt for CombinedRAGChain similar to RAGChain.generate_prompt."""
        import re
        req_page = None
        m = re.search(
            r"(?:summari[sz]e|explain|describe|discuss|what(?:'s| is)?\s+on)\s+(?:the\s+)?page\s*(\d+)"
            r"|page\s*(\d+)\s+(?:summary|summari[sz]e|explanation|explain|details)",
            query,
            re.IGNORECASE,
        )
        if m:
            try:
                req_page = int(m.group(1) or m.group(2))
            except Exception:
                req_page = None
        if req_page is None:
            m2 = re.search(r"\bpage\s*(\d{1,4})\b", query, re.IGNORECASE)
            if m2:
                try:
                    req_page = int(m2.group(1))
                except Exception:
                    req_page = None
        used_page = None
        used_filename = None
        if documents:
            try:
                first = documents[0]
                used_page = first.metadata.get("page_number")
                used_filename = first.metadata.get("filename")
            except Exception:
                pass
        if req_page is not None and used_page is not None:
            note = ""
            if req_page != used_page:
                note = f"(Requested page {req_page}, summarizing nearest available page {used_page}.)\n\n"
            return _with_output_format(
                "You are NetSec Tutor, restricted to the provided Network Security materials.\n"
                "Summarize ONLY the specified page from the context. Do NOT use outside knowledge. If the page content is not present, reply exactly: "
                "\"I can only assist with content from the provided Network Security course materials.\"\n\n"
                f"Target: {used_filename or 'Unknown source'} — page {used_page}.\n"
                f"{note}"
                "Provide a concise, structured summary with the following sections (omit a section if not applicable):\n"
                "- Key points (bullets)\n"
                "- Important terms/definitions (term: definition)\n"
                "- Examples or figures mentioned\n"
                "- Equations or formulas (if any)\n"
                "- Practical takeaways\n\n"
                f"Context (page text):\n{context}\n\n"
                f"User request:\n{query}\n\n"
                "Final page summary:"
            )
        # No page intent: return a generic placeholder, caller will build the default conversation prompt
        return ""

    def _similarity_search_all(self, query: str, k: int = 4) -> List[Document]:
        """Query each store with scores, merge results and return top-k documents.
        Note: LangChain's Chroma returns lower scores for closer matches, so sort ascending.
        """
        if not self.vector_stores:
            return []
        # pull k from each store, then cut back to k overall
        all_scored: List[tuple] = []
        per_store_k = max(1, k)
        for vs in self.vector_stores:
            try:
                results = vs.similarity_search_with_score(query, k=per_store_k) or []
                all_scored.extend(results)
            except Exception:
                continue
        if not all_scored:
            return []
        # sort by score ascending (best first), take top k
        all_scored.sort(key=lambda t: (t[1] if len(t) > 1 else 1e9))
        top_docs = [t[0] for t in all_scored[:k]]
        return top_docs

    def _format_context(self, documents: List[Document]) -> str:
        if not documents:
            return "No relevant context found."
        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("filename", "Unknown source")
            page = doc.metadata.get("page_number")
            page_info = f", page {page}" if page else ""
            content = (doc.page_content or "").strip()
            parts.append(f"Document {i} (Source: {source}{page_info}):\n{content}")
        return "\n\n".join(parts)

    def _format_recent_turns(self, chat_history: Optional[List[Dict]], max_messages: int = 6) -> str:
        if not chat_history:
            return "(no prior conversation)"
        recent = chat_history[-max_messages:]
        lines = []
        for m in recent:
            role = m.get("role", "assistant") if isinstance(m, dict) else "assistant"
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            role_name = "User" if role == "user" else "Assistant"
            lines.append(f"{role_name}: {content}")
        return "\n".join(lines)

    def chat_with_context(self, query: str, chat_history: Optional[List[Dict]] = None, k: int = 4) -> Dict[str, Any]:
        try:
            chat_history = chat_history or []
            # Simple history-aware rewrite similar to RAGChain
            user_msgs = [m.get("content", "") for m in chat_history if isinstance(m, dict) and m.get("role") == "user"]
            last_two = user_msgs[-3:-1] if len(user_msgs) > 1 else []
            concatenated = " \n".join([q for q in last_two if q] + [query])
            retrieval_query = concatenated.strip() or query

            # Try page-specific retrieval across all stores
            page_number = None
            m = re.search(
                r"(?:summari[sz]e|explain|describe|discuss|what(?:'s| is)?\s+on)\s+(?:the\s+)?page\s*(\d+)"
                r"|page\s*(\d+)\s+(?:summary|summari[sz]e|explanation|explain|details)",
                retrieval_query,
                re.IGNORECASE,
            )
            if m:
                page_number = int(m.group(1) or m.group(2)) if (m.group(1) or m.group(2)) else None
            if page_number is None:
                m2 = re.search(r"\bpage\s*(\d{1,4})\b", retrieval_query, re.IGNORECASE)
                if m2:
                    try:
                        page_number = int(m2.group(1))
                    except Exception:
                        page_number = None
            name_hint = None
            if page_number is not None:
                name_match = re.search(r"from\s+([^\n]+?)(?:\s+textbook|$)", retrieval_query, re.IGNORECASE)
                name_hint = name_match.group(1).strip() if name_match else None

            documents = []
            if page_number is not None:
                # Try exact or nearest pages across all stores
                candidates = [0, -1, 1, -2, 2]
                for off in candidates:
                    pn = page_number + off
                    if pn <= 0:
                        continue
                    for vs in self.vector_stores:
                        try:
                            docs = vs.get_page_document(pn, filename_contains=name_hint)  # type: ignore[attr-defined]
                        except Exception:
                            docs = []
                        if docs:
                            documents = docs
                            break
                    if documents:
                        break

            if not documents:
                # If this was a page-intent query, constrain to top-1 to avoid mixing content from multiple pages
                if page_number is not None and name_hint:
                    # restrict fallback to the hinted book across stores
                    all_scored = []
                    for vs in self.vector_stores:
                        try:
                            scored = vs.similarity_search_with_score_filename_contains(retrieval_query, k=1, filename_contains=name_hint)  # type: ignore[attr-defined]
                            all_scored.extend(scored or [])
                        except Exception:
                            continue
                    if all_scored:
                        all_scored.sort(key=lambda t: (t[1] if len(t) > 1 else 1e9))
                        documents = [all_scored[0][0]]
                    else:
                        documents = []
                else:
                    documents = self._similarity_search_all(retrieval_query, k=1 if page_number is not None else k)
            if not documents or not _question_supported_by_context(query, documents):
                return {
                    "answer": REFUSAL_TEXT,
                    "sources": [],
                    "context": "",
                    "query": query,
                    "retrieval_query": retrieval_query,
                    "show_sources": False,
                }

            context = self._format_context(documents)
            recent_turns = self._format_recent_turns(chat_history, max_messages=6)
            # Prefer page-summary prompt if page intent detected
            page_aware_prompt = self._generate_prompt_combined(query, context, documents)
            if "Final page summary:" in page_aware_prompt:
                raw_answer = self.model.invoke(page_aware_prompt)
            else:
                base_prompt = (
                    "You are NetSec Tutor, restricted to the provided Network Security context. "
                    "Answer ONLY using the retrieved snippets below. If not answerable from them, reply exactly: "
                    "\"I can only assist with content from the provided Network Security course materials.\"\n\n"
                    "Recent conversation:\n"
                    f"{recent_turns}\n\n"
                    "Retrieved knowledge base snippets:\n"
                    f"{context}\n\n"
                    "User's latest question (answer this):\n"
                    f"{query}\n\n"
                    "Final answer:"
                )
                final_prompt = _with_output_format(base_prompt)
                raw_answer = self.model.invoke(final_prompt)
            show_sources, answer = _split_structured_answer(raw_answer)

            sources: List[Dict[str, Any]] = []
            for doc in documents:
                src_path = doc.metadata.get('source') or ''
                page_num = doc.metadata.get('page_number')
                url = None
                try:
                    if src_path and page_num:
                        url = f"file://{src_path}#page={page_num}"
                    elif src_path:
                        url = f"file://{src_path}"
                except Exception:
                    url = None
                sources.append({
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "page_number": page_num,
                    "source": src_path,
                    "url": url,
                    "content_preview": (doc.page_content[:200] + "...") if len(doc.page_content) > 200 else doc.page_content,
                })

            return {
                "answer": answer,
                "sources": sources,
                "context": context,
                "query": query,
                "retrieval_query": retrieval_query,
                "show_sources": show_sources,
            }
        except Exception as e:
            return {
                "answer": f"An error occurred while generating the answer: {str(e)}",
                "sources": [],
                "context": "",
                "query": query,
                "show_sources": False,
            }
