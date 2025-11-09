import ast
import json
import random
import re
from typing import Any, Dict, List, Optional

import requests

from config import Config
from .vector_store import VectorStore
from .prompts import build_quiz_generation_prompt
from ..logging_utils import get_app_logger, summarize_text

logger = get_app_logger()


class QuizAgent:
    """Generates and grades quizzes grounded in the local network security knowledge base."""

    UNCERTAIN_PHRASES = {
        "i don't know",
        "i dont know",
        "idk",
        "not sure",
        "no idea",
        "can't remember",
        "cannot remember",
        "unsure",
        "i am not sure",
        "i'm not sure",
        "dont know",
        "do not know",
    }

    DEFAULT_RANDOM_TOPICS = [
        "network security fundamentals",
        "encryption methods",
        "public key cryptography",
        "private key cryptography",
        "symmetric key distribution",
        "asymmetric key distribution",
        "kerbos protocol",
        "X.509, PKI",
        "web security",
        "https and SSL/TLS",
        "TCP, UDP Protocols",
        "Message Authentication Codes (MACs)",
        "Stream ciphers, block ciphers",
        "secure hash algorithms",
        "HMACs",
    ]

    def __init__(self, vector_stores: List[VectorStore]):
        self.vector_stores = [
            vs for vs in vector_stores if vs is not None and getattr(vs, "vectorstore", None) is not None
        ]
        if not self.vector_stores:
            raise ValueError("QuizAgent requires at least one initialized vector store.")
        self.base_url = Config.OLLAMA_BASE_URL.rstrip("/")
        self.model = Config.OLLAMA_MODEL
        self.temperature = Config.OLLAMA_TEMPERATURE

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_quiz(
        self,
        num_mcq: int = 2,
        num_true_false: int = 2,
        num_open_ended: int = 1,
        topics: Optional[List[str]] = None,
        mode: str = "random",
        difficulty: str = "medium",
        source_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Generate a quiz with mixed question formats grounded in retrieved context."""
        total_questions = num_mcq + num_true_false + num_open_ended
        if total_questions <= 0:
            raise ValueError("At least one question must be requested.")
        if total_questions > 15:
            raise ValueError("Please request 15 or fewer total questions per quiz.")
        logger.info(
            "QuizAgent.generate_quiz start mode=%s difficulty=%s counts={'mcq':%d,'tf':%d,'open':%d}",
            mode,
            difficulty,
            num_mcq,
            num_true_false,
            num_open_ended,
        )

        topics_clean = self._prepare_topics(topics, mode)
        logger.info("QuizAgent.generate_quiz topics=%s", topics_clean)
        difficulty_clean = (difficulty or "medium").strip().lower()
        if difficulty_clean not in {"easy", "medium", "hard"}:
            difficulty_clean = "medium"
        source_map = self._gather_contexts(topics_clean, total_chunks=max(6, total_questions * 2))
        context_prompt = self._format_context_for_prompt(source_map)
        logger.info("QuizAgent.generate_quiz gathered_contexts=%d", len(source_map))

        logger.info(
            "QuizAgent.generate_quiz building prompt topics=%d difficulty=%s context_chars=%d",
            len(topics_clean),
            difficulty_clean,
            len(context_prompt),
        )
        prompt = build_quiz_generation_prompt(
            context_prompt=context_prompt,
            num_mcq=num_mcq,
            num_true_false=num_true_false,
            num_open=num_open_ended,
            topics=topics_clean,
            difficulty=difficulty_clean,
        )

        logger.info("QuizAgent.generate_quiz invoking LLM prompt_chars=%d", len(prompt))
        llm_response = self._call_llm(prompt)
        logger.info("QuizAgent.generate_quiz llm_response_chars=%d", len(llm_response))
        quiz_payload = self._parse_quiz_json(llm_response)
        logger.info("QuizAgent.generate_quiz parsed_questions=%d", len(quiz_payload.get("questions", [])))

        expected_counts = {
            "multiple_choice": num_mcq,
            "true_false": num_true_false,
            "open_ended": num_open_ended,
        }

        questions, type_counter = self._normalize_questions(
            quiz_payload.get("questions", []),
            expected_counts=expected_counts,
            enforce=False,
        )

        questions, type_counter = self._ensure_question_completeness(
            context_prompt=context_prompt,
            base_questions=questions,
            type_counter=type_counter,
            expected_counts=expected_counts,
            topics=topics_clean,
            difficulty=difficulty_clean,
        )

        questions = self._trim_to_expected(questions, expected_counts)

        final_counts = {"multiple_choice": 0, "true_false": 0, "open_ended": 0}
        for item in questions:
            q_type = item["type"]
            if q_type in final_counts:
                final_counts[q_type] += 1
        remaining = self._calculate_missing(final_counts, expected_counts)
        if any(remaining.values()):
            deficits = ", ".join(
                f"{k.replace('_', ' ')} (missing {v})"
                for k, v in remaining.items()
                if v > 0
            )
            raise ValueError(
                f"Quiz generation could not supply the required question mix even after retries: {deficits}."
            )

        questions = self._renumber_questions(questions)

        quiz_data = {
            "title": quiz_payload.get("quiz_title")
            or self._default_quiz_title(mode=mode, topics=topics_clean),
            "mode": mode,
            "topics": topics_clean,
            "questions": questions,
            "sources": source_map,
            "difficulty": difficulty_clean,
            "source_categories": list(source_categories) if source_categories else ["textbooks", "slides"],
        }
        logger.info("QuizAgent.generate_quiz complete title='%s' counts=%s", quiz_data["title"], final_counts)
        return quiz_data

    def grade_quiz(self, quiz_data: Dict[str, Any], user_answers: Dict[str, Any]) -> Dict[str, Any]:
        """Grade user responses against the quiz answer key, providing feedback with citations."""
        questions = quiz_data.get("questions", [])
        sources = quiz_data.get("sources", {})
        if not questions:
            raise ValueError("Quiz data contains no questions.")

        logger.info(
            "QuizAgent.grade_quiz start questions=%d answers=%d",
            len(questions),
            len(user_answers or {}),
        )
        graded_questions = []
        earned_score = 0.0
        max_score = float(len(questions))

        for item in questions:
            qid = item["id"]
            user_answer = user_answers.get(qid)
            q_type = item["type"]

            if q_type == "multiple_choice":
                result = self._grade_multiple_choice(item, user_answer, sources)
            elif q_type == "true_false":
                result = self._grade_true_false(item, user_answer, sources)
            elif q_type == "open_ended":
                result = self._grade_open_ended(item, user_answer, sources)
            else:
                result = {
                    "question": item,
                    "user_answer": user_answer,
                    "is_correct": False,
                    "score": 0.0,
                    "feedback": "Unsupported question type.",
                    "citations": [],
                }

            score_value = float(result.get("score", 0.0))
            score_possible = 1.0
            result["score_possible"] = score_possible
            result["score_display"] = self._format_score_display(score_value, score_possible)
            result["score_breakdown"] = {
                "earned": score_value,
                "possible": score_possible,
            }

            graded_questions.append(result)
            earned_score += max(0.0, min(1.0, result.get("score", 0.0)))

        percentage = (earned_score / max_score) * 100 if max_score else 0.0
        grading_summary = {
            "questions": graded_questions,
            "earned_score": earned_score,
            "max_score": max_score,
            "percentage": percentage,
        }
        logger.info(
            "QuizAgent.grade_quiz complete earned=%.2f/%d percentage=%.2f",
            earned_score,
            len(questions),
            percentage,
        )
        return grading_summary

    # ------------------------------------------------------------------ #
    # Question generation helpers
    # ------------------------------------------------------------------ #
    def _prepare_topics(self, topics: Optional[List[str]], mode: str) -> List[str]:
        if topics:
            cleaned = [t.strip() for t in topics if t and t.strip()]
            return cleaned or self._prepare_topics(None, mode)
        if mode.lower() == "random":
            sample_size = min(3, len(self.DEFAULT_RANDOM_TOPICS))
            return random.sample(self.DEFAULT_RANDOM_TOPICS, sample_size)
        return []

    def _gather_contexts(self, topics: List[str], total_chunks: int = 6) -> Dict[str, Dict[str, Any]]:
        """Retrieve relevant documents from the vector stores."""
        collected: List[Dict[str, Any]] = []
        seen = set()

        queries = topics or ["network security fundamentals"]
        chunks_per_topic = max(1, total_chunks // max(1, len(queries)))

        for topic in queries:
            for store in self.vector_stores:
                try:
                    docs = store.similarity_search(topic, k=chunks_per_topic)
                except Exception as exc:
                    logger.exception("QuizAgent._gather_contexts similarity error topic='%s': %s", topic, exc)
                    docs = []
                added = 0
                for doc in docs:
                    metadata = doc.metadata or {}
                    key = (
                        metadata.get("filename"),
                        metadata.get("page_number"),
                        metadata.get("source"),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    text = (doc.page_content or "").strip()
                    if not text:
                        continue
                    collected.append(
                        {
                            "content": text,
                            "metadata": {
                                "filename": metadata.get("filename"),
                                "page_number": metadata.get("page_number"),
                                "source": metadata.get("source"),
                                "topic": topic,
                            },
                        }
                    )
                    if len(collected) >= total_chunks:
                        break
                    added += 1
                if len(collected) >= total_chunks:
                    break
            logger.info(
                "QuizAgent._gather_contexts topic='%s' added=%d total=%d/%d",
                topic,
                added,
                len(collected),
                total_chunks,
            )
            if len(collected) >= total_chunks:
                break

        if not collected:
            # Check if vector stores are empty
            store_info = []
            for store in self.vector_stores:
                try:
                    info = store.get_collection_info()
                    count = info.get("count", 0)
                    store_info.append(f"{count} documents")
                except Exception:
                    store_info.append("unknown")
            
            logger.warning(
                "QuizAgent._gather_contexts failed store_info=%s",
                store_info or [],
            )
            raise ValueError(
                "Unable to retrieve any context from the knowledge base. "
                "This usually means:\n"
                "1. No documents have been uploaded to the knowledge base yet, OR\n"
                "2. The vector stores have not been built/rebuilt after uploading documents.\n\n"
                f"Vector store status: {', '.join(store_info) if store_info else 'No stores available'}.\n\n"
                "To fix this:\n"
                "1. Go to the Knowledge Base page\n"
                "2. Upload some documents (PDF, DOCX, or TXT files)\n"
                "3. Click 'Ingest' to build the vector stores\n"
                "4. Try generating a quiz again"
            )

        source_map: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(collected, start=1):
            source_id = f"S{idx}"
            source_map[source_id] = {**entry, "source_id": source_id}
        logger.info("QuizAgent._gather_contexts source_map=%d", len(source_map))
        return source_map

    def _format_context_for_prompt(self, source_map: Dict[str, Dict[str, Any]]) -> str:
        """Format retrieved contexts for the generation prompt."""
        formatted_blocks: List[str] = []
        for source_id, entry in source_map.items():
            meta = entry.get("metadata", {})
            header = f"[{source_id}] Course material excerpt"

            snippet = entry.get("content", "")
            snippet = snippet.strip()
            snippet = re.sub(r"^Page\s+\d+\s*\|\s*[^\n]+\n", "", snippet)
            if len(snippet) > 1200:
                snippet = snippet[:1150].rstrip() + " ..."

            formatted_blocks.append(f"{header}\n{snippet}")
        return "\n\n".join(formatted_blocks)

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
        }
        try:
            logger.info(
                "QuizAgent._call_llm model=%s temperature=%s prompt_chars=%d",
                self.model,
                self.temperature,
                len(prompt),
            )
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            text = data.get("response") or ""
            trimmed = text.strip()
            logger.info("QuizAgent._call_llm success chars=%d", len(trimmed))
            return trimmed
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Please ensure Ollama is running and accessible. Error: {str(e)}"
            ) from e
        except requests.exceptions.HTTPError as e:
            raise ValueError(
                f"Ollama API error: {e.response.status_code} - {e.response.text}. "
                f"Model '{self.model}' may not be available. "
                f"Try running: ollama pull {self.model}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(
                f"Ollama request timed out after 120 seconds. "
                f"The model may be too slow or the request too large."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error calling Ollama: {str(e)}") from e

    def _parse_quiz_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError("Quiz generation returned an empty response.")
        try:
            parsed = json.loads(text)
            logger.info(
                "QuizAgent._parse_quiz_json parsed questions=%d",
                len(parsed.get("questions", [])) if isinstance(parsed, dict) else 0,
            )
            return parsed
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end >= start:
                snippet = text[start : end + 1]
                try:
                    parsed = json.loads(snippet)
                    logger.info(
                        "QuizAgent._parse_quiz_json recovered_json questions=%d",
                        len(parsed.get("questions", [])) if isinstance(parsed, dict) else 0,
                    )
                    return parsed
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(snippet)
                        if isinstance(parsed, dict):
                            logger.info(
                                "QuizAgent._parse_quiz_json literal_eval fallback questions=%d",
                                len(parsed.get("questions", [])),
                            )
                            return json.loads(json.dumps(parsed))
                    except (ValueError, SyntaxError):
                        pass
                    raise ValueError(
                        "Failed to parse quiz JSON: Expecting property name enclosed in double quotes."
                    )
            logger.exception(
                "QuizAgent._parse_quiz_json invalid JSON response='%s'",
                summarize_text(text, 200),
            )
            raise ValueError("Quiz generation response was not valid JSON.")

    def _normalize_questions(
        self,
        questions: List[Dict[str, Any]],
        expected_counts: Dict[str, int],
        enforce: bool = True,
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        if not questions:
            raise ValueError("Quiz generation produced no questions.")

        normalized: List[Dict[str, Any]] = []
        type_counter = {"multiple_choice": 0, "true_false": 0, "open_ended": 0}

        for idx, raw in enumerate(questions, start=1):
            q_type = (raw.get("type") or "").strip().lower()
            if q_type not in type_counter:
                continue

            question_text = (raw.get("question") or "").strip()
            answer = raw.get("answer")
            explanation = (raw.get("answer_explanation") or "").strip()
            citations = [c.strip() for c in raw.get("citations", []) if c and c.strip()]

            if not question_text or answer in (None, "") or not explanation or not citations:
                continue

            record: Dict[str, Any] = {
                "id": raw.get("id") or f"Q{idx}",
                "type": q_type,
                "question": question_text,
                "answer": answer if isinstance(answer, str) else str(answer),
                "answer_explanation": explanation,
                "citations": citations,
            }

            if q_type == "multiple_choice":
                choices = raw.get("choices", [])
                if not isinstance(choices, list) or len(choices) != 4:
                    continue
                record["choices"] = [str(choice).strip() for choice in choices]
                raw_answer = record["answer"].strip()
                choice_map = self._choice_label_map(record)
                normalized_answer = self._resolve_choice_value(
                    raw_answer,
                    choice_map,
                    record["choices"],
                )
                if normalized_answer is None:
                    continue
                record["answer"] = normalized_answer
            elif q_type == "true_false":
                record["answer"] = record["answer"].strip().capitalize()
            else:  # open_ended
                record["answer"] = record["answer"].strip()

            type_counter[q_type] += 1
            normalized.append(record)

        if enforce:
            for q_type, expected in expected_counts.items():
                if type_counter[q_type] < expected:
                    raise ValueError(
                        f"Quiz generation produced insufficient {q_type.replace('_', ' ')} questions "
                        f"(expected {expected}, received {type_counter[q_type]})."
                    )

        return normalized, type_counter

    def _ensure_question_completeness(
        self,
        context_prompt: str,
        base_questions: List[Dict[str, Any]],
        type_counter: Dict[str, int],
        expected_counts: Dict[str, int],
        topics: List[str],
        difficulty: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        missing = self._calculate_missing(type_counter, expected_counts)
        if not any(missing.values()):
            return base_questions, type_counter

        augmented, counts = self._augment_question_types(
            context_prompt=context_prompt,
            existing_questions=base_questions,
            type_counter=type_counter,
            expected_counts=expected_counts,
            missing=missing,
            topics=topics,
            difficulty=difficulty,
        )
        return augmented, counts

    @staticmethod
    def _calculate_missing(
        type_counter: Dict[str, int], expected_counts: Dict[str, int]
    ) -> Dict[str, int]:
        missing: Dict[str, int] = {}
        for key, expected in expected_counts.items():
            current = type_counter.get(key, 0)
            deficit = max(0, expected - current)
            missing[key] = deficit
        return missing

    def _augment_question_types(
        self,
        context_prompt: str,
        existing_questions: List[Dict[str, Any]],
        type_counter: Dict[str, int],
        expected_counts: Dict[str, int],
        missing: Dict[str, int],
        topics: List[str],
        difficulty: str,
        max_attempts: int = 3,
    ) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        questions = list(existing_questions)
        counts = dict(type_counter)
        remaining = dict(missing)
        source_ids = self._extract_source_ids(context_prompt)

        for q_type in ["multiple_choice", "true_false", "open_ended"]:
            needed = remaining.get(q_type, 0)
            attempts = 0
            while needed > 0 and attempts < max_attempts:
                attempts += 1
                prompt = self._build_type_prompt(
                    context_prompt=context_prompt,
                    question_type=q_type,
                    amount=needed,
                    topics=topics,
                    source_ids=source_ids,
                    difficulty=difficulty,
                )
                llm_response = self._call_llm(prompt)
                payload = self._parse_quiz_json(llm_response)
                extras_raw = payload.get("questions", [])
                if not isinstance(extras_raw, list):
                    continue

                extras, _ = self._normalize_questions(
                    extras_raw,
                    expected_counts=expected_counts,
                    enforce=False,
                )

                for item in extras:
                    if item["type"] != q_type:
                        continue
                    if remaining[q_type] <= 0:
                        break
                    questions.append(item)
                    counts[q_type] = counts.get(q_type, 0) + 1
                    remaining[q_type] -= 1
                    needed = remaining[q_type]
                # loop continues if still need more
            if remaining.get(q_type, 0) > 0:
                if q_type == "open_ended":
                    added = False
                    while remaining[q_type] > 0:
                        fallback = self._fallback_open_ended_question(questions)
                        if not fallback:
                            break
                        questions.append(fallback)
                        counts[q_type] = counts.get(q_type, 0) + 1
                        remaining[q_type] -= 1
                        added = True
                    if remaining[q_type] <= 0:
                        continue

                label = q_type.replace("_", " ")
                raise ValueError(
                    f"Quiz generation could not supply the required question mix: {label} (missing {remaining[q_type]})."
                )

        return questions, counts

    def _build_type_prompt(
        self,
        context_prompt: str,
        question_type: str,
        amount: int,
        topics: List[str],
        source_ids: List[str],
        difficulty: str,
    ) -> str:
        type_labels = {
            "multiple_choice": "multiple-choice",
            "true_false": "true/false",
            "open_ended": "open-ended",
        }
        label = type_labels.get(question_type, question_type.replace("_", " "))
        topics_text = ", ".join(topics) if topics else "the network security course materials"
        source_hint = ", ".join(source_ids) if source_ids else "S1, S2, ..."
        difficulty_guidance = {
            "easy": "Focus on foundational definitions, basic facts, and straightforward recall questions.",
            "medium": "Blend conceptual understanding with applied reasoning that checks comprehension.",
            "hard": "Emphasize scenario-based reasoning, multi-step analysis, or nuanced comparisons.",
        }
        difficulty_text = difficulty_guidance.get(difficulty, difficulty_guidance["medium"])

        requirements = [
            f"- Produce exactly {amount} {label} question(s). No extra questions.",
            "- Every question object must include fields: id, type, question, choices (only for multiple-choice), "
            "answer, answer_explanation, citations.",
            "- Set the type field to the requested question type.",
            "- Use ONLY the provided source IDs in citations: "
            f"{source_hint}.",
            "- The answer_explanation must cite the sources using [S#] inline (e.g., [S1]).",
            f"- Target difficulty: {difficulty_text}",
            "- Questions must NOT mention filenames, PDFs, or page numbers. Keep them self-contained and conceptual.",
            "- Do not place [S#] or citations in the question text or answer field.",
        ]
        if question_type == "multiple_choice":
            requirements.extend(
                [
                    "- Supply exactly four choices labelled A, B, C, D (strings).",
                    "- The answer must be the correct letter (A/B/C/D).",
                ]
            )
        elif question_type == "true_false":
            requirements.append('- The answer must be either "True" or "False".')
        else:  # open_ended
            requirements.extend(
                [
                    "- Provide a concise paragraph (3-4 sentences) as the reference answer.",
                    "- The answer field should contain the reference answer text.",
                ]
            )

        instructions = "\n".join(requirements)

        return (
            "You previously generated a quiz but some required question types were missing.\n"
            "Using ONLY the context snippets below, create the missing questions so the quiz covers "
            f"{topics_text}.\n\n"
            f"{instructions}\n"
            "- Output ONLY a JSON object shaped as {\"questions\": [...]} with no markdown.\n"
            "- Avoid repeating earlier questions if possible.\n\n"
            "Context sources:\n"
            f"{context_prompt}\n\n"
            "Return the JSON object now:\n"
        )

    @staticmethod
    def _extract_source_ids(context_prompt: str) -> List[str]:
        if not context_prompt:
            return []
        ids = re.findall(r"\[(S\d+)\]", context_prompt)
        seen = set()
        ordered: List[str] = []
        for cid in ids:
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)
        return ordered

    @staticmethod
    def _trim_to_expected(
        questions: List[Dict[str, Any]],
        expected_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        counters = {key: 0 for key in expected_counts}
        trimmed: List[Dict[str, Any]] = []
        for question in questions:
            q_type = question["type"]
            allowed = expected_counts.get(q_type)
            if allowed is None:
                continue
            if counters[q_type] >= allowed:
                continue
            trimmed.append(question)
            counters[q_type] += 1
        return trimmed

    @staticmethod
    def _renumber_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for idx, question in enumerate(questions, start=1):
            question["id"] = f"Q{idx}"
        return questions

    def _default_quiz_title(self, mode: str, topics: List[str]) -> str:
        if mode.lower() == "random" or not topics:
            return "Network Security Mixed Quiz"
        return f"Network Security Quiz — {', '.join(topics)}"

    # ------------------------------------------------------------------ #
    # Grading helpers
    # ------------------------------------------------------------------ #
    def _grade_multiple_choice(
        self,
        question: Dict[str, Any],
        user_answer: Optional[str],
        sources: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        choices_map = self._choice_label_map(question)
        raw_choices = question.get("choices", []) or []

        raw_correct = (question.get("answer") or "").strip()
        raw_user = (user_answer or "").strip()

        correct_letter = self._resolve_choice_value(raw_correct, choices_map, raw_choices)
        user_letter = self._resolve_choice_value(raw_user, choices_map, raw_choices)

        is_correct = bool(correct_letter) and user_letter == correct_letter

        if not is_correct and raw_user and raw_correct:
            if self._normalize_choice_text(raw_user) == self._normalize_choice_text(raw_correct):
                is_correct = True
                if not user_letter:
                    user_letter = correct_letter

        correct_choice_text = self._choice_text_from_letter(correct_letter, raw_choices)
        user_choice_text = self._choice_text_from_letter(user_letter, raw_choices)
        if correct_choice_text:
            correct_display = correct_choice_text.upper()
        else:
            correct_display = raw_correct.upper() if raw_correct else "Not answered"

        if user_letter:
            if user_choice_text:
                user_display = user_choice_text
            else:
                user_display = raw_user or "Not answered"
        else:
            user_display = raw_user or "Not answered"

        if user_choice_text:
            stored_user_answer = user_choice_text.upper()
        elif raw_user:
            stored_user_answer = raw_user.upper()
        else:
            stored_user_answer = user_letter or ""

        feedback = (
            ("✅ Correct! " if is_correct else "❌ Not quite. ")
            + "Explanation:\n"
            + question["answer_explanation"]
        )
        citations = self._map_citations(question["citations"], sources)
        return {
            "question": question,
            "user_answer": stored_user_answer,
            "user_answer_display": user_display,
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "feedback": feedback,
            "citations": citations,
            "correct_answer_display": correct_display,
        }

    def _grade_true_false(
        self,
        question: Dict[str, Any],
        user_answer: Optional[str],
        sources: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        verdict = self._determine_true_false_answer(question, sources)
        correct_value = verdict.get("answer") if verdict else question["answer"]
        explanation = verdict.get("explanation") if verdict else question["answer_explanation"]
        citation_ids = verdict.get("citations") if verdict else question["citations"]
        if not citation_ids:
            citation_ids = question.get("citations", [])

        formatted = (user_answer or "").strip().capitalize()
        is_correct = formatted in {"True", "False"} and formatted == correct_value
        feedback = (
            ("✅ Correct! " if is_correct else "❌ Not quite. ")
            + "Explanation:\n"
            + explanation
        )
        citations = self._map_citations(citation_ids, sources)
        question["answer"] = correct_value
        question["answer_explanation"] = explanation
        question["citations"] = citation_ids
        return {
            "question": question,
            "user_answer": formatted,
            "user_answer_display": formatted if formatted else "Not answered",
            "is_correct": is_correct,
            "score": 1.0 if is_correct else 0.0,
            "feedback": feedback,
            "citations": citations,
            "correct_answer_display": correct_value,
        }

    def _grade_open_ended(
        self,
        question: Dict[str, Any],
        user_answer: Optional[str],
        sources: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not user_answer or not user_answer.strip():
            feedback = (
                "❌ No answer provided. Reference answer:\n" + question["answer_explanation"]
            )
            citations = self._map_citations(question["citations"], sources)
            return {
                "question": question,
                "user_answer": user_answer,
                "user_answer_display": "Not answered",
                "is_correct": False,
                "score": 0.0,
                "feedback": feedback,
                "citations": citations,
                "correct_answer_display": question["answer"],
            }

        evaluation = self._evaluate_open_response(question, user_answer, sources)
        citations = self._map_citations(evaluation.get("citations", []), sources)

        return {
            "question": question,
            "user_answer": user_answer,
            "user_answer_display": user_answer.strip(),
            "is_correct": evaluation.get("is_correct", False),
            "score": float(evaluation.get("score", 0.0)),
            "feedback": evaluation.get("feedback")
            or ("Reference answer:\n" + question["answer_explanation"]),
            "citations": citations,
            "correct_answer_display": question["answer"],
        }

    def _evaluate_open_response(
        self,
        question: Dict[str, Any],
        user_answer: str,
        sources: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        cleaned = user_answer.strip()
        normalized = cleaned.lower()
        word_count = len(re.findall(r"\b\w+\b", cleaned))
        if any(phrase in normalized for phrase in self.UNCERTAIN_PHRASES):
            return {
                "is_correct": False,
                "score": 0.0,
                "feedback": (
                    "❌ Answers that express uncertainty (e.g., 'I don't know') "
                    "do not earn credit. Review the reference explanation:\n"
                    + question["answer_explanation"]
                ),
                "citations": question.get("citations", []),
            }
        if word_count < 6:
            return {
                "is_correct": False,
                "score": 0.0,
                "feedback": (
                    "❌ The response is too brief to demonstrate understanding. "
                    "Provide a complete explanation. Reference answer:\n"
                    + question["answer_explanation"]
                ),
                "citations": question.get("citations", []),
            }

        allowed_ids = question.get("citations", [])
        context_prompt = self._format_context_for_prompt(
            {cid: sources[cid] for cid in allowed_ids if cid in sources}
        )
        prompt = (
            "You are grading a student's open-ended answer using ONLY the provided sources.\n"
            "Decide if the answer demonstrates sufficient understanding.\n"
            "Responses that confess uncertainty (e.g., \"I don't know\", \"not sure\", \"no idea\")"
            " must be marked incorrect with score 0.0.\n"
            "Do NOT award credit unless the student's explanation clearly matches the reference answer's key points.\n"
            "Provide JSON with fields: is_correct (true/false), score (0.0-1.0), feedback (2-3 sentences with [S#] citations), "
            "citations (list of source IDs used).\n\n"
            f"Question:\n{question['question']}\n\n"
            f"Reference answer:\n{question['answer']}\n\n"
            f"Student answer:\n{user_answer}\n\n"
            f"Allowed sources:\n{context_prompt}\n\n"
            "Return JSON now:\n"
        )

        try:
            response_text = self._call_llm(prompt)
            data = json.loads(response_text)
            data["citations"] = [
                cid for cid in data.get("citations", []) if cid in allowed_ids
            ]
            return data
        except Exception:
            # Fallback when LLM evaluation fails
            return {
                "is_correct": False,
                "score": 0.0,
                "feedback": (
                    "❌ Unable to automatically grade the response. "
                    "Here is the reference explanation:\n" + question["answer_explanation"]
                ),
                "citations": allowed_ids,
            }

    def _map_citations(
        self,
        citation_ids: List[str],
        sources: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        mapped: List[Dict[str, Any]] = []
        for cid in citation_ids:
            source = sources.get(cid)
            if not source:
                continue
            meta = source.get("metadata", {})
            mapped.append(
                {
                    "id": cid,
                    "filename": meta.get("filename"),
                    "page_number": meta.get("page_number"),
                    "snippet": source.get("content", "")[:400].strip(),
                }
            )
        return mapped

    @staticmethod
    def _format_score_display(score: float, possible: float) -> str:
        earned_text = f"{score:.2f}".rstrip("0").rstrip(".")
        possible_text = f"{possible:.2f}".rstrip("0").rstrip(".")
        if not earned_text:
            earned_text = "0"
        if not possible_text:
            possible_text = "0"
        return f"{earned_text}/{possible_text}"

    def _choice_label_map(self, question: Dict[str, Any]) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        choices = question.get("choices", []) or []
        letters = ["A", "B", "C", "D"]
        for idx, letter in enumerate(letters):
            if idx >= len(choices):
                break
            raw = str(choices[idx]).strip()
            if raw.upper().startswith(f"{letter}."):
                mapping[letter] = raw
            else:
                mapping[letter] = f"{letter}. {raw}"
        return mapping

    @staticmethod
    def _normalize_choice_text(value: str) -> str:
        trimmed = value.strip()
        if re.match(r"^[A-D](?:[\.\)\-:]+\s*|\s+)", trimmed, flags=re.IGNORECASE):
            trimmed = re.sub(r"^[A-D][\.\)\-:]*\s*", "", trimmed, count=1, flags=re.IGNORECASE)
        trimmed = trimmed.replace("–", "-").replace("—", "-")
        cleaned = re.sub(r"\s+", " ", trimmed)
        cleaned = cleaned.rstrip(" .,:;!?")
        return cleaned.lower()

    @staticmethod
    def _choice_text_from_letter(letter: Optional[str], raw_choices: List[Any]) -> Optional[str]:
        if not letter:
            return None
        idx = ord(letter.upper()) - ord("A")
        if 0 <= idx < len(raw_choices):
            return str(raw_choices[idx]).strip()
        return None

    def _fallback_open_ended_question(self, questions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        seed = None
        for candidate in questions:
            if candidate.get("type") == "multiple_choice":
                seed = candidate
                break
        if seed is None:
            for candidate in questions:
                if candidate.get("type") == "true_false":
                    seed = candidate
                    break
        if seed is None:
            return None

        base_text = (seed.get("question") or "").strip()
        base_text = base_text.rstrip(" ?.")
        prompt = f"In your own words, explain the concept: {base_text}."

        explanation = self._clean_fallback_text(seed.get("answer_explanation") or "")
        if not explanation:
            explanation = self._clean_fallback_text(seed.get("answer") or "")
        citations = [c for c in seed.get("citations", []) if c]
        if not explanation or not citations:
            return None

        question_id = self._next_question_id(questions)
        return {
            "id": question_id,
            "type": "open_ended",
            "question": prompt,
            "answer": explanation,
            "answer_explanation": explanation,
            "citations": citations,
        }

    @staticmethod
    def _clean_fallback_text(text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^[✅❌]\s*", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def _determine_true_false_answer(
        self,
        question: Dict[str, Any],
        sources: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        allowed_ids = question.get("citations", [])
        context_prompt = self._format_context_for_prompt(
            {cid: sources[cid] for cid in allowed_ids if cid in sources}
        )
        if not context_prompt.strip():
            return None
        prompt = (
            "You are verifying a true/false statement using ONLY the provided sources.\n"
            "Return strict JSON with fields: answer (\"True\" or \"False\"), "
            "explanation (2 sentences with [S#] citations), citations (list of source IDs).\n"
            "Do not include extra fields or commentary.\n\n"
            f"Statement:\n{question['question']}\n\n"
            f"Existing answer key says: {question.get('answer', 'Unknown')}\n"
            f"Sources:\n{context_prompt}\n\n"
            "JSON:"
        )
        try:
            response_text = self._call_llm(prompt)
            data = json.loads(response_text)
            answer = (data.get("answer") or "").strip().capitalize()
            if answer not in {"True", "False"}:
                return None
            explanation = (data.get("explanation") or "").strip()
            citations = [cid for cid in data.get("citations", []) if cid in allowed_ids]
            if not citations:
                citations = allowed_ids
            return {
                "answer": answer,
                "explanation": explanation or question.get("answer_explanation"),
                "citations": citations,
            }
        except Exception:
            return None

    def _next_question_id(self, existing: List[Dict[str, Any]]) -> str:
        used = {str(item.get("id") or "").strip() for item in existing}
        counter = len(existing) + 1
        while True:
            candidate = f"FALLBACK_{counter}"
            if candidate not in used:
                return candidate
            counter += 1

    def _resolve_choice_value(
        self,
        value: Optional[str],
        choices_map: Dict[str, str],
        raw_choices: List[Any],
    ) -> Optional[str]:
        if not value:
            return None
        candidate = value.strip()
        if not candidate:
            return None

        upper_candidate = candidate.upper()
        if upper_candidate in choices_map:
            return upper_candidate

        # Handle inputs like "A" / "a" / "A)" directly
        if len(upper_candidate) == 1 and upper_candidate in {"A", "B", "C", "D"}:
            return upper_candidate
        if (
            len(upper_candidate) >= 2
            and upper_candidate[0] in {"A", "B", "C", "D"}
            and upper_candidate[1] in {".", ")", ":", "-", " "}
        ):
            return upper_candidate[0]

        normalized_candidate = self._normalize_choice_text(candidate)
        for letter, display in choices_map.items():
            if self._normalize_choice_text(display) == normalized_candidate:
                return letter

        for idx, choice in enumerate(raw_choices):
            letter = chr(ord("A") + idx)
            if self._normalize_choice_text(str(choice)) == normalized_candidate:
                return letter

        return None
