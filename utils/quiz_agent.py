import json
import random
from typing import Any, Dict, List, Optional

import requests

from config import Config
from utils.vector_store import VectorStore


class QuizAgent:
    """Generates and grades quizzes grounded in the local network security knowledge base."""

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
    ) -> Dict[str, Any]:
        """Generate a quiz with mixed question formats grounded in retrieved context."""
        total_questions = num_mcq + num_true_false + num_open_ended
        if total_questions <= 0:
            raise ValueError("At least one question must be requested.")

        topics_clean = self._prepare_topics(topics, mode)
        source_map = self._gather_contexts(topics_clean, total_chunks=max(6, total_questions * 2))
        context_prompt = self._format_context_for_prompt(source_map)

        prompt = self._build_generation_prompt(
            context_prompt=context_prompt,
            num_mcq=num_mcq,
            num_true_false=num_true_false,
            num_open=num_open_ended,
            topics=topics_clean,
        )

        llm_response = self._call_llm(prompt)
        quiz_payload = self._parse_quiz_json(llm_response)

        questions = self._normalize_questions(
            quiz_payload.get("questions", []),
            expected_counts={"multiple_choice": num_mcq, "true_false": num_true_false, "open_ended": num_open_ended},
        )

        quiz_data = {
            "title": quiz_payload.get("quiz_title")
            or self._default_quiz_title(mode=mode, topics=topics_clean),
            "mode": mode,
            "topics": topics_clean,
            "questions": questions,
            "sources": source_map,
        }
        return quiz_data

    def grade_quiz(self, quiz_data: Dict[str, Any], user_answers: Dict[str, Any]) -> Dict[str, Any]:
        """Grade user responses against the quiz answer key, providing feedback with citations."""
        questions = quiz_data.get("questions", [])
        sources = quiz_data.get("sources", {})
        if not questions:
            raise ValueError("Quiz data contains no questions.")

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

            graded_questions.append(result)
            earned_score += max(0.0, min(1.0, result.get("score", 0.0)))

        percentage = (earned_score / max_score) * 100 if max_score else 0.0
        grading_summary = {
            "questions": graded_questions,
            "earned_score": earned_score,
            "max_score": max_score,
            "percentage": percentage,
        }
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
                except Exception:
                    docs = []
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
                if len(collected) >= total_chunks:
                    break
            if len(collected) >= total_chunks:
                break

        if not collected:
            raise ValueError("Unable to retrieve any context from the knowledge base.")

        source_map: Dict[str, Dict[str, Any]] = {}
        for idx, entry in enumerate(collected, start=1):
            source_id = f"S{idx}"
            source_map[source_id] = {**entry, "source_id": source_id}
        return source_map

    def _format_context_for_prompt(self, source_map: Dict[str, Dict[str, Any]]) -> str:
        """Format retrieved contexts for the generation prompt."""
        formatted_blocks: List[str] = []
        for source_id, entry in source_map.items():
            meta = entry.get("metadata", {})
            filename = meta.get("filename") or "Unknown source"
            page_num = meta.get("page_number")
            header = f"[{source_id}] {filename}"
            if page_num:
                header += f" — page {page_num}"

            snippet = entry.get("content", "")
            snippet = snippet.strip()
            if len(snippet) > 1200:
                snippet = snippet[:1150].rstrip() + " ..."

            formatted_blocks.append(f"{header}\n{snippet}")
        return "\n\n".join(formatted_blocks)

    def _build_generation_prompt(
        self,
        context_prompt: str,
        num_mcq: int,
        num_true_false: int,
        num_open: int,
        topics: List[str],
    ) -> str:
        topics_text = ", ".join(topics) if topics else "the network security course materials"

        return (
            "You are NetSec Quiz Agent, an expert tutor who designs assessments strictly from the provided sources.\n"
            "Using ONLY the context sources below, create a well-balanced quiz covering "
            f"{topics_text}.\n\n"
            "Requirements:\n"
            f"- Provide exactly {num_mcq} multiple-choice, {num_true_false} true/false, and {num_open} open-ended questions.\n"
            "- Multiple-choice questions must supply four options labelled A, B, C, D and specify the correct option letter.\n"
            "- True/false questions must have answers \"True\" or \"False\".\n"
            "- Open-ended questions require a short paragraph answer (3-4 sentences) as the reference solution.\n"
            "- Every question must include an answer explanation that cites the supporting sources using the IDs [S#].\n"
            "- Each question must include a \"citations\" array listing the relevant source IDs (e.g., [\"S1\", \"S3\"]).\n"
            "- Do NOT fabricate information. If a topic cannot be supported by the sources, adapt the question to what is supported.\n"
            "- Output MUST be valid JSON with UTF-8 characters that can be parsed directly, with no markdown fencing.\n\n"
            "Expected JSON schema:\n"
            "{\n"
            "  \"quiz_title\": string,\n"
            "  \"questions\": [\n"
            "    {\n"
            "      \"id\": string,\n"
            "      \"type\": \"multiple_choice\" | \"true_false\" | \"open_ended\",\n"
            "      \"question\": string,\n"
            "      \"choices\": [string, string, string, string]  // required for multiple_choice only\n"
            "      \"answer\": string,  // letter for MCQ, \"True\"/\"False\" for T/F, reference answer text for open_ended\n"
            "      \"answer_explanation\": string,\n"
            "      \"citations\": [string, ...]\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Context sources:\n"
            f"{context_prompt}\n\n"
            "Return the JSON object now:\n"
        )

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
        }
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        text = data.get("response") or ""
        return text.strip()

    def _parse_quiz_json(self, text: str) -> Dict[str, Any]:
        if not text:
            raise ValueError("Quiz generation returned an empty response.")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end >= start:
                snippet = text[start : end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError as err:
                    raise ValueError(f"Failed to parse quiz JSON: {err}") from err
            raise ValueError("Quiz generation response was not valid JSON.")

    def _normalize_questions(
        self,
        questions: List[Dict[str, Any]],
        expected_counts: Dict[str, int],
    ) -> List[Dict[str, Any]]:
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
                record["answer"] = record["answer"].strip().upper()
            elif q_type == "true_false":
                record["answer"] = record["answer"].strip().capitalize()
            else:  # open_ended
                record["answer"] = record["answer"].strip()

            type_counter[q_type] += 1
            normalized.append(record)

        for q_type, expected in expected_counts.items():
            if type_counter[q_type] < expected:
                raise ValueError(
                    f"Quiz generation produced insufficient {q_type.replace('_', ' ')} questions "
                    f"(expected {expected}, received {type_counter[q_type]})."
                )

        return normalized

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
        correct_letter = question["answer"].strip().upper()
        user_letter = (user_answer or "").strip().upper()
        is_correct = bool(user_letter) and user_letter == correct_letter
        choices_map = self._choice_label_map(question)
        correct_display = choices_map.get(correct_letter, correct_letter)
        user_display = choices_map.get(user_letter, "Not answered") if user_letter else "Not answered"
        feedback = (
            ("✅ Correct! " if is_correct else "❌ Not quite. ")
            + "Explanation:\n"
            + question["answer_explanation"]
        )
        citations = self._map_citations(question["citations"], sources)
        return {
            "question": question,
            "user_answer": user_letter,
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
        correct_value = question["answer"]
        formatted = (user_answer or "").strip().capitalize()
        is_correct = formatted in {"True", "False"} and formatted == correct_value
        feedback = (
            ("✅ Correct! " if is_correct else "❌ Not quite. ")
            + "Explanation:\n"
            + question["answer_explanation"]
        )
        citations = self._map_citations(question["citations"], sources)
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
        allowed_ids = question.get("citations", [])
        context_prompt = self._format_context_for_prompt(
            {cid: sources[cid] for cid in allowed_ids if cid in sources}
        )
        prompt = (
            "You are grading a student's open-ended answer using ONLY the provided sources.\n"
            "Decide if the answer demonstrates sufficient understanding.\n"
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
