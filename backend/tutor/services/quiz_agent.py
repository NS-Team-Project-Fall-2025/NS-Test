import ast
import json
import random
import re
from typing import Any, Dict, List, Optional

import requests

from config import Config
from .vector_store import VectorStore


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

        topics_clean = self._prepare_topics(topics, mode)
        difficulty_clean = (difficulty or "medium").strip().lower()
        if difficulty_clean not in {"easy", "medium", "hard"}:
            difficulty_clean = "medium"
        source_map = self._gather_contexts(topics_clean, total_chunks=max(6, total_questions * 2))
        context_prompt = self._format_context_for_prompt(source_map)

        prompt = self._build_generation_prompt(
            context_prompt=context_prompt,
            num_mcq=num_mcq,
            num_true_false=num_true_false,
            num_open=num_open_ended,
            topics=topics_clean,
            difficulty=difficulty_clean,
        )

        llm_response = self._call_llm(prompt)
        quiz_payload = self._parse_quiz_json(llm_response)

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
            header = f"[{source_id}] Course material excerpt"

            snippet = entry.get("content", "")
            snippet = snippet.strip()
            snippet = re.sub(r"^Page\s+\d+\s*\|\s*[^\n]+\n", "", snippet)
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
        difficulty: str,
    ) -> str:
        topics_text = ", ".join(topics) if topics else "the network security course materials"
        difficulty_guidance = {
            "easy": "Focus on foundational definitions, basic facts, and straightforward recall questions.",
            "medium": "Blend conceptual understanding with applied reasoning that checks comprehension.",
            "hard": "Emphasize scenario-based reasoning, multi-step analysis, or nuanced comparisons.",
        }
        difficulty_text = difficulty_guidance.get(difficulty, difficulty_guidance["medium"])

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
            f"- Target difficulty: {difficulty_text}\n"
            "- Questions must NOT mention filenames, PDFs, or page numbers. Keep them self-contained and conceptual.\n"
            "- Citations belong only inside answer_explanation; do not place [S#] in the question text or answer field.\n"
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
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(snippet)
                        if isinstance(parsed, dict):
                            return json.loads(json.dumps(parsed))
                    except (ValueError, SyntaxError):
                        pass
                    raise ValueError(
                        "Failed to parse quiz JSON: Expecting property name enclosed in double quotes."
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
                record["answer"] = record["answer"].strip().upper()
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
