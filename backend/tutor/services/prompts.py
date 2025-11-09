"""Centralized prompt builders for NetSec Tutor."""
from __future__ import annotations

from typing import Optional, List

OUTPUT_FORMAT_INSTRUCTIONS = (
    "\nOutput formatting requirements:\n"
    "1. The very first line must be a single-line JSON object containing a boolean field named \"show_sources\" "
    '(example: {"show_sources": true}).\n'
    "   - Use true when your answer is grounded in the provided context snippets.\n"
    "   - Use false only when you must refuse because the answer is not present in the provided context.\n"
    "2. After a blank line, write the natural-language answer exactly as it should appear to the user.\n"
    "3. Do not emit any other JSON or text before the JSON control line.\n"
)


def _with_output_format(prompt: str) -> str:
    return f"{prompt}{OUTPUT_FORMAT_INSTRUCTIONS}"


def build_page_summary_prompt(
    *,
    context: str,
    query: str,
    used_filename: Optional[str],
    used_page: Optional[int],
    note: str = "",
) -> str:
    return _with_output_format(
        "You are NetSec Tutor, restricted to the provided Network Security materials.\n"
        "Summarize ONLY the specified page from the context. Do NOT use outside knowledge. If the page content is not present, reply exactly: "
        "\"I can only assist with content from the provided Network Security course materials.\"\n\n"
        f"Target: {used_filename or 'Unknown source'} — page {used_page or 'unknown'}.\n"
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


def build_strict_qa_prompt(*, context: str, query: str) -> str:
    return _with_output_format(
        "You are NetSec Tutor, an expert assistant restricted to the provided Network Security materials.\n\n"
        "Strict rules:\n"
        "- Answer ONLY using the Context. If the Context does not contain the answer, reply exactly: "
        "\"I can only assist with content from the provided Network Security course materials.\"\n"
        "- Do NOT use outside knowledge. Do NOT hallucinate. If unsure, refuse.\n"
        "- If asked to summarize a page, provide a concise summary of ONLY that page.\n"
        "- Keep the answer precise and well-structured.\n\n"
        f"Context:\n{context}\n\n"
        f"User question:\n{query}\n\n"
        "Final answer (follow the rules):\n"
    )


def build_conversation_prompt(*, recent_turns: str, context: str, latest_question: str) -> str:
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


__all__ = [
    "OUTPUT_FORMAT_INSTRUCTIONS",
    "build_page_summary_prompt",
    "build_strict_qa_prompt",
    "build_conversation_prompt",
    "build_quiz_generation_prompt",
]


def build_quiz_generation_prompt(
    *,
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
