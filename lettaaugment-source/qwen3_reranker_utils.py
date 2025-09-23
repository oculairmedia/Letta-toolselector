"""Utility helpers for working with Qwen3 reranker formatting and outputs."""

from __future__ import annotations

import math
from typing import Any, Optional

DEFAULT_RERANK_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

SYSTEM_PROMPT_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)

ASSISTANT_PROMPT_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAnswer:"
)

MAX_DOCUMENT_CHARS = 3000
YES_TOKEN = "yes"
NO_TOKEN = "no"


def truncate_document(document: Optional[str], max_chars: int = MAX_DOCUMENT_CHARS) -> str:
    """Trim documents to a manageable length for the reranker."""
    if not document:
        return ""

    doc = document.strip()
    if len(doc) <= max_chars:
        return doc
    return doc[:max_chars]


def format_instruction(
    instruction: Optional[str],
    query: str,
    document: str,
) -> str:
    """Return the Qwen3-formatted instruction block."""
    task_instruction = instruction or DEFAULT_RERANK_INSTRUCTION
    return (
        f"<Instruct>: {task_instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )


def build_prompt(
    query: str,
    document: str,
    instruction: Optional[str] = None,
) -> str:
    """Produce the full chat-style prompt for the reranker."""
    truncated_document = truncate_document(document)
    formatted_instruction = format_instruction(instruction, query, truncated_document)
    return f"{SYSTEM_PROMPT_PREFIX}{formatted_instruction}\n{ASSISTANT_PROMPT_SUFFIX}"


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def extract_yes_probability(response_json: dict[str, Any]) -> float:
    """Extract the probability that the model answered "yes" using logprobs when available."""
    logprob_entries = response_json.get("logprobs") or []
    yes_prob: Optional[float] = None
    no_prob: Optional[float] = None

    for entry in logprob_entries:
        token = (entry.get("token") or "").strip().lower()
        logprob = entry.get("logprob")
        if token == YES_TOKEN and logprob is not None:
            yes_prob = math.exp(logprob)
        elif token == NO_TOKEN and logprob is not None:
            no_prob = math.exp(logprob)

        for alt in entry.get("top_logprobs") or entry.get("top_tokens") or []:
            alt_token = (alt.get("token") or "").strip().lower()
            alt_logprob = alt.get("logprob")
            if alt_logprob is None:
                continue
            probability = math.exp(alt_logprob)
            if alt_token == YES_TOKEN:
                yes_prob = max(yes_prob or 0.0, probability)
            elif alt_token == NO_TOKEN:
                no_prob = max(no_prob or 0.0, probability)

        if yes_prob is not None and no_prob is not None:
            break

    if yes_prob is not None and no_prob is not None:
        total = yes_prob + no_prob
        if total > 0:
            return clamp_probability(yes_prob / total)

    fallback_text = (response_json.get("response") or "").strip().lower()
    if fallback_text.startswith(YES_TOKEN):
        return 0.99
    if fallback_text.startswith(NO_TOKEN):
        return 0.01

    return 0.5


__all__ = [
    "DEFAULT_RERANK_INSTRUCTION",
    "SYSTEM_PROMPT_PREFIX",
    "ASSISTANT_PROMPT_SUFFIX",
    "MAX_DOCUMENT_CHARS",
    "format_instruction",
    "build_prompt",
    "truncate_document",
    "extract_yes_probability",
]
