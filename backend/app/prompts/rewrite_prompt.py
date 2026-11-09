"""Prompt for turning a follow-up question into a standalone search query.

Retrieval sees only one query, so "What about its limitations?" must become
"What are the limitations of FedProx?" before searching. Questions that do
not follow a conversation skip this step entirely.
"""

import re
from collections.abc import Sequence

from app.services.llm_service import ChatTurn

REWRITE_SYSTEM_PROMPT = """You rewrite the latest question in a conversation about research \
papers into a standalone question that can be searched on its own.
- Resolve references such as "it", "this method", "they" or "the second paper" using the \
conversation.
- Keep names, acronyms, datasets and numbers exactly as written.
- If the question is already standalone, return it unchanged.
- Output only the rewritten question on a single line, with no explanation."""

# Earlier answers are shortened; the rewrite only needs their topic.
MAX_HISTORY_CHARS = 600
MAX_REWRITE_CHARS = 500

_CITATION = re.compile(r"\s*\[S\d+\]")
_PREFIX = re.compile(r"^(standalone question|rewritten question|question)\s*:\s*", re.IGNORECASE)


def build_rewrite_turns(question: str, history: Sequence[ChatTurn]) -> list[ChatTurn]:
    lines = []
    for turn in history:
        speaker = "User" if turn.role == "user" else "Assistant"
        content = _CITATION.sub("", turn.content)  # old source ids mean nothing here
        if len(content) > MAX_HISTORY_CHARS:
            content = content[:MAX_HISTORY_CHARS].rsplit(" ", 1)[0] + " …"
        lines.append(f"{speaker}: {' '.join(content.split())}")
    conversation = "\n".join(lines)
    return [
        ChatTurn(
            "user",
            f"Conversation:\n{conversation}\n\nLatest question: {question}\n\nStandalone question:",
        )
    ]


def clean_rewrite(text: str, fallback: str) -> str:
    """First line of the model's output without labels or quotes; ``fallback`` if unusable."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return fallback
    rewritten = _PREFIX.sub("", lines[0]).strip().strip("\"'“”").strip()
    if not rewritten or len(rewritten) > MAX_REWRITE_CHARS:
        return fallback
    return rewritten
