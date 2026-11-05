"""Deterministic stand-ins for model-backed components, so most tests run offline."""

import re
from collections.abc import Sequence

from app.exceptions import ServiceUnavailableError

_WORD = re.compile(r"\S+")


class WhitespaceTokenizer:
    """Counts whitespace-separated words as tokens. Same protocol as the real tokenizer."""

    name = "test-whitespace"

    def count_many(self, texts: Sequence[str]) -> list[int]:
        return [len(_WORD.findall(text)) for text in texts]

    def split(self, text: str, max_tokens: int) -> list[str]:
        words = _WORD.findall(text)
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def unavailable_tokenizer():
    raise ServiceUnavailableError(
        "Could not load the tokenizer (simulated outage).", code="tokenizer_unavailable"
    )
