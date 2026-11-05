"""Token counting with the embedding model's own tokenizer.

Chunk sizes are measured in the tokens the embedding model actually sees, so
a chunk configured to fit the model's input window really does fit, instead
of being silently truncated during embedding.
"""

import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from app.exceptions import ServiceUnavailableError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Tokenizer(Protocol):
    name: str

    def count_many(self, texts: Sequence[str]) -> list[int]:
        """Token count of each text, excluding special tokens such as [CLS]."""
        ...

    def split(self, text: str, max_tokens: int) -> list[str]:
        """Cut text into consecutive pieces of at most ``max_tokens`` tokens."""
        ...


class HuggingFaceTokenizer:
    """Wraps the fast (Rust) tokenizer published with a Hugging Face model."""

    def __init__(self, model_name: str) -> None:
        # Imported lazily so modules that only need the protocol stay light.
        from tokenizers import Tokenizer as RustTokenizer

        self.name = model_name
        try:
            self._tokenizer = RustTokenizer.from_pretrained(model_name)
        except Exception as exc:
            raise ServiceUnavailableError(
                f"Could not load the tokenizer for '{model_name}'. It is downloaded from "
                "Hugging Face on first use, so check the internet connection and the model name.",
                code="tokenizer_unavailable",
            ) from exc
        # Counting must see the full text; truncation would under-count.
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()

    def count_many(self, texts: Sequence[str]) -> list[int]:
        if not texts:
            return []
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [len(encoding.ids) for encoding in encodings]

    def split(self, text: str, max_tokens: int) -> list[str]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        offsets, word_ids = encoding.offsets, encoding.word_ids
        pieces = []
        start = 0
        while start < len(offsets):
            end = min(start + max_tokens, len(offsets))
            # Move the cut back to a word boundary: a piece starting mid-word
            # ("##ero") re-tokenises differently and could exceed the limit.
            boundary = end
            while (
                end < len(offsets)
                and boundary > start
                and (word_ids[boundary] == word_ids[boundary - 1])
            ):
                boundary -= 1
            if boundary > start:
                end = boundary
            piece = text[offsets[start][0] : offsets[end - 1][1]].strip()
            if piece:
                pieces.append(piece)
            start = end
        return pieces


_load_lock = threading.Lock()


@lru_cache(maxsize=4)
def _cached_tokenizer(model_name: str) -> HuggingFaceTokenizer:
    return HuggingFaceTokenizer(model_name)


def get_huggingface_tokenizer(model_name: str) -> HuggingFaceTokenizer:
    """Load once per process. Failures are not cached, so a later call can retry."""
    with _load_lock:
        tokenizer = _cached_tokenizer(model_name)
    logger.debug("Using tokenizer %s", model_name)
    return tokenizer
