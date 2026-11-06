"""Deterministic stand-ins for model-backed components, so most tests run offline."""

import re
import zlib
from collections.abc import Sequence

import numpy as np

from app.exceptions import ServiceUnavailableError

_WORD = re.compile(r"\S+")
_TERM = re.compile(r"[a-z0-9]+")


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


class HashingEmbedder:
    """Bag-of-words vectors hashed into a few dimensions, then L2-normalised.

    Texts sharing words get similar vectors, which is enough to test indexing
    and search without a neural model. Records every batch it embeds.
    """

    max_seq_length = 512

    def __init__(self, name: str = "test-hashing", dimension: int = 64) -> None:
        self.name = name
        self.dimension = dimension
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.batches.append(list(texts))
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for term in _TERM.findall(text.lower()):
                vectors[row, zlib.crc32(term.encode()) % self.dimension] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms == 0, 1.0, norms)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.batches)


def unavailable_embedder():
    raise ServiceUnavailableError(
        "Could not load the embedding model (simulated outage).",
        code="embedding_model_unavailable",
    )
