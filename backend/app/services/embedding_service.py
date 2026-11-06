"""Dense embeddings with Sentence Transformers.

Every vector is L2-normalised, so the inner product the vector index computes
equals cosine similarity. The model is loaded once per process and reused.
"""

import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.exceptions import ServiceUnavailableError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    name: str
    dimension: int
    # Longest input in tokens the model reads; anything beyond is truncated.
    max_seq_length: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """float32 array of shape (len(texts), dimension), rows L2-normalised."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """float32 vector of shape (dimension,), L2-normalised."""
        ...


class SentenceTransformerEmbedder:
    def __init__(
        self, model_name: str, *, device: str, batch_size: int, query_prefix: str = ""
    ) -> None:
        # Heavy import (torch) kept local so the API can start without it.
        from sentence_transformers import SentenceTransformer

        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise ServiceUnavailableError(
                f"Could not load the embedding model '{model_name}'. It is downloaded from "
                "Hugging Face on first use, so check the internet connection and the model name.",
                code="embedding_model_unavailable",
            ) from exc
        self.name = model_name
        self.dimension = int(self._model.get_embedding_dimension())
        self.max_seq_length = int(self._model.max_seq_length)
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        # Concurrent encode calls would just compete for the same CPU cores.
        self._lock = threading.Lock()

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        with self._lock:
            vectors = self._model.encode(
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([self._query_prefix + text])[0]


_load_lock = threading.Lock()


@lru_cache(maxsize=2)
def _cached_embedder(
    model_name: str, device: str, batch_size: int, query_prefix: str
) -> SentenceTransformerEmbedder:
    embedder = SentenceTransformerEmbedder(
        model_name, device=device, batch_size=batch_size, query_prefix=query_prefix
    )
    logger.info(
        "Loaded embedding model %s (dimension %d, max %d tokens, device %s)",
        model_name,
        embedder.dimension,
        embedder.max_seq_length,
        device,
    )
    return embedder


def get_sentence_transformer_embedder(
    model_name: str, *, device: str, batch_size: int, query_prefix: str = ""
) -> SentenceTransformerEmbedder:
    """Load once per configuration. Failures are not cached, so a later call can retry."""
    with _load_lock:
        return _cached_embedder(model_name, device, batch_size, query_prefix)


def warn_if_chunks_exceed_model(embedder: Embedder, chunk_size_tokens: int) -> None:
    # The model adds two special tokens ([CLS] and [SEP]) to every input.
    if chunk_size_tokens + 2 > embedder.max_seq_length:
        logger.warning(
            "CHUNK_SIZE_TOKENS=%d exceeds what %s reads (%d tokens including special "
            "tokens); the end of each chunk will be ignored by dense retrieval.",
            chunk_size_tokens,
            embedder.name,
            embedder.max_seq_length,
        )
