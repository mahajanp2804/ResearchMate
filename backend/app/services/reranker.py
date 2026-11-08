"""Cross-encoder reranking.

The first-stage retrievers embed the query and each passage separately
(bi-encoder) or only count shared words (BM25). A cross-encoder reads the
question and a passage together, so it judges relevance far more precisely,
but it costs one model call per pair. It is therefore run only on the few
dozen candidates that fusion already ranked highly.

Scores are the model's raw output (logits): higher is more relevant, and
they are comparable within one query. See ``min_rerank_score`` in config
for how they are used as an evidence threshold.
"""

import threading
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from app.exceptions import ServiceUnavailableError
from app.utils.logging import get_logger

logger = get_logger(__name__)

RERANK_BATCH_SIZE = 16


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Relevance of each passage to the query; higher is better."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str) -> None:
        # Heavy import (torch) kept local so the API can start without it.
        from sentence_transformers import CrossEncoder

        try:
            self._model = CrossEncoder(model_name, device=device)
        except Exception as exc:
            raise ServiceUnavailableError(
                f"Could not load the reranking model '{model_name}'. It is downloaded from "
                "Hugging Face on first use, so check the internet connection and the model name.",
                code="reranker_unavailable",
            ) from exc
        self.name = model_name
        # Concurrent predict calls would just compete for the same CPU cores.
        self._lock = threading.Lock()

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        with self._lock:
            scores = self._model.predict(
                [(query, passage) for passage in passages],
                batch_size=RERANK_BATCH_SIZE,
                show_progress_bar=False,
            )
        return [float(score) for score in scores]


_load_lock = threading.Lock()


@lru_cache(maxsize=2)
def _cached_reranker(model_name: str, device: str) -> CrossEncoderReranker:
    reranker = CrossEncoderReranker(model_name, device=device)
    logger.info("Loaded reranking model %s (device %s)", model_name, device)
    return reranker


def get_cross_encoder_reranker(model_name: str, *, device: str) -> CrossEncoderReranker:
    """Load once per configuration. Failures are not cached, so a later call can retry."""
    with _load_lock:
        return _cached_reranker(model_name, device)
