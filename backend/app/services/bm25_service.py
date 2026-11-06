"""Keyword retrieval with BM25 over every searchable chunk.

Dense retrieval matches meaning; BM25 matches the exact words a user typed,
which matters for names, acronyms and numbers that embeddings blur together
("FedProx", "CIFAR-10", "Sent140", "0.95").

The index lives in memory and is rebuilt only when the set of searchable
chunks changes. Change detection uses a cheap aggregate over chunk ids,
which is reliable because SQLite never reuses chunk ids (AUTOINCREMENT).
"""

import math
import re
import threading
import time
import unicodedata
from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk, Document, IndexingStatus
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Decimals ("0.95") first, then words that may contain hyphens or underscores.
_TOKEN = re.compile(r"\d+\.\d+|[a-z0-9]+(?:[-_][a-z0-9]+)*")
_COMPOUND_SEPARATOR = re.compile(r"[-_]")

# Common English function words carry no topical signal for BM25.
STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such than
    that the their theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why will with
    would you your yours yourself yourselves via et al
    """.split()  # noqa: SIM905 - a word list reads better than 130 quoted strings
)


def _fold_plural(token: str) -> str:
    """Map simple plurals to their singular so "clients" matches "client".

    Deliberately tiny (no stemming library): it is applied identically to
    documents and queries, so consistency matters more than linguistics.
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def analyze(text: str) -> list[str]:
    """Turn text into BM25 terms.

    A hyphenated term is kept whole and also split, so "non-IID" matches
    queries for "non-IID" as well as "IID".
    """
    terms: list[str] = []
    for match in _TOKEN.finditer(unicodedata.normalize("NFKC", text).lower()):
        word = match.group()
        candidates = [word]
        if _COMPOUND_SEPARATOR.search(word):
            candidates.extend(_COMPOUND_SEPARATOR.split(word))
        for candidate in candidates:
            if len(candidate) >= 2 and candidate not in STOPWORDS:
                terms.append(_fold_plural(candidate))
    return terms


class LuceneIdfBM25(BM25Okapi):
    """BM25Okapi with the always-positive IDF used by Lucene and Elasticsearch.

    Okapi's IDF, log((N - n + 0.5) / (n + 0.5)), is zero or negative once a term
    appears in half the chunks. In a small library (a couple of short papers)
    that makes relevant terms such as "drift" score nothing. log(1 + ...) keeps
    every matching term positive and still ranks rare terms above common ones.
    """

    def _calc_idf(self, nd: dict[str, int]) -> None:
        for word, document_frequency in nd.items():
            self.idf[word] = math.log(
                1 + (self.corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
        self.average_idf = sum(self.idf.values()) / len(self.idf) if self.idf else 0.0


@dataclass(frozen=True)
class KeywordHit:
    chunk_id: int
    score: float  # BM25 score; only comparable within one query


@dataclass(frozen=True)
class _Snapshot:
    fingerprint: tuple[int, int, int]
    chunk_ids: np.ndarray  # int64, aligned with the BM25 corpus
    document_ids: np.ndarray  # str, aligned with the BM25 corpus
    bm25: LuceneIdfBM25 | None  # None when there is nothing to search


def _searchable_chunks():
    return (
        select(Chunk)
        .join(Document, Document.id == Chunk.document_id)
        .where(Document.indexing_status == IndexingStatus.INDEXED)
    )


def _fingerprint(db: Session) -> tuple[int, int, int]:
    count, id_sum, id_max = db.execute(
        _searchable_chunks()
        .with_only_columns(
            func.count(Chunk.id),
            func.coalesce(func.sum(Chunk.id), 0),
            func.coalesce(func.max(Chunk.id), 0),
        )
        .order_by(None)
    ).one()
    return int(count), int(id_sum), int(id_max)


class BM25Service:
    def __init__(self, *, k1: float, b: float) -> None:
        self._k1 = k1
        self._b = b
        self._lock = threading.Lock()
        self._snapshot: _Snapshot | None = None

    def _current(self, db: Session) -> _Snapshot:
        fingerprint = _fingerprint(db)
        with self._lock:
            if self._snapshot is not None and self._snapshot.fingerprint == fingerprint:
                return self._snapshot

            started = time.perf_counter()
            rows = db.execute(
                _searchable_chunks()
                .with_only_columns(Chunk.id, Chunk.document_id, Chunk.text)
                .order_by(Chunk.id)
            ).all()
            corpus = [analyze(text) for _, _, text in rows]
            has_terms = any(corpus)  # BM25 divides by the average length
            self._snapshot = _Snapshot(
                fingerprint=fingerprint,
                chunk_ids=np.array([row[0] for row in rows], dtype=np.int64),
                document_ids=np.array([row[1] for row in rows], dtype=object),
                bm25=LuceneIdfBM25(corpus, k1=self._k1, b=self._b) if has_terms else None,
            )
            logger.info(
                "Built BM25 index over %d chunks in %.0f ms",
                len(rows),
                (time.perf_counter() - started) * 1000,
            )
            return self._snapshot

    def search(
        self,
        db: Session,
        query: str,
        top_k: int,
        document_ids: Collection[str] | None = None,
    ) -> list[KeywordHit]:
        """Top-``top_k`` chunks by BM25 score, optionally only from ``document_ids``.

        Chunks sharing no term with the query are never returned.
        """
        terms = analyze(query)
        if not terms or top_k <= 0:
            return []
        snapshot = self._current(db)
        if snapshot.bm25 is None:
            return []

        scores = snapshot.bm25.get_scores(terms)
        if document_ids is not None:
            allowed = np.isin(snapshot.document_ids, list(document_ids))
            scores = np.where(allowed, scores, 0.0)
        matching = np.flatnonzero(scores > 0)
        best = matching[np.argsort(-scores[matching], kind="stable")][:top_k]
        return [
            KeywordHit(chunk_id=int(snapshot.chunk_ids[index]), score=float(scores[index]))
            for index in best
        ]
