"""Choose which retrieved passages are sent to the LLM.

Candidates arrive best first. Walking down that list, a passage is kept
unless it falls below the optional evidence threshold, nearly duplicates a
passage already kept, or would exceed the token budget. The kept passages
stay in relevance order; they become the numbered sources [S1], [S2], ...
that answers cite.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # retrieval_service imports this module, so only for type hints
    from app.services.retrieval_service import RetrievedChunk

# Two passages are near-duplicates (e.g. a section repeated in a paper, or the
# same paper uploaded twice in two versions) if they share most of their
# distinct words (Jaccard similarity), or if nearly all words of the shorter
# one appear in the other (it adds almost nothing). Adjacent chunks only share
# their overlap, about 50 of 250 tokens, far below both thresholds.
NEAR_DUPLICATE_SIMILARITY = 0.8
NEAR_DUPLICATE_CONTAINMENT = 0.9

_WORD = re.compile(r"\w+")


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def _is_near_duplicate(first: frozenset[str], second: frozenset[str]) -> bool:
    if not first or not second:
        return False
    shared = len(first & second)
    return (
        shared / len(first | second) >= NEAR_DUPLICATE_SIMILARITY
        or shared / min(len(first), len(second)) >= NEAR_DUPLICATE_CONTAINMENT
    )


@dataclass(frozen=True)
class ContextSelection:
    chunks: list[RetrievedChunk]
    total_tokens: int
    dropped_low_score: int = 0
    dropped_duplicates: int = 0
    dropped_over_budget: int = 0

    @property
    def has_evidence(self) -> bool:
        return bool(self.chunks)


def select_context(
    candidates: Sequence[RetrievedChunk],
    *,
    max_chunks: int,
    max_tokens: int,
    min_score: float | None = None,
) -> ContextSelection:
    """Pick up to ``max_chunks`` passages totalling at most ``max_tokens`` tokens.

    ``min_score`` applies to the candidates' scores (cross-encoder logits
    after reranking); None disables the evidence threshold.
    """
    selected: list[RetrievedChunk] = []
    selected_words: list[frozenset[str]] = []
    total_tokens = 0
    low_score = duplicates = over_budget = 0

    for candidate in candidates:
        if len(selected) == max_chunks:
            break
        if min_score is not None and candidate.score < min_score:
            low_score += 1
            continue
        words = _words(candidate.text)
        if any(_is_near_duplicate(words, kept) for kept in selected_words):
            duplicates += 1
            continue
        if total_tokens + candidate.token_count > max_tokens:
            over_budget += 1  # a shorter passage further down may still fit
            continue
        selected.append(candidate)
        selected_words.append(words)
        total_tokens += candidate.token_count

    return ContextSelection(
        chunks=selected,
        total_tokens=total_tokens,
        dropped_low_score=low_score,
        dropped_duplicates=duplicates,
        dropped_over_budget=over_budget,
    )
