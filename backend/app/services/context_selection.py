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

# Passages sharing this share of their distinct words (Jaccard similarity)
# count as near-duplicates, e.g. a section repeated in a paper, or the same
# paper uploaded twice in two versions. Adjacent chunks only share their
# overlap (about 50 of 250 tokens), well below this.
NEAR_DUPLICATE_SIMILARITY = 0.8

_WORD = re.compile(r"\w+")


def _words(text: str) -> frozenset[str]:
    return frozenset(_WORD.findall(text.lower()))


def _jaccard(first: frozenset[str], second: frozenset[str]) -> float:
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


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
        if any(_jaccard(words, kept) >= NEAR_DUPLICATE_SIMILARITY for kept in selected_words):
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
