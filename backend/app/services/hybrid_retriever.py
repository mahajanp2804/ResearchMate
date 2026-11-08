"""Reciprocal Rank Fusion of several ranked lists.

RRF scores each item by its ranks alone: score(d) = sum over lists of
1 / (k + rank(d)), using 1-based ranks. It needs no score normalisation, so
cosine similarities and BM25 scores (which live on different scales) are
never compared directly, and an item found by several retrievers rises above
items found by only one. k (60 by default, from Cormack et al., 2009) dampens
how much a first place outweighs a tenth place.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FusedCandidate:
    chunk_id: int
    score: float
    # 1-based rank in each list that contained this chunk, e.g. {"dense": 3, "bm25": 1}.
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def best_rank(self) -> int:
        return min(self.ranks.values())


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[int]], *, k: int = 60
) -> list[FusedCandidate]:
    """Fuse ranked lists of chunk ids into one list, best first.

    Ties are broken by the best individual rank, then by chunk id, so the
    output is deterministic.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for method, chunk_ids in ranked_lists.items():
        for position, chunk_id in enumerate(chunk_ids, start=1):
            if method in ranks.get(chunk_id, {}):
                continue  # count each list's first occurrence only
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
            ranks.setdefault(chunk_id, {})[method] = position

    fused = [
        FusedCandidate(chunk_id=chunk_id, score=score, ranks=ranks[chunk_id])
        for chunk_id, score in scores.items()
    ]
    fused.sort(key=lambda candidate: (-candidate.score, candidate.best_rank, candidate.chunk_id))
    return fused
