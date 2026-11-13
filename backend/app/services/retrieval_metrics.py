"""Retrieval metrics: how well a ranked list of passages covers a question's evidence.

Relevance is decided from the dataset's evidence, not by a model: a
retrieved passage is relevant if it comes from the evidence's paper and
contains the evidence text (compared on letters and digits only, so line
breaks, hyphenation and punctuation don't matter). Chunks keep whole
sentences, so a sentence of evidence is never split between two passages.

For a cut-off K:
- Precision@K: share of the top K passages that are relevant. With one
  relevant passage and K = 5 the best possible value is 0.2.
- Recall@K: share of the question's evidence passages found in the top K.
- Hit@K: 1 if any of the top K passages is relevant, else 0.
- Reciprocal rank: 1 / rank of the first relevant passage (0 if none was
  retrieved). Its mean over questions is MRR.

Questions without evidence (unanswerable ones) have no retrieval metrics.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean

from app.services.citation_service import letters_and_digits


@dataclass(frozen=True)
class Passage:
    document_id: str
    text: str


@dataclass(frozen=True)
class RankJudgement:
    """Which retrieved passages are relevant, and where each evidence item first appears."""

    # One flag per retrieved passage, best first.
    relevant: list[bool]
    # For each evidence item, the 1-based rank of the first passage containing
    # it, or None if it was not retrieved.
    evidence_ranks: list[int | None]

    @property
    def first_relevant_rank(self) -> int | None:
        return next((rank for rank, hit in enumerate(self.relevant, start=1) if hit), None)

    @property
    def reciprocal_rank(self) -> float:
        rank = self.first_relevant_rank
        return 1.0 / rank if rank else 0.0

    def precision_at(self, k: int) -> float:
        return sum(self.relevant[:k]) / k

    def recall_at(self, k: int) -> float:
        found = sum(1 for rank in self.evidence_ranks if rank is not None and rank <= k)
        return found / len(self.evidence_ranks)

    def hit_at(self, k: int) -> float:
        return 1.0 if any(self.relevant[:k]) else 0.0


def judge_ranking(ranked: Sequence[Passage], evidence: Sequence[Passage]) -> RankJudgement:
    """Mark each retrieved passage relevant or not, given the evidence passages."""
    if not evidence:
        raise ValueError("Retrieval metrics need at least one evidence passage")
    needles = [(item.document_id, letters_and_digits(item.text)) for item in evidence]
    haystacks = [(passage.document_id, letters_and_digits(passage.text)) for passage in ranked]

    relevant = [
        any(doc == evidence_doc and needle in text for evidence_doc, needle in needles)
        for doc, text in haystacks
    ]
    evidence_ranks = [
        next(
            (
                rank
                for rank, (doc, text) in enumerate(haystacks, start=1)
                if doc == evidence_doc and needle in text
            ),
            None,
        )
        for evidence_doc, needle in needles
    ]
    return RankJudgement(relevant=relevant, evidence_ranks=evidence_ranks)


def summarise(judgements: Iterable[RankJudgement], ks: Sequence[int]) -> dict[str, float | int]:
    """Mean metrics over questions: precision@k, recall@k and hit@k for each k, and mrr."""
    judgements = list(judgements)
    if not judgements:
        return {"questions": 0}
    summary: dict[str, float | int] = {"questions": len(judgements)}
    for k in ks:
        summary[f"precision@{k}"] = round(fmean(j.precision_at(k) for j in judgements), 4)
        summary[f"recall@{k}"] = round(fmean(j.recall_at(k) for j in judgements), 4)
        summary[f"hit@{k}"] = round(fmean(j.hit_at(k) for j in judgements), 4)
    summary["mrr"] = round(fmean(j.reciprocal_rank for j in judgements), 4)
    return summary
