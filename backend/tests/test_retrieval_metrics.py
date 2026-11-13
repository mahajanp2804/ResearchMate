"""Retrieval metrics on hand-checked examples."""

import pytest

from app.services.retrieval_metrics import Passage, judge_ranking, summarise

EVIDENCE_A = Passage("paper-a", "The proximal term keeps local models close.")
EVIDENCE_B = Passage("paper-b", "Control variates correct client drift.")


def ranked(*items: tuple[str, str]) -> list[Passage]:
    return [Passage(document_id, text) for document_id, text in items]


def test_relevance_ignores_layout_but_not_the_paper() -> None:
    passages = ranked(
        ("paper-b", "Background. The proximal term keeps local models close."),  # wrong paper
        # Line breaks, a line-break hyphen and punctuation don't matter.
        ("paper-a", "Intro text.\nThe proxi-\nmal term keeps local  models close!"),
        ("paper-a", "Unrelated passage about datasets."),
    )
    judgement = judge_ranking(passages, [EVIDENCE_A])
    assert judgement.relevant == [False, True, False]
    assert judgement.first_relevant_rank == 2
    assert judgement.reciprocal_rank == 0.5


def test_metrics_for_one_question_with_two_evidence_passages() -> None:
    passages = ranked(
        ("paper-a", "Other text."),
        ("paper-a", "The proximal term keeps local models close."),
        ("paper-c", "Distractor."),
        ("paper-b", "Control variates correct client drift. More text."),
        ("paper-a", "More other text."),
    )
    judgement = judge_ranking(passages, [EVIDENCE_A, EVIDENCE_B])
    assert judgement.evidence_ranks == [2, 4]

    assert judgement.precision_at(1) == 0.0
    assert judgement.precision_at(3) == pytest.approx(1 / 3)
    assert judgement.precision_at(5) == pytest.approx(2 / 5)
    assert judgement.recall_at(1) == 0.0
    assert judgement.recall_at(3) == 0.5
    assert judgement.recall_at(5) == 1.0
    assert (judgement.hit_at(1), judgement.hit_at(3)) == (0.0, 1.0)


def test_nothing_relevant_retrieved() -> None:
    judgement = judge_ranking(ranked(("paper-c", "Distractor.")), [EVIDENCE_A])
    assert judgement.first_relevant_rank is None and judgement.reciprocal_rank == 0.0
    assert judgement.evidence_ranks == [None]
    # Precision divides by K even when fewer passages were returned.
    assert judgement.precision_at(5) == 0.0


def test_one_passage_containing_two_evidence_items_counts_once_for_precision() -> None:
    both = Passage("paper-a", "The proximal term keeps local models close. Second fact here.")
    judgement = judge_ranking([both], [EVIDENCE_A, Passage("paper-a", "Second fact here.")])
    assert judgement.precision_at(1) == 1.0
    assert judgement.recall_at(1) == 1.0


def test_summary_averages_over_questions() -> None:
    first = judge_ranking(
        ranked(("paper-a", "The proximal term keeps local models close.")), [EVIDENCE_A]
    )
    second = judge_ranking(
        ranked(
            ("paper-c", "x"),
            ("paper-c", "y"),
            ("paper-b", "Control variates correct client drift."),
        ),
        [EVIDENCE_B],
    )
    summary = summarise([first, second], ks=[1, 3])
    assert summary["questions"] == 2
    assert summary["mrr"] == pytest.approx((1 + 1 / 3) / 2, abs=1e-4)
    assert summary["recall@1"] == 0.5 and summary["recall@3"] == 1.0
    assert summary["precision@1"] == 0.5
    assert summary["precision@3"] == pytest.approx((1 / 3 + 1 / 3) / 2, abs=1e-4)
    assert summary["hit@3"] == 1.0
    assert summarise([], ks=[1]) == {"questions": 0}


def test_evidence_is_required() -> None:
    with pytest.raises(ValueError):
        judge_ranking(ranked(("paper-a", "text")), [])
