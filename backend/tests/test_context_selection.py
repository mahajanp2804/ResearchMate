from app.services.context_selection import select_context
from app.services.retrieval_service import RetrievedChunk


def candidate(chunk_id: int, text: str, score: float = 0.0, tokens: int = 100) -> RetrievedChunk:
    return RetrievedChunk(
        rank=chunk_id,
        score=score,
        chunk_id=chunk_id,
        document_id="doc",
        document_title="Paper",
        chunk_index=chunk_id,
        page_start=1,
        page_end=1,
        section=None,
        text=text,
        token_count=tokens,
    )


DISTINCT = [
    "FedProx adds a proximal term to the local objective.",
    "SCAFFOLD corrects client drift with control variates.",
    "Experiments use MNIST and Sent140 with non-IID partitions.",
    "The analysis assumes bounded dissimilarity between devices.",
]


def ids(selection) -> list[int]:
    return [chunk.chunk_id for chunk in selection.chunks]


def test_keeps_relevance_order_up_to_the_chunk_limit() -> None:
    selection = select_context(
        [candidate(i, text) for i, text in enumerate(DISTINCT, start=1)],
        max_chunks=3,
        max_tokens=10_000,
    )
    assert ids(selection) == [1, 2, 3]
    assert selection.total_tokens == 300
    assert selection.has_evidence


def test_near_duplicates_are_dropped() -> None:
    repeated = (
        "FedProx tolerates partial work: each device may perform a variable number of updates."
    )
    selection = select_context(
        [
            candidate(1, repeated),
            candidate(2, repeated.replace("variable", "varying")),  # same passage, one word changed
            candidate(3, DISTINCT[1]),
        ],
        max_chunks=5,
        max_tokens=10_000,
    )
    assert ids(selection) == [1, 3]
    assert selection.dropped_duplicates == 1


def test_passage_contained_in_a_kept_one_is_dropped() -> None:
    # The second passage repeats the first and adds one sentence. Jaccard is
    # 0.73, below the 0.8 threshold, but every word of the first appears in it,
    # so only the containment rule catches it.
    paragraph = (
        "FedProx tolerates partial work: each device may perform a variable number of local "
        "updates. The proximal term limits the impact of local updates by keeping them close "
        "to the global model. We prove convergence under a bounded dissimilarity assumption."
    )
    extended = (
        "Clients upload compressed gradients over slow cellular networks during nightly "
        f"charging windows. {paragraph}"
    )
    selection = select_context(
        [candidate(1, paragraph), candidate(2, extended), candidate(3, DISTINCT[1])],
        max_chunks=5,
        max_tokens=10_000,
    )
    assert ids(selection) == [1, 3]
    assert selection.dropped_duplicates == 1


def test_overlapping_neighbouring_chunks_are_kept() -> None:
    # Consecutive chunks share a sentence of overlap but are otherwise different.
    shared = "The proximal term keeps local models close to the global model."
    first = f"FedProx tolerates partial work from slow devices. {shared}"
    second = f"{shared} We prove convergence under bounded dissimilarity for non-IID data."
    selection = select_context(
        [candidate(1, first), candidate(2, second)], max_chunks=5, max_tokens=10_000
    )
    assert ids(selection) == [1, 2]


def test_token_budget_skips_passages_that_do_not_fit() -> None:
    selection = select_context(
        [
            candidate(1, DISTINCT[0], tokens=200),
            candidate(2, DISTINCT[1], tokens=250),  # would exceed 300
            candidate(3, DISTINCT[2], tokens=90),  # still fits
        ],
        max_chunks=5,
        max_tokens=300,
    )
    assert ids(selection) == [1, 3]
    assert selection.total_tokens == 290
    assert selection.dropped_over_budget == 1


def test_evidence_threshold_drops_weak_passages() -> None:
    selection = select_context(
        [candidate(1, DISTINCT[0], score=4.0), candidate(2, DISTINCT[1], score=-10.5)],
        max_chunks=5,
        max_tokens=10_000,
        min_score=-10.0,
    )
    assert ids(selection) == [1]
    assert selection.dropped_low_score == 1


def test_no_passage_passing_the_threshold_means_no_evidence() -> None:
    selection = select_context(
        [candidate(1, DISTINCT[0], score=-11.0)], max_chunks=5, max_tokens=10_000, min_score=-10.0
    )
    assert selection.chunks == []
    assert not selection.has_evidence


def test_threshold_disabled_by_default() -> None:
    selection = select_context(
        [candidate(1, DISTINCT[0], score=-11.5)], max_chunks=5, max_tokens=10_000
    )
    assert ids(selection) == [1]


def test_no_candidates() -> None:
    selection = select_context([], max_chunks=5, max_tokens=10_000)
    assert not selection.has_evidence
    assert selection.total_tokens == 0
