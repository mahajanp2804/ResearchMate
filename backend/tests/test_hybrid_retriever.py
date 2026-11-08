import pytest

from app.services.hybrid_retriever import reciprocal_rank_fusion


def test_fusion_arithmetic() -> None:
    fused = reciprocal_rank_fusion({"dense": [1, 2, 3], "bm25": [3, 1, 4]}, k=60)

    scores = {candidate.chunk_id: candidate.score for candidate in fused}
    assert scores[1] == pytest.approx(1 / 61 + 1 / 62)
    assert scores[2] == pytest.approx(1 / 62)
    assert scores[3] == pytest.approx(1 / 63 + 1 / 61)
    assert scores[4] == pytest.approx(1 / 63)
    # Found by both retrievers beats found by one, whatever the single rank.
    assert [candidate.chunk_id for candidate in fused] == [1, 3, 2, 4]


def test_records_rank_in_each_list() -> None:
    fused = reciprocal_rank_fusion({"dense": [1, 2], "bm25": [2]})
    by_id = {candidate.chunk_id: candidate for candidate in fused}
    assert by_id[1].ranks == {"dense": 1}
    assert by_id[2].ranks == {"dense": 2, "bm25": 1}
    assert by_id[2].best_rank == 1


def test_single_list_keeps_its_order() -> None:
    fused = reciprocal_rank_fusion({"dense": [7, 3, 9]})
    assert [candidate.chunk_id for candidate in fused] == [7, 3, 9]


def test_ties_are_broken_deterministically() -> None:
    # Each chunk is first in one list, so all scores tie at 1/61.
    fused = reciprocal_rank_fusion({"dense": [5], "bm25": [2], "other": [9]})
    assert [candidate.chunk_id for candidate in fused] == [2, 5, 9]


def test_duplicates_within_a_list_count_once() -> None:
    fused = reciprocal_rank_fusion({"dense": [1, 1, 2]})
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[0].ranks == {"dense": 1}


def test_smaller_k_rewards_top_ranks_more() -> None:
    # Chunk 1 is first in one list; chunk 2 is fourth in both.
    lists = {"dense": [1, 10, 11, 2], "bm25": [3, 12, 13, 2]}
    # k=1: 1/2 = 0.50 for chunk 1 beats 2/5 = 0.40 for chunk 2.
    assert reciprocal_rank_fusion(lists, k=1)[0].chunk_id == 1
    # k=60: 2/64 = 0.031 for chunk 2 beats 1/61 = 0.016 for chunk 1.
    assert reciprocal_rank_fusion(lists, k=60)[0].chunk_id == 2


def test_empty_and_invalid_input() -> None:
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": [1]}, k=0)
