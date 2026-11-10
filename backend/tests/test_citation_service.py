import pytest

from app.services.citation_service import parse_marker, validate_citations

SOURCES = ["S1", "S2", "S3"]


@pytest.mark.parametrize(
    ("marker", "ids"),
    [
        ("[S1]", ["S1"]),
        ("[S1, S3]", ["S1", "S3"]),
        ("[S2; S4]", ["S2", "S4"]),
        ("[ s2 ]", ["S2"]),
        ("[S1-S3]", ["S1", "S2", "S3"]),
        ("[S1–S2, S5]", ["S1", "S2", "S5"]),
        ("[S3-S1]", ["S3"]),  # backwards range: keep the first id only
        ("[S1-S99]", ["S1"]),  # absurd range: treated as a mistake
    ],
)
def test_parse_marker(marker: str, ids: list[str]) -> None:
    assert parse_marker(marker) == ids


def test_valid_citations_are_kept_in_order_of_first_use() -> None:
    check = validate_citations("FedProx adds a term [S2]. It helps [S1][S2].", SOURCES)
    assert check.text == "FedProx adds a term [S2]. It helps [S1][S2]."
    assert check.cited_ids == ["S2", "S1"]
    assert check.invalid_ids == []


def test_grouped_markers_are_normalised() -> None:
    check = validate_citations("Both methods differ [S1, S3] and agree [s2; S1].", SOURCES)
    assert check.text == "Both methods differ [S1][S3] and agree [S2][S1]."
    assert check.cited_ids == ["S1", "S3", "S2"]


def test_invented_citation_is_removed_and_reported() -> None:
    check = validate_citations("The papers do not report hardware [S9].", SOURCES)
    assert check.text == "The papers do not report hardware."
    assert check.cited_ids == []
    assert check.invalid_ids == ["S9"]


def test_invalid_ids_are_dropped_from_a_group() -> None:
    check = validate_citations("SCAFFOLD corrects drift [S1, S7] [S8].", SOURCES)
    assert check.text == "SCAFFOLD corrects drift [S1]."
    assert check.cited_ids == ["S1"]
    assert check.invalid_ids == ["S7", "S8"]


def test_range_partially_outside_the_sources() -> None:
    check = validate_citations("Results vary [S2-S5].", SOURCES)
    assert check.text == "Results vary [S2][S3]."
    assert check.invalid_ids == ["S4", "S5"]


def test_answer_without_citations() -> None:
    check = validate_citations("FedProx adds a proximal term.", SOURCES)
    assert check.text == "FedProx adds a proximal term."
    assert (check.cited_ids, check.invalid_ids) == ([], [])


def test_markdown_structure_survives_cleaning() -> None:
    answer = "**Summary** [S9]\n\n- First point [S1]\n- Second point [S4]\n\nDone."
    check = validate_citations(answer, SOURCES)
    assert check.text == "**Summary**\n\n- First point [S1]\n- Second point\n\nDone."


def test_text_that_only_looks_like_a_citation_is_untouched() -> None:
    answer = "See Table [2] and the S1 baseline in [Section 3]."
    assert validate_citations(answer, SOURCES).text == answer
