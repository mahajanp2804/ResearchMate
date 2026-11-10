import pytest

from app.services.citation_service import (
    SourcePassage,
    best_excerpt,
    build_citations,
    citation_coverage,
    locate_page,
    parse_marker,
    statements_citing,
    validate_citations,
)

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


# --- Excerpts and pages --------------------------------------------------------

PASSAGE = (
    "2 Proposed Method\nFedProx tolerates partial work: each device may perform a variable "
    "number of local updates. The proximal term limits the impact of local updates by keeping "
    "them close to the global model. We prove convergence under bounded dissimilarity."
)


def test_statements_are_grouped_by_the_ids_they_cite() -> None:
    answer = "FedProx adds a proximal term [S1]. It converges [S1][S2].\n- Devices vary [S2]"
    assert statements_citing(answer) == {
        "S1": ["FedProx adds a proximal term.", "It converges."],
        "S2": ["It converges.", "Devices vary"],
    }


def test_excerpt_is_the_best_matching_sentence() -> None:
    excerpt, support = best_excerpt(
        PASSAGE, ["The proximal term keeps local updates close to the global model."]
    )
    assert excerpt.startswith("The proximal term limits the impact")
    assert support > 0.7


def test_excerpt_for_an_unrelated_claim_has_low_support() -> None:
    excerpt, support = best_excerpt(PASSAGE, ["GPUs were rented from a cloud provider."])
    assert excerpt  # still returns the passage's first sentence
    assert support < 0.3


def test_short_best_sentence_is_extended_and_long_ones_shortened() -> None:
    short = "See Table 2. Accuracy reaches 0.95 on MNIST with 100 clients in every setting."
    excerpt, _ = best_excerpt(short, ["Table 2"])
    assert excerpt.startswith("See Table 2. Accuracy reaches")
    long_text = "Federated " + "learning " * 120 + "."
    excerpt, _ = best_excerpt(long_text, ["federated learning"])
    assert len(excerpt) <= 401 and excerpt.endswith("…")


def test_single_page_passage_uses_its_page() -> None:
    assert locate_page({}, "anything", 4, 4) == 4


def test_multi_page_passage_finds_the_excerpt_page() -> None:
    pages = {
        3: "Clients train locally. The server aggregates the up-",
        4: "dates from every client. The proximal term limits the im-\npact of local updates.",
    }
    # Hyphenated line breaks and spacing in the raw page do not prevent a match.
    assert locate_page(pages, "The proximal term limits the impact of local updates.", 3, 4) == 4
    assert locate_page(pages, "Clients train locally.", 3, 4) == 3


def test_ligatures_in_raw_text_still_match() -> None:
    pages = {1: "Intro.", 2: "We study eﬃcient ﬁne-tuning of models."}
    assert locate_page(pages, "We study efficient fine-tuning of models.", 1, 2) == 2


def test_unfound_excerpt_falls_back_to_the_first_page() -> None:
    assert locate_page({1: "abc", 2: "def"}, "Something else entirely.", 1, 2) == 1


def test_build_citations_resolves_each_cited_source() -> None:
    sources = [
        SourcePassage("S1", PASSAGE, 2, 2),
        SourcePassage("S2", "SCAFFOLD corrects client drift with control variates.", 1, 1),
        SourcePassage("S3", "Unused passage about datasets.", 5, 5),
    ]
    answer = "SCAFFOLD uses control variates [S2]. FedProx tolerates partial work [S1]."
    citations = build_citations(answer, sources)
    assert [citation.source_id for citation in citations] == ["S2", "S1"]
    assert citations[1].excerpt.startswith("FedProx tolerates partial work")
    assert citations[1].page == 2
    assert all(citation.support > 0.5 for citation in citations)


# --- Coverage ------------------------------------------------------------------


def test_fully_cited_answer() -> None:
    coverage = citation_coverage(
        "FedProx adds a proximal term to the objective [S1]. It tolerates partial work [S2]."
    )
    assert coverage.status == "cited"
    assert coverage.cited_statements == 2
    assert coverage.uncited_statements == []


def test_partially_cited_answer_lists_the_unsupported_claims() -> None:
    coverage = citation_coverage(
        "FedProx adds a proximal term to the objective [S1]. "
        "It was later adopted by every major federated learning framework."
    )
    assert coverage.status == "partially_cited"
    assert coverage.uncited_statements == [
        "It was later adopted by every major federated learning framework."
    ]


def test_answer_without_any_citation_is_uncited() -> None:
    coverage = citation_coverage("FedProx converges faster than FedAvg on every benchmark tested.")
    assert coverage.status == "uncited"
    assert coverage.cited_statements == 0


def test_statements_about_missing_information_need_no_citation() -> None:
    answer = (
        "SCAFFOLD uses control variates to correct drift [S1]. "
        "The papers do not report the hardware used for training. "
        "Runtime costs are not specified in the sources. "
        "There are no details about the learning rate schedule."
    )
    assert citation_coverage(answer).status == "cited"


def test_headings_list_intros_and_fragments_are_not_claims() -> None:
    answer = (
        "### Key findings\n"
        "**In short**\n"
        "The two methods differ in the following ways:\n"
        "- FedProx adds a proximal term to the local objective [S1]\n"
        "- SCAFFOLD corrects client drift with control variates [S2]"
    )
    coverage = citation_coverage(answer)
    assert coverage.status == "cited"
    assert coverage.cited_statements == 2
