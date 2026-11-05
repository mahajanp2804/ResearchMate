import pytest

from app.services.text_cleaning import (
    clean_pages,
    is_heading,
    join_lines,
    normalize_text,
    remove_headers_and_footers,
)


def test_normalize_expands_ligatures_and_drops_invisible_characters() -> None:
    assert normalize_text("eﬃcient ﬁne-tuning") == "efficient fine-tuning"
    assert normalize_text("feder­ated​ learn﻿ing") == "federated learning"
    assert normalize_text("ＧＰＵ\x07") == "GPU"


def test_join_lines_rejoins_words_hyphenated_at_line_breaks() -> None:
    assert join_lines(["a communication-effi-", "cient method"]) == (
        "a communication-efficient method"
    )
    # A capital after the hyphen marks a real compound, so the hyphen stays.
    assert join_lines(["handles non-", "IID data"]) == "handles non-IID data"
    assert join_lines(["first   line", "", "  second line "]) == "first line second line"


@pytest.mark.parametrize(
    "line",
    [
        "Abstract",
        "1 Introduction",
        "3.2 Experimental Setup",
        "4.1.2 Results on CIFAR-10",
        "IV. EXPERIMENTS",
        "References",
        "6 Conclusion and Future Work",
        "A Appendix",
    ],
)
def test_recognises_headings(line: str) -> None:
    assert is_heading(line)


@pytest.mark.parametrize(
    "line",
    [
        "A Survey of Federated Optimization Methods",  # article "A", not a numbered section
        "1 FedAvg 85.3 90.1",  # table row
        "3 clients are sampled per round.",  # sentence
        "2 Clients upload their model updates,",  # wrapped clause
        "We present a new method for federated learning that converges faster",  # too long
        "",
    ],
)
def test_rejects_non_headings(line: str) -> None:
    assert not is_heading(line)


BODY = [
    ["Clients train locally.", "Updates are averaged.", "The server broadcasts."],
    ["Data is non-IID.", "Drift slows convergence.", "A proximal term helps."],
    ["We use MNIST.", "Models are small CNNs.", "Training runs on CPUs."],
    ["Accuracy improves.", "Communication drops.", "Variance is reduced."],
    ["Limitations remain.", "Privacy is not proven.", "Future work follows."],
]


def test_removes_running_headers_footers_and_page_numbers() -> None:
    pages = [
        ["Journal of Federated Learning, Vol. 3", *body, f"Page {number}"]
        for number, body in enumerate(BODY, start=1)
    ]
    cleaned = remove_headers_and_footers(pages)

    for number, (lines, body) in enumerate(zip(cleaned, BODY, strict=True), start=1):
        assert "Journal of Federated Learning, Vol. 3" not in lines
        assert f"Page {number}" not in lines
        assert lines == body


def test_lone_page_numbers_are_removed_even_without_repetition() -> None:
    cleaned = remove_headers_and_footers([["7", "Only page of a short note.", "More text."]])
    assert cleaned == [["Only page of a short note.", "More text."]]


def test_short_documents_keep_repeated_lines() -> None:
    # With fewer than three pages a repeated line could be real content.
    pages = [["Shared line", "Body one."], ["Shared line", "Body two."]]
    assert remove_headers_and_footers(pages) == pages


def test_clean_pages_produces_heading_and_prose_blocks() -> None:
    pages = clean_pages(
        [(1, "Abstract\nWe study federated\nlearning.\n1 Introduction\nClients train locally.")]
    )
    blocks = [(block.is_heading, block.text) for block in pages[0].blocks]
    assert blocks == [
        (True, "Abstract"),
        (False, "We study federated learning."),
        (True, "1 Introduction"),
        (False, "Clients train locally."),
    ]
