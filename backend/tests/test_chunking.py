import re

import pytest

from app.services.chunking_service import (
    ChunkDraft,
    chunk_pages,
    chunking_signature,
    split_sentences,
)
from tests.fakes import WhitespaceTokenizer

TOKENIZER = WhitespaceTokenizer()
MARKER = re.compile(r"\bp(\d+)s(\d+)\b")


WORDS = ["federated", "clients", "servers", "gradients", "rounds", "updates", "models"]


def sentence(page: int, index: int) -> str:
    # 6 whitespace tokens, with a marker recording where it came from. The
    # wording varies so lines don't look like a repeated running header.
    first = WORDS[(page * 3 + index) % len(WORDS)]
    second = WORDS[(page + index * 2) % len(WORDS)]
    return f"Marker p{page}s{index} {first} and {second} improve."


def page_text(page: int, sentences: int) -> str:
    # Hard-wrapped like real PDF text: one sentence per line.
    return "\n".join(sentence(page, index) for index in range(sentences))


def make_pages(page_count: int = 5, sentences_per_page: int = 20) -> list[tuple[int, str]]:
    return [(page, page_text(page, sentences_per_page)) for page in range(1, page_count + 1)]


def chunk(pages, size=50, overlap=10) -> list[ChunkDraft]:
    return chunk_pages(pages, TOKENIZER, chunk_size=size, chunk_overlap=overlap)


def markers(text: str) -> list[tuple[int, int]]:
    return [(int(page), int(index)) for page, index in MARKER.findall(text)]


def test_chunks_respect_the_token_limit() -> None:
    chunks = chunk(make_pages(), size=50, overlap=10)
    assert len(chunks) > 10
    for draft in chunks:
        assert draft.token_count <= 50
        assert draft.token_count == TOKENIZER.count_many([draft.text])[0]
        assert draft.char_count == len(draft.text)


def test_page_range_matches_where_the_text_came_from() -> None:
    for draft in chunk(make_pages(), size=50, overlap=10):
        pages = [page for page, _ in markers(draft.text)]
        assert draft.page_start == min(pages)
        assert draft.page_end == max(pages)


def test_no_text_is_lost_or_reordered() -> None:
    chunks = chunk(make_pages(page_count=3, sentences_per_page=12), size=40, overlap=10)
    seen: list[tuple[int, int]] = []
    for draft in chunks:
        for marker in markers(draft.text):
            if marker not in seen:
                seen.append(marker)
    expected = [(page, index) for page in range(1, 4) for index in range(12)]
    assert seen == expected


def test_consecutive_chunks_share_whole_sentences_up_to_the_overlap() -> None:
    chunks = chunk(make_pages(page_count=2), size=50, overlap=15)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        previous_sentences = split_sentences(previous.text)
        current_sentences = split_sentences(current.text)
        shared = 0
        for size in range(1, len(previous_sentences)):
            if previous_sentences[-size:] == current_sentences[:size]:
                shared = size
        shared_tokens = sum(TOKENIZER.count_many(previous_sentences[-shared:])) if shared else 0
        assert 0 < shared_tokens <= 15


def test_zero_overlap_shares_nothing() -> None:
    chunks = chunk(make_pages(page_count=2), size=50, overlap=0)
    all_markers = [marker for draft in chunks for marker in markers(draft.text)]
    assert len(all_markers) == len(set(all_markers)) == 40


def test_chunk_indexes_are_sequential_and_output_is_deterministic() -> None:
    first = chunk(make_pages(), size=60, overlap=10)
    assert [draft.chunk_index for draft in first] == list(range(len(first)))
    assert chunk(make_pages(), size=60, overlap=10) == first


def test_sentence_cut_by_a_page_break_is_rejoined_and_spans_both_pages() -> None:
    pages = [
        (1, "Clients train locally. The server aggregates the"),
        (2, "updates from every client. Training then continues."),
    ]
    chunks = chunk(pages, size=100, overlap=0)
    assert len(chunks) == 1
    assert "The server aggregates the updates from every client." in chunks[0].text
    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_word_hyphenated_across_a_page_break_is_rejoined() -> None:
    pages = [(1, "We propose a communication-effi-"), (2, "cient algorithm for training.")]
    text = chunk(pages, size=100, overlap=0)[0].text
    assert "communication-efficient algorithm" in text


def test_last_chunk_can_be_on_a_single_later_page() -> None:
    pages = [(1, page_text(1, 10)), (2, page_text(2, 10))]
    last = chunk(pages, size=30, overlap=0)[-1]
    assert (last.page_start, last.page_end) == (2, 2)


def test_section_heading_starts_a_new_chunk_without_overlap() -> None:
    introduction = " ".join(sentence(1, index) for index in range(6))
    method = " ".join(sentence(2, index) for index in range(6))
    pages = [(1, f"1 Introduction\n{introduction}"), (2, f"2 Method\n{method}")]

    # The introduction (38 tokens) is above the 20% fill needed to close a chunk.
    chunks = chunk(pages, size=120, overlap=30)

    assert [draft.section for draft in chunks] == ["1 Introduction", "2 Method"]
    assert chunks[1].text.startswith("2 Method\n")
    assert all(page == 2 for page, _ in markers(chunks[1].text))


def test_tiny_section_is_merged_into_the_next() -> None:
    pages = [(1, "Abstract\nShort summary here.\n1 Introduction\n" + page_text(1, 4))]
    chunks = chunk(pages, size=100, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].section == "Abstract"
    assert "1 Introduction" in chunks[0].text


def test_front_matter_chunk_takes_the_first_section_it_contains() -> None:
    pages = [(1, "A Study of Federated Learning\nAbstract\n" + page_text(1, 3))]
    assert chunk(pages, size=100, overlap=0)[0].section == "Abstract"


def test_sentence_longer_than_the_limit_is_split() -> None:
    long_sentence = " ".join(f"word{index}" for index in range(230))
    chunks = chunk([(1, long_sentence)], size=100, overlap=10)
    assert [draft.token_count for draft in chunks] == [100, 100, 30]
    assert " ".join(draft.text for draft in chunks) == long_sentence


def test_running_headers_and_page_numbers_are_not_in_chunks() -> None:
    pages = [
        (page, f"Journal of Federated Learning\n{page_text(page, 5)}\n{page}")
        for page in range(1, 6)
    ]
    for draft in chunk(pages, size=60, overlap=0):
        assert "Journal of Federated Learning" not in draft.text
        assert not re.search(r"(^|\s)\d(\s|$)", draft.text)


def test_empty_input_produces_no_chunks() -> None:
    assert chunk([]) == []
    assert chunk([(1, "   \n  ")]) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "McMahan et al. propose FedAvg. It works on MNIST.",
            ["McMahan et al. propose FedAvg.", "It works on MNIST."],
        ),
        ("See Fig. 3 for details. Accuracy is 0.95 overall.", None),
        ('He said "it converges." Then it diverged.', None),
        ("Results (Tab. 2) improve. J. Smith agrees.", None),
    ],
)
def test_sentence_splitting_handles_abbreviations(text: str, expected) -> None:
    sentences = split_sentences(text)
    if expected is not None:
        assert sentences == expected
    else:
        assert len(sentences) == 2
    assert " ".join(sentences) == text


def test_signature_identifies_tokenizer_and_settings() -> None:
    assert chunking_signature("tok", 250, 50) == "v1:tok:250/50"
