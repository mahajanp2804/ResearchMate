"""Checks against the real embedding-model tokenizer (downloaded once, then cached)."""

import pytest

from app.services.chunking_service import chunk_pages
from app.services.tokenization import get_huggingface_tokenizer

pytestmark = pytest.mark.models

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def tokenizer():
    return get_huggingface_tokenizer(MODEL)


def test_counts_word_pieces_without_special_tokens(tokenizer) -> None:
    # "fedprox" is split into word pieces: fed ##pro ##x
    assert tokenizer.count_many(["FedProx", "", "data"]) == [3, 0, 1]


def test_split_produces_pieces_within_the_limit(tokenizer) -> None:
    text = " ".join(["heterogeneous federated optimisation"] * 100)
    pieces = tokenizer.split(text, 64)
    assert all(count <= 64 for count in tokenizer.count_many(pieces))
    assert " ".join(pieces).split() == text.split()


def test_chunks_fit_the_model_input_window(tokenizer) -> None:
    page = (
        "Federated learning enables many clients to collaboratively train a model "
        "under the orchestration of a central server, while keeping data decentralized. "
    ) * 40
    chunks = chunk_pages([(1, page), (2, page)], tokenizer, chunk_size=250, chunk_overlap=50)
    assert chunks
    # 250 content tokens + [CLS] + [SEP] fits MiniLM's 256-token window.
    assert all(draft.token_count <= 250 for draft in chunks)
