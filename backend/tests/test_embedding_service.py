import logging

import numpy as np
import pytest

from app.services.embedding_service import (
    get_sentence_transformer_embedder,
    warn_if_chunks_exceed_model,
)
from tests.fakes import HashingEmbedder

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def test_warns_when_chunks_are_longer_than_the_model_reads(caplog) -> None:
    embedder = HashingEmbedder()
    embedder.max_seq_length = 256
    with caplog.at_level(logging.WARNING):
        warn_if_chunks_exceed_model(embedder, 254)
        assert caplog.records == []
        warn_if_chunks_exceed_model(embedder, 800)
    assert "CHUNK_SIZE_TOKENS=800" in caplog.text


def test_fake_embedder_is_deterministic_and_normalised() -> None:
    embedder = HashingEmbedder()
    vectors = embedder.embed_documents(["federated averaging", "federated averaging", ""])
    np.testing.assert_allclose(np.linalg.norm(vectors[:2], axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(vectors[0], vectors[1])
    assert not vectors[2].any()


@pytest.fixture(scope="module")
def embedder():
    return get_sentence_transformer_embedder(MODEL, device="cpu", batch_size=8)


@pytest.mark.models
class TestSentenceTransformerEmbedder:
    def test_reports_model_properties(self, embedder) -> None:
        assert embedder.name == MODEL
        assert embedder.dimension == 384
        assert embedder.max_seq_length == 256

    def test_vectors_are_normalised_float32(self, embedder) -> None:
        vectors = embedder.embed_documents(["FedProx", "FedAvg", "SCAFFOLD"])
        assert vectors.shape == (3, 384)
        assert vectors.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, rtol=1e-5)

    def test_paraphrases_score_higher_than_unrelated_text(self, embedder) -> None:
        query = embedder.embed_query("How does FedProx handle heterogeneous clients?")
        related, unrelated = embedder.embed_documents(
            [
                "FedProx adds a proximal term so devices with non-IID data stay close "
                "to the global model.",
                "Preheat the oven and mix two cups of flour with sugar.",
            ]
        )
        assert float(query @ related) > float(query @ unrelated) + 0.2

    def test_loaded_once_per_configuration(self, embedder) -> None:
        assert get_sentence_transformer_embedder(MODEL, device="cpu", batch_size=8) is embedder

    def test_empty_input(self, embedder) -> None:
        assert embedder.embed_documents([]).shape == (0, 384)
