import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.config import BACKEND_DIR
from app.services.vector_store import VectorStore
from tests.fakes import HashingEmbedder

EMBEDDER = HashingEmbedder()
TEXTS = {
    101: "FedProx adds a proximal term for heterogeneous clients",
    102: "FedAvg averages local model updates from many devices",
    103: "SCAFFOLD uses control variates to correct client drift",
    104: "The dataset is partitioned in a non-IID way across clients",
}


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    store = VectorStore(tmp_path / "indexes")
    store.ensure_model(EMBEDDER.name, EMBEDDER.dimension)
    store.add(list(TEXTS), EMBEDDER.embed_documents(list(TEXTS.values())))
    return store


def query(text: str) -> np.ndarray:
    return EMBEDDER.embed_query(text)


def test_empty_store_returns_nothing(tmp_path: Path) -> None:
    store = VectorStore(tmp_path)
    assert store.vector_count == 0
    assert store.ids() == set()
    assert store.search(query("anything"), 5) == []


def test_search_ranks_the_most_similar_chunk_first(store: VectorStore) -> None:
    hits = store.search(query("control variates correct client drift"), 3)
    assert hits[0].chunk_id == 103
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_identical_text_scores_cosine_one(store: VectorStore) -> None:
    hit = store.search(query(TEXTS[102]), 1)[0]
    assert hit.chunk_id == 102
    assert hit.score == pytest.approx(1.0, abs=1e-5)


def test_k_larger_than_the_index_returns_everything(store: VectorStore) -> None:
    assert {hit.chunk_id for hit in store.search(query("clients"), 50)} == set(TEXTS)


def test_allowed_ids_restrict_the_search(store: VectorStore) -> None:
    hits = store.search(query(TEXTS[103]), 10, allowed_ids={101, 102})
    assert {hit.chunk_id for hit in hits} == {101, 102}
    assert store.search(query(TEXTS[103]), 10, allowed_ids=set()) == []


def test_remove_deletes_only_the_given_ids(store: VectorStore) -> None:
    assert store.remove({102, 104, 999}) == 2
    assert store.ids() == {101, 103}
    assert all(hit.chunk_id != 102 for hit in store.search(query(TEXTS[102]), 10))


def test_adding_an_existing_id_replaces_its_vector(store: VectorStore) -> None:
    store.add([101], EMBEDDER.embed_documents(["completely different words about cooking"]))
    assert store.vector_count == len(TEXTS)
    assert store.search(query("cooking"), 1)[0].chunk_id == 101


def test_index_survives_a_restart(store: VectorStore, tmp_path: Path) -> None:
    before = store.search(query("non-IID partitions"), 4)
    reloaded = VectorStore(tmp_path / "indexes")
    assert reloaded.model_name == EMBEDDER.name
    assert reloaded.dimension == EMBEDDER.dimension
    assert reloaded.ids() == set(TEXTS)
    assert reloaded.search(query("non-IID partitions"), 4) == before


def test_same_model_keeps_vectors(store: VectorStore) -> None:
    assert store.ensure_model(EMBEDDER.name, EMBEDDER.dimension) is False
    assert store.vector_count == len(TEXTS)


def test_different_model_discards_vectors(store: VectorStore, tmp_path: Path) -> None:
    assert store.ensure_model("another-model", 32) is True
    assert store.vector_count == 0
    assert VectorStore(tmp_path / "indexes").model_name == "another-model"


def test_unreadable_index_starts_empty(store: VectorStore, tmp_path: Path) -> None:
    store.index_path.write_bytes(b"not a faiss index")
    reloaded = VectorStore(tmp_path / "indexes")
    assert reloaded.vector_count == 0
    assert reloaded.model_name is None


def test_wrong_vector_shape_is_rejected(store: VectorStore) -> None:
    with pytest.raises(ValueError, match="shape"):
        store.add([200], np.zeros((1, 8), dtype=np.float32))


def test_add_requires_a_model(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        VectorStore(tmp_path).add([1], np.zeros((1, 4), dtype=np.float32))


@pytest.mark.models
def test_pytorch_and_faiss_work_together_in_one_process() -> None:
    # On macOS the two libraries ship separate OpenMP runtimes; the wrong setup
    # segfaults, hangs, or could return wrong results. Run in a subprocess so a
    # crash or hang fails this test instead of the test session, compare FAISS
    # with NumPy's exact scores, and repeat the work on worker threads the way
    # FastAPI runs requests and background tasks.
    script = """
from concurrent.futures import ThreadPoolExecutor
from app.services.vector_store import VectorStore
from app.services.embedding_service import get_sentence_transformer_embedder
import pathlib, tempfile
import numpy as np
embedder = get_sentence_transformer_embedder(
    "sentence-transformers/all-MiniLM-L6-v2", device="cpu", batch_size=32)
rng = np.random.default_rng(0)
words = ["federated", "clients", "server", "gradient", "privacy", "drift", "proximal"]
texts = [" ".join(rng.choice(words, 12)) for _ in range(300)]

def round_trip():  # embed -> index -> search, checked against NumPy
    vectors = embedder.embed_documents(texts)
    store = VectorStore(pathlib.Path(tempfile.mkdtemp()))
    store.ensure_model(embedder.name, embedder.dimension)
    store.add(list(range(len(texts))), vectors)
    for row in range(0, 300, 37):
        hits = store.search(embedder.embed_query(texts[row]), 5)
        exact = np.sort(vectors @ vectors[row])[::-1][:5]
        assert np.allclose([hit.score for hit in hits], exact, atol=1e-4)
    return True

assert round_trip()  # main thread
with ThreadPoolExecutor(max_workers=3) as pool:  # worker threads, concurrently
    assert all(future.result() for future in [pool.submit(round_trip) for _ in range(4)])
print("ok")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=BACKEND_DIR,
            capture_output=True,
            text=True,
            timeout=240,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("PyTorch + FAISS hung (OpenMP deadlock)")
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().endswith("ok")
