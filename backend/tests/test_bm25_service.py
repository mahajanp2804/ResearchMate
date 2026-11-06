import pytest
from sqlalchemy.orm import Session

from app.models import Chunk, Document, IndexingStatus
from app.services.bm25_service import BM25Service, analyze
from tests.test_models import make_document


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The Clients and the Server", ["client", "server"]),
        ("non-IID data", ["non-iid", "non", "iid", "data"]),
        ("CIFAR-10 accuracy of 0.95", ["cifar-10", "cifar", "10", "accuracy", "0.95"]),
        ("FedProx studies", ["fedprox", "study"]),
        ("processes analyses analysis", ["process", "analyse", "analysis"]),
        ("ﬁne-tuning", ["fine-tuning", "fine", "tuning"]),
        ("a I of the", []),
    ],
)
def test_analyze(text: str, expected: list[str]) -> None:
    assert analyze(text) == expected


CORPUS = {
    "fedprox": [
        "FedProx adds a proximal term to the local objective of each device.",
        "We evaluate on Sent140 and Shakespeare with non-IID partitions.",
    ],
    "fedavg": [
        "Federated averaging trains a model on many devices and averages the updates.",
        "Experiments use MNIST and CIFAR-10 with clients holding different data.",
    ],
    "cooking": ["Bake the bread at high heat until the crust is golden."],
}


def add_paper(db: Session, key: str, texts: list[str], status=IndexingStatus.INDEXED) -> Document:
    document = make_document(key.ljust(64, "0"), title=key, indexing_status=status)
    document.chunks = [
        Chunk(
            chunk_index=index,
            page_start=1,
            page_end=1,
            text=text,
            token_count=len(text.split()),
            char_count=len(text),
        )
        for index, text in enumerate(texts)
    ]
    db.add(document)
    db.commit()
    return document


@pytest.fixture
def papers(db_session: Session) -> dict[str, Document]:
    return {key: add_paper(db_session, key, texts) for key, texts in CORPUS.items()}


@pytest.fixture
def service() -> BM25Service:
    return BM25Service(k1=1.5, b=0.75)


def text_of(db: Session, chunk_id: int) -> str:
    return db.get(Chunk, chunk_id).text


def test_exact_keyword_ranks_first(db_session, papers, service) -> None:
    hits = service.search(db_session, "Which paper uses Sent140?", 5)
    assert "Sent140" in text_of(db_session, hits[0].chunk_id)
    # "Experiments use MNIST" also matches ("uses" -> "use"), but ranks lower.
    assert all(hit.score < hits[0].score for hit in hits[1:])


def test_rare_terms_outweigh_common_ones(db_session, papers, service) -> None:
    hits = service.search(db_session, "proximal term devices", 5)
    assert "proximal" in text_of(db_session, hits[0].chunk_id)
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_plural_and_case_insensitive_matching(db_session, papers, service) -> None:
    hits = service.search(db_session, "CLIENT", 5)
    assert "clients" in text_of(db_session, hits[0].chunk_id)


def test_queries_without_matching_terms_return_nothing(db_session, papers, service) -> None:
    assert service.search(db_session, "quantum chromodynamics", 5) == []
    assert service.search(db_session, "what is the", 5) == []  # only stopwords


def test_document_filter_excludes_other_papers(db_session, papers, service) -> None:
    fedavg_chunks = {chunk.id for chunk in papers["fedavg"].chunks}
    hits = service.search(db_session, "devices data", 10, document_ids={papers["fedavg"].id})
    assert hits
    assert {hit.chunk_id for hit in hits} <= fedavg_chunks
    assert service.search(db_session, "bread", 10, document_ids={papers["fedavg"].id}) == []


def test_papers_that_are_not_indexed_are_not_searched(db_session, papers, service) -> None:
    add_paper(db_session, "pending", ["Zygomorphic flowers are unusual."], IndexingStatus.INDEXING)
    assert service.search(db_session, "zygomorphic", 5) == []


def test_top_k_is_respected(db_session, papers, service) -> None:
    assert len(service.search(db_session, "devices data model", 1)) == 1


def test_index_is_reused_until_the_corpus_changes(db_session, papers, service) -> None:
    service.search(db_session, "fedprox", 5)
    first = service._snapshot
    service.search(db_session, "mnist", 5)
    assert service._snapshot is first

    add_paper(db_session, "scaffold", ["SCAFFOLD corrects client drift with control variates."])
    hits = service.search(db_session, "control variates", 5)
    assert service._snapshot is not first
    assert "SCAFFOLD" in text_of(db_session, hits[0].chunk_id)

    db_session.delete(papers["cooking"])
    db_session.commit()
    assert service.search(db_session, "bread crust", 5) == []


def test_status_change_rebuilds_the_index(db_session, papers, service) -> None:
    assert service.search(db_session, "bread", 5)
    papers["cooking"].indexing_status = IndexingStatus.NOT_INDEXED
    db_session.commit()
    assert service.search(db_session, "bread", 5) == []


def test_empty_corpus(db_session, service) -> None:
    assert service.search(db_session, "anything at all", 5) == []
