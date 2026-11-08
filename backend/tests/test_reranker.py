import pytest

from app.services.reranker import get_cross_encoder_reranker
from tests.fakes import OverlapReranker

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

PASSAGES = {
    "objective": "We propose FedProx, a generalization of FedAvg. FedProx adds a proximal "
    "term to the local objective, which improves stability.",
    "partial": "FedProx tolerates partial work: each device may perform a variable number of "
    "local updates, and the proximal term keeps them close to the global model.",
    "datasets": "We evaluate on MNIST, FEMNIST, Shakespeare and Sent140 with non-IID "
    "partitions. Models are multinomial logistic regression and LSTMs.",
    "limits": "Our analysis assumes bounded dissimilarity, which may not hold in all "
    "federations. Future work includes adaptive selection of the proximal coefficient.",
    "drift": "Federated Averaging suffers from client drift when data is heterogeneous. "
    "SCAFFOLD uses control variates to correct for the drift in local updates.",
    "cooking": "Preheat the oven and mix two cups of flour with sugar and butter.",
}


def test_fake_reranker_scores_by_overlap() -> None:
    reranker = OverlapReranker()
    scores = reranker.score("client drift", ["client drift happens", "drift", "nothing"])
    assert scores == pytest.approx([9.0, -1.0, -11.0])
    assert reranker.calls == [("client drift", 3)]


@pytest.fixture(scope="module")
def reranker():
    return get_cross_encoder_reranker(MODEL, device="cpu")


@pytest.mark.models
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What does FedProx add to the local objective?", "objective"),
        ("Which datasets were used in the experiments?", "datasets"),
        ("What are the limitations of the analysis?", "limits"),
        ("How does SCAFFOLD correct client drift?", "drift"),
        ("How do devices with limited computation participate?", "partial"),
    ],
)
def test_cross_encoder_ranks_the_answering_passage_first(reranker, question, expected) -> None:
    names = list(PASSAGES)
    scores = reranker.score(question, list(PASSAGES.values()))
    assert names[scores.index(max(scores))] == expected


@pytest.mark.models
def test_absolute_scores_do_not_separate_answerable_questions(reranker) -> None:
    # Why MIN_RERANK_SCORE is off by default: this answerable paraphrase scores
    # below an unanswerable question, so any threshold either refuses the first
    # or admits the second. Ranking, not the absolute score, is what helps.
    passages = list(PASSAGES.values())
    answerable = max(reranker.score("How are slow phones handled during training?", passages))
    unanswerable = max(reranker.score("What GPU was used to train the model?", passages))
    assert answerable < unanswerable


@pytest.mark.models
def test_cross_encoder_is_loaded_once_and_handles_empty_input(reranker) -> None:
    assert get_cross_encoder_reranker(MODEL, device="cpu") is reranker
    assert reranker.score("anything", []) == []
