"""The bundled sample papers and evaluation dataset are consistent and usable."""

import json
from collections import Counter
from pathlib import Path

import pymupdf
import pytest
from pydantic import ValidationError

from app.exceptions import NotFoundError
from app.services.citation_service import letters_and_digits
from app.services.evaluation_data import (
    build_sample_pdf,
    list_dataset_specs,
    load_dataset_spec,
    load_sample_collection,
)

SAMPLE = load_sample_collection()
DATASET = load_dataset_spec("sample_papers")
PAPERS = {paper.key: paper for paper in SAMPLE.papers}


def test_bundled_dataset_covers_every_kind_of_question() -> None:
    assert [spec.slug for spec in list_dataset_specs()] == ["sample_papers"]
    assert DATASET.sample_collection and len(DATASET.questions) == 35
    assert Counter(question.category for question in DATASET.questions) == {
        "factual": 12,
        "paraphrase": 10,
        "keyword": 6,
        "multi_paper": 3,
        "unanswerable": 4,
    }
    # The dataset's papers are exactly the sample collection.
    assert DATASET.papers == {paper.key: paper.title for paper in SAMPLE.papers}


@pytest.mark.parametrize("question", DATASET.questions, ids=lambda question: question.key)
def test_every_evidence_passage_is_in_its_paper(question) -> None:
    for evidence in question.evidence:
        pages = PAPERS[evidence.paper].pages
        found = [
            number
            for number, page in enumerate(pages, start=1)
            if letters_and_digits(evidence.text) in letters_and_digits(page)
        ]
        assert found, f"{question.key}: evidence not found in {evidence.paper}"


@pytest.mark.parametrize("paper", SAMPLE.papers, ids=lambda paper: paper.key)
def test_sample_pdfs_keep_their_text_and_title(paper) -> None:
    document = pymupdf.open(stream=build_sample_pdf(paper), filetype="pdf")
    assert document.page_count == len(paper.pages)
    assert document.metadata["title"] == paper.title
    extracted = letters_and_digits(" ".join(page.get_text() for page in document))
    for question in DATASET.questions:
        for evidence in question.evidence:
            if evidence.paper == paper.key:
                assert letters_and_digits(evidence.text) in extracted, question.key


def write_dataset(directory: Path, **changes) -> str:
    data = json.loads(
        (Path(__file__).parents[1] / "evaluation/datasets/sample_papers.json").read_text()
    )
    data.update(changes)
    (directory / "custom.json").write_text(json.dumps(data))
    return "custom"


def test_an_unanswerable_question_cannot_have_evidence(tmp_path: Path) -> None:
    questions = [
        {
            "key": "x",
            "category": "unanswerable",
            "question": "Anything?",
            "evidence": [{"paper": "proxavg", "text": "Some supporting text."}],
        }
    ]
    with pytest.raises(ValidationError, match="unanswerable"):
        load_dataset_spec(write_dataset(tmp_path, questions=questions), tmp_path)


def test_evidence_must_name_a_paper_of_the_dataset(tmp_path: Path) -> None:
    questions = [
        {
            "key": "x",
            "category": "factual",
            "question": "Anything?",
            "evidence": [{"paper": "missing", "text": "Some supporting text."}],
        }
    ]
    with pytest.raises(ValidationError, match="unknown paper"):
        load_dataset_spec(write_dataset(tmp_path, questions=questions), tmp_path)


@pytest.mark.parametrize("slug", ["missing", "../sample_papers", "sample papers"])
def test_unknown_or_unsafe_dataset_names_are_not_found(slug: str) -> None:
    with pytest.raises(NotFoundError):
        load_dataset_spec(slug)
