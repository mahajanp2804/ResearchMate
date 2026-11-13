"""The bundled sample collection and evaluation datasets.

Files under ``backend/evaluation/``:

- ``sample_papers.json``: five short synthetic papers (fictional methods and
  results). They are turned into PDFs on demand, so evaluation is
  reproducible without checking binary or copyrighted papers into the repo.
- ``datasets/*.json``: curated questions. Each question names the passages
  that answer it as exact ``evidence`` text, plus a reference answer and a
  category. Papers are referred to by title, so a dataset also works over a
  user's own papers.

Evidence is matched as text (ignoring spacing and punctuation) rather than
by chunk id, because chunk ids change whenever chunking settings change.
"""

import json
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.config import BACKEND_DIR
from app.exceptions import NotFoundError

EVALUATION_DIR = BACKEND_DIR / "evaluation"
SAMPLE_PAPERS_FILE = EVALUATION_DIR / "sample_papers.json"
DATASETS_DIR = EVALUATION_DIR / "datasets"

QuestionCategory = Literal["factual", "paraphrase", "keyword", "multi_paper", "unanswerable"]
CATEGORY_LABELS: dict[str, str] = {
    "factual": "Factual",
    "paraphrase": "Paraphrase",
    "keyword": "Exact term or number",
    "multi_paper": "Needs two papers",
    "unanswerable": "Not answerable",
}


class SamplePaper(BaseModel):
    key: str
    title: str
    authors: str
    pages: list[str] = Field(min_length=1)


class SampleCollection(BaseModel):
    description: str
    papers: list[SamplePaper]


class Evidence(BaseModel):
    paper: str  # key into the dataset's papers
    text: str = Field(min_length=10)


class QuestionSpec(BaseModel):
    key: str
    category: QuestionCategory
    question: str = Field(min_length=3)
    evidence: list[Evidence]
    reference_answer: str | None = None

    @model_validator(mode="after")
    def _evidence_matches_category(self) -> "QuestionSpec":
        if self.category == "unanswerable" and self.evidence:
            raise ValueError(f"{self.key}: an unanswerable question cannot have evidence")
        if self.category != "unanswerable" and not self.evidence:
            raise ValueError(f"{self.key}: an answerable question needs evidence")
        return self


class DatasetSpec(BaseModel):
    """A dataset file. ``slug`` is its file name without the extension."""

    slug: str = ""
    name: str
    description: str
    # True if the papers can be installed from the bundled sample collection.
    sample_collection: bool = False
    # Paper key -> exact paper title.
    papers: dict[str, str] = Field(min_length=1)
    questions: list[QuestionSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _references_are_valid(self) -> "DatasetSpec":
        keys = [question.key for question in self.questions]
        if len(keys) != len(set(keys)):
            raise ValueError("question keys must be unique")
        for question in self.questions:
            for evidence in question.evidence:
                if evidence.paper not in self.papers:
                    raise ValueError(f"{question.key}: unknown paper {evidence.paper!r}")
        return self


@cache
def load_sample_collection(path: Path = SAMPLE_PAPERS_FILE) -> SampleCollection:
    return SampleCollection.model_validate_json(path.read_text(encoding="utf-8"))


def list_dataset_specs(directory: Path = DATASETS_DIR) -> list[DatasetSpec]:
    return [load_dataset_spec(path.stem, directory) for path in sorted(directory.glob("*.json"))]


def load_dataset_spec(slug: str, directory: Path = DATASETS_DIR) -> DatasetSpec:
    path = directory / f"{slug}.json"
    # The slug comes from a request: only plain names inside the folder.
    if not slug.replace("_", "").replace("-", "").isalnum() or not path.is_file():
        raise NotFoundError("Evaluation dataset not found.", details={"dataset": slug})
    data = json.loads(path.read_text(encoding="utf-8"))
    return DatasetSpec.model_validate({**data, "slug": slug})


def build_sample_pdf(paper: SamplePaper) -> bytes:
    """A simple A4 PDF of a sample paper, with its title and authors as metadata."""
    import pymupdf

    width, height, margin = 595, 842, 64
    document = pymupdf.open()
    for number, body in enumerate(paper.pages):
        page = document.new_page(width=width, height=height)
        top = margin
        if number == 0:
            title_box = pymupdf.Rect(margin, top, width - margin, top + 80)
            if page.insert_textbox(title_box, paper.title, fontsize=16) < 0:
                raise ValueError(f"The title of {paper.key} does not fit on the page")
            author_box = pymupdf.Rect(margin, top + 84, width - margin, top + 104)
            page.insert_textbox(author_box, paper.authors, fontsize=10)
            top += 116
        body_box = pymupdf.Rect(margin, top, width - margin, height - margin)
        # insert_textbox writes nothing at all when the text does not fit.
        if page.insert_textbox(body_box, body, fontsize=9.5) < 0:
            raise ValueError(f"Page {number + 1} of {paper.key} does not fit on one PDF page")
    document.set_metadata(
        {
            "title": paper.title,
            "author": paper.authors,
            "subject": "Synthetic sample paper for ResearchMate evaluation",
        }
    )
    data = document.tobytes()
    document.close()
    return data
